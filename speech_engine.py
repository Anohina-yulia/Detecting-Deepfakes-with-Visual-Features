# -*- coding: utf-8 -*-
"""Распознавание речи: загрузка моделей и фоновый цикл субтитров.

Параметры распознавания (язык, VAD, beam, температура) те же, что в прежнем
app.App.whisper_loop. Изменена схема фиксации текста:

* раньше каждые 1.5 с распознавалось окно 3.8 с, и окна склеивались по
  совпадающим словам. Если окно резало слово ("…ашел"), склейка ломалась:
  в субтитрах появлялись обрывки и повторы, а фраза, не успевшая
  зафиксироваться до выхода из окна, терялась. Повторы к тому же завышали
  «повторы слов» в речевой части индекса отклонения;
* теперь буфер хранит звук от последнего зафиксированного слова, распознаётся
  с метками времени слов, и слово фиксируется, когда два последовательных
  распознавания совпали (схема Local Agreement для потокового Whisper).
  Стоимость одного вызова почти не зависит от длины буфера: Whisper всё равно
  дополняет вход до 30 с.

Также: модели грузятся один раз и переиспользуются между сессиями, фильтр
типичных "галлюцинаций" Whisper, статусы для интерфейса, пауза и остановка.
Всё выполняется локально: звук никуда не отправляется.
"""
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from models import EV_SPEECH, EV_SYSTEM, SessionEvent

MODELS_DISABLED = "выключено"
MODELS_LOADING = "загрузка"
MODELS_READY = "готово"
MODELS_FAILED = "недоступно"

HOP_SECONDS = 1.5          # как часто распознавать накопленный звук
MIN_FIRST_SECONDS = 1.2    # меньше звука в буфере - не распознаём
MAX_BUFFER_SECONDS = 15.0  # длиннее - буфер обрезается по последнему зафиксированному слову
SILENCE_KEEP = 1.0         # сколько звука оставлять, когда в буфере нет речи
PHRASE_PAUSE = 1.2         # пауза между словами, после которой начинается новая фраза, с
PHRASE_IDLE = 2.0          # столько секунд без новых слов - фраза закрывается
PHRASE_MAX_WORDS = 18
PROMPT_WORDS = 25          # сколько последних слов подсказывать модели как контекст
SENTENCE_END = re.compile(r"[.!?…]$")


def load_file_audio(path, samplerate):
    """Звук из видеофайла (декодер входит в faster-whisper). None - звука нет."""
    try:
        from faster_whisper.audio import decode_audio

        audio = np.asarray(decode_audio(path, sampling_rate=samplerate),
                           dtype=np.float32).reshape(-1)
        return audio if len(audio) else None
    except Exception as error:
        print("Не удалось прочитать звук из видеофайла:", error)
        return None


class SpeechModels:
    """Whisper + модель коррекции текста. Загрузка блокирующая - вызывайте в фоне."""

    def __init__(self):
        self.whisper = None
        self.corrector = None
        self.state = MODELS_DISABLED
        self.error = ""
        self.model_size = ""
        self.correction_requested = False
        self._lock = threading.Lock()
        self._thread = None

    @property
    def ready(self):
        return self.state == MODELS_READY and self.whisper is not None

    @property
    def loading(self):
        return self.state == MODELS_LOADING

    @property
    def correction_active(self):
        return self.corrector is not None and getattr(self.corrector, "model", None) is not None

    def disable(self):
        with self._lock:
            if self.state != MODELS_LOADING:
                self.state = MODELS_DISABLED

    def load(self, model_size="small", use_correction=True, allow_download=True):
        from text_correction import SafeMeaningCorrectionModule

        with self._lock:
            self.state = MODELS_LOADING
            self.error = ""
            self.correction_requested = use_correction
        try:
            whisper = self.whisper
            if whisper is None or getattr(whisper, "model_size", None) != model_size:
                from whisper_subtitle import WhisperSubtitleModule

                whisper = WhisperSubtitleModule(model_size=model_size,
                                                allow_download=allow_download)
            corrector = self.corrector
            needs_corrector = (corrector is None
                               or (use_correction and corrector.model is None)
                               or (not use_correction and corrector.model is not None))
            if needs_corrector:
                corrector = SafeMeaningCorrectionModule(allow_download=allow_download,
                                                        load_model=use_correction)
            with self._lock:
                self.whisper, self.corrector = whisper, corrector
                self.model_size = model_size
                self.state = MODELS_READY
        except Exception as error:
            message = str(error) or error.__class__.__name__
            if isinstance(error, ImportError):
                message = "не установлен пакет faster-whisper"
            print("Распознавание речи недоступно:", message)
            with self._lock:
                self.state = MODELS_FAILED
                self.error = message

    def load_async(self, model_size="small", use_correction=True, allow_download=True,
                   on_done=None):
        if self._thread is not None and self._thread.is_alive():
            return False
        with self._lock:  # сразу, чтобы интерфейс не успел показать "не загружена"
            self.state = MODELS_LOADING
            self.error = ""

        def run():
            self.load(model_size, use_correction, allow_download)
            if on_done is not None:
                try:
                    on_done()
                except Exception as error:
                    print("Ошибка уведомления о загрузке моделей:", error)

        self._thread = threading.Thread(target=run, name="speech-models", daemon=True)
        self._thread.start()
        return True


@dataclass
class Word:
    start: float          # время звука от начала подачи, с
    end: float
    text: str
    logprob: Optional[float] = None


def _norm(word):
    return re.sub(r"[^\w]", "", word.lower())


def agreed_prefix(previous, current):
    """Слова, одинаковые в начале двух последовательных распознаваний."""
    stable = []
    for a, b in zip(previous, current):
        if not _norm(b.text) or _norm(a.text) != _norm(b.text):
            break
        stable.append(b)
    return stable


class SpeechSession:
    """Фоновое распознавание речи одной сессии."""

    def __init__(self, models, audio, storage, time_fn, on_event=None, hop_seconds=HOP_SECONDS):
        self.whisper = models.whisper
        self.corrector = models.corrector
        self.audio = audio
        self.storage = storage
        self.time_fn = time_fn
        self.on_event = on_event
        self.sr = audio.samplerate
        self.hop_seconds = hop_seconds

        self.buffer = np.empty(0, dtype=np.float32)
        self.buffer_start = 0.0      # время звука, с которого начинается буфер
        self.audio_time = 0.0        # сколько звука подано всего, с
        self._anchor = None          # время сессии, соответствующее audio_time = 0
        self.pending_samples = 0
        self.committed_until = 0.0   # до этого момента слова зафиксированы
        self.hypothesis = []         # незафиксированные слова прошлого распознавания
        self.phrase = []             # зафиксированные слова текущей фразы
        self.history = []            # последние зафиксированные слова (контекст модели)

        self.speech_queue = queue.Queue()  # сырой текст речи -> индикатор отклонения
        self.running = False
        self.paused = False
        self.phrase_count = 0
        self.last_error = ""
        self._last_error_time = 0.0
        self._reset_requested = False
        self._lock = threading.Lock()
        self._thread = None
        self.whisper.reset()

    # ---------- управление ----------

    def start(self):
        if self._thread is not None:
            return
        self.running = True
        self._thread = threading.Thread(target=self.loop, name="speech", daemon=True)
        self._thread.start()

    def stop(self, timeout=3.0):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                print("Поток распознавания речи не успел завершиться (идёт расшифровка).")
            self._thread = None

    def set_paused(self, paused):
        """Не блокирует вызывающий поток: буфер сбросит сам поток распознавания."""
        self.paused = paused
        if paused:
            self._reset_requested = True

    def _reset_buffer(self):
        self._close_phrase()
        self.hypothesis = []
        self.buffer = np.empty(0, dtype=np.float32)
        self.buffer_start = self.audio_time
        self.committed_until = self.audio_time
        self.pending_samples = 0
        self._anchor = None  # после паузы микрофона в звуке разрыв - привязка заново

    def get_display_lines(self):
        return self.whisper.get_display_lines()

    def drain_speech(self):
        items = []
        while True:
            try:
                items.append(self.speech_queue.get_nowait())
            except queue.Empty:
                return items

    def _emit(self, kind, text, level="info", t=None):
        if self.on_event is None:
            return
        try:
            when = float(self.time_fn()) if t is None else float(t)
            self.on_event(SessionEvent(t=when, kind=kind, text=text, level=level))
        except Exception as error:
            print("Ошибка передачи события речи:", error)

    def _session_time(self, audio_t):
        """Время сессии, когда прозвучал момент audio_t."""
        if self._anchor is None:
            return max(float(self.time_fn()) - (self.audio_time - audio_t), 0.0)
        return max(self._anchor + audio_t, 0.0)

    def _update_anchor(self, chunk_len):
        """Звук подаётся вровень с видео, но ждёт в очереди, пока идёт распознавание:
        конец поданного звука = текущее время сессии, а взятый блок отстаёт от него
        на длину очереди. Задержки обработки только увеличивают оценку - берём минимум."""
        queued = 0.0
        q = getattr(self.audio, "audio_queue", None)
        if q is not None:
            try:
                queued = q.qsize() * chunk_len / self.sr
            except Exception:
                queued = 0.0
        anchor = float(self.time_fn()) - self.audio_time - queued
        self._anchor = anchor if self._anchor is None else min(self._anchor, anchor)

    # ---------- текст ----------

    def should_apply_meaning_correction(self, text, avg_logprob, avg_no_speech):
        if not text or len(text.split()) < 3:
            return False
        return (self.corrector.looks_broken(text)
                or (avg_logprob is not None and avg_logprob < -0.8)
                or (avg_no_speech is not None and avg_no_speech > 0.5))

    def process_live_text(self, text):
        text = self.corrector.basic_cleanup(text)
        words = text.split()
        return " ".join(words[-18:]) if len(words) > 18 else text

    def process_final_text(self, text, avg_logprob, avg_no_speech):
        cleaned = self.corrector.basic_cleanup(text)
        if cleaned and self.should_apply_meaning_correction(cleaned, avg_logprob, avg_no_speech):
            return self.corrector.correct(cleaned)
        return cleaned

    # ---------- распознавание ----------

    def _transcribe(self):
        # контекст - только слова, звука которых в буфере уже нет: иначе модель
        # "пропускает" их при распознавании и сдвигает метки времени соседних слов
        context = [w for w in self.history if w.end <= self.buffer_start + 0.01]
        prompt = " ".join(w.text for w in context[-PROMPT_WORDS:]) or None
        segments, _ = self.whisper.model.transcribe(
            self.buffer,
            language="ru",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=350, speech_pad_ms=120),
            beam_size=1,
            best_of=1,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            word_timestamps=True,
        )
        words = []
        for segment in segments:
            if getattr(segment, "no_speech_prob", 0.0) > 0.75:
                continue
            if not self.whisper.strip_hallucinations(segment.text):
                continue
            logprob = getattr(segment, "avg_logprob", None)
            for w in segment.words or []:
                text = w.word.strip()
                if text and w.end - w.start >= 0.02:  # слова нулевой длины - артефакт конца буфера
                    words.append(Word(self.buffer_start + w.start, self.buffer_start + w.end,
                                      text, logprob))
        return words

    def _commit(self, words):
        for word in words:
            if self.phrase and word.start - self.phrase[-1].end > PHRASE_PAUSE:
                self._close_phrase()
            self.phrase.append(word)
            self.history.append(word)
            if SENTENCE_END.search(word.text) or len(self.phrase) >= PHRASE_MAX_WORDS:
                self._close_phrase()
        self.history = self.history[-100:]

    def _close_phrase(self):
        if not self.phrase:
            return
        words, self.phrase = self.phrase, []
        raw = " ".join(w.text for w in words).strip()
        logprobs = [w.logprob for w in words if w.logprob is not None]
        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else None
        text = self.whisper.cleanup_text(self.process_final_text(raw, avg_logprob, None))
        if not text or not any(len(_norm(w.text)) > 1 for w in words):
            return
        when = self._session_time(words[0].start)
        self.storage.append_final(text, when)
        self.whisper.update_live_text(text)
        self.phrase_count += 1
        self._emit(EV_SPEECH, text, t=when)
        self.speech_queue.put((when, raw))

    def _trim(self, cut_to):
        cut_to = min(max(cut_to, self.buffer_start), self.audio_time)
        drop = int((cut_to - self.buffer_start) * self.sr)
        if drop <= 0:
            return
        self.buffer = self.buffer[drop:]
        self.buffer_start += drop / self.sr
        self.hypothesis = [w for w in self.hypothesis if w.start >= self.buffer_start]

    def _new_words(self, words):
        """Слова после зафиксированного места.

        Метки времени при повторном распознавании сдвигаются на десятые доли
        секунды, поэтому новизна определяется по середине слова, а стык ещё раз
        проверяется по совпадению последних зафиксированных слов.
        """
        fresh = [w for w in words if (w.start + w.end) / 2 > self.committed_until]
        if fresh and self.history and fresh[0].start < self.committed_until + 0.5:
            tail = [_norm(w.text) for w in self.history[-3:]]
            for k in range(min(3, len(tail), len(fresh)), 0, -1):
                if [_norm(w.text) for w in fresh[:k]] == tail[-k:]:
                    return fresh[k:]  # те же слова, что уже зафиксированы
        return fresh

    def _step(self):
        """Одно распознавание буфера: фиксация согласованных слов и обновление живой строки."""
        words = self._new_words(self._transcribe())
        stable = agreed_prefix(self.hypothesis, words)
        if stable:
            self.committed_until = stable[-1].end
            self._commit(stable)
        self.hypothesis = words[len(stable):]

        live = " ".join(w.text for w in self.phrase + self.hypothesis)
        if live:
            shown = self.whisper.update_live_text(self.process_live_text(live))
            if shown:
                self.storage.append_live(shown)

        last_word_end = (self.hypothesis or self.phrase or [None])[-1]
        idle = self.audio_time - (last_word_end.end if last_word_end else self.committed_until)
        if self.phrase and not self.hypothesis and idle > PHRASE_IDLE:
            self._close_phrase()

        if not words and not self.phrase:
            self._trim(self.audio_time - SILENCE_KEEP)            # в буфере тишина
        elif not self.phrase and not self.hypothesis:
            self._trim(self.committed_until)                      # фраза закрыта - начать заново
        elif self.audio_time - self.buffer_start > MAX_BUFFER_SECONDS:
            self._close_phrase()
            self._trim(self.committed_until)

    def _flush(self):
        """Конец сессии: последние слова фиксируются как есть, кроме обрезанного."""
        with self._lock:
            words = list(self.hypothesis)
            while words and (re.search(r"(\.\.\.|…|-)$", words[-1].text)
                             or words[-1].end >= self.audio_time - 0.05):
                words.pop()  # слово оборвано концом звука
            if words:
                self._commit(words)
            self.hypothesis = []
            self._close_phrase()

    def feed(self, chunk):
        """Добавить блок звука; когда накопился шаг - распознать буфер."""
        with self._lock:
            audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
            self.buffer = np.concatenate([self.buffer, audio])
            self.audio_time += len(audio) / self.sr
            self.pending_samples += len(audio)
            self._update_anchor(len(audio))
            if (len(self.buffer) < int(self.sr * MIN_FIRST_SECONDS)
                    or self.pending_samples < int(self.sr * self.hop_seconds)):
                return
            self.pending_samples = 0
            self._step()

    def loop(self):
        while self.running:
            if self._reset_requested:
                self._reset_requested = False
                with self._lock:
                    self._reset_buffer()
            try:
                chunk = self.audio.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.paused:
                continue

            try:
                self.feed(chunk)
            except Exception as error:
                self.last_error = str(error)
                print("Ошибка распознавания речи:", error)
                now = time.time()
                if now - self._last_error_time > 10.0:  # не засыпать журнал одинаковыми ошибками
                    self._last_error_time = now
                    self._emit(EV_SYSTEM, f"Ошибка распознавания речи: {error}", "error")

        try:
            self._flush()
        except Exception as error:
            print("Ошибка при завершении распознавания речи:", error)
