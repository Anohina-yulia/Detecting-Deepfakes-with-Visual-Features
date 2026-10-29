# -*- coding: utf-8 -*-
import os
import queue
import textwrap
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from analysis_logger import AnalysisLogger
from audio_module import AudioModule
from capture_protect import exclude_window_from_capture
from eye_tracker import EyeTrackerModule
from micro_expression import MicroExpressionModule
from mimic_analyzer import MimicAnalyzerModule
from neural_face_analyzer import NeuralFaceAnalyzerModule
from quality_analyzer import QualityAnalyzerModule
from redness_analyzer import RednessAnalyzerModule
from subtitle_storage import SubtitleStorage
from suspicion_analyzer import SuspicionAnalyzerModule
from text_correction import SafeMeaningCorrectionModule
from models import LABEL_LIE, LABEL_TRUTH
from ui_widgets import (GRAY, GREEN, RED, WHITE, YELLOW, DisplayMapper, PanelManager,
                        draw_bar, load_font, new_panel, to_bgra)
from video_source import ScreenSource, VideoFileSource, choose_video_source
from whisper_subtitle import WhisperSubtitleModule

CANVAS_W, CANVAS_H = 1280, 720

# Для видеофайла: воспроизводить в реальном времени, чтобы субтитры (Whisper)
# успевали за звуком. False - обрабатывать как можно быстрее (субтитры отстанут).
PACE_VIDEO_FILE = True

REDNESS_PERIOD = 6       # раз в сколько кадров пересчитывать покраснение
SUSPICION_PERIOD = 5     # ... индикатор напряжения
QUALITY_PERIOD = 3       # ... качество кадра
PANEL_RENDER_PERIOD = 2  # ... перерисовывать панели
QUALITY_PATIENCE_FRAMES = 60  # столько кадров "плохого" качества терпим, потом калибруем всё равно


def make_session_dir(cap):
    """Каждый запуск пишет результаты в свою папку и ничего не затирает."""
    if isinstance(cap, VideoFileSource):
        base = os.path.splitext(os.path.basename(cap.filepath))[0]
    elif isinstance(cap, ScreenSource):
        base = "screen"
    else:
        base = "webcam"
    path = os.path.join("results", f"{base}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(path, exist_ok=True)
    return path


def load_file_audio(path, samplerate):
    """Звук из видеофайла (декодер входит в faster-whisper)."""
    try:
        from faster_whisper.audio import decode_audio

        audio = np.asarray(decode_audio(path, sampling_rate=samplerate),
                           dtype=np.float32).reshape(-1)
        return audio if len(audio) else None
    except Exception as error:
        print("Не удалось прочитать звук из видеофайла:", error)
        return None


class App:
    # Заголовок окна только латиницей: OpenCV на Windows плохо работает
    # с кириллицей в названии окна, а по нему ищется окно для защиты от захвата.
    WINDOW_TITLE = "Mimic Analyzer"

    def __init__(self, cap, participant=""):
        cv2.setUseOptimized(True)

        self.cap = cap
        self.participant = participant
        self.label = ""            # текущая метка разметки: правда / ложь / пусто
        self.label_changed = False
        self.is_screen_mode = isinstance(cap, ScreenSource)
        self.is_video_file = isinstance(cap, VideoFileSource)
        self.window_protected = False
        self.protect_attempts = 0

        self.session_dir = make_session_dir(cap)
        print("Результаты сеанса:", self.session_dir)

        # время на таймлайне (секунды): для файла - время кадра, для камеры - с начала сеанса
        self.t0 = time.time()
        self.video_time = 0.0

        self.video_writer = None
        self.output_video_path = None

        # --- звук и субтитры ---
        self.audio = AudioModule()
        self.whisper = WhisperSubtitleModule(model_size="small")
        self.corrector = SafeMeaningCorrectionModule()
        self.storage = SubtitleStorage(
            live_file=os.path.join(self.session_dir, "subtitles_live.txt"),
            final_file=os.path.join(self.session_dir, "subtitles.txt"),
            time_fn=lambda: self.video_time,
        )
        self.speech_queue = queue.Queue()  # сырой текст речи -> индикатор напряжения

        self.file_audio = None
        self.file_audio_pos = 0
        if self.is_video_file:
            print("Читаю звук из видеофайла...")
            self.file_audio = load_file_audio(cap.filepath, self.audio.samplerate)
            if self.file_audio is None:
                print("Звука в файле нет - субтитров не будет.")

        # --- анализ ---
        self.neural_face = NeuralFaceAnalyzerModule(model_path="models/face_landmarker.task")
        self.eye_tracker = EyeTrackerModule()
        self.mimic_analyzer = MimicAnalyzerModule()
        self.micro_module = MicroExpressionModule()
        self.redness_analyzer = RednessAnalyzerModule()
        self.suspicion_analyzer = SuspicionAnalyzerModule()
        self.quality_analyzer = QualityAnalyzerModule()

        self.analysis_logger = AnalysisLogger(
            filepath=os.path.join(self.session_dir, "analysis_log.csv"),
            min_interval_sec=0.7,
        )

        self.last_quality = None
        self.bad_quality_frames = 0
        self.last_redness = None
        self.last_suspicion = None

        # --- интерфейс ---
        self.panels = PanelManager(CANVAS_W, CANVAS_H, layout_file="panel_layout.json")
        self.panels.register("volume", (15, 8))
        self.panels.register("eyes", (15, 50))
        self.panels.register("redness", (15, 305))
        self.panels.register("mimic", (CANVAS_W - 435, 50))
        self.panels.register("suspicion", (CANVAS_W - 435, 345))
        self.panels.register("subtitles", ((CANVAS_W - 560) // 2, CANVAS_H - 82 - 12))
        self.panels.register("label", (15, 600))
        self.mapper = DisplayMapper(self.WINDOW_TITLE, CANVAS_W, CANVAS_H)
        self.fullscreen = False

        # --- состояние ---
        self.running = True
        self.frame_index = 0

        self.transcribe_window_seconds = 3.8
        self.transcribe_hop_seconds = 1.5
        self.audio_buf = np.empty(0, dtype=np.float32)
        self.pending_samples = 0

    # =====================================================
    # ВРЕМЯ И ЗВУК ИЗ ФАЙЛА
    # =====================================================

    def timeline(self):
        if self.is_video_file:
            return self.cap.get_time()
        return time.time() - self.t0

    def feed_file_audio(self, t):
        if self.file_audio is None:
            return
        end = min(int(t * self.audio.samplerate), len(self.file_audio))
        while self.file_audio_pos < end:
            nxt = min(self.file_audio_pos + self.audio.blocksize, end)
            self.audio.feed(self.file_audio[self.file_audio_pos:nxt])
            self.file_audio_pos = nxt

    # =====================================================
    # ОТРИСОВКА ПАНЕЛЕЙ-ИЗОБРАЖЕНИЙ
    # =====================================================

    @staticmethod
    def render_volume(volume):
        img, draw = new_panel(230, 34)
        draw_bar(draw, 10, 9, min(volume * 14.0, 1.0), "Звук", GREEN, load_font(15),
                 label_w=50, width=150, show_value=False)
        return to_bgra(img)

    @staticmethod
    def render_subtitles(lines):
        if not lines:
            return None
        wrapped = []
        for line in lines:
            wrapped.extend(textwrap.wrap(line, width=46) or [""])
        wrapped = wrapped[-2:]

        img, draw = new_panel(560, 82, outline=False)
        font = load_font(18)
        y = 8
        for line in wrapped:
            draw.text((14, y), line, font=font, fill=WHITE)
            y += 24
        return to_bgra(img)

    def set_label(self, label):
        if label == self.label:
            return
        self.label = label
        self.label_changed = True
        print(f"[{self.video_time:6.1f} c] разметка: {label or 'снята'}")

    def render_label(self):
        img, draw = new_panel(330, 80)
        font, small = load_font(17), load_font(15)
        text, color = {"": ("нет", GRAY), LABEL_TRUTH: ("ПРАВДА", GREEN),
                       LABEL_LIE: ("ЛОЖЬ", RED)}[self.label]
        draw.text((12, 8), f"Разметка: {text}", font=font, fill=color)
        draw.text((12, 34), "t - правда   l - ложь   n - снять", font=small, fill=GRAY)
        ready = self.last_suspicion is not None and self.last_suspicion.baseline_ready
        if not ready:
            draw.text((12, 56), "база собирается - пока не размечайте", font=small, fill=GRAY)
        return to_bgra(img)

    @staticmethod
    def draw_face_box(frame, face_data):
        if face_data.detected and face_data.bbox:
            x1, y1, x2, y2 = face_data.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def on_mouse(self, event, x, y, flags, param):
        cx, cy = self.mapper.to_canvas(x, y)
        self.panels.handle_mouse(event, cx, cy)

    # =====================================================
    # РАСПОЗНАВАНИЕ РЕЧИ
    # =====================================================

    @staticmethod
    def collect_transcription(segments):
        parts, logprobs, no_speech = [], [], []
        for segment in segments:
            part = segment.text.strip()
            if part:
                parts.append(part)
            if hasattr(segment, "avg_logprob"):
                logprobs.append(segment.avg_logprob)
            if hasattr(segment, "no_speech_prob"):
                no_speech.append(segment.no_speech_prob)

        text = " ".join(parts).strip()
        avg_logprob = sum(logprobs) / len(logprobs) if logprobs else None
        avg_no_speech = sum(no_speech) / len(no_speech) if no_speech else None
        return text, avg_logprob, avg_no_speech

    def should_apply_meaning_correction(self, text, avg_logprob, avg_no_speech):
        if not text or len(text.split()) < 3:
            return False
        return (self.corrector.looks_broken(text)
                or (avg_logprob is not None and avg_logprob < -0.8)
                or (avg_no_speech is not None and avg_no_speech > 0.5))

    def process_live_text(self, text):
        text = self.corrector.basic_cleanup(text)
        words = text.split()
        return " ".join(words[:18]) if len(words) > 18 else text

    def process_final_text(self, text, avg_logprob, avg_no_speech):
        cleaned = self.corrector.basic_cleanup(text)
        if cleaned and self.should_apply_meaning_correction(cleaned, avg_logprob, avg_no_speech):
            return self.corrector.correct(cleaned)
        return cleaned

    @staticmethod
    def should_commit_final_text(text, avg_logprob, avg_no_speech):
        text = (text or "").strip()
        if not text:
            return False
        words = text.split()
        if avg_no_speech is not None and avg_no_speech > 0.75:
            return False
        if len(words) >= 5:
            return True
        if text[-1] in ".!?" and len(words) >= 2:
            return True
        return avg_logprob is not None and avg_logprob > -0.45 and len(words) >= 3

    def append_audio_chunk(self, chunk):
        audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
        max_samples = int(self.audio.samplerate * self.transcribe_window_seconds)
        self.audio_buf = np.concatenate([self.audio_buf, audio])[-max_samples:]
        self.pending_samples += len(audio)

    def whisper_loop(self):
        hop_samples = int(self.audio.samplerate * self.transcribe_hop_seconds)
        min_first = int(self.audio.samplerate * 1.6)

        while self.running:
            try:
                chunk = self.audio.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.append_audio_chunk(chunk)
                if len(self.audio_buf) < min_first or self.pending_samples < hop_samples:
                    continue

                audio_data = self.audio_buf.copy()
                self.pending_samples = 0

                segments, _ = self.whisper.model.transcribe(
                    audio_data,
                    language="ru",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=350, speech_pad_ms=120),
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    compression_ratio_threshold=2.4,
                    condition_on_previous_text=True,
                )
                raw_text, avg_logprob, avg_no_speech = self.collect_transcription(segments)
                if not raw_text:
                    continue

                shown = self.whisper.update_live_text(self.process_live_text(raw_text))
                if shown:
                    self.storage.append_live(shown)

                final_text = self.process_final_text(raw_text, avg_logprob, avg_no_speech)
                if self.should_commit_final_text(final_text, avg_logprob, avg_no_speech):
                    result = self.whisper.commit_final_text(final_text, raw_text=raw_text)
                    if result:
                        tail, raw_tail = result
                        self.storage.append_final(tail)
                        if raw_tail:
                            self.speech_queue.put((self.video_time, raw_tail))

            except Exception as error:
                print("Ошибка распознавания речи:", error)

    # =====================================================
    # ВЫХОДНОЕ ВИДЕО
    # =====================================================

    def setup_video_writer(self):
        if not self.is_video_file:
            return

        name = os.path.splitext(os.path.basename(self.cap.filepath))[0]
        self.output_video_path = os.path.join(self.session_dir, name + "_analyzed.mp4")

        fps = self.cap.get_fps() or 30.0
        writer = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (CANVAS_W, CANVAS_H))
        if not writer.isOpened():
            print("Ошибка: не удалось создать выходное видео.")
            return
        self.video_writer = writer
        print("Выходное видео (без звука):", self.output_video_path)

    # =====================================================
    # ЗАЩИТА ОКНА ОТ ЗАХВАТА (режим экрана)
    # =====================================================

    def try_protect_window(self):
        if not self.is_screen_mode or self.window_protected or self.protect_attempts >= 30:
            return
        self.protect_attempts += 1
        if exclude_window_from_capture(self.WINDOW_TITLE):
            self.window_protected = True
            print("Окно предпросмотра исключено из захвата экрана.")
        elif self.protect_attempts == 30:
            print("Не удалось исключить окно предпросмотра из захвата экрана.")

    # =====================================================
    # ГЛАВНЫЙ ЦИКЛ
    # =====================================================

    def run(self):
        self.setup_video_writer()

        if self.is_video_file:
            pass  # звук подаётся из файла в основном цикле
        else:
            try:
                self.audio.start()
            except Exception as error:
                print("Не удалось открыть микрофон:", error)

        whisper_thread = threading.Thread(target=self.whisper_loop, daemon=True)
        whisper_thread.start()

        print()
        print("Видео, звук, субтитры, лицо, взгляд, мимика, покраснение и напряжение запущены.")
        print("Клавиши:  q / Esc - выход   c - калибровка мимики   r - калибровка покраснения")
        print("          b - личная калибровка напряжения (необязательно)   p - вернуть панели на места")
        print("          h - скрыть/показать панели   f - полный экран")
        print("Разметка:  t - сейчас говорится правда   l - сейчас ложь   n - снять метку")
        print("Панели переносятся мышью, окно можно растягивать.")
        print()

        cv2.namedWindow(self.WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_TITLE, CANVAS_W, CANVAS_H)
        cv2.setMouseCallback(self.WINDOW_TITLE, self.on_mouse)

        wall0 = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Обработка видео завершена." if self.is_video_file
                      else "Ошибка чтения кадра.")
                break

            frame_time = self.timeline()
            self.video_time = frame_time
            self.feed_file_audio(frame_time)

            volume = self.audio.get_volume()
            lines = self.whisper.get_display_lines()
            subtitle_text = lines[-1] if lines else ""

            # --- лицо и качество кадра ---
            face_data, _ = self.neural_face.process(frame, self.frame_index, frame_time)
            face_ok = face_data.detected

            if (self.last_quality is None or not face_ok
                    or self.frame_index % QUALITY_PERIOD == 0):
                self.last_quality = self.quality_analyzer.process(frame, face_data)
            raw_good = face_ok and self.last_quality.good
            self.bad_quality_frames = 0 if (raw_good or not face_ok) else self.bad_quality_frames + 1
            # страховка: калибровка не должна зависать навсегда из-за строгой оценки качества
            quality_good = face_ok and (raw_good or self.bad_quality_frames > QUALITY_PATIENCE_FRAMES)
            if self.bad_quality_frames == QUALITY_PATIENCE_FRAMES + 1:
                print("Качество кадра низкое (" + self.last_quality.note + "), калибрую всё равно.")

            # --- мимика (калибруется только на хороших кадрах) ---
            mimic = self.mimic_analyzer.process(face_data, allow_calibration=quality_good)
            if face_ok:
                micro_event = self.micro_module.update(mimic, frame_time)
            else:
                micro_event = self.micro_module.get_active(frame_time)

            # --- покраснение ---
            if (self.last_redness is None or not face_ok
                    or (quality_good and self.frame_index % REDNESS_PERIOD == 0)):
                self.last_redness, _ = self.redness_analyzer.process(
                    frame, face_data, allow_calibration=quality_good)

            # --- глаза ---
            frame = self.draw_face_box(frame, face_data)
            frame = self.eye_tracker.process(frame, face_data)

            # --- напряжение ---
            while True:
                try:
                    t_speech, raw_tail = self.speech_queue.get_nowait()
                except queue.Empty:
                    break
                self.suspicion_analyzer.feed_speech(raw_tail, t_speech)

            if self.last_suspicion is None or self.frame_index % SUSPICION_PERIOD == 0:
                self.last_suspicion = self.suspicion_analyzer.process(
                    self.eye_tracker, mimic, frame_time, valid=quality_good)

            # --- журнал ---
            self.analysis_logger.log(
                face_data=face_data,
                eye_tracker=self.eye_tracker,
                mimic_data=mimic,
                micro_event=micro_event,
                subtitle_text=subtitle_text,
                redness_data=self.last_redness,
                suspicion_data=self.last_suspicion,
                video_time=frame_time,
                quality_note=self.last_quality.note,
                label=self.label,
                participant=self.participant,
                force=self.label_changed,
            )
            self.label_changed = False

            # --- панели ---
            if self.frame_index % PANEL_RENDER_PERIOD == 0:
                p = self.panels
                p.set_image("volume", self.render_volume(volume))
                p.set_image("eyes", self.eye_tracker.render_panel(self.last_quality.note))
                p.set_image("redness", self.redness_analyzer.render_panel(self.last_redness))
                p.set_image("mimic", self.mimic_analyzer.render_panel(mimic, micro_event))
                p.set_image("suspicion", self.suspicion_analyzer.render_panel(self.last_suspicion))
                p.set_image("subtitles", self.render_subtitles(lines))
                p.set_image("label", self.render_label())
            self.panels.draw(frame)

            if self.video_writer is not None:
                self.video_writer.write(frame)

            cv2.imshow(self.WINDOW_TITLE, self.mapper.render(frame))
            self.try_protect_window()
            self.frame_index += 1

            # --- реальное время для видеофайла ---
            if self.is_video_file and PACE_VIDEO_FILE:
                ahead = frame_time - (time.time() - wall0)
                if ahead > 0:
                    time.sleep(min(ahead, 0.1))

            # --- клавиатура ---
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                self.mimic_analyzer.start_calibration()
            elif key == ord("r"):
                self.redness_analyzer.start_calibration()
            elif key == ord("b"):
                self.suspicion_analyzer.start_calibration()
            elif key == ord("t"):
                self.set_label(LABEL_TRUTH)
            elif key == ord("l"):
                self.set_label(LABEL_LIE)
            elif key == ord("n"):
                self.set_label("")
            elif key == ord("p"):
                self.panels.reset_layout()
            elif key == ord("h"):
                self.panels.toggle_visible()
            elif key == ord("f"):
                self.fullscreen = not self.fullscreen
                cv2.setWindowProperty(
                    self.WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)

            # окно закрыто крестиком
            if cv2.getWindowProperty(self.WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break

        self.shutdown(whisper_thread)

    def shutdown(self, whisper_thread):
        self.running = False
        whisper_thread.join(timeout=3.0)

        self.cap.release()
        self.audio.stop()
        self.neural_face.close()
        self.panels.save()

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print("Обработанное видео сохранено:", self.output_video_path)

        cv2.destroyAllWindows()
        print("Результаты сохранены в папке:", self.session_dir)


if __name__ == "__main__":
    source = choose_video_source(width=CANVAS_W, height=CANVAS_H)
    if not source.isOpened():
        print("Не удалось открыть источник видео. Проверьте камеру, её индекс или путь к файлу.")
    else:
        who = input("ID участника для оценки (Enter - пропустить): ").strip()
        App(source, participant=who).run()