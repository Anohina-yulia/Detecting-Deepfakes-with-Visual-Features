# -*- coding: utf-8 -*-
"""Фоновый поток сессии: захват видео, анализ, звук, речь, запись результатов.

Окно не выполняет тяжёлых вычислений: оно получает от потока готовые кадры
(QImage) и снимки показателей через сигналы Qt. Управление потоком (пауза,
калибровка, разметка) идёт через очередь команд, поэтому модули анализа
используются только из одного потока.
"""
import os
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from analysis_logger import AnalysisLogger
from analysis_pipeline import AnalysisPipeline, describe_face_error
from app_paths import FACE_MODEL_PATH, RESULTS_DIR, make_session_dir
from audio_module import AudioModule
from models import EV_SYSTEM, SessionEvent
from speech_engine import MODELS_FAILED, SpeechSession, load_file_audio
from subtitle_storage import SubtitleStorage

TICK_INTERVAL = 0.1          # как часто отправлять показатели в окно, с
MAX_READ_FAILURES = 60       # подряд неудачных чтений кадра до остановки
MAX_PROCESS_ERRORS = 30      # подряд ошибок анализа до остановки

SPEECH_OFF = "выключено"
SPEECH_WAIT_MODEL = "загрузка модели"
SPEECH_LISTENING = "слушаю"
SPEECH_PAUSED = "пауза"
SPEECH_NO_MIC = "нет микрофона"
SPEECH_NO_AUDIO = "нет звука в файле"
SPEECH_FAILED = "недоступно"


@dataclass
class SessionConfig:
    source_kind: str = "camera"          # camera | file | screen
    camera_index: int = 0
    file_path: str = ""
    participant: str = ""
    save_log: bool = True
    save_subtitles: bool = True
    save_video: bool = False
    pace_video_file: bool = True
    speech_enabled: bool = True
    audio_device: Optional[int] = None
    personal_baseline: bool = True
    with_mesh: bool = False
    results_dir: str = RESULTS_DIR
    face_model_path: str = FACE_MODEL_PATH
    width: int = 1280
    height: int = 720

    @property
    def source_label(self):
        if self.source_kind == "file":
            return os.path.basename(self.file_path) or "Видеофайл"
        if self.source_kind == "screen":
            return "Экран"
        return f"Камера {self.camera_index}"


@dataclass
class SessionTick:
    snapshot: object = None
    fps: float = 0.0
    proc_ms: float = 0.0
    volume: float = 0.0
    speech_state: str = SPEECH_OFF
    speech_detail: str = ""
    live_text: str = ""
    phrase_count: int = 0
    video_pos: Optional[float] = None
    video_duration: Optional[float] = None
    paused: bool = False
    camera_active: bool = False
    mic_active: bool = False
    saving: bool = False
    t: float = 0.0
    session_dir: Optional[str] = None


@dataclass
class SessionResult:
    session_dir: Optional[str] = None
    reason: str = "stopped"              # stopped | eof | error
    duration: float = 0.0
    source_kind: str = "camera"
    source_label: str = ""
    participant: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    output_video: Optional[str] = None
    error: str = ""


class SessionThread(QThread):
    frame_ready = Signal()
    tick = Signal(object)                 # SessionTick
    event_emitted = Signal(object)        # SessionEvent
    phase_changed = Signal(str, str)      # фаза, пояснение
    failed = Signal(str, str)             # заголовок, подробности
    session_finished = Signal(object)     # SessionResult

    def __init__(self, config, speech_models, parent=None):
        super().__init__(parent)
        self.cfg = config
        self.models = speech_models
        self._stop = threading.Event()
        self._commands = queue.Queue()
        self._frame_lock = threading.Lock()
        self._latest = None
        self._frame_pending = False

        self.source = None
        self.pipeline = None
        self.audio = None
        self.speech = None
        self.writer = None
        self.session_dir = None
        self.output_video = None
        self.file_audio = None
        self.file_audio_pos = 0

        self.paused = False
        self.with_mesh = config.with_mesh
        self.t = 0.0
        self.speech_state = SPEECH_OFF if not config.speech_enabled else SPEECH_WAIT_MODEL
        self.speech_detail = ""
        self._speech_failed = False
        self._last_speech_try = 0.0
        self._last_snapshot = None
        self._last_tick = 0.0
        self._fps = 0.0
        self._proc_ms = 0.0
        self._last_frame_wall = None

    # =====================================================
    # управление из окна (потокобезопасно)
    # =====================================================

    def request_stop(self):
        self._stop.set()

    def set_paused(self, paused):
        self._commands.put(("pause", bool(paused)))

    def recalibrate(self, mimic=True, redness=True, baseline=True):
        self._commands.put(("recalibrate", (mimic, redness, baseline)))

    def set_label(self, label):
        self._commands.put(("label", label))

    def set_mesh(self, enabled):
        self._commands.put(("mesh", bool(enabled)))

    def take_frame(self):
        with self._frame_lock:
            item = self._latest
            self._frame_pending = False
        return item

    # =====================================================
    # поток
    # =====================================================

    def run(self):
        result = SessionResult(source_kind=self.cfg.source_kind, source_label=self.cfg.source_label,
                               participant=self.cfg.participant)
        try:
            self._run(result)
        except Exception as error:
            traceback.print_exc()
            result.reason = "error"
            result.error = str(error)
            self.failed.emit("Сессия прервана из-за ошибки", str(error))
        finally:
            self.phase_changed.emit("stopping", "Освобождение камеры и микрофона…")
            self._cleanup()
            result.duration = self.t
            result.session_dir = self.session_dir
            result.output_video = self.output_video
            self.session_finished.emit(result)

    def _event(self, kind, text, level="info", t=None):
        self.event_emitted.emit(SessionEvent(t=self.t if t is None else t, kind=kind,
                                             text=text, level=level))

    def _open_source(self):
        from video_source import ScreenSource, VideoFileSource, WebcamSource

        cfg = self.cfg
        if cfg.source_kind == "file":
            self.phase_changed.emit("opening", "Открытие видеофайла…")
            if not cfg.file_path or not os.path.exists(cfg.file_path):
                self.failed.emit("Видеофайл не найден",
                                 f"Файл «{cfg.file_path or '—'}» не существует или был перемещён.")
                return None
            source = VideoFileSource(cfg.file_path, width=cfg.width, height=cfg.height)
            if not source.isOpened():
                source.release()
                self.failed.emit("Не удалось открыть видеофайл",
                                 "Формат не поддерживается или файл повреждён. "
                                 "Попробуйте MP4 (H.264), AVI или MOV.")
                return None
            return source
        if cfg.source_kind == "screen":
            self.phase_changed.emit("opening", "Подготовка захвата экрана…")
            source = ScreenSource(width=cfg.width, height=cfg.height)
            ok, _ = source.read()
            if not ok:
                self.failed.emit("Захват экрана недоступен",
                                 "Windows не разрешила снимок экрана. Проверьте параметры "
                                 "конфиденциальности и повторите попытку.")
                return None
            return source

        self.phase_changed.emit("opening", f"Подключение к камере {cfg.camera_index}…")
        source = WebcamSource(index=cfg.camera_index, width=cfg.width, height=cfg.height)
        if not source.isOpened():
            source.release()
            self.failed.emit("Камера недоступна",
                             f"Не удалось открыть камеру №{cfg.camera_index}. Проверьте, что она "
                             "подключена и не занята другим приложением (Zoom, Teams, браузер), "
                             "или выберите другую камеру в настройках.")
            return None
        ok = False
        for _ in range(20):  # первые кадры с камеры иногда приходят не сразу
            ok, _ = source.read()
            if ok or self._stop.is_set():
                break
            time.sleep(0.05)
        if not ok:
            source.release()
            self.failed.emit("Камера не передаёт изображение",
                             "Камера открылась, но кадры не поступают. Закройте другие программы, "
                             "использующие камеру, и повторите попытку.")
            return None
        return source

    def _run(self, result):
        cfg = self.cfg
        is_file = cfg.source_kind == "file"

        self.source = self._open_source()
        if self.source is None:
            result.reason = "error"
            result.error = "источник видео недоступен"
            return
        if self._stop.is_set():  # сессию отменили, пока подключалась камера
            result.reason = "cancelled"
            return

        # --- папка результатов (только если пользователь разрешил что-то сохранять) ---
        wants_files = cfg.save_log or (cfg.save_subtitles and cfg.speech_enabled) \
            or (cfg.save_video and is_file)
        if wants_files:
            if is_file:
                base = os.path.splitext(os.path.basename(cfg.file_path))[0] or "video"
            else:
                base = "screen" if cfg.source_kind == "screen" else "webcam"
            self.session_dir = make_session_dir(base, cfg.results_dir)
            print("Результаты сеанса:", self.session_dir)

        # --- анализ ---
        self.phase_changed.emit("opening", "Загрузка модели лица…")
        logger = None
        if cfg.save_log and self.session_dir:
            logger = AnalysisLogger(filepath=os.path.join(self.session_dir, "analysis_log.csv"),
                                    min_interval_sec=0.7)
        self.pipeline = AnalysisPipeline(model_path=cfg.face_model_path, logger=logger,
                                         participant=cfg.participant)
        if not self.pipeline.face_available:
            self._event(EV_SYSTEM, "Анализ лица недоступен: "
                        + describe_face_error(self.pipeline.face_error), "error")
        if cfg.personal_baseline:
            self.pipeline.start_personal_baseline(0.0)
        for ev in self.pipeline.drain_events():
            self.event_emitted.emit(ev)

        # --- звук ---
        self.audio = AudioModule()
        if is_file and cfg.speech_enabled:
            self.phase_changed.emit("opening", "Чтение звуковой дорожки…")
            self.file_audio = load_file_audio(cfg.file_path, self.audio.samplerate)
            if self.file_audio is None:
                self.speech_state = SPEECH_NO_AUDIO
                self._speech_failed = True
                self._event(EV_SYSTEM, "В видеофайле нет звука - субтитров не будет", "warning")
        self._try_start_speech()

        # --- обработанное видео (только по явному разрешению) ---
        if is_file and cfg.save_video and self.session_dir:
            name = os.path.splitext(os.path.basename(cfg.file_path))[0]
            path = os.path.join(self.session_dir, name + "_analyzed.mp4")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     self.source.get_fps() or 30.0, (cfg.width, cfg.height))
            if writer.isOpened():
                self.writer, self.output_video = writer, path
            else:
                writer.release()
                self._event(EV_SYSTEM, "Не удалось создать файл обработанного видео", "error")

        self.phase_changed.emit("running", "")
        self._loop(result, is_file)

    def _loop(self, result, is_file):
        cfg = self.cfg
        t0 = time.time()
        wall0 = time.time()
        pause_started = None
        read_failures = 0
        process_errors = 0
        duration = self.source.get_duration() if is_file else None

        while not self._stop.is_set():
            # --- команды окна ---
            while True:
                try:
                    command, value = self._commands.get_nowait()
                except queue.Empty:
                    break
                if command == "pause" and value != self.paused:
                    self.paused = value
                    if value:
                        pause_started = time.time()
                        self._pause_audio()
                        self.phase_changed.emit("paused", "")
                    else:
                        if pause_started is not None:
                            wall0 += time.time() - pause_started
                        pause_started = None
                        self._resume_audio()
                        self.phase_changed.emit("running", "")
                elif command == "recalibrate":
                    mimic, redness, baseline = value
                    self.pipeline.recalibrate(self.t, mimic=mimic, redness=redness, baseline=baseline)
                elif command == "label":
                    self.pipeline.set_label(value, self.t)
                elif command == "mesh":
                    self.with_mesh = value
            for ev in self.pipeline.drain_events():
                self.event_emitted.emit(ev)

            # --- пауза: видеофайл стоит, камера/экран показывают картинку без анализа ---
            if self.paused:
                if is_file:
                    self._emit_tick(is_file, duration)
                    time.sleep(0.04)
                    continue
                ok, frame = self.source.read()
                if ok:
                    self.t = time.time() - t0
                    self._publish(frame, None)
                self._emit_tick(is_file, duration)
                time.sleep(0.03 if cfg.source_kind == "camera" else 0.12)
                continue

            ok, frame = self.source.read()
            if not ok:
                if is_file:
                    result.reason = "eof"
                    print("Обработка видео завершена.")
                    break
                read_failures += 1
                if read_failures >= MAX_READ_FAILURES:
                    result.reason = "error"
                    result.error = "источник видео перестал передавать кадры"
                    self.failed.emit("Источник видео перестал отвечать",
                                     "Камера отключена или занята другой программой. "
                                     "Результаты сессии сохранены.")
                    break
                time.sleep(0.03)
                continue
            read_failures = 0

            self.t = self.source.get_time() if is_file else time.time() - t0
            if self.file_audio is not None and self.speech is not None:
                self._feed_file_audio(self.t)

            if time.time() - self._last_speech_try > 1.0:
                self._try_start_speech()

            subtitle_text = ""
            if self.speech is not None:
                lines = self.speech.get_display_lines()
                subtitle_text = lines[-1] if lines else ""
                for t_speech, raw_tail in self.speech.drain_speech():
                    self.pipeline.feed_speech(raw_tail, t_speech)

            started = time.perf_counter()
            try:
                snapshot = self.pipeline.process(frame, self.t, subtitle_text=subtitle_text)
                process_errors = 0
            except Exception as error:
                process_errors += 1
                print("Ошибка анализа кадра:", error)
                if process_errors == 1:
                    traceback.print_exc()
                if process_errors >= MAX_PROCESS_ERRORS:
                    raise
                self._publish(frame, None)
                continue
            proc_ms = (time.perf_counter() - started) * 1000.0
            self._proc_ms = proc_ms if self._proc_ms == 0 else self._proc_ms * 0.9 + proc_ms * 0.1

            now = time.time()
            if self._last_frame_wall is not None:
                dt = max(now - self._last_frame_wall, 1e-3)
                fps = 1.0 / dt
                self._fps = fps if self._fps == 0 else self._fps * 0.9 + fps * 0.1
            self._last_frame_wall = now

            for ev in self.pipeline.drain_events():
                self.event_emitted.emit(ev)

            overlay = self.pipeline.build_overlay(with_mesh=self.with_mesh)
            if self.writer is not None:
                marked = frame.copy()
                self.pipeline.render_overlays(marked)
                self.writer.write(marked)

            self._publish(frame, overlay)
            self._last_snapshot = snapshot
            self._emit_tick(is_file, duration)

            # видеофайл воспроизводится в реальном времени, чтобы субтитры успевали за звуком
            if is_file and cfg.pace_video_file:
                ahead = self.t - (time.time() - wall0)
                if ahead > 0:
                    time.sleep(min(ahead, 0.1))

        if self._stop.is_set() and result.reason not in ("error", "eof"):
            result.reason = "stopped"

    # =====================================================
    # звук и речь
    # =====================================================

    def _try_start_speech(self):
        self._last_speech_try = time.time()
        cfg = self.cfg
        if self.speech is not None or self._speech_failed:
            return
        if not cfg.speech_enabled:
            self.speech_state = SPEECH_OFF
            return
        if not self.models.ready:
            if self.models.loading:
                self.speech_state = SPEECH_WAIT_MODEL
            else:
                self.speech_state = SPEECH_FAILED
                self.speech_detail = self.models.error or "модель речи не загружена"
                if self.models.state == MODELS_FAILED:
                    self._speech_failed = True
            return

        save_text = bool(self.session_dir and cfg.save_subtitles)
        folder = self.session_dir or ""
        storage = SubtitleStorage(
            live_file=os.path.join(folder, "subtitles_live.txt"),
            final_file=os.path.join(folder, "subtitles.txt"),
            time_fn=lambda: self.t,
            enabled=save_text,
        )
        speech = SpeechSession(self.models, self.audio, storage, time_fn=lambda: self.t,
                               on_event=self.event_emitted.emit)
        if cfg.source_kind == "file":
            # модель догрузилась посреди видео - распознаём с текущего места, а не весь хвост
            self.file_audio_pos = int(self.t * self.audio.samplerate)
        else:
            try:
                self.audio.start(device=cfg.audio_device)
            except Exception as error:
                self._speech_failed = True
                self.speech_state = SPEECH_NO_MIC
                self.speech_detail = str(error)
                self._event(EV_SYSTEM, f"Микрофон недоступен: {error}", "error")
                return
        self.speech = speech
        self.speech.start()
        self.speech_state = SPEECH_PAUSED if self.paused else SPEECH_LISTENING
        self.speech_detail = f"Whisper {self.models.model_size}"
        if self.paused:
            self._pause_audio()

    def _pause_audio(self):
        if self.speech is not None:
            self.speech_state = SPEECH_PAUSED
        if self.cfg.source_kind == "file":
            # звук файла непрерывен во времени файла: на паузе он просто не подаётся,
            # а уже поданный дораспознаётся - фраза на стыке паузы не теряется
            return
        if self.speech is not None:
            self.speech.set_paused(True)
        self.audio.stop()
        self.audio.clear_queue()

    def _resume_audio(self):
        if self.speech is None:
            return
        if self.cfg.source_kind != "file":
            try:
                self.audio.start(device=self.cfg.audio_device)
            except Exception as error:
                self.speech_state = SPEECH_NO_MIC
                self._event(EV_SYSTEM, f"Микрофон недоступен: {error}", "error")
                return
            self.speech.set_paused(False)
        self.speech_state = SPEECH_LISTENING

    def _feed_file_audio(self, t):
        end = min(int(t * self.audio.samplerate), len(self.file_audio))
        while self.file_audio_pos < end:
            nxt = min(self.file_audio_pos + self.audio.blocksize, end)
            self.audio.feed(self.file_audio[self.file_audio_pos:nxt])
            self.file_audio_pos = nxt

    # =====================================================
    # передача в окно
    # =====================================================

    def _publish(self, frame, overlay):
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        h, w = frame.shape[:2]
        image = QImage(frame.data, w, h, int(frame.strides[0]), QImage.Format_BGR888).copy()
        with self._frame_lock:
            self._latest = (image, overlay)
            pending = self._frame_pending
            self._frame_pending = True
        if not pending:
            self.frame_ready.emit()

    def _emit_tick(self, is_file, duration):
        now = time.time()
        if now - self._last_tick < TICK_INTERVAL:
            return
        self._last_tick = now
        live = ""
        phrases = 0
        if self.speech is not None:
            lines = self.speech.get_display_lines()
            live = lines[-1] if lines else ""
            phrases = self.speech.phrase_count
        cfg = self.cfg
        self.tick.emit(SessionTick(
            snapshot=self._last_snapshot,
            fps=self._fps,
            proc_ms=self._proc_ms,
            volume=self.audio.get_volume() if self.audio is not None else 0.0,
            speech_state=self.speech_state,
            speech_detail=self.speech_detail,
            live_text=live,
            phrase_count=phrases,
            video_pos=self.t if is_file else None,
            video_duration=duration if is_file else None,
            paused=self.paused,
            camera_active=cfg.source_kind == "camera",
            mic_active=bool(self.audio is not None and self.audio.active),
            saving=bool(self.session_dir) and not self.paused,
            t=self.t,
            session_dir=self.session_dir,
        ))

    # =====================================================
    # освобождение ресурсов
    # =====================================================

    def _cleanup(self):
        if self.speech is not None:
            self.speech.stop(timeout=3.0)
        if self.audio is not None:
            self.audio.stop()
        if self.source is not None:
            try:
                self.source.release()
            except Exception as error:
                print("Ошибка при освобождении источника видео:", error)
        if self.pipeline is not None:
            self.pipeline.close()
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print("Обработанное видео сохранено:", self.output_video)
        if self.session_dir:
            print("Результаты сохранены в папке:", self.session_dir)
