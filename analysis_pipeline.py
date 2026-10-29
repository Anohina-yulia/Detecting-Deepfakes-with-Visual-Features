# -*- coding: utf-8 -*-
"""Покадровый конвейер анализа (вынесен из app.App.run).

Порядок вызовов и периоды пересчёта сохранены как в исходном цикле:
лицо -> качество кадра -> мимика -> быстрые изменения мимики -> покраснение ->
глаза -> индикатор отклонения от базы -> журнал.

Поверх этого считаются производные показатели, которых раньше не было на экране,
но которые следуют из уже измеряемых данных: частота морганий в минуту,
длительность закрытия глаз, доля взгляда вне центра, активность мимики.

Конвейер не зависит от интерфейса: им пользуются новое окно (gui/) и
классический режим OpenCV (app.py --classic).
"""
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2

from eye_tracker import EyeTrackerModule
from micro_expression import MicroExpressionModule
from mimic_analyzer import CALIBRATION_FRAMES as MIMIC_CALIBRATION_FRAMES
from mimic_analyzer import MimicAnalyzerModule
from models import (EV_CALIBRATION, EV_EYES, EV_FACE, EV_LABEL, EV_MICRO, EV_QUALITY,
                    EXPR_NO_FACE, GAZE_CENTER, GAZE_NONE, POS_NONE,
                    FaceData, MicroEvent, MimicData, RednessData, SessionEvent)
from quality_analyzer import QualityAnalyzerModule, QualityData
from redness_analyzer import CALIBRATION_FRAMES as REDNESS_CALIBRATION_FRAMES
from redness_analyzer import RednessAnalyzerModule
from suspicion_analyzer import BASELINE_SECONDS, SuspicionAnalyzerModule, SuspicionData

REDNESS_PERIOD = 6       # раз в сколько кадров пересчитывать покраснение
SUSPICION_PERIOD = 5     # ... индикатор отклонения от базы
QUALITY_PERIOD = 3       # ... качество кадра
QUALITY_PATIENCE_FRAMES = 60  # столько кадров "плохого" качества терпим, потом калибруем всё равно

BLINK_RATE_WINDOW = 60.0       # окно частоты морганий, с
BLINK_RATE_MIN_VISIBLE = 15.0  # меньше секунд с лицом в окне - частота ненадёжна
GAZE_WINDOW = 15.0             # окно доли взгляда вне центра, с
GAZE_MIN_VISIBLE = 3.0
ACTIVITY_TAU = 1.0             # сглаживание активности мимики, с
ACTIVITY_SCALE = 0.6           # скорость изменения мимики (ед./с), равная 100 %
MICRO_WINDOW = 60.0            # окно подсчёта быстрых изменений мимики, с
MAX_FRAME_GAP = 0.5            # больший разрыв между кадрами не считается временем наблюдения
FACE_LOST_DELAY = 1.0          # через сколько секунд без лица сообщать о потере
QUALITY_ISSUE_DELAY = 2.0      # сколько секунд плохого кадра до предупреждения
LONG_CLOSURE_SECONDS = 1.0     # закрытие глаз дольше - отдельное событие

QUALITY_OK_NOTES = ("норма", "лицо не по центру")
MIMIC_FEATURES = ("smile", "mouth_open", "brow_raise", "brow_frown", "eye_squint")


@dataclass
class OverlayData:
    """Что подсветить поверх видео (координаты кадра)."""
    frame_size: Tuple[int, int] = (0, 0)
    bbox: Optional[Tuple[int, int, int, int]] = None
    left_eye: Optional[List[Tuple[int, int]]] = None
    right_eye: Optional[List[Tuple[int, int]]] = None
    left_iris: Optional[Tuple[float, float, float]] = None   # x, y, радиус
    right_iris: Optional[Tuple[float, float, float]] = None
    landmarks: Optional[List[Tuple[int, int]]] = None
    eyes_closed: bool = False


@dataclass
class AnalysisSnapshot:
    """Состояние всех показателей после очередного кадра."""
    t: float = 0.0
    frame_index: int = 0
    face_model_ok: bool = True
    face_detected: bool = False
    face_position: str = POS_NONE
    visible_time: float = 0.0           # всего секунд с лицом в кадре
    quality: Optional[QualityData] = None
    quality_good: bool = False

    gaze: str = GAZE_NONE
    gaze_h: Optional[float] = None
    gaze_v: Optional[float] = None
    gaze_off_ratio: Optional[float] = None   # доля времени взгляда вне центра за окно
    gaze_visible_s: float = 0.0

    ear: Optional[float] = None
    eyes_closed: bool = False
    blink_count: int = 0
    blink_rate: Optional[float] = None       # морганий в минуту за окно
    blink_visible_s: float = 0.0
    last_closure_ms: Optional[float] = None

    mimic: Optional[MimicData] = None
    mimic_ready: bool = False
    mimic_progress: float = 0.0
    expression: str = EXPR_NO_FACE
    expression_intensity: Optional[float] = None
    activity: Optional[float] = None
    micro_event: Optional[MicroEvent] = None
    micro_per_min: int = 0

    redness: Optional[RednessData] = None
    redness_ready: bool = False
    redness_progress: float = 0.0

    deviation: Optional[SuspicionData] = None
    baseline_personal: bool = False
    baseline_collecting: bool = False
    baseline_progress: float = 1.0
    baseline_blink_rate: Optional[float] = None
    baseline_gaze_off: Optional[float] = None
    baseline_activity: Optional[float] = None

    label: str = ""


class _Window:
    """Скользящее окно (время, наблюдаемые секунды, значение) с бегущими суммами."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.items = deque()
        self.visible = 0.0
        self.total = 0.0

    def add(self, t, visible_dt, value):
        self.items.append((t, visible_dt, value))
        self.visible += visible_dt
        self.total += value
        while self.items and t - self.items[0][0] > self.seconds:
            _, v, x = self.items.popleft()
            self.visible -= v
            self.total -= x
        if self.visible < 1e-9:
            self.visible = 0.0

    def clear(self):
        self.items.clear()
        self.visible = self.total = 0.0


def clamp01(value):
    return float(max(0.0, min(1.0, value)))


class AnalysisPipeline:
    def __init__(self, model_path, logger=None, participant=""):
        self.face_error = ""
        self.neural_face = None
        try:
            from neural_face_analyzer import NeuralFaceAnalyzerModule

            self.neural_face = NeuralFaceAnalyzerModule(model_path=model_path)
        except Exception as error:  # нет модели или mediapipe - работаем без анализа лица
            self.face_error = str(error) or error.__class__.__name__
            print("Анализ лица недоступен:", self.face_error)

        self.eye_tracker = EyeTrackerModule()
        self.mimic_analyzer = MimicAnalyzerModule()
        self.micro_module = MicroExpressionModule()
        self.redness_analyzer = RednessAnalyzerModule()
        self.suspicion_analyzer = SuspicionAnalyzerModule()
        self.quality_analyzer = QualityAnalyzerModule()

        self.logger = logger
        self.participant = participant
        self.label = ""
        self._label_changed = False

        self.frame_index = 0
        self.last_face = FaceData()
        self.last_quality = None
        self.bad_quality_frames = 0
        self.last_redness = None
        self.last_redness_boxes = {}
        self.last_suspicion = None
        self.last_mimic = MimicData(dominant_expression=EXPR_NO_FACE)
        self.last_micro = None
        self.quality_good = False

        # производные показатели
        self.visible_time = 0.0
        self._prev_t = None
        self._last_blink_count = 0
        self._blink_window = _Window(BLINK_RATE_WINDOW)
        self._gaze_window = _Window(GAZE_WINDOW)
        self._micro_times = deque()
        self._last_micro_ts = None
        self._prev_scores = None
        self._activity_ema = None
        self._activity = None

        # события
        self.events = []
        self._face_present = False
        self._face_seen_once = False
        self._face_missing_since = None
        self._quality_issue = ""
        self._quality_issue_since = None
        self._quality_reported = ""
        self._long_closure_reported = False
        self._ready = (False, False, True)

        # базовые уровни для показателей интерфейса (собираются вместе с личной базой)
        self._baseline_gaze = []
        self._baseline_activity = []
        self.display_baseline = {"gaze_off": None, "activity": None}

    # =====================================================
    # управление
    # =====================================================

    @property
    def face_available(self):
        return self.neural_face is not None

    def close(self):
        if self.neural_face is not None:
            try:
                self.neural_face.close()
            except Exception as error:
                print("Ошибка при закрытии модели лица:", error)
            self.neural_face = None

    def emit(self, t, kind, text, level="info"):
        self.events.append(SessionEvent(t=float(t), kind=kind, text=text, level=level))

    def drain_events(self):
        events, self.events = self.events, []
        return events

    def start_personal_baseline(self, t=0.0, announce=True):
        """Личная база (по умолчанию используются типичные значения)."""
        self.suspicion_analyzer.start_calibration(silent=not announce)
        self._baseline_gaze = []
        self._baseline_activity = []
        self.display_baseline = {"gaze_off": None, "activity": None}
        self._ready = (self._ready[0], self._ready[1], False)
        if announce:
            self.emit(t, EV_CALIBRATION,
                      f"Сбор личной базы ({BASELINE_SECONDS:.0f} с): ведите себя как обычно")

    def recalibrate(self, t=0.0, mimic=True, redness=True, baseline=True):
        parts = []
        if mimic:
            self.mimic_analyzer.start_calibration()
            parts.append("мимика")
        if redness:
            self.redness_analyzer.start_calibration()
            parts.append("цвет кожи")
        if baseline:
            self.start_personal_baseline(t, announce=False)
            parts.append("личная база")
        self._ready = (self.mimic_analyzer.neutral_ready, self.redness_analyzer.neutral_ready,
                       self.suspicion_analyzer.baseline_ready)
        if parts:
            self.emit(t, EV_CALIBRATION, "Перекалибровка: " + ", ".join(parts)
                      + ". Смотрите в камеру, лицо нейтральное.", "notice")

    def set_label(self, label, t=0.0):
        if label == self.label:
            return
        self.label = label
        self._label_changed = True
        self.emit(t, EV_LABEL, f"Разметка: {label}" if label else "Разметка снята", "notice")

    def feed_speech(self, raw_text, t):
        self.suspicion_analyzer.feed_speech(raw_text, t)

    # =====================================================
    # кадр
    # =====================================================

    def _detect(self, frame, frame_time):
        height, width = frame.shape[:2]
        if self.neural_face is None:
            return FaceData(detected=False, frame_width=width, frame_height=height,
                            timestamp=frame_time, face_position=POS_NONE)
        face_data, _ = self.neural_face.process(frame, self.frame_index, frame_time)
        return face_data

    @staticmethod
    def draw_face_box(frame, face_data):
        if face_data.detected and face_data.bbox:
            x1, y1, x2, y2 = face_data.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def render_overlays(self, frame, face_data=None):
        """Рамка лица и контуры глаз на кадре (как в прежней версии)."""
        face_data = face_data or self.last_face
        self.draw_face_box(frame, face_data)
        self.eye_tracker.draw_eyes(frame, face_data)
        return frame

    def process(self, frame, frame_time, subtitle_text="", draw_overlays=False):
        """Анализирует кадр. draw_overlays=True рисует рамку лица и глаза прямо на кадре."""
        face_data = self._detect(frame, frame_time)
        self.last_face = face_data
        face_ok = face_data.detected

        # --- качество кадра ---
        if (self.last_quality is None or not face_ok
                or self.frame_index % QUALITY_PERIOD == 0):
            self.last_quality = self.quality_analyzer.process(frame, face_data)
        raw_good = face_ok and self.last_quality.good
        self.bad_quality_frames = 0 if (raw_good or not face_ok) else self.bad_quality_frames + 1
        # страховка: калибровка не должна зависать навсегда из-за строгой оценки качества
        quality_good = face_ok and (raw_good or self.bad_quality_frames > QUALITY_PATIENCE_FRAMES)
        if self.bad_quality_frames == QUALITY_PATIENCE_FRAMES + 1:
            print("Качество кадра низкое (" + self.last_quality.note + "), калибрую всё равно.")
        self.quality_good = quality_good

        # --- мимика (калибруется только на хороших кадрах) ---
        mimic = self.mimic_analyzer.process(face_data, allow_calibration=quality_good)
        if face_ok:
            micro_event = self.micro_module.update(mimic, frame_time)
        else:
            micro_event = self.micro_module.get_active(frame_time)
        self.last_mimic, self.last_micro = mimic, micro_event

        # --- покраснение (до рисования на кадре: важен чистый цвет кожи) ---
        if (self.last_redness is None or not face_ok
                or (quality_good and self.frame_index % REDNESS_PERIOD == 0)):
            self.last_redness, self.last_redness_boxes = self.redness_analyzer.process(
                frame, face_data, allow_calibration=quality_good)

        # --- глаза ---
        if draw_overlays:
            self.draw_face_box(frame, face_data)
        self.eye_tracker.process(frame, face_data, draw=draw_overlays)

        # --- отклонение от базы ---
        if self.last_suspicion is None or self.frame_index % SUSPICION_PERIOD == 0:
            self.last_suspicion = self.suspicion_analyzer.process(
                self.eye_tracker, mimic, frame_time, valid=quality_good)

        # --- журнал ---
        if self.logger is not None:
            self.logger.log(
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
                force=self._label_changed,
            )
        self._label_changed = False

        snapshot = self._update_derived(frame_time, face_data, mimic, micro_event)
        self.frame_index += 1
        return snapshot

    # =====================================================
    # производные показатели и события
    # =====================================================

    def _update_derived(self, t, face_data, mimic, micro_event):
        face_ok = face_data.detected
        dt = 0.0 if self._prev_t is None else min(max(t - self._prev_t, 0.0), MAX_FRAME_GAP)
        self._prev_t = t
        visible_dt = dt if face_ok else 0.0
        self.visible_time += visible_dt
        eyes = self.eye_tracker

        # частота морганий
        new_blinks = max(eyes.blink_count - self._last_blink_count, 0)
        self._last_blink_count = eyes.blink_count
        self._blink_window.add(t, visible_dt, new_blinks)
        blink_rate = None
        if self._blink_window.visible >= BLINK_RATE_MIN_VISIBLE:
            blink_rate = self._blink_window.total / self._blink_window.visible * 60.0

        # доля взгляда вне центра
        off = face_ok and eyes.last_gaze not in (GAZE_CENTER, GAZE_NONE)
        self._gaze_window.add(t, visible_dt, visible_dt if off else 0.0)
        gaze_off = None
        if self._gaze_window.visible >= GAZE_MIN_VISIBLE:
            gaze_off = clamp01(self._gaze_window.total / self._gaze_window.visible)

        # активность мимики: средняя скорость изменения пяти показателей мимики
        if face_ok:
            scores = [getattr(mimic, name) for name in MIMIC_FEATURES]
            if self._prev_scores is not None and dt > 0:
                speed = sum(abs(a - b) for a, b in zip(scores, self._prev_scores)) / len(scores) / dt
                alpha = 1.0 - math.exp(-dt / ACTIVITY_TAU)
                if self._activity_ema is None:
                    self._activity_ema = speed
                else:
                    self._activity_ema += alpha * (speed - self._activity_ema)
            self._prev_scores = scores
            if self._activity_ema is not None:
                self._activity = clamp01(self._activity_ema / ACTIVITY_SCALE)
        else:
            self._prev_scores = None
            if self._face_missing_since is not None and t - self._face_missing_since > FACE_LOST_DELAY:
                self._activity_ema = None
                self._activity = None

        # быстрые изменения мимики
        if micro_event is not None and micro_event.timestamp != self._last_micro_ts:
            self._last_micro_ts = micro_event.timestamp
            self._micro_times.append(t)
            self.emit(t, EV_MICRO, f"Быстрое изменение мимики: {micro_event.name}")
        while self._micro_times and t - self._micro_times[0] > MICRO_WINDOW:
            self._micro_times.popleft()

        self._update_face_events(t, face_ok)
        self._update_quality_events(t, face_ok)
        self._update_closure_events(t, face_ok)
        self._update_calibration_events(t)
        self._collect_display_baseline(gaze_off)

        # интенсивность выражения относительно нейтрального
        intensity = None
        expression = mimic.dominant_expression
        if face_ok:
            values = {name: getattr(mimic, name) for name in MIMIC_FEATURES}
            intensity = clamp01(max(values.values())) if values else 0.0

        sus = self.suspicion_analyzer
        baseline_progress = 1.0
        if not sus.baseline_ready:
            baseline_progress = clamp01(sus.valid_time / BASELINE_SECONDS)

        redness = self.last_redness or RednessData()
        quality = self.last_quality
        return AnalysisSnapshot(
            t=t,
            frame_index=self.frame_index,
            face_model_ok=self.face_available,
            face_detected=face_ok,
            face_position=face_data.face_position,
            visible_time=self.visible_time,
            quality=quality,
            quality_good=self.quality_good,
            gaze=eyes.last_gaze,
            gaze_h=eyes.last_horizontal_ratio if face_ok else None,
            gaze_v=eyes.last_vertical_ratio if face_ok else None,
            gaze_off_ratio=gaze_off if face_ok else None,
            gaze_visible_s=self._gaze_window.visible,
            ear=eyes.last_ear if face_ok else None,
            eyes_closed=bool(face_ok and eyes.eyes_closed),
            blink_count=eyes.blink_count,
            blink_rate=blink_rate if face_ok else None,
            blink_visible_s=self._blink_window.visible,
            last_closure_ms=eyes.last_closure_ms,
            mimic=mimic,
            mimic_ready=self.mimic_analyzer.neutral_ready,
            mimic_progress=clamp01(self.mimic_analyzer.calibration_progress / MIMIC_CALIBRATION_FRAMES),
            expression=expression if face_ok else EXPR_NO_FACE,
            expression_intensity=intensity,
            activity=self._activity if face_ok else None,
            micro_event=micro_event,
            micro_per_min=len(self._micro_times),
            redness=redness,
            redness_ready=bool(self.redness_analyzer.neutral_ready and redness.calibrated),
            redness_progress=clamp01(len(self.redness_analyzer.neutral_samples) / REDNESS_CALIBRATION_FRAMES)
            if not self.redness_analyzer.neutral_ready else 1.0,
            deviation=self.last_suspicion,
            baseline_personal=bool(getattr(sus, "baseline_personal", False)),
            baseline_collecting=not sus.baseline_ready,
            baseline_progress=baseline_progress,
            baseline_blink_rate=sus.base_blink_rate if sus.baseline_ready else None,
            baseline_gaze_off=self.display_baseline["gaze_off"],
            baseline_activity=self.display_baseline["activity"],
            label=self.label,
        )

    def _update_face_events(self, t, face_ok):
        if face_ok:
            self._face_missing_since = None
            if not self._face_present:
                self._face_present = True
                if not self._face_seen_once:
                    self._face_seen_once = True
                    self.emit(t, EV_FACE, "Лицо обнаружено")
                else:
                    self.emit(t, EV_FACE, "Лицо снова в кадре")
            return
        if self._face_missing_since is None:
            self._face_missing_since = t
        if self._face_present and t - self._face_missing_since >= FACE_LOST_DELAY:
            self._face_present = False
            self.emit(self._face_missing_since, EV_FACE, "Лицо пропало из кадра", "warning")

    def _update_quality_events(self, t, face_ok):
        note = self.last_quality.note if (face_ok and self.last_quality) else ""
        issue = note if note and note not in QUALITY_OK_NOTES else ""
        if issue != self._quality_issue:
            self._quality_issue = issue
            self._quality_issue_since = t
        if not issue:
            if self._quality_issue_since is not None and t - self._quality_issue_since >= QUALITY_ISSUE_DELAY:
                self._quality_reported = ""
            return
        if issue != self._quality_reported and t - self._quality_issue_since >= QUALITY_ISSUE_DELAY:
            self._quality_reported = issue
            self.emit(t, EV_QUALITY, f"Качество кадра: {issue}", "warning")

    def _update_closure_events(self, t, face_ok):
        closed_for = self.eye_tracker.current_closure_seconds(t) if face_ok else 0.0
        if closed_for >= LONG_CLOSURE_SECONDS and not self._long_closure_reported:
            self._long_closure_reported = True
            self.emit(t, EV_EYES, f"Глаза закрыты дольше {LONG_CLOSURE_SECONDS:.0f} с")
        elif closed_for == 0.0:
            self._long_closure_reported = False

    def _update_calibration_events(self, t):
        ready = (self.mimic_analyzer.neutral_ready, self.redness_analyzer.neutral_ready,
                 self.suspicion_analyzer.baseline_ready)
        mimic_was, redness_was, base_was = self._ready
        if ready[0] and not mimic_was:
            self.emit(t, EV_CALIBRATION, "Мимика: нейтральное состояние зафиксировано")
        if ready[1] and not redness_was:
            self.emit(t, EV_CALIBRATION, "Покраснение: исходный цвет кожи зафиксирован")
        if ready[2] and not base_was:
            self._finish_display_baseline()
            rate = self.suspicion_analyzer.base_blink_rate
            self.emit(t, EV_CALIBRATION,
                      f"Личная база собрана: моргание {rate:.0f}/мин", "notice")
        self._ready = ready

    def _collect_display_baseline(self, gaze_off):
        sus = self.suspicion_analyzer
        if sus.baseline_ready or not self.quality_good:
            return
        if gaze_off is not None:
            self._baseline_gaze.append(gaze_off)
        if self._activity is not None:
            self._baseline_activity.append(self._activity)

    def _finish_display_baseline(self):
        def mean(values):
            return float(sum(values) / len(values)) if values else None

        self.display_baseline = {"gaze_off": mean(self._baseline_gaze),
                                 "activity": mean(self._baseline_activity)}
        self._baseline_gaze, self._baseline_activity = [], []

    # =====================================================
    # подсветка для интерфейса
    # =====================================================

    def build_overlay(self, with_mesh=False):
        face = self.last_face
        overlay = OverlayData(frame_size=(face.frame_width, face.frame_height))
        if not face.detected or not face.landmarks:
            return overlay
        pts = face.landmarks
        pick = lambda ids: [pts[i] for i in ids]
        eyes = self.eye_tracker

        def iris(ids):
            points = pick(ids)
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            radius = sum(math.hypot(p[0] - cx, p[1] - cy) for p in points) / len(points)
            return cx, cy, max(radius, 2.0)

        overlay.bbox = face.bbox
        overlay.left_eye = pick(eyes.LEFT_EYE_CONTOUR)
        overlay.right_eye = pick(eyes.RIGHT_EYE_CONTOUR)
        if len(pts) > max(eyes.RIGHT_IRIS):
            overlay.left_iris = iris(eyes.LEFT_IRIS)
            overlay.right_iris = iris(eyes.RIGHT_IRIS)
        overlay.eyes_closed = eyes.eyes_closed
        if with_mesh:
            overlay.landmarks = list(pts)
        return overlay


def describe_face_error(error_text):
    """Короткое объяснение для интерфейса."""
    text = (error_text or "").lower()
    if "mediapipe" in text or "no module" in text:
        return "не установлен пакет mediapipe"
    if "не найдена модель" in text or "no such file" in text:
        return "нет файла models/face_landmarker.task"
    return error_text or "неизвестная ошибка"
