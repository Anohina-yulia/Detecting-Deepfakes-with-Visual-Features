# -*- coding: utf-8 -*-
from collections import deque

import numpy as np

from models import (EXPR_BROWS, EXPR_FROWN, EXPR_MOUTH, EXPR_NEUTRAL, EXPR_NO_FACE,
                    EXPR_SMILE, EXPR_SQUINT, MimicData)
from ui_widgets import (BLUE, GRAY, GREEN, ORANGE, WHITE, YELLOW, clamp01, draw_bar,
                        load_font, new_panel, to_bgra)

CALIBRATION_FRAMES = 20

# (подпись = название выражения, поле MimicData, цвет при доминировании)
BARS = [
    (EXPR_SMILE, "smile", GREEN),
    (EXPR_MOUTH, "mouth_open", BLUE),
    (EXPR_BROWS, "brow_raise", YELLOW),
    (EXPR_FROWN, "brow_frown", ORANGE),
    (EXPR_SQUINT, "eye_squint", YELLOW),
]
FEATURES = ["smile", "mouth_open", "brow_raise", "brow_frown", "eye_squint"]


class MimicAnalyzerModule:
    def __init__(self):
        self.last_result = MimicData()
        self.neutral_ready = False
        self.neutral_samples = []
        self.neutral = {name: 0.0 for name in FEATURES}
        self.smooth = {name: deque(maxlen=5) for name in FEATURES}

    # ---------- признаки ----------

    @staticmethod
    def _score(blend, names):
        values = [float(blend.get(n, 0.0)) for n in names]
        return float(sum(values) / len(values)) if values else 0.0

    def extract_raw_features(self, face_data):
        blend = getattr(face_data, "blendshapes", {}) or {}
        return {
            "smile": self._score(blend, ["mouthSmileLeft", "mouthSmileRight"]),
            "mouth_open": self._score(blend, ["jawOpen"]),
            "brow_raise": (0.55 * self._score(blend, ["browInnerUp"])
                           + 0.225 * self._score(blend, ["browOuterUpLeft"])
                           + 0.225 * self._score(blend, ["browOuterUpRight"])),
            "brow_frown": self._score(blend, ["browDownLeft", "browDownRight"]),
            "eye_squint": self._score(blend, ["eyeSquintLeft", "eyeSquintRight"]),
        }

    # ---------- калибровка ----------

    def start_calibration(self):
        self.neutral_samples = []
        self.neutral_ready = False
        for history in self.smooth.values():
            history.clear()
        print("Калибровка мимики: смотрите прямо, лицо нейтральное, не говорите.")

    def update_calibration(self, face_data):
        self.neutral_samples.append(self.extract_raw_features(face_data))
        if len(self.neutral_samples) < CALIBRATION_FRAMES:
            return False

        # медиана устойчивее среднего к случайной улыбке или слову во время калибровки
        for name in FEATURES:
            self.neutral[name] = float(np.median([s[name] for s in self.neutral_samples]))

        self.neutral_ready = True
        self.neutral_samples = []
        print("Калибровка мимики завершена.")
        return True

    @property
    def calibration_progress(self):
        return CALIBRATION_FRAMES if self.neutral_ready else len(self.neutral_samples)

    # ---------- обработка ----------

    @staticmethod
    def _from_baseline(raw, baseline, gain, deadzone):
        return clamp01((raw - baseline - deadzone) * gain)

    def process(self, face_data, allow_calibration=True):
        """allow_calibration=False - не набирать калибровку на плохих кадрах."""
        if not face_data.detected:
            self.last_result = MimicData(dominant_expression=EXPR_NO_FACE)
            return self.last_result

        if not self.neutral_ready and allow_calibration:
            self.update_calibration(face_data)

        raw = self.extract_raw_features(face_data)

        if self.neutral_ready:
            n = self.neutral
            scores = {
                "smile": self._from_baseline(raw["smile"], n["smile"], 5.5, 0.03),
                "mouth_open": self._from_baseline(raw["mouth_open"], n["mouth_open"], 5.5, 0.03),
                "brow_raise": self._from_baseline(raw["brow_raise"], n["brow_raise"], 6.0, 0.02),
                "brow_frown": self._from_baseline(raw["brow_frown"], n["brow_frown"], 5.5, 0.02),
                "eye_squint": self._from_baseline(raw["eye_squint"], n["eye_squint"], 6.0, 0.02),
            }
        else:
            scores = {
                "smile": clamp01((raw["smile"] - 0.08) * 4.0),
                "mouth_open": clamp01((raw["mouth_open"] - 0.08) * 4.0),
                "brow_raise": clamp01((raw["brow_raise"] - 0.07) * 5.0),
                "brow_frown": clamp01((raw["brow_frown"] - 0.07) * 5.0),
                "eye_squint": clamp01((raw["eye_squint"] - 0.08) * 5.0),
            }

        for name in FEATURES:
            self.smooth[name].append(scores[name])
            scores[name] = float(sum(self.smooth[name]) / len(self.smooth[name]))

        if scores["smile"] > 0.38:
            dominant = EXPR_SMILE
        elif scores["mouth_open"] > 0.35:
            dominant = EXPR_MOUTH
        elif scores["brow_frown"] > 0.30:
            dominant = EXPR_FROWN
        elif scores["brow_raise"] > 0.28:
            dominant = EXPR_BROWS
        elif scores["eye_squint"] > 0.30:
            dominant = EXPR_SQUINT
        else:
            dominant = EXPR_NEUTRAL

        self.last_result = MimicData(dominant_expression=dominant, **scores)
        return self.last_result

    # ---------- панель ----------

    @staticmethod
    def _bar_color(label, value, dominant, active_color):
        if dominant == label:
            return active_color
        return BLUE if value > 0.22 else GRAY

    def render_panel(self, mimic_data, micro_event=None):
        img, draw = new_panel(420, 285)
        font, small = load_font(17), load_font(15)

        colors = {label: color for label, _, color in BARS}
        dominant = mimic_data.dominant_expression
        title_color = colors.get(dominant, WHITE)
        draw.text((12, 10), f"Мимика: {dominant}", font=font, fill=title_color)

        calib = ("готово" if self.neutral_ready
                 else f"{self.calibration_progress}/{CALIBRATION_FRAMES}")
        draw.text((12, 38), f"Калибровка: {calib}", font=small, fill=GRAY)

        for i, (label, attr, active) in enumerate(BARS):
            value = getattr(mimic_data, attr)
            color = self._bar_color(label, value, dominant, active)
            draw_bar(draw, 12, 72 + 32 * i, value, label, color, small,
                     label_w=115, width=170, label_color=color)

        event_name = "нет"
        if micro_event is not None:
            event_name = getattr(micro_event, "name", "нет")
        draw.text((12, 285 - 32), f"Микровыражение: {event_name}", font=small, fill=WHITE)
        return to_bgra(img)