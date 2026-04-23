# -*- coding: utf-8 -*-
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models import MimicData


class MimicAnalyzerModule:
    def __init__(self):
        self.last_result = MimicData()

        self.neutral_ready = False
        self.neutral_samples = []

        self.neutral_smile = 0.0
        self.neutral_mouth_open = 0.0
        self.neutral_brow_raise = 0.0
        self.neutral_brow_frown = 0.0
        self.neutral_eye_squint = 0.0

        self.smooth_smile = deque(maxlen=5)
        self.smooth_mouth_open = deque(maxlen=5)
        self.smooth_brow_raise = deque(maxlen=5)
        self.smooth_brow_frown = deque(maxlen=5)
        self.smooth_eye_squint = deque(maxlen=5)

        try:
            self.font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
            self.small_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
        except Exception:
            self.font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()

    def clamp01(self, value):
        return float(max(0.0, min(1.0, value)))

    def shorten_text(self, text, max_len=28):
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."

    def mean_smooth(self, history, value):
        history.append(float(value))
        return float(sum(history) / len(history))

    def get_score(self, blendshapes, names, default=0.0):
        if not blendshapes:
            return float(default)

        values = [float(blendshapes.get(name, 0.0)) for name in names]
        if not values:
            return float(default)

        return float(sum(values) / len(values))

    def extract_raw_features(self, face_data):
        blend = getattr(face_data, "blendshapes", {}) or {}

        smile_raw = self.get_score(blend, ["mouthSmileLeft", "mouthSmileRight"])
        mouth_open_raw = max(
            self.get_score(blend, ["jawOpen"]),
            self.get_score(blend, ["mouthOpen"])
        )
        brow_raise_raw = (
            0.55 * self.get_score(blend, ["browInnerUp"]) +
            0.225 * self.get_score(blend, ["browOuterUpLeft"]) +
            0.225 * self.get_score(blend, ["browOuterUpRight"])
        )
        brow_frown_raw = self.get_score(blend, ["browDownLeft", "browDownRight"])
        eye_squint_raw = self.get_score(blend, ["eyeSquintLeft", "eyeSquintRight"])

        return {
            "smile": smile_raw,
            "mouth_open": mouth_open_raw,
            "brow_raise": brow_raise_raw,
            "brow_frown": brow_frown_raw,
            "eye_squint": eye_squint_raw,
        }

    def start_calibration(self):
        self.neutral_samples = []
        self.neutral_ready = False
        print("Калибровка мимики началась: смотри прямо и держи нейтральное лицо")

    def update_calibration(self, face_data):
        if not face_data.detected:
            return False

        raw = self.extract_raw_features(face_data)
        self.neutral_samples.append(raw)

        if len(self.neutral_samples) < 20:
            return False

        self.neutral_smile = float(np.mean([x["smile"] for x in self.neutral_samples]))
        self.neutral_mouth_open = float(np.mean([x["mouth_open"] for x in self.neutral_samples]))
        self.neutral_brow_raise = float(np.mean([x["brow_raise"] for x in self.neutral_samples]))
        self.neutral_brow_frown = float(np.mean([x["brow_frown"] for x in self.neutral_samples]))
        self.neutral_eye_squint = float(np.mean([x["eye_squint"] for x in self.neutral_samples]))

        self.neutral_ready = True
        self.neutral_samples = []
        print("Калибровка мимики завершена")
        return True

    def auto_prepare_neutral(self, face_data):
        if self.neutral_ready:
            return
        self.update_calibration(face_data)

    def score_from_baseline(self, raw_value, baseline, gain, deadzone=0.02):
        return self.clamp01((raw_value - baseline - deadzone) * gain)

    def process(self, face_data):
        if not face_data.detected:
            self.last_result = MimicData(dominant_expression="лица нет")
            return self.last_result

        self.auto_prepare_neutral(face_data)

        raw = self.extract_raw_features(face_data)

        if self.neutral_ready:
            smile_score = self.score_from_baseline(raw["smile"], self.neutral_smile, gain=5.5, deadzone=0.03)
            mouth_open_score = self.score_from_baseline(raw["mouth_open"], self.neutral_mouth_open, gain=5.5, deadzone=0.03)
            brow_raise_score = self.score_from_baseline(raw["brow_raise"], self.neutral_brow_raise, gain=6.0, deadzone=0.02)
            brow_frown_score = self.score_from_baseline(raw["brow_frown"], self.neutral_brow_frown, gain=5.5, deadzone=0.02)
            eye_squint_score = self.score_from_baseline(raw["eye_squint"], self.neutral_eye_squint, gain=6.0, deadzone=0.02)
        else:
            smile_score = self.clamp01((raw["smile"] - 0.08) * 4.0)
            mouth_open_score = self.clamp01((raw["mouth_open"] - 0.08) * 4.0)
            brow_raise_score = self.clamp01((raw["brow_raise"] - 0.07) * 5.0)
            brow_frown_score = self.clamp01((raw["brow_frown"] - 0.07) * 5.0)
            eye_squint_score = self.clamp01((raw["eye_squint"] - 0.08) * 5.0)

        smile_score = self.mean_smooth(self.smooth_smile, smile_score)
        mouth_open_score = self.mean_smooth(self.smooth_mouth_open, mouth_open_score)
        brow_raise_score = self.mean_smooth(self.smooth_brow_raise, brow_raise_score)
        brow_frown_score = self.mean_smooth(self.smooth_brow_frown, brow_frown_score)
        eye_squint_score = self.mean_smooth(self.smooth_eye_squint, eye_squint_score)

        if smile_score > 0.38:
            dominant = "улыбка"
        elif mouth_open_score > 0.35:
            dominant = "рот открыт"
        elif brow_frown_score > 0.30:
            dominant = "нахмуренность"
        elif brow_raise_score > 0.28:
            dominant = "подняты брови"
        elif eye_squint_score > 0.30:
            dominant = "прищур"
        else:
            dominant = "нейтрально"

        self.last_result = MimicData(
            smile=smile_score,
            mouth_open=mouth_open_score,
            brow_raise=brow_raise_score,
            brow_frown=brow_frown_score,
            eye_squint=eye_squint_score,
            dominant_expression=dominant
        )
        return self.last_result

    def get_emotion_color(self, label, value, dominant_expression):
        inactive = (210, 210, 210)
        active_blue = (80, 190, 255)
        active_green = (110, 230, 140)
        active_orange = (255, 185, 90)
        active_yellow = (245, 225, 100)

        if dominant_expression == "улыбка" and label == "Улыбка":
            return active_green
        if dominant_expression == "рот открыт" and label == "Рот открыт":
            return active_blue
        if dominant_expression == "подняты брови" and label == "Брови вверх":
            return active_yellow
        if dominant_expression == "нахмуренность" and label == "Нахмур.":
            return active_orange
        if dominant_expression == "прищур" and label == "Прищур":
            return active_yellow

        if value > 0.22:
            return active_blue

        return inactive

    def draw_progress_bar(self, draw, x, y, width, height, value, label, dominant_expression):
        color = self.get_emotion_color(label, value, dominant_expression)

        draw.text((x, y - 2), label, font=self.small_font, fill=color)

        bar_x1 = x + 115
        bar_y1 = y + 2
        bar_x2 = bar_x1 + width
        bar_y2 = bar_y1 + height

        draw.rounded_rectangle(
            (bar_x1, bar_y1, bar_x2, bar_y2),
            radius=5,
            fill=(45, 45, 45),
            outline=(110, 110, 110),
            width=1
        )

        fill_width = int(width * self.clamp01(value))
        if fill_width > 0:
            draw.rounded_rectangle(
                (bar_x1, bar_y1, bar_x1 + fill_width, bar_y2),
                radius=5,
                fill=color
            )

        draw.text((bar_x2 + 8, y - 2), f"{value:.2f}", font=self.small_font, fill=(255, 255, 255))

    def draw_panel(self, frame, mimic_data, micro_event=None):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)

        frame_width = frame.shape[1]

        panel_width = 420
        panel_height = 285
        x1 = frame_width - panel_width - 15
        y1 = 50
        x2 = frame_width - 15
        y2 = y1 + panel_height

        draw.rounded_rectangle(
            (x1, y1, x2, y2),
            radius=10,
            fill=(0, 0, 0),
            outline=(255, 255, 255),
            width=2
        )

        event_name = "нет"
        if micro_event is not None:
            event_name = self.shorten_text(getattr(micro_event, "name", "нет"), max_len=28)

        calib_text = "да" if self.neutral_ready else f"{len(self.neutral_samples)}/20"

        dominant_color = self.get_emotion_color(
            label={
                "улыбка": "Улыбка",
                "рот открыт": "Рот открыт",
                "подняты брови": "Брови вверх",
                "нахмуренность": "Нахмур.",
                "прищур": "Прищур",
                "нейтрально": ""
            }.get(mimic_data.dominant_expression, ""),
            value=max(
                mimic_data.smile,
                mimic_data.mouth_open,
                mimic_data.brow_raise,
                mimic_data.brow_frown,
                mimic_data.eye_squint
            ),
            dominant_expression=mimic_data.dominant_expression
        )

        draw.text(
            (x1 + 12, y1 + 10),
            f"Мимика: {mimic_data.dominant_expression}",
            font=self.font,
            fill=dominant_color if mimic_data.dominant_expression != "нейтрально" else (255, 255, 255)
        )
        draw.text(
            (x1 + 12, y1 + 38),
            f"Калибровка: {calib_text}",
            font=self.small_font,
            fill=(220, 220, 220)
        )

        start_y = y1 + 72
        bar_width = 170
        bar_height = 14
        gap = 32

        self.draw_progress_bar(draw, x1 + 12, start_y + gap * 0, bar_width, bar_height, mimic_data.smile, "Улыбка", mimic_data.dominant_expression)
        self.draw_progress_bar(draw, x1 + 12, start_y + gap * 1, bar_width, bar_height, mimic_data.mouth_open, "Рот открыт", mimic_data.dominant_expression)
        self.draw_progress_bar(draw, x1 + 12, start_y + gap * 2, bar_width, bar_height, mimic_data.brow_raise, "Брови вверх", mimic_data.dominant_expression)
        self.draw_progress_bar(draw, x1 + 12, start_y + gap * 3, bar_width, bar_height, mimic_data.brow_frown, "Нахмур.", mimic_data.dominant_expression)
        self.draw_progress_bar(draw, x1 + 12, start_y + gap * 4, bar_width, bar_height, mimic_data.eye_squint, "Прищур", mimic_data.dominant_expression)

        draw.text(
            (x1 + 12, y2 - 32),
            f"Микро: {event_name}",
            font=self.small_font,
            fill=(255, 255, 255)
        )

        result = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return result