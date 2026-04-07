# -*- coding: utf-8 -*-
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models import MimicData


class MimicAnalyzerModule:
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291
    UPPER_LIP = 13
    LOWER_LIP = 14

    LEFT_BROW_POINTS = [70, 63, 105, 66, 107]
    RIGHT_BROW_POINTS = [300, 293, 334, 296, 336]

    LEFT_BROW_INNER_POINTS = [66, 107]
    RIGHT_BROW_INNER_POINTS = [296, 336]

    LEFT_EYE_TOP_POINTS = [159, 158]
    RIGHT_EYE_TOP_POINTS = [386, 385]

    LEFT_EYE_BOTTOM_POINTS = [145, 153]
    RIGHT_EYE_BOTTOM_POINTS = [374, 380]

    def __init__(self):
        self.last_result = MimicData()

        self.neutral_ready = False
        self.neutral_samples = []

        self.neutral_brow_eye_norm = None
        self.neutral_brow_inner_norm = None
        self.neutral_mouth_width_norm = None
        self.neutral_mouth_open_norm = None
        self.neutral_eye_open_norm = None

        self.smooth_smile = deque(maxlen=5)
        self.smooth_mouth_open = deque(maxlen=5)
        self.smooth_brow_raise = deque(maxlen=5)
        self.smooth_brow_frown = deque(maxlen=5)
        self.smooth_eye_squint = deque(maxlen=5)

        try:
            self.font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
        except Exception:
            self.font = ImageFont.load_default()

    def dist(self, p1, p2):
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def clamp01(self, value):
        return float(max(0.0, min(1.0, value)))

    def shorten_text(self, text, max_len=28):
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."

    def mean_point(self, points):
        arr = np.array(points, dtype=np.float32)
        center = np.mean(arr, axis=0)
        return (float(center[0]), float(center[1]))

    def get_points_by_indices(self, pts, indices):
        return [pts[i] for i in indices]

    def mean_smooth(self, history, value):
        history.append(value)
        return float(sum(history) / len(history))

    def extract_features(self, face_data):
        pts = face_data.landmarks

        left_mouth = pts[self.LEFT_MOUTH]
        right_mouth = pts[self.RIGHT_MOUTH]
        upper_lip = pts[self.UPPER_LIP]
        lower_lip = pts[self.LOWER_LIP]

        left_brow_points = self.get_points_by_indices(pts, self.LEFT_BROW_POINTS)
        right_brow_points = self.get_points_by_indices(pts, self.RIGHT_BROW_POINTS)

        left_brow_inner_points = self.get_points_by_indices(pts, self.LEFT_BROW_INNER_POINTS)
        right_brow_inner_points = self.get_points_by_indices(pts, self.RIGHT_BROW_INNER_POINTS)

        left_eye_top_points = self.get_points_by_indices(pts, self.LEFT_EYE_TOP_POINTS)
        right_eye_top_points = self.get_points_by_indices(pts, self.RIGHT_EYE_TOP_POINTS)

        left_eye_bottom_points = self.get_points_by_indices(pts, self.LEFT_EYE_BOTTOM_POINTS)
        right_eye_bottom_points = self.get_points_by_indices(pts, self.RIGHT_EYE_BOTTOM_POINTS)

        left_brow_center = self.mean_point(left_brow_points)
        right_brow_center = self.mean_point(right_brow_points)

        left_brow_inner = self.mean_point(left_brow_inner_points)
        right_brow_inner = self.mean_point(right_brow_inner_points)

        left_eye_top = self.mean_point(left_eye_top_points)
        right_eye_top = self.mean_point(right_eye_top_points)

        left_eye_bottom = self.mean_point(left_eye_bottom_points)
        right_eye_bottom = self.mean_point(right_eye_bottom_points)

        face_w = max(face_data.bbox[2] - face_data.bbox[0], 1)
        face_h = max(face_data.bbox[3] - face_data.bbox[1], 1)

        mouth_width = self.dist(left_mouth, right_mouth)
        mouth_open = self.dist(upper_lip, lower_lip)

        left_brow_eye = self.dist(left_brow_center, left_eye_top)
        right_brow_eye = self.dist(right_brow_center, right_eye_top)
        avg_brow_eye = (left_brow_eye + right_brow_eye) / 2.0

        brow_inner_dist = self.dist(left_brow_inner, right_brow_inner)

        left_eye_open = self.dist(left_eye_top, left_eye_bottom)
        right_eye_open = self.dist(right_eye_top, right_eye_bottom)
        avg_eye_open = (left_eye_open + right_eye_open) / 2.0

        return {
            "mouth_width_norm": mouth_width / face_w,
            "mouth_open_norm": mouth_open / face_h,
            "brow_eye_norm": avg_brow_eye / face_h,
            "brow_inner_norm": brow_inner_dist / face_w,
            "eye_open_norm": avg_eye_open / face_h,
        }

    def start_calibration(self):
        self.neutral_samples = []
        self.neutral_ready = False
        print("Калибровка началась: смотри прямо и держи нейтральное лицо")

    def update_calibration(self, face_data):
        if not face_data.detected or not face_data.landmarks or not face_data.bbox:
            return False

        features = self.extract_features(face_data)
        self.neutral_samples.append(features)

        if len(self.neutral_samples) < 25:
            return False

        self.neutral_brow_eye_norm = float(np.mean([x["brow_eye_norm"] for x in self.neutral_samples]))
        self.neutral_brow_inner_norm = float(np.mean([x["brow_inner_norm"] for x in self.neutral_samples]))
        self.neutral_mouth_width_norm = float(np.mean([x["mouth_width_norm"] for x in self.neutral_samples]))
        self.neutral_mouth_open_norm = float(np.mean([x["mouth_open_norm"] for x in self.neutral_samples]))
        self.neutral_eye_open_norm = float(np.mean([x["eye_open_norm"] for x in self.neutral_samples]))

        self.neutral_ready = True
        self.neutral_samples = []

        print("Калибровка завершена")
        return True

    def auto_prepare_neutral(self, face_data):
        if self.neutral_ready:
            return
        self.update_calibration(face_data)

    def process(self, face_data):
        if not face_data.detected or not face_data.landmarks or not face_data.bbox:
            self.last_result = MimicData(dominant_expression="лица нет")
            return self.last_result

        self.auto_prepare_neutral(face_data)

        features = self.extract_features(face_data)

        mouth_width_norm = features["mouth_width_norm"]
        mouth_open_norm = features["mouth_open_norm"]
        brow_eye_norm = features["brow_eye_norm"]
        brow_inner_norm = features["brow_inner_norm"]
        eye_open_norm = features["eye_open_norm"]

        if self.neutral_ready:
            brow_raise_delta = brow_eye_norm - self.neutral_brow_eye_norm
            brow_down_delta = self.neutral_brow_eye_norm - brow_eye_norm
            brow_close_delta = self.neutral_brow_inner_norm - brow_inner_norm

            smile_delta = mouth_width_norm - self.neutral_mouth_width_norm
            mouth_open_delta = mouth_open_norm - self.neutral_mouth_open_norm
            eye_squint_delta = self.neutral_eye_open_norm - eye_open_norm

            smile_score = self.clamp01(smile_delta * 18.0)
            mouth_open_score = self.clamp01(mouth_open_delta * 28.0)
            brow_raise_score = self.clamp01(brow_raise_delta * 40.0)

            brow_close_score = self.clamp01(brow_close_delta * 55.0)
            brow_down_score = self.clamp01(brow_down_delta * 45.0)
            brow_frown_score = self.clamp01(0.6 * brow_close_score + 0.4 * brow_down_score)

            eye_squint_score = self.clamp01(eye_squint_delta * 35.0)
        else:
            smile_score = self.clamp01((mouth_width_norm - 0.22) * 5.0)
            mouth_open_score = self.clamp01((mouth_open_norm - 0.02) * 12.0)
            brow_raise_score = self.clamp01((brow_eye_norm - 0.050) * 18.0)
            brow_frown_score = self.clamp01((0.22 - brow_inner_norm) * 8.0)
            eye_squint_score = self.clamp01((0.045 - eye_open_norm) * 22.0)

        smile_score = self.mean_smooth(self.smooth_smile, smile_score)
        mouth_open_score = self.mean_smooth(self.smooth_mouth_open, mouth_open_score)
        brow_raise_score = self.mean_smooth(self.smooth_brow_raise, brow_raise_score)
        brow_frown_score = self.mean_smooth(self.smooth_brow_frown, brow_frown_score)
        eye_squint_score = self.mean_smooth(self.smooth_eye_squint, eye_squint_score)

        if smile_score > 0.50:
            dominant = "улыбка"
        elif mouth_open_score > 0.45:
            dominant = "рот открыт"
        elif brow_frown_score > 0.22:
            dominant = "нахмуренность"
        elif brow_raise_score > 0.22:
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

    def draw_panel(self, frame, mimic_data, micro_event=None):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)

        frame_width = frame.shape[1]

        panel_width = 360
        panel_height = 235
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
            event_name = self.shorten_text(micro_event.name, max_len=24)

        calib_text = "да" if self.neutral_ready else f"{len(self.neutral_samples)}/25"

        lines = [
            f"Мимика: {mimic_data.dominant_expression}",
            f"Калибровка: {calib_text}",
            f"Улыбка: {mimic_data.smile:.2f}",
            f"Рот открыт: {mimic_data.mouth_open:.2f}",
            f"Брови вверх: {mimic_data.brow_raise:.2f}",
            f"Нахмур.: {mimic_data.brow_frown:.2f}",
            f"Прищур: {mimic_data.eye_squint:.2f}",
            f"Микро: {event_name}"
        ]

        y = y1 + 10
        for line in lines:
            draw.text((x1 + 12, y), line, font=self.font, fill=(255, 255, 255))
            y += 26

        result = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return result