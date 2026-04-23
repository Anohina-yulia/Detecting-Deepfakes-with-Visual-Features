# -*- coding: utf-8 -*-
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QualityData:
    good: bool = False
    score: float = 0.0
    face_size: float = 0.0
    brightness: float = 0.0
    sharpness: float = 0.0
    symmetry: float = 0.0
    note: str = "лица нет"


class QualityAnalyzerModule:
    def __init__(self):
        pass

    def clamp01(self, value):
        return float(max(0.0, min(1.0, value)))

    def brightness_score(self, brightness):
        # идеальный диапазон примерно 95..185
        if brightness < 45:
            return 0.0
        if brightness < 95:
            return self.clamp01((brightness - 45) / 50.0)
        if brightness <= 185:
            return 1.0
        if brightness <= 235:
            return self.clamp01(1.0 - (brightness - 185) / 50.0)
        return 0.0

    def sharpness_score(self, sharpness):
        return self.clamp01((sharpness - 30.0) / 120.0)

    def face_size_score(self, face_ratio):
        return self.clamp01((face_ratio - 0.05) / 0.10)

    def symmetry_score(self, diff):
        return self.clamp01(1.0 - diff / 65.0)

    def process(self, frame, face_data):
        if not face_data.detected or not face_data.bbox:
            return QualityData(
                good=False,
                score=0.0,
                note="лица нет"
            )

        x1, y1, x2, y2 = face_data.bbox
        h, w = frame.shape[:2]

        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return QualityData(
                good=False,
                score=0.0,
                note="пустой crop"
            )

        face_area = max((x2 - x1) * (y2 - y1), 1)
        frame_area = max(w * h, 1)
        face_ratio = face_area / frame_area

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        mid = gray.shape[1] // 2
        if mid > 5:
            left_brightness = float(np.mean(gray[:, :mid]))
            right_brightness = float(np.mean(gray[:, mid:]))
            symmetry_diff = abs(left_brightness - right_brightness)
        else:
            symmetry_diff = 100.0

        face_size_s = self.face_size_score(face_ratio)
        brightness_s = self.brightness_score(brightness)
        sharpness_s = self.sharpness_score(sharpness)
        symmetry_s = self.symmetry_score(symmetry_diff)

        center_s = 1.0 if face_data.face_position == "по центру" else 0.55

        total = (
            0.34 * face_size_s +
            0.24 * brightness_s +
            0.22 * sharpness_s +
            0.10 * symmetry_s +
            0.10 * center_s
        )

        note = "норма"
        if face_size_s < 0.35:
            note = "подойди ближе"
        elif brightness_s < 0.35:
            note = "поправь свет"
        elif sharpness_s < 0.30:
            note = "мыло/дрожание"
        elif center_s < 0.8:
            note = "лицо не по центру"
        elif symmetry_s < 0.35:
            note = "свет неравномерный"

        good = (
            total >= 0.58
            and face_ratio >= 0.05
            and sharpness >= 25.0
        )

        return QualityData(
            good=good,
            score=total,
            face_size=face_ratio,
            brightness=brightness,
            sharpness=sharpness,
            symmetry=symmetry_s,
            note=note
        )