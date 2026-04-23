# -*- coding: utf-8 -*-
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models import RednessData


class RednessAnalyzerModule:
    def __init__(self):
        self.history_left = deque(maxlen=6)
        self.history_right = deque(maxlen=6)
        self.history_nose = deque(maxlen=6)
        self.history_forehead = deque(maxlen=6)

        self.neutral_ready = False
        self.neutral_samples = []

        self.neutral_left = 0.0
        self.neutral_right = 0.0
        self.neutral_nose = 0.0
        self.neutral_forehead = 0.0

        try:
            self.font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
            self.small_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
        except Exception:
            self.font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()

    def clamp01(self, value):
        return float(max(0.0, min(1.0, value)))

    def mean_point(self, points):
        arr = np.array(points, dtype=np.float32)
        center = np.mean(arr, axis=0)
        return int(center[0]), int(center[1])

    def smooth(self, history, value):
        history.append(float(value))
        return float(sum(history) / len(history))

    def start_calibration(self):
        self.neutral_ready = False
        self.neutral_samples = []
        print("Калибровка покраснений началась: смотри прямо, нейтральное лицо, ровный свет")

    def compute_raw_redness(self, patch):
        if patch is None or patch.size == 0:
            return 0.0

        b, g, r = cv2.split(patch)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)

        _, s_ch, _ = cv2.split(hsv)
        y_ch, cr_ch, cb_ch = cv2.split(ycrcb)

        skin_mask = (
            (cr_ch > 133) & (cr_ch < 178) &
            (cb_ch > 78) & (cb_ch < 132) &
            (y_ch > 55) & (y_ch < 225) &
            (s_ch > 15) & (s_ch < 170) &
            (r > 55) & (g > 35) & (b > 20)
        )

        if np.count_nonzero(skin_mask) < 40:
            return 0.0

        r_m = float(np.mean(r[skin_mask]))
        g_m = float(np.mean(g[skin_mask]))
        b_m = float(np.mean(b[skin_mask]))
        cr_m = float(np.mean(cr_ch[skin_mask]))
        cb_m = float(np.mean(cb_ch[skin_mask]))
        y_m = float(np.mean(y_ch[skin_mask]))

        total = max(r_m + g_m + b_m, 1.0)

        warmth = (r_m - g_m) / total
        ex_red = (1.4 * r_m - g_m - 0.4 * b_m) / total
        cr_bias = (cr_m - cb_m) / 255.0

        brightness_penalty = 0.0
        if y_m < 80:
            brightness_penalty = 0.03

        raw = (
            0.45 * warmth +
            0.35 * ex_red +
            0.20 * cr_bias -
            brightness_penalty
        )

        return self.clamp01((raw - 0.015) * 4.2)

    def sample_patch_score(self, frame, center, patch_w=24, patch_h=18):
        h, w = frame.shape[:2]
        cx, cy = center

        x1 = max(cx - patch_w, 0)
        y1 = max(cy - patch_h, 0)
        x2 = min(cx + patch_w, w - 1)
        y2 = min(cy + patch_h, h - 1)

        if x2 <= x1 or y2 <= y1:
            return 0.0, (x1, y1, x2, y2)

        patch = frame[y1:y2, x1:x2]
        raw_score = self.compute_raw_redness(patch)
        return raw_score, (x1, y1, x2, y2)

    def status_from_score(self, score):
        if score < 0.10:
            return "норма"
        if score < 0.22:
            return "легкое"
        if score < 0.38:
            return "заметное"
        return "сильное"

    def update_calibration(self, raw_left, raw_right, raw_nose, raw_forehead):
        sample = {
            "left": raw_left,
            "right": raw_right,
            "nose": raw_nose,
            "forehead": raw_forehead
        }

        valid_count = sum(1 for v in sample.values() if v > 0.0)
        if valid_count < 3:
            return False

        self.neutral_samples.append(sample)

        if len(self.neutral_samples) < 25:
            return False

        self.neutral_left = float(np.mean([x["left"] for x in self.neutral_samples]))
        self.neutral_right = float(np.mean([x["right"] for x in self.neutral_samples]))
        self.neutral_nose = float(np.mean([x["nose"] for x in self.neutral_samples]))
        self.neutral_forehead = float(np.mean([x["forehead"] for x in self.neutral_samples]))

        self.neutral_ready = True
        self.neutral_samples = []
        print("Калибровка покраснений завершена")
        return True

    def process(self, frame, face_data):
        if not face_data.detected or not face_data.landmarks:
            return RednessData(status="лица нет"), {}

        pts = face_data.landmarks

        left_cheek_center = self.mean_point([pts[205], pts[50], pts[123], pts[116]])
        right_cheek_center = self.mean_point([pts[425], pts[280], pts[352], pts[345]])
        nose_center = self.mean_point([pts[1], pts[5], pts[195], pts[4]])
        forehead_center = self.mean_point([pts[10], pts[151], pts[9]])

        raw_left, left_box = self.sample_patch_score(frame, left_cheek_center, 28, 20)
        raw_right, right_box = self.sample_patch_score(frame, right_cheek_center, 28, 20)
        raw_nose, nose_box = self.sample_patch_score(frame, nose_center, 20, 16)
        raw_forehead, forehead_box = self.sample_patch_score(frame, forehead_center, 30, 18)

        boxes = {
            "left_cheek": left_box,
            "right_cheek": right_box,
            "nose": nose_box,
            "forehead": forehead_box
        }

        if not self.neutral_ready:
            self.update_calibration(raw_left, raw_right, raw_nose, raw_forehead)
            progress = len(self.neutral_samples)
            return RednessData(
                overall=0.0,
                left_cheek=0.0,
                right_cheek=0.0,
                nose=0.0,
                forehead=0.0,
                status=f"калибр. {progress}/25"
            ), boxes

        deadzone = 0.035

        left_score = self.clamp01((raw_left - self.neutral_left - deadzone) * 7.5)
        right_score = self.clamp01((raw_right - self.neutral_right - deadzone) * 7.5)
        nose_score = self.clamp01((raw_nose - self.neutral_nose - deadzone) * 7.0)
        forehead_score = self.clamp01((raw_forehead - self.neutral_forehead - deadzone) * 6.5)

        left_score = self.smooth(self.history_left, left_score)
        right_score = self.smooth(self.history_right, right_score)
        nose_score = self.smooth(self.history_nose, nose_score)
        forehead_score = self.smooth(self.history_forehead, forehead_score)

        overall = (
            0.34 * left_score +
            0.34 * right_score +
            0.20 * nose_score +
            0.12 * forehead_score
        )

        data = RednessData(
            overall=overall,
            left_cheek=left_score,
            right_cheek=right_score,
            nose=nose_score,
            forehead=forehead_score,
            status=self.status_from_score(overall)
        )

        return data, boxes

    def draw_overlay(self, frame, redness_data, boxes):
        if not boxes:
            return frame

        overlay = frame.copy()

        def color_from_score(score):
            intensity = int(70 + 160 * self.clamp01(score))
            return (0, 0, intensity)

        for key, box in boxes.items():
            x1, y1, x2, y2 = box
            score = getattr(redness_data, key, 0.0)

            if score < 0.12:
                continue

            color = color_from_score(score)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        frame = cv2.addWeighted(overlay, 0.14, frame, 0.86, 0)
        return frame

    def get_color(self, label, value, overall_status):
        inactive = (210, 210, 210)
        blue = (80, 190, 255)
        orange = (255, 185, 90)
        red = (255, 120, 120)

        if overall_status == "сильное" and value > 0.35:
            return red
        if overall_status == "заметное" and value > 0.20:
            return orange
        if value > 0.10:
            return blue
        return inactive

    def draw_progress_bar(self, draw, x, y, width, height, value, label, overall_status):
        color = self.get_color(label, value, overall_status)

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

    def draw_panel(self, frame, redness_data):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)

        x1 = 15
        y1 = 300
        x2 = 435
        y2 = 555

        draw.rounded_rectangle(
            (x1, y1, x2, y2),
            radius=10,
            fill=(0, 0, 0),
            outline=(255, 255, 255),
            width=2
        )

        status_color = (255, 255, 255)
        if "калибр." in redness_data.status:
            status_color = (245, 225, 100)
        elif redness_data.status == "легкое":
            status_color = (80, 190, 255)
        elif redness_data.status == "заметное":
            status_color = (255, 185, 90)
        elif redness_data.status == "сильное":
            status_color = (255, 120, 120)

        draw.text(
            (x1 + 12, y1 + 10),
            f"Покраснение: {redness_data.status}",
            font=self.font,
            fill=status_color
        )

        calib_text = "да" if self.neutral_ready else f"{len(self.neutral_samples)}/25"
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

        self.draw_progress_bar(draw, x1 + 12, start_y + gap * 0, bar_width, bar_height, redness_data.overall, "Общее", redness_data.status)
        self.draw_progress_bar(draw, x1 + 12, start_y + gap * 1, bar_width, bar_height, redness_data.left_cheek, "Левая щека", redness_data.status)
        self.draw_progress_bar(draw, x1 + 12, start_y + gap * 2, bar_width, bar_height, redness_data.right_cheek, "Правая щека", redness_data.status)
        self.draw_progress_bar(draw, x1 + 12, start_y + gap * 3, bar_width, bar_height, redness_data.nose, "Нос", redness_data.status)
        self.draw_progress_bar(draw, x1 + 12, start_y + gap * 4, bar_width, bar_height, redness_data.forehead, "Лоб", redness_data.status)

        result = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return result