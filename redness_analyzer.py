# -*- coding: utf-8 -*-
from collections import deque

import cv2
import numpy as np

from models import (RED_CALIBRATION, RED_MILD, RED_NO_FACE, RED_NORMAL, RED_NOTICEABLE,
                    RED_STRONG, RednessData)
from ui_widgets import (BLUE, GRAY, ORANGE, RED, WHITE, YELLOW, clamp01, draw_bar,
                        load_font, new_panel, to_bgra)

CALIBRATION_FRAMES = 30
WEIGHTS = {"left": 0.34, "right": 0.34, "nose": 0.18, "forehead": 0.14}
ZONES = ["left", "right", "nose", "forehead"]


class RednessAnalyzerModule:
    def __init__(self):
        self.history = {name: deque(maxlen=6) for name in ZONES + ["overall"]}
        self.neutral_ready = False
        self.neutral_samples = []
        self.neutral = {name: 0.0 for name in ZONES + ["global"]}

    # ---------- вспомогательное ----------

    @staticmethod
    def mean_point(points):
        center = np.mean(np.array(points, dtype=np.float32), axis=0)
        return int(center[0]), int(center[1])

    @staticmethod
    def smooth(history, value):
        history.append(float(value))
        return float(sum(history) / len(history))

    @staticmethod
    def _global(raw):
        return sum(WEIGHTS[name] * raw[name] for name in ZONES)

    def start_calibration(self):
        self.neutral_ready = False
        self.neutral_samples = []
        for history in self.history.values():
            history.clear()
        print("Калибровка покраснения: смотрите прямо, лицо нейтральное, ровный свет.")

    # ---------- цветовая метрика ----------

    @staticmethod
    def gray_world_normalize(patch):
        patch_f = patch.astype(np.float32) + 1.0
        b, g, r = cv2.split(patch_f)
        mean_b, mean_g, mean_r = float(b.mean()), float(g.mean()), float(r.mean())
        gray = (mean_b + mean_g + mean_r) / 3.0
        merged = cv2.merge([b * gray / mean_b, g * gray / mean_g, r * gray / mean_r])
        return np.clip(merged, 0, 255).astype(np.uint8)

    def compute_patch_metric(self, patch):
        if patch is None or patch.size == 0:
            return 0.0

        patch = self.gray_world_normalize(patch)
        b, g, r = cv2.split(patch)
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)

        _, s_ch, _ = cv2.split(hsv)
        y_ch, cr_ch, cb_ch = cv2.split(ycrcb)
        _, a_ch, _ = cv2.split(lab)

        skin = ((cr_ch > 132) & (cr_ch < 177) & (cb_ch > 78) & (cb_ch < 132)
                & (y_ch > 55) & (y_ch < 230) & (s_ch > 12) & (s_ch < 175)
                & (r > 50) & (g > 30) & (b > 20))

        if np.count_nonzero(skin) < 45:
            return 0.0

        r_s, g_s, b_s = (x[skin].astype(np.float32) for x in (r, g, b))
        total = np.maximum(r_s + g_s + b_s, 1.0)

        rg_metric = np.mean((r_s - g_s) / total)
        crcb_metric = np.mean((cr_ch[skin].astype(np.float32) - cb_ch[skin].astype(np.float32)) / 255.0)
        lab_metric = np.mean((a_ch[skin].astype(np.float32) - 128.0) / 40.0)

        raw = 0.34 * rg_metric + 0.36 * crcb_metric + 0.30 * lab_metric
        return clamp01((raw - 0.01) * 3.8)

    def sample_patch_score(self, frame, center, half_w, half_h):
        h, w = frame.shape[:2]
        cx, cy = center
        x1, y1 = max(cx - half_w, 0), max(cy - half_h, 0)
        x2, y2 = min(cx + half_w, w - 1), min(cy + half_h, h - 1)
        if x2 <= x1 or y2 <= y1:
            return 0.0, (x1, y1, x2, y2)
        return self.compute_patch_metric(frame[y1:y2, x1:x2]), (x1, y1, x2, y2)

    @staticmethod
    def status_from_score(score):
        if score < 0.07:
            return RED_NORMAL
        if score < 0.16:
            return RED_MILD
        if score < 0.30:
            return RED_NOTICEABLE
        return RED_STRONG

    # ---------- калибровка ----------

    def update_calibration(self, raw):
        if sum(1 for name in ZONES if raw[name] > 0.0) < 3:
            return False

        sample = dict(raw)
        sample["global"] = self._global(raw)
        self.neutral_samples.append(sample)

        if len(self.neutral_samples) < CALIBRATION_FRAMES:
            return False

        for name in ZONES + ["global"]:
            self.neutral[name] = float(np.median([s[name] for s in self.neutral_samples]))

        self.neutral_ready = True
        self.neutral_samples = []
        print("Калибровка покраснения завершена.")
        return True

    def adaptive_baseline_update(self, raw, overall):
        """Очень медленная подстройка базы, пока покраснения нет."""
        if not self.neutral_ready or overall > 0.05:
            return
        alpha = 0.002
        for name in ZONES:
            self.neutral[name] = (1 - alpha) * self.neutral[name] + alpha * raw[name]
        self.neutral["global"] = (1 - alpha) * self.neutral["global"] + alpha * self._global(raw)

    # ---------- основной вызов ----------

    def process(self, frame, face_data, allow_calibration=True):
        if not face_data.detected or not face_data.landmarks:
            return RednessData(status=RED_NO_FACE, calibrated=self.neutral_ready,
                               progress=len(self.neutral_samples)), {}

        pts = face_data.landmarks
        x1, y1, x2, y2 = face_data.bbox
        k = float(np.clip((x2 - x1) / 300.0, 0.5, 2.5))  # размер патчей следует за размером лица

        def half(w, h):
            return max(6, int(w * k)), max(6, int(h * k))

        centers = {
            "left": self.mean_point([pts[205], pts[50], pts[123], pts[116]]),
            "right": self.mean_point([pts[425], pts[280], pts[352], pts[345]]),
            "nose": self.mean_point([pts[1], pts[5], pts[195], pts[4]]),
            "forehead": self.mean_point([pts[10], pts[151], pts[9]]),
        }
        sizes = {"left": half(28, 20), "right": half(28, 20),
                 "nose": half(20, 16), "forehead": half(30, 18)}

        raw, boxes = {}, {}
        box_names = {"left": "left_cheek", "right": "right_cheek",
                     "nose": "nose", "forehead": "forehead"}
        for name in ZONES:
            raw[name], boxes[box_names[name]] = self.sample_patch_score(
                frame, centers[name], *sizes[name])

        if not self.neutral_ready:
            if allow_calibration:
                self.update_calibration(raw)
            return RednessData(status=RED_CALIBRATION, calibrated=False,
                               progress=len(self.neutral_samples)), boxes

        global_shift = max(self._global(raw) - self.neutral["global"], 0.0)
        deadzone = 0.030
        gains = {"left": (8.0, 0.25), "right": (8.0, 0.25),
                 "nose": (7.2, 0.20), "forehead": (6.8, 0.20)}

        scores = {}
        for name in ZONES:
            gain, shift_w = gains[name]
            value = clamp01((raw[name] - self.neutral[name] - deadzone
                             - shift_w * global_shift) * gain)
            scores[name] = self.smooth(self.history[name], value)

        overall = self.smooth(self.history["overall"],
                              sum(WEIGHTS[n] * scores[n] for n in ZONES))
        self.adaptive_baseline_update(raw, overall)

        return RednessData(
            overall=overall,
            left_cheek=scores["left"],
            right_cheek=scores["right"],
            nose=scores["nose"],
            forehead=scores["forehead"],
            status=self.status_from_score(overall),
            calibrated=True,
            progress=CALIBRATION_FRAMES,
        ), boxes

    # ---------- панель ----------

    @staticmethod
    def _color(value, status):
        if status == RED_STRONG and value > 0.28:
            return RED
        if status == RED_NOTICEABLE and value > 0.16:
            return ORANGE
        if value > 0.08:
            return BLUE
        return GRAY

    def render_panel(self, data):
        img, draw = new_panel(420, 255)
        font, small = load_font(17), load_font(15)

        status_colors = {RED_CALIBRATION: YELLOW, RED_MILD: BLUE,
                         RED_NOTICEABLE: ORANGE, RED_STRONG: RED}
        draw.text((12, 10), f"Покраснение: {data.status}", font=font,
                  fill=status_colors.get(data.status, WHITE))

        calib = "готово" if self.neutral_ready else f"{len(self.neutral_samples)}/{CALIBRATION_FRAMES}"
        draw.text((12, 38), f"Калибровка: {calib}", font=small, fill=GRAY)

        rows = [("Общее", data.overall), ("Левая щека", data.left_cheek),
                ("Правая щека", data.right_cheek), ("Нос", data.nose), ("Лоб", data.forehead)]
        for i, (label, value) in enumerate(rows):
            color = self._color(value, data.status)
            draw_bar(draw, 12, 72 + 32 * i, value, label, color, small,
                     label_w=115, width=170, label_color=color)
        return to_bgra(img)