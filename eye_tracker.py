# -*- coding: utf-8 -*-
from collections import deque

import cv2
import numpy as np

from models import (GAZE_CENTER, GAZE_DOWN, GAZE_LEFT, GAZE_NONE, GAZE_RIGHT,
                    GAZE_UP, POS_NONE)
from ui_widgets import BLUE, GRAY, WHITE, draw_bar, load_font, new_panel, to_bgra


class EyeTrackerModule:
    LEFT_EYE_EAR = [33, 159, 158, 133, 153, 145]
    RIGHT_EYE_EAR = [362, 386, 385, 263, 380, 374]

    LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155,
                        133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_CONTOUR = [263, 249, 390, 373, 374, 380, 381, 382,
                         362, 398, 384, 385, 386, 387, 388, 466]

    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    LEFT_EYE_CORNERS = [33, 133]
    RIGHT_EYE_CORNERS = [362, 263]

    def __init__(self):
        self.ear_threshold = 0.21
        self.min_closed_frames = 2

        self.closed_frames = 0
        self.blink_count = 0
        self.blink_in_progress = False

        self.ear_history = deque(maxlen=4)
        self.horizontal_history = deque(maxlen=3)
        self.vertical_history = deque(maxlen=3)

        self.last_gaze = GAZE_NONE
        self.last_ear = 0.0
        self.face_found = False
        self.last_face_position = POS_NONE
        self.last_horizontal_ratio = 0.5
        self.last_vertical_ratio = 0.5

    # ---------- вычисления ----------

    @staticmethod
    def distance(p1, p2):
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def compute_ear(self, points):
        p1, p2, p3, p4, p5, p6 = points
        horizontal = self.distance(p1, p4)
        if horizontal == 0:
            return 0.0
        return (self.distance(p2, p6) + self.distance(p3, p5)) / (2.0 * horizontal)

    @staticmethod
    def _ratio(value, low, high):
        size = high - low
        if size <= 1:
            return 0.5
        return float(np.clip((value - low) / size, 0.0, 1.0))

    def horizontal_ratio(self, iris, corners):
        cx = float(np.mean(np.array(iris), axis=0)[0])
        xs = [p[0] for p in corners]
        return self._ratio(cx, float(min(xs)), float(max(xs)))

    def vertical_ratio(self, iris, contour):
        cy = float(np.mean(np.array(iris), axis=0)[1])
        ys = [p[1] for p in contour]
        return self._ratio(cy, float(min(ys)), float(max(ys)))

    @staticmethod
    def classify_gaze(h_ratio, v_ratio):
        h = v = GAZE_CENTER
        if h_ratio < 0.42:
            h = GAZE_LEFT
        elif h_ratio > 0.58:
            h = GAZE_RIGHT
        if v_ratio < 0.38:
            v = GAZE_UP
        elif v_ratio > 0.62:
            v = GAZE_DOWN

        if h != GAZE_CENTER and v != GAZE_CENTER:
            return f"{h}-{v.lower()}"
        if h != GAZE_CENTER:
            return h
        if v != GAZE_CENTER:
            return v
        return GAZE_CENTER

    def update_blink_counter(self, ear):
        self.ear_history.append(ear)
        smooth = float(sum(self.ear_history) / len(self.ear_history))
        self.last_ear = smooth

        if smooth < self.ear_threshold:
            self.closed_frames += 1
            if self.closed_frames >= self.min_closed_frames:
                self.blink_in_progress = True
        else:
            if self.blink_in_progress:
                self.blink_count += 1
            self.closed_frames = 0
            self.blink_in_progress = False

    def _reset_blink_state(self):
        """Лицо пропало - незавершённое моргание не засчитываем."""
        self.closed_frames = 0
        self.blink_in_progress = False
        self.ear_history.clear()

    @staticmethod
    def _smooth(history, value):
        history.append(float(value))
        return float(sum(history) / len(history))

    # ---------- рисование на кадре ----------

    @staticmethod
    def _draw_eye_shape(frame, points, color):
        if len(points) >= 2:
            cv2.polylines(frame, [np.array(points, dtype=np.int32)], True, color, 1, cv2.LINE_AA)

    def _draw_iris(self, frame, points):
        center = np.mean(np.array(points), axis=0).astype(int)
        radius = max(2, int(np.mean([self.distance(center, p) for p in points])))
        cv2.circle(frame, tuple(int(v) for v in center), radius, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, tuple(int(v) for v in center), 1, (0, 255, 255), -1)

    # ---------- основной вызов ----------

    def process(self, frame, face_data):
        """Обновляет моргания и взгляд, рисует контуры глаз. Панель - в render_panel."""
        self.face_found = face_data.detected
        self.last_face_position = face_data.face_position

        if not face_data.detected or not face_data.landmarks:
            self.last_gaze = GAZE_NONE
            self._reset_blink_state()
            return frame

        pts = face_data.landmarks
        pick = lambda ids: [pts[i] for i in ids]

        left_ear = self.compute_ear(pick(self.LEFT_EYE_EAR))
        right_ear = self.compute_ear(pick(self.RIGHT_EYE_EAR))
        self.update_blink_counter((left_ear + right_ear) / 2.0)

        left_iris, right_iris = pick(self.LEFT_IRIS), pick(self.RIGHT_IRIS)
        left_contour, right_contour = pick(self.LEFT_EYE_CONTOUR), pick(self.RIGHT_EYE_CONTOUR)

        h = (self.horizontal_ratio(left_iris, pick(self.LEFT_EYE_CORNERS)) +
             self.horizontal_ratio(right_iris, pick(self.RIGHT_EYE_CORNERS))) / 2.0
        v = (self.vertical_ratio(left_iris, left_contour) +
             self.vertical_ratio(right_iris, right_contour)) / 2.0

        self.last_horizontal_ratio = self._smooth(self.horizontal_history, h)
        self.last_vertical_ratio = self._smooth(self.vertical_history, v)
        self.last_gaze = self.classify_gaze(self.last_horizontal_ratio,
                                            self.last_vertical_ratio)

        self._draw_eye_shape(frame, left_contour, (255, 0, 255))
        self._draw_eye_shape(frame, right_contour, (255, 0, 255))
        self._draw_iris(frame, left_iris)
        self._draw_iris(frame, right_iris)
        return frame

    # ---------- панель ----------

    def render_panel(self, quality_note=""):
        img, draw = new_panel(375, 245)
        font, small = load_font(17), load_font(15)

        draw.text((12, 10), "Лицо: " + ("найдено" if self.face_found else "не найдено"),
                  font=font, fill=WHITE)
        draw.text((12, 36), f"Положение: {self.last_face_position}", font=small, fill=GRAY)
        draw.text((12, 58), f"Взгляд: {self.last_gaze}", font=small, fill=GRAY)
        draw.text((12, 80), f"Моргания: {self.blink_count}", font=small, fill=GRAY)
        if quality_note:
            draw.text((12, 102), f"Качество кадра: {quality_note}", font=small, fill=GRAY)

        y0, gap = 134, 38
        draw_bar(draw, 12, y0, self.last_horizontal_ratio, "Горизонталь", BLUE, small,
                 label_w=105, width=170, marker=True, end_labels=("Л", "П"))
        draw_bar(draw, 12, y0 + gap, self.last_vertical_ratio, "Вертикаль", BLUE, small,
                 label_w=105, width=170, marker=True, end_labels=("В", "Н"))
        draw_bar(draw, 12, y0 + gap * 2, min(self.last_ear / 0.45, 1.0), "Глаз открыт",
                 BLUE, small, label_w=105, width=170)
        return to_bgra(img)