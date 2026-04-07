from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class EyeTrackerModule:
    LEFT_EYE_EAR = [33, 159, 158, 133, 153, 145]
    RIGHT_EYE_EAR = [362, 386, 385, 263, 380, 374]

    LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_CONTOUR = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]

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
        self.ear_history = deque(maxlen=5)

        self.last_gaze = "не найдено"
        self.last_ear = 0.0
        self.face_found = False
        self.last_face_position = "не найдено"

        try:
            self.font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
        except Exception:
            self.font = ImageFont.load_default()

    def distance(self, p1, p2):
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def compute_ear(self, points):
        p1, p2, p3, p4, p5, p6 = points

        horizontal = self.distance(p1, p4)
        if horizontal == 0:
            return 0.0

        vertical_1 = self.distance(p2, p6)
        vertical_2 = self.distance(p3, p5)

        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def estimate_horizontal_gaze_ratio(self, iris_points, corner_points):
        iris_center = np.mean(np.array(iris_points), axis=0)
        xs = [point[0] for point in corner_points]

        min_x = float(min(xs))
        max_x = float(max(xs))
        eye_width = max_x - min_x

        if eye_width <= 1:
            return 0.5

        ratio = (iris_center[0] - min_x) / eye_width
        return float(np.clip(ratio, 0.0, 1.0))

    def estimate_vertical_gaze_ratio(self, iris_points, eye_contour_points):
        iris_center = np.mean(np.array(iris_points), axis=0)
        ys = [point[1] for point in eye_contour_points]

        min_y = float(min(ys))
        max_y = float(max(ys))
        eye_height = max_y - min_y

        if eye_height <= 1:
            return 0.5

        ratio = (iris_center[1] - min_y) / eye_height
        return float(np.clip(ratio, 0.0, 1.0))

    def classify_gaze(self, horizontal_ratio, vertical_ratio):
        horizontal = "центр"
        vertical = "центр"

        if horizontal_ratio < 0.42:
            horizontal = "влево"
        elif horizontal_ratio > 0.58:
            horizontal = "вправо"

        if vertical_ratio < 0.38:
            vertical = "вверх"
        elif vertical_ratio > 0.62:
            vertical = "вниз"

        if horizontal != "центр" and vertical != "центр":
            return f"{horizontal}-{vertical}"
        if horizontal != "центр":
            return horizontal
        if vertical != "центр":
            return vertical
        return "центр"

    def update_blink_counter(self, ear):
        self.ear_history.append(ear)
        smooth_ear = float(sum(self.ear_history) / len(self.ear_history))
        self.last_ear = smooth_ear

        if smooth_ear < self.ear_threshold:
            self.closed_frames += 1
            if self.closed_frames >= self.min_closed_frames:
                self.blink_in_progress = True
        else:
            if self.blink_in_progress:
                self.blink_count += 1
            self.closed_frames = 0
            self.blink_in_progress = False

    def draw_eye_shape(self, frame, points, color):
        if len(points) < 2:
            return
        contour = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [contour], True, color, 1, cv2.LINE_AA)

    def draw_iris(self, frame, points):
        if not points:
            return

        center = np.mean(np.array(points), axis=0).astype(int)
        radius = max(2, int(np.mean([
            self.distance(center, point) for point in points
        ])))

        cv2.circle(frame, tuple(center), radius, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, tuple(center), 1, (0, 255, 255), -1, cv2.LINE_AA)

    def draw_panel(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        
        x1, y1 = 15, 50
        x2, y2 = 320, 185
        
        draw.rounded_rectangle(
            (x1, y1, x2, y2),
            radius=10,
            fill=(0, 0, 0),
            outline=(255, 255, 255),
            width=2
        )
        
        face_text = "да" if self.face_found else "нет"
        
        lines = [
            f"Лицо: {face_text}",
            f"Положение: {self.last_face_position}",
            f"Взгляд: {self.last_gaze}",
            f"EAR: {self.last_ear:.3f}",
            f"Моргания: {self.blink_count}"
        ]
        
        y = y1 + 10
        for line in lines:
            draw.text((x1 + 12, y), line, font=self.font, fill=(255, 255, 255))
            y += 24
            
        result = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return result

    def process(self, frame, face_data):
        self.face_found = face_data.detected
        self.last_face_position = face_data.face_position

        if not face_data.detected or not face_data.landmarks:
            self.last_gaze = "не найдено"
            frame = self.draw_panel(frame)
            return frame

        points = face_data.landmarks

        left_eye_points = [points[i] for i in self.LEFT_EYE_EAR]
        right_eye_points = [points[i] for i in self.RIGHT_EYE_EAR]

        left_eye_contour = [points[i] for i in self.LEFT_EYE_CONTOUR]
        right_eye_contour = [points[i] for i in self.RIGHT_EYE_CONTOUR]

        left_iris = [points[i] for i in self.LEFT_IRIS]
        right_iris = [points[i] for i in self.RIGHT_IRIS]

        left_corners = [points[i] for i in self.LEFT_EYE_CORNERS]
        right_corners = [points[i] for i in self.RIGHT_EYE_CORNERS]

        left_ear = self.compute_ear(left_eye_points)
        right_ear = self.compute_ear(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0
        self.update_blink_counter(avg_ear)

        left_horizontal = self.estimate_horizontal_gaze_ratio(left_iris, left_corners)
        right_horizontal = self.estimate_horizontal_gaze_ratio(right_iris, right_corners)
        horizontal_ratio = (left_horizontal + right_horizontal) / 2.0

        left_vertical = self.estimate_vertical_gaze_ratio(left_iris, left_eye_contour)
        right_vertical = self.estimate_vertical_gaze_ratio(right_iris, right_eye_contour)
        vertical_ratio = (left_vertical + right_vertical) / 2.0

        self.last_gaze = self.classify_gaze(horizontal_ratio, vertical_ratio)

        self.draw_eye_shape(frame, left_eye_contour, (255, 0, 255))
        self.draw_eye_shape(frame, right_eye_contour, (255, 0, 255))
        self.draw_iris(frame, left_iris)
        self.draw_iris(frame, right_iris)

        frame = self.draw_panel(frame)
        return frame
