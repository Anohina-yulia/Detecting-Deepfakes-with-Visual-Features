# -*- coding: utf-8 -*-
import os
import time

import cv2
import mediapipe as mp

from models import FaceData, POS_CENTER, POS_LEFT, POS_NONE, POS_RIGHT


def _resolve_model_path(model_path):
    """Относительный путь ищется сначала от текущей папки, затем от папки проекта."""
    if os.path.isabs(model_path) or os.path.exists(model_path):
        return model_path
    candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
    return candidate if os.path.exists(candidate) else model_path


class NeuralFaceAnalyzerModule:
    def __init__(self, model_path="models/face_landmarker.task"):
        self.mp = mp
        vision = mp.tasks.vision

        model_path = _resolve_model_path(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Не найдена модель лица: {model_path}. "
                "Положите файл face_landmarker.task в папку models проекта.")
        # модель читается в память: MediaPipe на Windows не открывает пути с кириллицей
        with open(model_path, "rb") as file:
            model_data = file.read()

        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=model_data),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        self._base_time = None
        self._last_ts_ms = -1
        self.last_face_data = FaceData()

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    @staticmethod
    def _to_pixel(landmark, width, height):
        x = min(max(int(landmark.x * width), 0), width - 1)
        y = min(max(int(landmark.y * height), 0), height - 1)
        return (x, y)

    @staticmethod
    def _classify_position(center_x, frame_width):
        if center_x < frame_width * 0.4:
            return POS_LEFT
        if center_x > frame_width * 0.6:
            return POS_RIGHT
        return POS_CENTER

    @staticmethod
    def _blendshape_dict(result):
        if not result.face_blendshapes:
            return {}
        return {item.category_name: float(item.score)
                for item in result.face_blendshapes[0]}

    def process(self, frame, frame_index, frame_time=None):
        """frame_time - время кадра в секундах (для файла: по таймлайну видео).

        Если не указано, берётся текущее системное время.
        Возвращает (face_data, None): второе значение оставлено для совместимости.
        """
        height, width = frame.shape[:2]
        timestamp = float(frame_time) if frame_time is not None else time.time()

        if self._base_time is None:
            self._base_time = timestamp
        ts_ms = int((timestamp - self._base_time) * 1000)
        ts_ms = max(ts_ms, self._last_ts_ms + 1)  # MediaPipe требует строгого роста
        self._last_ts_ms = ts_ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        if not result.face_landmarks:
            self.last_face_data = FaceData(
                detected=False, frame_width=width, frame_height=height,
                timestamp=timestamp, face_position=POS_NONE,
            )
            return self.last_face_data, None

        points = [self._to_pixel(lm, width, height) for lm in result.face_landmarks[0]]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x1, y1 = max(min(xs) - 10, 0), max(min(ys) - 10, 0)
        x2, y2 = min(max(xs) + 10, width - 1), min(max(ys) + 10, height - 1)

        self.last_face_data = FaceData(
            detected=True,
            landmarks=points,
            bbox=(x1, y1, x2, y2),
            frame_width=width,
            frame_height=height,
            timestamp=timestamp,
            face_position=self._classify_position((x1 + x2) / 2.0, width),
            blendshapes=self._blendshape_dict(result),
        )
        return self.last_face_data, None