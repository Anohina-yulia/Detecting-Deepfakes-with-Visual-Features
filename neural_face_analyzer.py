# -*- coding: utf-8 -*-
import time
import cv2
import mediapipe as mp

from models import FaceData, MimicData


class NeuralFaceAnalyzerModule:
    def __init__(self, model_path="models/face_landmarker.task"):
        self.mp = mp
        self.base_options = mp.tasks.BaseOptions
        self.vision = mp.tasks.vision
        self.running_mode = self.vision.RunningMode.VIDEO

        options = self.vision.FaceLandmarkerOptions(
            base_options=self.base_options(model_asset_path=model_path),
            running_mode=self.running_mode,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )

        self.landmarker = self.vision.FaceLandmarker.create_from_options(options)

        self.last_blendshapes = {}
        self.last_face_data = FaceData()
        self.last_mimic_data = MimicData()

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()

    def to_pixel(self, landmark, width, height):
        x = min(max(int(landmark.x * width), 0), width - 1)
        y = min(max(int(landmark.y * height), 0), height - 1)
        return (x, y)

    def classify_face_position(self, box_center_x, frame_width):
        left_border = frame_width * 0.4
        right_border = frame_width * 0.6

        if box_center_x < left_border:
            return "слева в кадре"
        if box_center_x > right_border:
            return "справа в кадре"
        return "по центру"

    def _blendshape_dict(self, result):
        data = {}

        if not result.face_blendshapes:
            return data

        first_face = result.face_blendshapes[0]

        for item in first_face:
            data[item.category_name] = float(item.score)

        return data

    def _build_face_data(self, result, frame):
        height, width = frame.shape[:2]

        if not result.face_landmarks:
            return FaceData(
                detected=False,
                frame_width=width,
                frame_height=height,
                timestamp=time.time(),
                face_position="не найдено"
            )

        face_landmarks = result.face_landmarks[0]
        points = [self.to_pixel(landmark, width, height) for landmark in face_landmarks]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x1 = max(min(xs) - 10, 0)
        y1 = max(min(ys) - 10, 0)
        x2 = min(max(xs) + 10, width - 1)
        y2 = min(max(ys) + 10, height - 1)

        box_center_x = (x1 + x2) / 2.0
        face_position = self.classify_face_position(box_center_x, width)

        return FaceData(
            detected=True,
            landmarks=points,
            bbox=(x1, y1, x2, y2),
            frame_width=width,
            frame_height=height,
            timestamp=time.time(),
            face_position=face_position,
            blendshapes=self.last_blendshapes.copy()
        )

    def _get_score(self, names, default=0.0):
        values = [self.last_blendshapes.get(name, 0.0) for name in names]

        if not values:
            return default

        return float(sum(values) / len(values))

    def _build_mimic_data(self):
        smile_score = self._get_score(["mouthSmileLeft", "mouthSmileRight"])
        mouth_open_score = self._get_score(["jawOpen", "mouthOpen"])
        brow_raise_score = self._get_score(["browInnerUp", "browOuterUpLeft", "browOuterUpRight"])
        brow_frown_score = self._get_score(["browDownLeft", "browDownRight"])
        eye_squint_score = self._get_score(["eyeSquintLeft", "eyeSquintRight"])

        dominant = "нейтрально"

        pairs = {
            "улыбка": smile_score,
            "рот открыт": mouth_open_score,
            "подняты брови": brow_raise_score,
            "нахмуренность": brow_frown_score,
            "прищур": eye_squint_score,
        }

        best_name = max(pairs, key=pairs.get)
        if pairs[best_name] > 0.25:
            dominant = best_name

        return MimicData(
            smile=smile_score,
            mouth_open=mouth_open_score,
            brow_raise=brow_raise_score,
            brow_frown=brow_frown_score,
            eye_squint=eye_squint_score,
            dominant_expression=dominant
        )

    def process(self, frame, frame_index):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(frame_index * 33)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        self.last_blendshapes = self._blendshape_dict(result)
        self.last_face_data = self._build_face_data(result, frame)
        self.last_mimic_data = self._build_mimic_data()

        return self.last_face_data, self.last_mimic_data
