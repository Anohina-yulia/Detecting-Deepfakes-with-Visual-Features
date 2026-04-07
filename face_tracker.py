import time

import cv2
import mediapipe as mp

from models import FaceData


class FaceTrackerModule:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()

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

    def process(self, frame):
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return FaceData(
                detected=False,
                frame_width=width,
                frame_height=height,
                timestamp=time.time(),
                face_position="не найдено"
            )

        face_landmarks = results.multi_face_landmarks[0]
        points = [
            self.to_pixel(landmark, width, height)
            for landmark in face_landmarks.landmark
        ]

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]

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
            face_position=face_position
        )

    def draw_face_box(self, frame, face_data):
        if not face_data.detected or not face_data.bbox:
            return frame
        
        x1, y1, x2, y2 = face_data.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
        return frame
