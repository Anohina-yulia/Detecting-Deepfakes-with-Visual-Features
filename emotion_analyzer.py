# -*- coding: utf-8 -*-
import queue
import threading
import time
from dataclasses import dataclass, field

import cv2
from deepface import DeepFace


@dataclass
class EmotionData:
    dominant_emotion: str = "не определено"
    confidence: float = 0.0
    scores: dict = field(default_factory=dict)
    updated_at: float = 0.0


class EmotionAnalyzerModule:
    def __init__(self, min_interval_sec=0.8, face_size=224):
        self.min_interval_sec = min_interval_sec
        self.face_size = face_size

        self.last_submit_time = 0.0
        self.lock = threading.Lock()
        self.latest_result = EmotionData()

        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _normalize_emotion_name(self, name):
        mapping = {
            "angry": "злость",
            "disgust": "отвращение",
            "fear": "страх",
            "happy": "радость",
            "sad": "грусть",
            "surprise": "удивление",
            "neutral": "нейтрально",
        }
        return mapping.get(str(name).lower(), str(name))

    def _prepare_face(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return None

        h, w = face_bgr.shape[:2]
        if h < 20 or w < 20:
            return None

        resized = cv2.resize(face_bgr, (self.face_size, self.face_size))
        return resized

    def _worker_loop(self):
        while self.running:
            try:
                face_bgr = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                result = DeepFace.analyze(
                    img_path=face_bgr,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="skip",
                    silent=True,
                )

                if isinstance(result, list):
                    result = result[0] if result else {}

                emotion_scores = result.get("emotion", {})
                dominant = result.get("dominant_emotion", "не определено")

                if emotion_scores:
                    best_score = float(max(emotion_scores.values()))
                else:
                    best_score = 0.0

                data = EmotionData(
                    dominant_emotion=self._normalize_emotion_name(dominant),
                    confidence=best_score,
                    scores=dict(emotion_scores),
                    updated_at=time.time()
                )

                with self.lock:
                    self.latest_result = data

            except Exception as error:
                print("Ошибка анализа эмоций:", error)

    def submit_face(self, frame, face_data):
        if not face_data.detected or not face_data.bbox:
            return

        now = time.time()
        if (now - self.last_submit_time) < self.min_interval_sec:
            return

        x1, y1, x2, y2 = face_data.bbox

        pad_x = int((x2 - x1) * 0.08)
        pad_y = int((y2 - y1) * 0.12)

        x1 = max(x1 - pad_x, 0)
        y1 = max(y1 - pad_y, 0)
        x2 = min(x2 + pad_x, frame.shape[1] - 1)
        y2 = min(y2 + pad_y, frame.shape[0] - 1)

        face_crop = frame[y1:y2, x1:x2]
        prepared = self._prepare_face(face_crop)

        if prepared is None:
            return

        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass

        self.frame_queue.put(prepared)
        self.last_submit_time = now

    def get_result(self):
        with self.lock:
            return self.latest_result

    def close(self):
        self.running = False
        if self.worker.is_alive():
            self.worker.join(timeout=1.0)
