# -*- coding: utf-8 -*-
import os
import cv2
import json
import time
import queue
import uuid
import tempfile
import threading
import subprocess
from dataclasses import dataclass, field


@dataclass
class EmotionData:
    dominant_emotion: str = "не определено"
    confidence: float = 0.0
    scores: dict = field(default_factory=dict)
    updated_at: float = 0.0
    status: str = "ожидание"
    error: str = ""


class EmotionBridgeModule:
    def __init__(
        self,
        worker_python_path=r".\.venv_emotion\Scripts\python.exe",
        worker_script_path="emotion_worker.py",
        min_interval_sec=1.0,
        face_size=224
    ):
        self.worker_python_path = os.path.abspath(worker_python_path)
        self.worker_script_path = os.path.abspath(worker_script_path)
        self.min_interval_sec = min_interval_sec
        self.face_size = face_size

        self.frame_queue = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.latest_result = EmotionData(status="ожидание")

        self.last_submit_time = 0.0
        self.running = True

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _prepare_face(self, frame, face_data):
        if not face_data.detected or not face_data.bbox:
            return None

        x1, y1, x2, y2 = face_data.bbox

        pad_x = int((x2 - x1) * 0.10)
        pad_y = int((y2 - y1) * 0.14)

        x1 = max(x1 - pad_x, 0)
        y1 = max(y1 - pad_y, 0)
        x2 = min(x2 + pad_x, frame.shape[1] - 1)
        y2 = min(y2 + pad_y, frame.shape[0] - 1)

        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None

        h, w = crop.shape[:2]
        if h < 40 or w < 40:
            return None

        resized = cv2.resize(crop, (self.face_size, self.face_size))
        return resized

    def _save_temp_face(self, face_bgr):
        temp_dir = tempfile.gettempdir()
        filename = f"emotion_face_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(temp_dir, filename)

        ok = cv2.imwrite(filepath, face_bgr)
        if not ok:
            return None

        return filepath

    def _call_worker(self, image_path):
        try:
            completed = subprocess.run(
                [self.worker_python_path, self.worker_script_path, image_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=90
            )
        except Exception as error:
            return {
                "ok": False,
                "error": f"subprocess error: {error}"
            }

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()

        if not stdout:
            return {
                "ok": False,
                "error": f"worker returned empty stdout; stderr={stderr[:300]}"
            }

        lines = stdout.splitlines()

        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except Exception:
                continue

        return {
            "ok": False,
            "error": f"worker returned non-json stdout: {stdout[:300]}"
        }

    def _loop(self):
        while self.running:
            try:
                face_bgr = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            temp_path = None
            try:
                temp_path = self._save_temp_face(face_bgr)
                if temp_path is None:
                    with self.lock:
                        self.latest_result = EmotionData(
                            status="ошибка",
                            error="не удалось сохранить временное изображение",
                            updated_at=time.time()
                        )
                    continue

                response = self._call_worker(temp_path)

                if not response:
                    with self.lock:
                        self.latest_result = EmotionData(
                            status="ошибка",
                            error="worker не вернул ответ",
                            updated_at=time.time()
                        )
                    continue

                if not response.get("ok"):
                    error_text = response.get("error", "неизвестная ошибка")
                    print("Emotion worker error:", error_text)

                    with self.lock:
                        self.latest_result = EmotionData(
                            status="ошибка",
                            error=error_text,
                            updated_at=time.time()
                        )
                    continue

                data = EmotionData(
                    dominant_emotion=response.get("dominant_emotion", "не определено"),
                    confidence=float(response.get("confidence", 0.0)),
                    scores=dict(response.get("scores", {})),
                    updated_at=time.time(),
                    status="ok",
                    error=""
                )

                with self.lock:
                    self.latest_result = data

            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

    def submit_face(self, frame, face_data):
        now = time.time()
        if (now - self.last_submit_time) < self.min_interval_sec:
            return

        prepared_face = self._prepare_face(frame, face_data)
        if prepared_face is None:
            with self.lock:
                self.latest_result.status = "нет лица"
            return

        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass

        self.frame_queue.put(prepared_face)
        self.last_submit_time = now

    def get_result(self):
        with self.lock:
            return self.latest_result

    def close(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)