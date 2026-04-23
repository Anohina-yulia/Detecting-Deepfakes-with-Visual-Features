# -*- coding: utf-8 -*-
import queue
import threading
from dataclasses import dataclass, field

from models import RednessData


@dataclass
class AnalysisSnapshot:
    face_data: object = None
    mimic_data: object = None
    micro_event: object = None
    redness_data: object = None
    redness_boxes: dict = field(default_factory=dict)
    quality_data: object = None
    frame_index: int = 0


class VideoAnalysisWorker:
    def __init__(
        self,
        neural_face,
        mimic_analyzer,
        micro_module,
        redness_analyzer,
        quality_analyzer,
        submit_every_n=2
    ):
        self.neural_face = neural_face
        self.mimic_analyzer = mimic_analyzer
        self.micro_module = micro_module
        self.redness_analyzer = redness_analyzer
        self.quality_analyzer = quality_analyzer
        self.submit_every_n = submit_every_n

        self.queue = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.running = True

        self.last_good_mimic_data = getattr(self.mimic_analyzer, "last_result", None)
        self.last_good_micro_event = None
        self.last_good_redness_data = RednessData(status="ожидание")
        self.last_good_redness_boxes = {}

        self.latest_snapshot = AnalysisSnapshot()

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, frame, frame_index):
        if frame is None:
            return

        if frame_index % self.submit_every_n != 0 and self.latest_snapshot.face_data is not None:
            return

        frame_copy = frame.copy()

        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass

        self.queue.put((frame_copy, frame_index))

    def _loop(self):
        while self.running:
            try:
                frame, frame_index = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                face_data, _ = self.neural_face.process(frame, frame_index)

                current_mimic_data = self.mimic_analyzer.process(face_data)
                current_redness_data, current_redness_boxes = self.redness_analyzer.process(frame, face_data)
                quality_data = self.quality_analyzer.process(frame, face_data)

                if quality_data.good:
                    mimic_data = current_mimic_data
                    micro_event = self.micro_module.update(mimic_data, face_data.timestamp)
                    redness_data = current_redness_data
                    redness_boxes = current_redness_boxes

                    self.last_good_mimic_data = mimic_data
                    self.last_good_micro_event = micro_event
                    self.last_good_redness_data = redness_data
                    self.last_good_redness_boxes = redness_boxes
                else:
                    if getattr(self.mimic_analyzer, "neutral_ready", False):
                        mimic_data = self.last_good_mimic_data
                        micro_event = self.last_good_micro_event
                    else:
                        mimic_data = current_mimic_data
                        micro_event = self.micro_module.update(mimic_data, face_data.timestamp)

                    if getattr(self.redness_analyzer, "neutral_ready", False):
                        redness_data = self.last_good_redness_data
                        redness_boxes = self.last_good_redness_boxes
                    else:
                        redness_data = current_redness_data
                        redness_boxes = current_redness_boxes

                snapshot = AnalysisSnapshot(
                    face_data=face_data,
                    mimic_data=mimic_data,
                    micro_event=micro_event,
                    redness_data=redness_data,
                    redness_boxes=redness_boxes,
                    quality_data=quality_data,
                    frame_index=frame_index
                )

                with self.lock:
                    self.latest_snapshot = snapshot

            except Exception as error:
                print("Ошибка фонового анализа:", error)

    def get_latest(self):
        with self.lock:
            return self.latest_snapshot

    def close(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)