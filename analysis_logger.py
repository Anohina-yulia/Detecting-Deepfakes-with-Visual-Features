# -*- coding: utf-8 -*-
import csv
import os
import re
import threading
import time
from datetime import datetime


class AnalysisLogger:
    def __init__(self, filepath="analysis_log.csv", min_interval_sec=0.7):
        self.filepath = filepath
        self.min_interval_sec = min_interval_sec
        self.lock = threading.Lock()

        self.last_write_time = 0.0
        self.last_signature = None

        folder = os.path.dirname(self.filepath)
        if folder:
            os.makedirs(folder, exist_ok=True)

        if not os.path.exists(self.filepath) or os.path.getsize(self.filepath) == 0:
            with open(self.filepath, "w", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file, delimiter=";")
                writer.writerow([
                    "timestamp",
                    "face_detected",
                    "face_position",
                    "gaze",
                    "blink_count",
                    "mimic",
                    "smile",
                    "mouth_open",
                    "brow_raise",
                    "brow_frown",
                    "eye_squint",
                    "micro_event",
                    "redness_status",
                    "redness_overall",
                    "redness_left_cheek",
                    "redness_right_cheek",
                    "redness_nose",
                    "redness_forehead",
                    "suspicion_status",
                    "suspicion_overall",
                    "suspicion_mimic_risk",
                    "suspicion_blink_risk",
                    "suspicion_gaze_risk",
                    "suspicion_text_risk",
                    "suspicion_note",
                    "subtitle"
                ])

    def normalize_text(self, text):
        if text is None:
            return ""
        text = str(text).strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def build_signature(
        self,
        face_detected,
        face_position,
        gaze,
        blink_count,
        mimic_name,
        micro_event_name,
        redness_status,
        redness_overall,
        suspicion_status,
        suspicion_overall,
        subtitle
    ):
        subtitle = self.normalize_text(subtitle)

        return (
            bool(face_detected),
            str(face_position),
            str(gaze),
            int(blink_count),
            str(mimic_name),
            str(micro_event_name),
            str(redness_status),
            round(float(redness_overall), 3),
            str(suspicion_status),
            round(float(suspicion_overall), 3),
            subtitle
        )

    def should_write(self, signature):
        now = time.time()

        if self.last_signature is None:
            self.last_signature = signature
            self.last_write_time = now
            return True

        changed = signature != self.last_signature
        enough_time = (now - self.last_write_time) >= self.min_interval_sec
        periodic_write = (now - self.last_write_time) >= 2.5

        if (changed and enough_time) or periodic_write:
            self.last_signature = signature
            self.last_write_time = now
            return True

        return False

    def log(
        self,
        face_data,
        eye_tracker,
        mimic_data,
        micro_event=None,
        subtitle_text="",
        redness_data=None,
        suspicion_data=None
    ):
        if face_data is None or eye_tracker is None or mimic_data is None:
            return

        micro_event_name = "нет"
        if micro_event is not None:
            micro_event_name = getattr(micro_event, "name", "нет")

        subtitle_text = self.normalize_text(subtitle_text)

        redness_status = "нет данных"
        redness_overall = 0.0
        redness_left_cheek = 0.0
        redness_right_cheek = 0.0
        redness_nose = 0.0
        redness_forehead = 0.0

        if redness_data is not None:
            redness_status = getattr(redness_data, "status", "нет данных")
            redness_overall = float(getattr(redness_data, "overall", 0.0))
            redness_left_cheek = float(getattr(redness_data, "left_cheek", 0.0))
            redness_right_cheek = float(getattr(redness_data, "right_cheek", 0.0))
            redness_nose = float(getattr(redness_data, "nose", 0.0))
            redness_forehead = float(getattr(redness_data, "forehead", 0.0))

        suspicion_status = "нет данных"
        suspicion_overall = 0.0
        suspicion_mimic_risk = 0.0
        suspicion_blink_risk = 0.0
        suspicion_gaze_risk = 0.0
        suspicion_text_risk = 0.0
        suspicion_note = ""

        if suspicion_data is not None:
            suspicion_status = getattr(suspicion_data, "status", "нет данных")
            suspicion_overall = float(getattr(suspicion_data, "overall", 0.0))
            suspicion_mimic_risk = float(getattr(suspicion_data, "mimic_risk", 0.0))
            suspicion_blink_risk = float(getattr(suspicion_data, "blink_risk", 0.0))
            suspicion_gaze_risk = float(getattr(suspicion_data, "gaze_risk", 0.0))
            suspicion_text_risk = float(getattr(suspicion_data, "text_risk", 0.0))
            suspicion_note = self.normalize_text(getattr(suspicion_data, "note", ""))

        signature = self.build_signature(
            face_detected=face_data.detected,
            face_position=face_data.face_position,
            gaze=eye_tracker.last_gaze,
            blink_count=eye_tracker.blink_count,
            mimic_name=mimic_data.dominant_expression,
            micro_event_name=micro_event_name,
            redness_status=redness_status,
            redness_overall=redness_overall,
            suspicion_status=suspicion_status,
            suspicion_overall=suspicion_overall,
            subtitle=subtitle_text
        )

        if not self.should_write(signature):
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row = [
            timestamp,
            "да" if face_data.detected else "нет",
            face_data.face_position,
            eye_tracker.last_gaze,
            eye_tracker.blink_count,
            mimic_data.dominant_expression,
            round(float(mimic_data.smile), 3),
            round(float(mimic_data.mouth_open), 3),
            round(float(mimic_data.brow_raise), 3),
            round(float(mimic_data.brow_frown), 3),
            round(float(mimic_data.eye_squint), 3),
            micro_event_name,
            redness_status,
            round(redness_overall, 3),
            round(redness_left_cheek, 3),
            round(redness_right_cheek, 3),
            round(redness_nose, 3),
            round(redness_forehead, 3),
            suspicion_status,
            round(suspicion_overall, 3),
            round(suspicion_mimic_risk, 3),
            round(suspicion_blink_risk, 3),
            round(suspicion_gaze_risk, 3),
            round(suspicion_text_risk, 3),
            suspicion_note,
            subtitle_text
        ]

        with self.lock:
            with open(self.filepath, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file, delimiter=";")
                writer.writerow(row)