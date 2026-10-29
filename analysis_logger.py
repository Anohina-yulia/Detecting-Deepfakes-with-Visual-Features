# -*- coding: utf-8 -*-
import csv
import os
import re
import threading
import time
from datetime import datetime

HEADER = [
    "время", "время_видео_с", "лицо_найдено", "положение_лица", "взгляд", "моргания",
    "мимика", "улыбка", "рот_открыт", "брови_вверх", "нахмуривание", "прищур",
    "микровыражение",
    "покраснение_статус", "покраснение_общее", "покраснение_левая_щека",
    "покраснение_правая_щека", "покраснение_нос", "покраснение_лоб",
    "напряжение_статус", "напряжение_общее", "напряжение_мимика",
    "напряжение_моргание", "напряжение_взгляд", "напряжение_речь",
    "напряжение_заметка", "качество_кадра", "метка", "участник", "субтитры",
]


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
            # utf-8-sig, чтобы Excel правильно открывал кириллицу
            with open(self.filepath, "w", newline="", encoding="utf-8-sig") as file:
                csv.writer(file, delimiter=";").writerow(HEADER)

    @staticmethod
    def normalize_text(text):
        return re.sub(r"\s+", " ", str(text or "").strip())

    def _should_write(self, signature, force=False):
        now = time.time()
        if self.last_signature is None or force:
            write = True
        else:
            changed = signature != self.last_signature
            enough = (now - self.last_write_time) >= self.min_interval_sec
            periodic = (now - self.last_write_time) >= 2.5
            write = (changed and enough) or periodic

        if write:
            self.last_signature = signature
            self.last_write_time = now
        return write

    def log(self, face_data, eye_tracker, mimic_data, micro_event=None,
            subtitle_text="", redness_data=None, suspicion_data=None,
            video_time=0.0, quality_note="", label="", participant="", force=False):
        if face_data is None or eye_tracker is None or mimic_data is None:
            return

        micro_name = getattr(micro_event, "name", "нет") if micro_event else "нет"
        subtitle = self.normalize_text(subtitle_text)

        r = redness_data
        s = suspicion_data
        red = [getattr(r, name, 0.0) if r else 0.0
               for name in ("overall", "left_cheek", "right_cheek", "nose", "forehead")]
        red_status = getattr(r, "status", "нет данных") if r else "нет данных"

        sus = [getattr(s, name, 0.0) if s else 0.0
               for name in ("overall", "mimic_risk", "blink_risk", "gaze_risk", "text_risk")]
        sus_status = getattr(s, "status", "нет данных") if s else "нет данных"
        sus_note = self.normalize_text(getattr(s, "note", "")) if s else ""

        signature = (
            bool(face_data.detected), str(face_data.face_position), str(eye_tracker.last_gaze),
            int(eye_tracker.blink_count), str(mimic_data.dominant_expression), micro_name,
            red_status, round(red[0], 3), sus_status, round(sus[0], 3), subtitle, label,
        )

        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            round(float(video_time), 2),
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
            micro_name,
            red_status, *[round(float(v), 3) for v in red],
            sus_status, *[round(float(v), 3) for v in sus],
            sus_note,
            quality_note,
            label,
            participant,
            subtitle,
        ]

        # проверка "писать ли" и сама запись - под одним замком
        with self.lock:
            if not self._should_write(signature, force):
                return
            with open(self.filepath, "a", newline="", encoding="utf-8") as file:
                csv.writer(file, delimiter=";").writerow(row)