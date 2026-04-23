# -*- coding: utf-8 -*-
import re
import time
from dataclasses import dataclass
from collections import deque, Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class SuspicionData:
    overall: float = 0.0
    status: str = "калибровка"
    verdict: str = "неопределённо"
    mimic_risk: float = 0.0
    blink_risk: float = 0.0
    gaze_risk: float = 0.0
    text_risk: float = 0.0
    note: str = ""
    baseline_ready: bool = False
    baseline_progress: int = 0


class SuspicionAnalyzerModule:
    def __init__(self):
        self.blink_history = deque(maxlen=180)
        self.gaze_history = deque(maxlen=90)
        self.text_history = deque(maxlen=30)

        self.baseline_ready = False
        self.baseline_samples = []

        self.base_mimic = 0.0
        self.base_blink_rate = 0.0
        self.base_gaze_score = 0.0
        self.base_text_score = 0.0

        # Стабилизация
        self.overall_history = deque(maxlen=12)
        self.verdict_history = deque(maxlen=10)
        self.last_stable_verdict = "неопределённо"
        self.last_stable_status = "калибровка"

        try:
            self.font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
            self.small_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
        except Exception:
            self.font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()

    def clamp01(self, value):
        return float(max(0.0, min(1.0, value)))

    def shorten_text(self, text, max_len=34):
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."

    def update_blink_history(self, blink_count):
        now = time.time()
        self.blink_history.append((now, int(blink_count)))

        while self.blink_history and (now - self.blink_history[0][0]) > 25.0:
            self.blink_history.popleft()

    def compute_blink_rate_per_min(self):
        if len(self.blink_history) < 2:
            return 0.0

        t0, b0 = self.blink_history[0]
        t1, b1 = self.blink_history[-1]

        dt = max(t1 - t0, 1e-6)
        return float((max(b1 - b0, 0) / dt) * 60.0)

    def compute_blink_risk(self):
        current_rate = self.compute_blink_rate_per_min()

        if not self.baseline_ready:
            if current_rate <= 18:
                return 0.0
            if current_rate <= 28:
                return self.clamp01((current_rate - 18) / 10.0 * 0.5)
            return self.clamp01(0.5 + (current_rate - 28) / 18.0)

        delta = current_rate - self.base_blink_rate

        if delta <= 4:
            return 0.0
        if delta <= 10:
            return self.clamp01((delta - 4) / 6.0 * 0.5)
        return self.clamp01(0.5 + (delta - 10) / 12.0)

    def update_gaze_history(self, gaze):
        now = time.time()
        self.gaze_history.append((now, gaze))

        while self.gaze_history and (now - self.gaze_history[0][0]) > 15.0:
            self.gaze_history.popleft()

    def compute_gaze_score(self):
        if len(self.gaze_history) < 3:
            return 0.0

        gazes = [g for _, g in self.gaze_history]
        non_center = sum(1 for g in gazes if g != "центр" and g != "не найдено")
        changes = 0

        for i in range(1, len(gazes)):
            if gazes[i] != gazes[i - 1]:
                changes += 1

        non_center_ratio = non_center / max(len(gazes), 1)
        change_ratio = changes / max(len(gazes) - 1, 1)

        return self.clamp01(0.55 * non_center_ratio + 0.45 * change_ratio)

    def compute_gaze_risk(self):
        current_score = self.compute_gaze_score()

        if not self.baseline_ready:
            return current_score

        delta = current_score - self.base_gaze_score
        return self.clamp01(delta * 2.2)

    def analyze_text(self, text):
        text = (text or "").strip().lower()
        if not text:
            return 0.0, "нет текста"

        self.text_history.append(text)

        fillers = [
            "ээ", "эм", "ну", "как бы", "типа", "короче",
            "это самое", "в общем", "ну вот"
        ]
        correction_words = ["нет", "точнее", "в смысле", "то есть", "вернее"]

        filler_hits = sum(text.count(token) for token in fillers)
        correction_hits = sum(text.count(token) for token in correction_words)

        words = re.findall(r"\w+", text, flags=re.UNICODE)
        repeated_words = 0
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                repeated_words += 1

        punctuation_absent = 0 if any(ch in text for ch in ".!,?") else 1
        very_short = 1 if len(words) <= 2 else 0

        raw = (
            0.30 * min(filler_hits, 3) / 3.0 +
            0.25 * min(correction_hits, 3) / 3.0 +
            0.25 * min(repeated_words, 2) / 2.0 +
            0.10 * punctuation_absent +
            0.10 * very_short
        )

        score = self.clamp01(raw)

        if filler_hits >= 2:
            note = "много пауз/слов-паразитов"
        elif correction_hits >= 2:
            note = "самоисправления"
        elif repeated_words >= 1:
            note = "повторы"
        else:
            note = "спокойная речь"

        return score, note

    def compute_text_risk(self, text):
        current_score, note = self.analyze_text(text)

        if not self.baseline_ready:
            return current_score, note

        delta = current_score - self.base_text_score
        return self.clamp01(delta * 2.0), note

    def analyze_mimic(self, mimic_data):
        smile_relief = 0.18 * float(mimic_data.smile)

        raw = (
            0.36 * float(mimic_data.brow_frown) +
            0.22 * float(mimic_data.eye_squint) +
            0.18 * float(mimic_data.mouth_open) +
            0.14 * float(mimic_data.brow_raise)
        ) - smile_relief

        return self.clamp01(raw)

    def compute_mimic_risk(self, mimic_data):
        current_score = self.analyze_mimic(mimic_data)

        if not self.baseline_ready:
            return current_score

        delta = current_score - self.base_mimic
        return self.clamp01(delta * 2.0)

    def update_baseline(self, mimic_data, eye_tracker, subtitle_text):
        mimic_score = self.analyze_mimic(mimic_data)
        blink_rate = self.compute_blink_rate_per_min()
        gaze_score = self.compute_gaze_score()
        text_score, _ = self.analyze_text(subtitle_text)

        self.baseline_samples.append({
            "mimic": mimic_score,
            "blink": blink_rate,
            "gaze": gaze_score,
            "text": text_score
        })

        if len(self.baseline_samples) < 25:
            return

        self.base_mimic = float(np.mean([x["mimic"] for x in self.baseline_samples]))
        self.base_blink_rate = float(np.mean([x["blink"] for x in self.baseline_samples]))
        self.base_gaze_score = float(np.mean([x["gaze"] for x in self.baseline_samples]))
        self.base_text_score = float(np.mean([x["text"] for x in self.baseline_samples]))

        self.baseline_ready = True
        self.baseline_samples = []

    def smooth_overall(self, current_overall):
        self.overall_history.append(float(current_overall))
        return float(sum(self.overall_history) / len(self.overall_history))

    def status_from_score(self, score):
        if score < 0.16:
            return "спокойно"
        if score < 0.32:
            return "умеренно"
        if score < 0.52:
            return "напряжение"
        return "подозрительно"

    def verdict_from_score(self, score):
        if score < 0.24:
            return "скорее правда"
        if score < 0.50:
            return "неопределённо"
        return "скорее ложь"

    def stabilize_verdict(self, current_verdict):
        self.verdict_history.append(current_verdict)

        if len(self.verdict_history) < 4:
            return self.last_stable_verdict

        counts = Counter(self.verdict_history)
        most_common_verdict, most_common_count = counts.most_common(1)[0]

        # Меняем вердикт только если он реально устойчив
        if most_common_count >= 4:
            self.last_stable_verdict = most_common_verdict

        return self.last_stable_verdict

    def stabilize_status(self, current_status):
        # Лёгкая стабилизация статуса через сглаженный overall
        self.last_stable_status = current_status
        return self.last_stable_status

    def color_from_status(self, status):
        if status == "спокойно":
            return (120, 230, 140)
        if status == "умеренно":
            return (120, 200, 255)
        if status == "напряжение":
            return (255, 200, 110)
        if status == "подозрительно":
            return (255, 120, 120)
        return (255, 255, 255)

    def color_from_verdict(self, verdict):
        if verdict == "скорее правда":
            return (120, 230, 140)
        if verdict == "неопределённо":
            return (255, 220, 120)
        if verdict == "скорее ложь":
            return (255, 120, 120)
        return (255, 255, 255)

    def process(self, eye_tracker, mimic_data, subtitle_text):
        self.update_blink_history(eye_tracker.blink_count)
        self.update_gaze_history(eye_tracker.last_gaze)

        if not self.baseline_ready:
            self.update_baseline(mimic_data, eye_tracker, subtitle_text)

        blink_risk = self.compute_blink_risk()
        gaze_risk = self.compute_gaze_risk()
        text_risk, note = self.compute_text_risk(subtitle_text)
        mimic_risk = self.compute_mimic_risk(mimic_data)

        raw_overall = self.clamp01(
            0.34 * mimic_risk +
            0.24 * blink_risk +
            0.20 * gaze_risk +
            0.22 * text_risk
        )

        smooth_overall = self.smooth_overall(raw_overall)

        if not self.baseline_ready:
            return SuspicionData(
                overall=smooth_overall,
                status="калибровка",
                verdict="неопределённо",
                mimic_risk=mimic_risk,
                blink_risk=blink_risk,
                gaze_risk=gaze_risk,
                text_risk=text_risk,
                note="настройка личной базы",
                baseline_ready=False,
                baseline_progress=len(self.baseline_samples)
            )

        current_status = self.status_from_score(smooth_overall)
        current_verdict = self.verdict_from_score(smooth_overall)

        stable_status = self.stabilize_status(current_status)
        stable_verdict = self.stabilize_verdict(current_verdict)

        return SuspicionData(
            overall=smooth_overall,
            status=stable_status,
            verdict=stable_verdict,
            mimic_risk=mimic_risk,
            blink_risk=blink_risk,
            gaze_risk=gaze_risk,
            text_risk=text_risk,
            note=note,
            baseline_ready=True,
            baseline_progress=25
        )

    def draw_progress_bar(self, draw, x, y, width, height, value, label, status):
        color = self.color_from_status(status)

        draw.text((x, y - 2), label, font=self.small_font, fill=color)

        bar_x1 = x + 110
        bar_y1 = y + 2
        bar_x2 = bar_x1 + width
        bar_y2 = bar_y1 + height

        draw.rounded_rectangle(
            (bar_x1, bar_y1, bar_x2, bar_y2),
            radius=5,
            fill=(45, 45, 45),
            outline=(110, 110, 110),
            width=1
        )

        fill_width = int(width * self.clamp01(value))
        if fill_width > 0:
            draw.rounded_rectangle(
                (bar_x1, bar_y1, bar_x1 + fill_width, bar_y2),
                radius=5,
                fill=color
            )

        draw.text((bar_x2 + 8, y - 2), f"{value:.2f}", font=self.small_font, fill=(255, 255, 255))

    def draw_panel(self, frame, suspicion_data):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)

        frame_width = frame.shape[1]

        x1 = frame_width - 420 - 15
        y1 = 350
        x2 = frame_width - 15
        y2 = 630

        draw.rounded_rectangle(
            (x1, y1, x2, y2),
            radius=10,
            fill=(0, 0, 0),
            outline=(255, 255, 255),
            width=2
        )

        status_color = self.color_from_status(suspicion_data.status)
        verdict_color = self.color_from_verdict(suspicion_data.verdict)

        draw.text(
            (x1 + 12, y1 + 10),
            f"Индикатор: {suspicion_data.status}",
            font=self.font,
            fill=status_color
        )
        draw.text(
            (x1 + 12, y1 + 38),
            f"Вердикт: {suspicion_data.verdict}",
            font=self.font,
            fill=verdict_color
        )

        if not suspicion_data.baseline_ready:
            note_text = f"Калибровка: {suspicion_data.baseline_progress}/25"
        else:
            note_text = f"Примечание: {self.shorten_text(suspicion_data.note)}"

        draw.text(
            (x1 + 12, y1 + 66),
            note_text,
            font=self.small_font,
            fill=(220, 220, 220)
        )

        start_y = y1 + 100
        gap = 34
        bar_width = 160
        bar_height = 14

        self.draw_progress_bar(draw, x1 + 12, start_y + gap * 0, bar_width, bar_height, suspicion_data.overall, "Общий риск", suspicion_data.status)
        self.draw_progress_bar(draw, x1 + 12, start_y + gap * 1, bar_width, bar_height, suspicion_data.mimic_risk, "Мимика", suspicion_data.status)
        self.draw_progress_bar(draw, x1 + 12, start_y + gap * 2, bar_width, bar_height, suspicion_data.blink_risk, "Моргания", suspicion_data.status)
        self.draw_progress_bar(draw, x1 + 12, start_y + gap * 3, bar_width, bar_height, suspicion_data.gaze_risk, "Взгляд", suspicion_data.status)
        self.draw_progress_bar(draw, x1 + 12, start_y + gap * 4, bar_width, bar_height, suspicion_data.text_risk, "Речь", suspicion_data.status)

        draw.text(
            (x1 + 12, y2 - 28),
            "Оценка, не детектор лжи",
            font=self.small_font,
            fill=(180, 180, 180)
        )

        result = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return result