# -*- coding: utf-8 -*-
"""Индикатор напряжения.

ВАЖНО: это эвристика по мимике, морганиям, взгляду и речи. Она показывает
отклонение человека от его же собственной базы, но НЕ определяет ложь:
стресс, усталость, свет и камера дают такие же изменения у честных людей.
"""
import re
from collections import Counter, deque
from dataclasses import dataclass, replace

import numpy as np

from models import GAZE_CENTER
from ui_widgets import (BLUE, GRAY, GRAY_DARK, GREEN, ORANGE, RED, WHITE, YELLOW,
                        clamp01, draw_bar, load_font, new_panel, to_bgra)

BASELINE_SECONDS = 40.0   # сколько секунд с хорошим кадром собирается личная база
BLINK_WINDOW = 25.0       # окно подсчёта морганий, с
GAZE_WINDOW = 15.0        # окно оценки взгляда, с
SPEECH_WINDOW = 30.0      # окно оценки речи, с
MIN_WORDS = 8             # меньше слов речь не оценивается

STATUS_CALIBRATION = "калибровка"
STATUS_CALM = "спокойно"
STATUS_MODERATE = "умеренно"
STATUS_RAISED = "повышено"
STATUS_HIGH = "высокое"

VERDICT_NORMAL = "в пределах нормы"
VERDICT_UNCERTAIN = "неопределённо"
VERDICT_ABOVE = "выше обычного"

FILLER_PATTERNS = [re.compile(p) for p in (
    r"\bээ+\b", r"\bэм+\b", r"\bну\b", r"\bкак бы\b", r"\bтипа\b",
    r"\bкороче\b", r"\bэто самое\b", r"\bв общем\b")]
CORRECTION_PATTERNS = [re.compile(p) for p in (
    r"\bточнее\b", r"\bвернее\b", r"\bто есть\b", r"\bв смысле\b", r"\bнет\s*,")]


@dataclass
class SuspicionData:
    overall: float = 0.0
    status: str = STATUS_CALIBRATION
    verdict: str = VERDICT_UNCERTAIN

    mimic_risk: float = 0.0
    blink_risk: float = 0.0
    gaze_risk: float = 0.0
    text_risk: float = 0.0

    note: str = ""
    baseline_ready: bool = False
    baseline_progress: int = 0  # процент готовности базы


class SuspicionAnalyzerModule:
    def __init__(self):
        self.blink_history = deque(maxlen=2000)
        self.gaze_history = deque(maxlen=1000)
        self.speech = deque()  # (время, текст)

        self.overall_history = deque(maxlen=10)
        self.verdict_history = deque(maxlen=8)
        self.status_history = deque(maxlen=8)
        self.last_stable_status = STATUS_CALM
        self.last_stable_verdict = VERDICT_UNCERTAIN

        self.last_data = SuspicionData()
        self.start_calibration(silent=True)
        self._use_defaults()  # работает сразу, без ожидания

    # ---------- калибровка ----------

    def start_calibration(self, silent=False):
        self.baseline_ready = False
        self.baseline_samples = []
        self.valid_time = 0.0
        self.last_t = None
        self.blink_start_count = None
        self.base_mimic = 0.0
        self.base_blink_rate = 0.0
        self.base_gaze = 0.0
        self.base_text = 0.0
        self.overall_history.clear()
        self.status_history.clear()
        self.verdict_history.clear()
        self.last_data = SuspicionData()
        if not silent:
            print("Калибровка индикатора напряжения: говорите и ведите себя как обычно.")

    def _use_defaults(self):
        """Типичные значения вместо личной базы: индикатор работает с первой секунды."""
        self.baseline_ready = True
        self.base_mimic = 0.05
        self.base_blink_rate = 17.0
        self.base_gaze = 0.25
        self.base_text = 0.10
        self.last_data = SuspicionData(status=STATUS_CALM, verdict=VERDICT_UNCERTAIN,
                                       baseline_ready=True, baseline_progress=100)

    # ---------- речь ----------

    def feed_speech(self, text, now):
        text = (text or "").strip().lower()
        if text:
            self.speech.append((now, text))

    def _trim_speech(self, now):
        while self.speech and now - self.speech[0][0] > SPEECH_WINDOW:
            self.speech.popleft()

    def analyze_speech(self):
        """Оценка по ВСЕМ фразам окна (каждая считается один раз)."""
        if not self.speech:
            return 0.0, "нет речи", False

        text = " ".join(t for _, t in self.speech)
        words = re.findall(r"\w+", text)
        if len(words) < MIN_WORDS:
            return 0.0, "мало речи", False

        filler_hits = sum(len(p.findall(text)) for p in FILLER_PATTERNS)
        correction_hits = sum(len(p.findall(text)) for p in CORRECTION_PATTERNS)
        repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i - 1])

        n = float(len(words))
        score = clamp01(0.35 * min(filler_hits / n / 0.08, 1.0)
                        + 0.30 * min(correction_hits / n / 0.04, 1.0)
                        + 0.35 * min(repeats / n / 0.04, 1.0))

        if filler_hits / n > 0.05:
            note = "много слов-паразитов"
        elif correction_hits / n > 0.025:
            note = "самоисправления"
        elif repeats / n > 0.025:
            note = "повторы слов"
        else:
            note = "спокойная речь"
        return score, note, True

    # ---------- сигналы ----------

    def _blink_rate(self, now):
        while self.blink_history and now - self.blink_history[0][0] > BLINK_WINDOW:
            self.blink_history.popleft()
        if len(self.blink_history) < 2:
            return None
        (t0, b0), (t1, b1) = self.blink_history[0], self.blink_history[-1]
        if t1 - t0 < 10.0:  # слишком короткое окно - частота ненадёжна
            return None
        return max(b1 - b0, 0) / (t1 - t0) * 60.0

    def _blink_risk(self, now):
        rate = self._blink_rate(now)
        if rate is None:
            return 0.0
        delta = abs(rate - self.base_blink_rate)  # и учащение, и урежение
        if delta <= 4:
            return 0.0
        if delta <= 10:
            return clamp01((delta - 4) / 6.0 * 0.55)
        return clamp01(0.55 + (delta - 10) / 12.0)

    def _gaze_score(self, now):
        while self.gaze_history and now - self.gaze_history[0][0] > GAZE_WINDOW:
            self.gaze_history.popleft()
        gazes = [g for _, g in self.gaze_history]
        if len(gazes) < 3:
            return 0.0
        non_center = sum(1 for g in gazes if g != GAZE_CENTER) / len(gazes)
        changes = sum(1 for i in range(1, len(gazes)) if gazes[i] != gazes[i - 1])
        return clamp01(0.55 * non_center + 0.45 * changes / max(len(gazes) - 1, 1))

    @staticmethod
    def _mimic_score(m):
        raw = (0.34 * m.brow_frown + 0.22 * m.eye_squint
               + 0.18 * m.mouth_open + 0.12 * m.brow_raise) - 0.22 * m.smile
        return clamp01(raw)

    # ---------- стабилизация ----------

    @staticmethod
    def status_from_score(score):
        if score < 0.15:
            return STATUS_CALM
        if score < 0.30:
            return STATUS_MODERATE
        if score < 0.48:
            return STATUS_RAISED
        return STATUS_HIGH

    @staticmethod
    def verdict_from_score(score):
        if score < 0.22:
            return VERDICT_NORMAL
        if score < 0.56:
            return VERDICT_UNCERTAIN
        return VERDICT_ABOVE

    @staticmethod
    def stabilize(history, current, last):
        history.append(current)
        if len(history) < 5:
            return last
        label, count = Counter(history).most_common(1)[0]
        return label if count >= 4 else last

    # ---------- основной вызов ----------

    def process(self, eye_tracker, mimic_data, now, valid=True):
        """now - время кадра (сек). valid=False - нет лица или плохой кадр."""
        self._trim_speech(now)

        if not valid:
            self.last_t = None  # паузу не засчитываем в калибровку
            self.last_data = replace(self.last_data, note="нет лица / плохой кадр")
            return self.last_data

        dt = 0.0 if self.last_t is None else min(max(now - self.last_t, 0.0), 1.0)
        self.last_t = now

        self.blink_history.append((now, int(eye_tracker.blink_count)))
        self.gaze_history.append((now, eye_tracker.last_gaze))

        mimic_score = self._mimic_score(mimic_data)
        gaze_score = self._gaze_score(now)
        text_score, text_note, has_speech = self.analyze_speech()

        if not self.baseline_ready:
            if self.blink_start_count is None:
                self.blink_start_count = int(eye_tracker.blink_count)
            self.valid_time += dt
            self.baseline_samples.append((mimic_score, gaze_score, text_score, has_speech))

            if self.valid_time >= BASELINE_SECONDS:
                self._finish_baseline(eye_tracker)
            else:
                self.last_data = SuspicionData(
                    status=STATUS_CALIBRATION,
                    verdict=VERDICT_UNCERTAIN,
                    note="сбор личной базы",
                    baseline_ready=False,
                    baseline_progress=int(100 * self.valid_time / BASELINE_SECONDS),
                )
                return self.last_data

        mimic_risk = clamp01((mimic_score - self.base_mimic) * 1.8)
        gaze_risk = clamp01((gaze_score - self.base_gaze) * 2.0)
        blink_risk = self._blink_risk(now)
        text_risk = clamp01((text_score - self.base_text) * 1.8) if has_speech else 0.0

        raw_overall = clamp01(0.32 * mimic_risk + 0.24 * blink_risk
                              + 0.18 * gaze_risk + 0.26 * text_risk)
        self.overall_history.append(raw_overall)
        overall = float(sum(self.overall_history) / len(self.overall_history))

        self.last_stable_status = self.stabilize(
            self.status_history, self.status_from_score(overall), self.last_stable_status)
        self.last_stable_verdict = self.stabilize(
            self.verdict_history, self.verdict_from_score(overall), self.last_stable_verdict)

        self.last_data = SuspicionData(
            overall=overall,
            status=self.last_stable_status,
            verdict=self.last_stable_verdict,
            mimic_risk=mimic_risk,
            blink_risk=blink_risk,
            gaze_risk=gaze_risk,
            text_risk=text_risk,
            note=text_note,
            baseline_ready=True,
            baseline_progress=100,
        )
        return self.last_data

    def _finish_baseline(self, eye_tracker):
        samples = self.baseline_samples
        self.base_mimic = float(np.mean([s[0] for s in samples]))
        self.base_gaze = float(np.mean([s[1] for s in samples]))
        speech_scores = [s[2] for s in samples if s[3]]
        self.base_text = float(np.mean(speech_scores)) if speech_scores else 0.0

        blinks = max(int(eye_tracker.blink_count) - (self.blink_start_count or 0), 0)
        self.base_blink_rate = blinks / max(self.valid_time, 1e-6) * 60.0

        self.baseline_ready = True
        self.baseline_samples = []
        print(f"Личная база собрана: моргания {self.base_blink_rate:.1f}/мин.")

    # ---------- панель ----------

    @staticmethod
    def color_from_status(status):
        return {STATUS_CALM: GREEN, STATUS_MODERATE: (120, 200, 255),
                STATUS_RAISED: ORANGE, STATUS_HIGH: RED,
                STATUS_CALIBRATION: YELLOW}.get(status, WHITE)

    @staticmethod
    def color_from_verdict(verdict):
        return {VERDICT_NORMAL: GREEN, VERDICT_UNCERTAIN: YELLOW,
                VERDICT_ABOVE: RED}.get(verdict, WHITE)

    def render_panel(self, d):
        img, draw = new_panel(420, 265)
        font, small = load_font(17), load_font(15)

        status_color = self.color_from_status(d.status)
        draw.text((12, 10), f"Напряжение: {d.status}", font=font, fill=status_color)
        draw.text((12, 38), f"Оценка: {d.verdict}", font=font,
                  fill=self.color_from_verdict(d.verdict))

        if not d.baseline_ready:
            info = f"Калибровка: {d.baseline_progress}%"
        else:
            info = f"Заметка: {d.note}"
        draw.text((12, 66), info[:46], font=small, fill=GRAY)

        rows = [("Общий", d.overall), ("Мимика", d.mimic_risk), ("Моргание", d.blink_risk),
                ("Взгляд", d.gaze_risk), ("Речь", d.text_risk)]
        for i, (label, value) in enumerate(rows):
            draw_bar(draw, 12, 100 + 34 * i, value, label, status_color, small,
                     label_w=110, width=160, label_color=status_color)

        return to_bgra(img)