# -*- coding: utf-8 -*-
"""Временные ряды и события сессии для временной шкалы и итогов.

Хранит только числовые показатели (4 отсчёта в секунду) и текстовые события -
никаких кадров и звука.
"""
import bisect
from collections import Counter

from models import EV_MICRO, EV_SPEECH, EXPR_NO_FACE

SAMPLE_PERIOD = 0.25

# ключ ряда -> (название, единицы)
SERIES = {
    "activity": ("Активность мимики", "%"),
    "blink_rate": ("Моргание", "/мин"),
    "gaze_off": ("Взгляд вне центра", "%"),
    "redness": ("Покраснение", "индекс"),
    "deviation": ("Отклонение от базы", "%"),
    "intensity": ("Выраженность мимики", "%"),
}


class SessionTimeline:
    def __init__(self, sample_period=SAMPLE_PERIOD):
        self.sample_period = sample_period
        self.t = []
        self.series = {key: [] for key in SERIES}
        self.face = []
        self.expression = []
        self.events = []
        self.baselines = {key: None for key in SERIES}
        self.last_snapshot = None
        self._next_t = None  # время следующего отсчёта (сетка с шагом sample_period)
        self.blink_total = 0
        self.visible_time = 0.0
        self.duration = 0.0
        self.version = 0  # растёт при каждом изменении (для перерисовки)

    def clear(self):
        self.__init__(self.sample_period)

    # ---------- запись ----------

    @staticmethod
    def values_from_snapshot(s):
        dev = s.deviation
        deviation = None
        if s.face_detected and dev is not None and dev.baseline_ready:
            deviation = dev.overall
        redness = None
        if s.face_detected and s.redness is not None and s.redness_ready:
            redness = s.redness.overall
        return {
            "activity": s.activity,
            "blink_rate": s.blink_rate,
            "gaze_off": s.gaze_off_ratio,
            "redness": redness,
            "deviation": deviation,
            "intensity": s.expression_intensity if s.face_detected else None,
        }

    def add_snapshot(self, s):
        """Возвращает True, если добавлен новый отсчёт."""
        self.last_snapshot = s
        self.duration = max(self.duration, s.t)
        self.blink_total = s.blink_count
        self.visible_time = s.visible_time
        self.baselines["blink_rate"] = s.baseline_blink_rate
        self.baselines["gaze_off"] = s.baseline_gaze_off
        self.baselines["activity"] = s.baseline_activity
        self.baselines["redness"] = 0.0 if s.redness_ready else None
        self.baselines["deviation"] = 0.0
        if self.t and s.t <= self.t[-1]:
            return False  # время не может идти назад
        if self._next_t is not None and s.t < self._next_t - 1e-6:
            return False
        # отсчёты по сетке: при снимках раз в 0.1 с получается ровно 4 в секунду, а не 3.3
        self._next_t = max((self._next_t or s.t) + self.sample_period, s.t)
        self.t.append(s.t)
        for key, value in self.values_from_snapshot(s).items():
            self.series[key].append(None if value is None else float(value))
        self.face.append(bool(s.face_detected))
        self.expression.append(s.expression if s.face_detected else EXPR_NO_FACE)
        self.version += 1
        return True

    def add_event(self, event):
        self.events.append(event)
        self.duration = max(self.duration, event.t)
        self.version += 1

    # ---------- чтение ----------

    def index_range(self, t_from, t_to):
        lo = bisect.bisect_left(self.t, t_from)
        hi = bisect.bisect_right(self.t, t_to)
        return lo, hi

    def value_at(self, key, t):
        if not self.t:
            return None
        i = bisect.bisect_left(self.t, t)
        candidates = [j for j in (i - 1, i) if 0 <= j < len(self.t)]
        if not candidates:
            return None
        j = min(candidates, key=lambda k: abs(self.t[k] - t))
        if abs(self.t[j] - t) > max(1.0, self.sample_period * 4):
            return None
        return self.series[key][j]

    def events_between(self, t_from, t_to, kinds=None):
        return [e for e in self.events
                if t_from <= e.t <= t_to and (kinds is None or e.kind in kinds)]

    def recent(self, key, seconds):
        """Значения ряда за последние seconds секунд (для мини-графиков)."""
        if not self.t:
            return [], []
        lo, hi = self.index_range(self.t[-1] - seconds, self.t[-1])
        return self.t[lo:hi], self.series[key][lo:hi]

    def bucketed(self, key, t_from, t_to, buckets):
        """Среднее по корзинам: для длинных сессий на графике не нужны все точки."""
        lo, hi = self.index_range(t_from, t_to)
        times, values = self.t[lo:hi], self.series[key][lo:hi]
        if len(times) <= buckets * 2 or buckets <= 0:
            return times, values
        span = max(t_to - t_from, 1e-6)
        sums = [0.0] * buckets
        counts = [0] * buckets
        for t, v in zip(times, values):
            if v is None:
                continue
            b = min(int((t - t_from) / span * buckets), buckets - 1)
            sums[b] += v
            counts[b] += 1
        out_t, out_v = [], []
        for b in range(buckets):
            out_t.append(t_from + (b + 0.5) * span / buckets)
            out_v.append(sums[b] / counts[b] if counts[b] else None)
        return out_t, out_v

    # ---------- итоги ----------

    def summary(self):
        n = len(self.t)
        face_share = (sum(self.face) / n) if n else None

        def mean(key):
            vals = [v for v in self.series[key] if v is not None]
            return (sum(vals) / len(vals)) if vals else None

        def peak(key):
            vals = [v for v in self.series[key] if v is not None]
            return max(vals) if vals else None

        expressions = Counter(e for e, f in zip(self.expression, self.face) if f)
        total_expr = sum(expressions.values())
        expr_share = [(name, count / total_expr) for name, count in expressions.most_common(4)] \
            if total_expr else []

        blink_rate = None
        if self.visible_time >= 15.0:
            blink_rate = self.blink_total / self.visible_time * 60.0

        return {
            "duration_s": round(self.duration, 1),
            "visible_s": round(self.visible_time, 1),
            "face_share": None if face_share is None else round(face_share, 3),
            "blink_total": int(self.blink_total),
            "blink_rate_mean": None if blink_rate is None else round(blink_rate, 1),
            "gaze_off_mean": _round(mean("gaze_off")),
            "activity_mean": _round(mean("activity")),
            "redness_peak": _round(peak("redness")),
            "deviation_mean": _round(mean("deviation")),
            "deviation_peak": _round(peak("deviation")),
            "rapid_changes": sum(1 for e in self.events if e.kind == EV_MICRO),
            "phrases": sum(1 for e in self.events if e.kind == EV_SPEECH),
            "expressions": [[name, round(share, 3)] for name, share in expr_share],
            "baseline_blink_rate": _round(self.baselines.get("blink_rate")),
        }


def _round(value, digits=3):
    return None if value is None else round(float(value), digits)
