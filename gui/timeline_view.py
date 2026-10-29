# -*- coding: utf-8 -*-
"""Временная шкала сессии: дорожки показателей, обычный диапазон, база и события."""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from gui.observations import (ACTIVITY_HIGH, ACTIVITY_MARGIN, BLINK_DELTA, DEVIATION_HIGH,
                              GAZE_HIGH, GAZE_MARGIN, REDNESS_HIGH)
from gui.theme import C, font, numeric_font, qc
from gui.widgets import Card, CardHeader, SegmentedControl
from models import (EV_CALIBRATION, EV_EYES, EV_LABEL, EV_MICRO, EV_QUALITY, EV_SPEECH,
                    EV_SYSTEM)

LABEL_W = 150
AXIS_H = 20
STRIP_H = 22


def _fmt_time(t):
    t = max(0, int(t))
    if t >= 3600:
        return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"
    return f"{t // 60:02d}:{t % 60:02d}"


def _pct(v):
    return f"{v * 100:.0f}%"


class Lane:
    SHORT = {"activity": "Активность", "blink_rate": "Моргание", "gaze_off": "Взгляд",
             "redness": "Покраснение", "deviation": "Отклонение"}

    def __init__(self, key, title, fmt, band_fn, scale_fn, baseline_fn):
        self.key, self.title, self.fmt = key, title, fmt
        self.short = self.SHORT.get(key, title)
        self.band_fn, self.scale_fn, self.baseline_fn = band_fn, scale_fn, baseline_fn


def _max(values, default):
    vals = [v for v in values if v is not None]
    return max(vals) if vals else default


LANES = [
    Lane("activity", "Активность мимики", _pct,
         lambda b: (0.0, (b["activity"] + ACTIVITY_MARGIN) if b.get("activity") is not None else ACTIVITY_HIGH),
         lambda vals, b: (0.0, 1.0),
         lambda b: b.get("activity")),
    Lane("blink_rate", "Моргание", lambda v: f"{v:.0f}/мин",
         lambda b: ((max(b["blink_rate"] - BLINK_DELTA, 0.0), b["blink_rate"] + BLINK_DELTA)
                    if b.get("blink_rate") is not None else None),
         lambda vals, b: (0.0, max(40.0, _max(vals, 0) * 1.1, (b.get("blink_rate") or 17.0) + 15.0)),
         lambda b: b.get("blink_rate")),
    Lane("gaze_off", "Взгляд вне центра", _pct,
         lambda b: (0.0, (b["gaze_off"] + GAZE_MARGIN) if b.get("gaze_off") is not None else GAZE_HIGH),
         lambda vals, b: (0.0, 1.0),
         lambda b: b.get("gaze_off")),
    Lane("redness", "Покраснение", lambda v: f"{v:.2f}",
         lambda b: (0.0, REDNESS_HIGH),
         lambda vals, b: (0.0, max(0.35, _max(vals, 0) * 1.1)),
         lambda b: b.get("redness")),
    Lane("deviation", "Отклонение от базы", _pct,
         lambda b: (0.0, DEVIATION_HIGH),
         lambda vals, b: (0.0, max(0.6, _max(vals, 0) * 1.1)),
         lambda b: 0.0 if b.get("deviation") is not None else None),
]


class TimelineView(QWidget):
    def __init__(self, timeline, parent=None):
        super().__init__(parent)
        self.timeline = timeline
        self.window_s = 180
        self.live = False
        self.hover_x = None
        self.setMouseTracking(True)
        self.setMinimumHeight(90)
        self._hatch = QBrush(qc(C.TEXT_4, 120), Qt.BDiagPattern)
        self._painted_version = -1

    def _lanes(self):
        """На низкой шкале - три главные дорожки, иначе линии становятся нитками."""
        plot_h = self.height() - STRIP_H - AXIS_H - 4
        if plot_h / len(LANES) >= 18:
            return LANES
        return [lane for lane in LANES if lane.key in ("activity", "blink_rate", "deviation")]

    def set_window(self, seconds):
        self.window_s = seconds
        self.update()

    def refresh(self):
        if self.timeline.version != self._painted_version:
            self.update()

    # ---------- геометрия ----------

    def _range(self):
        end = max(self.timeline.duration, self.timeline.t[-1] if self.timeline.t else 0.0)
        if self.window_s <= 0:
            return 0.0, max(end, 60.0)
        if end <= self.window_s:
            return 0.0, float(self.window_s)
        return end - self.window_s, end

    def _plot_rect(self):
        return QRectF(LABEL_W, STRIP_H + 2, self.width() - LABEL_W - 14, self.height() - STRIP_H - AXIS_H - 4)

    def _x(self, t, t0, t1, plot):
        return plot.x() + (t - t0) / max(t1 - t0, 1e-6) * plot.width()

    def mouseMoveEvent(self, event):
        x = event.position().x()
        plot = self._plot_rect()
        self.hover_x = x if plot.left() <= x <= plot.right() else None
        self.update()

    def leaveEvent(self, event):
        self.hover_x = None
        self.update()

    # ---------- рисование ----------

    def paintEvent(self, event):
        self._painted_version = self.timeline.version
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        tl = self.timeline
        plot = self._plot_rect()
        t0, t1 = self._range()
        lanes = self._lanes()
        lane_h = plot.height() / len(lanes)
        pad = min(5.0, lane_h * 0.15)

        # сетка времени
        step = self._tick_step(t1 - t0)
        p.setFont(numeric_font(11))
        first = (int(t0 // step) + 1) * step if t0 > 0 else 0
        t = first
        while t <= t1 + 1e-6:
            x = self._x(t, t0, t1, plot)
            p.setPen(QPen(qc(C.BORDER), 1))
            p.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            p.setPen(qc(C.TEXT_3))
            p.drawText(QRectF(x - 30, plot.bottom() + 3, 60, AXIS_H - 4), Qt.AlignCenter, _fmt_time(t))
            t += step

        # полоса событий
        p.setFont(font(11, 600))
        p.setPen(qc(C.TEXT_3))
        p.drawText(QRectF(0, 0, LABEL_W - 12, STRIP_H), Qt.AlignLeft | Qt.AlignVCenter, "События")

        if not tl.t:
            p.setPen(qc(C.TEXT_3))
            p.setFont(font(12))
            for i, lane in enumerate(lanes):
                y = plot.top() + i * lane_h
                self._paint_lane_label(p, lane, y, lane_h, None, False)
                if i:
                    p.setPen(QPen(qc(C.BORDER), 1))
                    p.drawLine(QPointF(0, y), QPointF(plot.right(), y))
            p.setPen(qc(C.TEXT_4))
            p.setFont(font(12))
            p.drawText(plot, Qt.AlignCenter, "Графики появятся после начала анализа")
            return

        # периоды без лица - штриховка через все дорожки
        self._paint_face_gaps(p, plot, t0, t1)

        buckets = int(plot.width())
        for i, lane in enumerate(lanes):
            y = plot.top() + i * lane_h
            if i:
                p.setPen(QPen(qc(C.BORDER), 1))
                p.drawLine(QPointF(0, y), QPointF(plot.right(), y))
            times, values = tl.bucketed(lane.key, t0, t1, buckets)
            current = tl.series[lane.key][-1] if tl.series[lane.key] else None
            band = lane.band_fn(tl.baselines)
            out = current is not None and band is not None and not (band[0] <= current <= band[1])
            self._paint_lane_label(p, lane, y, lane_h, current, out)
            lane_rect = QRectF(plot.x(), y + pad, plot.width(), lane_h - 2 * pad)
            self._paint_series(p, lane, lane_rect, times, values, t0, t1, band)

        self._paint_events(p, plot, t0, t1)

        if self.live and tl.t:
            x = self._x(tl.t[-1], t0, t1, plot)
            p.setPen(QPen(qc(C.ACCENT, 160), 1.2))
            p.drawLine(QPointF(x, STRIP_H - 4), QPointF(x, plot.bottom()))

        if self.hover_x is not None:
            self._paint_hover(p, plot, t0, t1, lane_h)

    @staticmethod
    def _tick_step(span):
        for step in (5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600):
            if span / step <= 8:
                return step
        return 7200

    def _paint_lane_label(self, p, lane, y, h, current, out):
        text = lane.fmt(current) if current is not None else "—"
        value_color = qc(C.ACCENT if out else (C.TEXT if current is not None else C.TEXT_4))
        if h >= 36:
            p.setFont(font(12))
            p.setPen(qc(C.TEXT_2))
            p.drawText(QRectF(0, y + h / 2 - 17, LABEL_W - 12, 16), Qt.AlignLeft | Qt.AlignVCenter,
                       lane.title)
            p.setFont(numeric_font(14))
            p.setPen(value_color)
            p.drawText(QRectF(0, y + h / 2 + 1, LABEL_W - 12, 18), Qt.AlignLeft | Qt.AlignVCenter, text)
            return
        # низкая дорожка - название и значение в одну строку
        value_font = numeric_font(12)
        value_w = QFontMetrics(value_font).horizontalAdvance(text) + 6
        title_font = font(11)
        p.setFont(title_font)
        p.setPen(qc(C.TEXT_2))
        title = QFontMetrics(title_font).elidedText(lane.short, Qt.ElideRight, int(LABEL_W - 16 - value_w))
        p.drawText(QRectF(0, y, LABEL_W - 12 - value_w, h), Qt.AlignLeft | Qt.AlignVCenter, title)
        p.setFont(value_font)
        p.setPen(value_color)
        p.drawText(QRectF(0, y, LABEL_W - 12, h), Qt.AlignRight | Qt.AlignVCenter, text)

    def _paint_face_gaps(self, p, plot, t0, t1):
        tl = self.timeline
        lo, hi = tl.index_range(t0, t1)
        start = None
        p.setPen(Qt.NoPen)
        for i in range(lo, hi):
            if not tl.face[i]:
                if start is None:
                    start = tl.t[i]
            elif start is not None:
                self._fill_gap(p, plot, t0, t1, start, tl.t[i])
                start = None
        if start is not None:
            self._fill_gap(p, plot, t0, t1, start, tl.t[hi - 1])

    def _fill_gap(self, p, plot, t0, t1, a, b):
        x0, x1 = self._x(a, t0, t1, plot), self._x(b, t0, t1, plot)
        r = QRectF(x0, plot.top(), max(x1 - x0, 2.0), plot.height())
        p.fillRect(r, qc(C.SURFACE_2))
        p.fillRect(r, self._hatch)

    def _paint_series(self, p, lane, r, times, values, t0, t1, band):
        lo, hi = lane.scale_fn(values, self.timeline.baselines)
        span = max(hi - lo, 1e-9)

        def y(v):
            v = max(lo, min(hi, v))
            return r.bottom() - (v - lo) / span * r.height()

        if band is not None:
            top, bottom = y(band[1]), y(band[0])
            p.fillRect(QRectF(r.x(), top, r.width(), max(bottom - top, 1.0)), qc(C.SURFACE_2))
        baseline = lane.baseline_fn(self.timeline.baselines)
        if baseline is not None:
            p.setPen(QPen(qc(C.TEXT_4), 1, Qt.DashLine))
            p.drawLine(QPointF(r.x(), y(baseline)), QPointF(r.right(), y(baseline)))

        path = QPainterPath()
        started = False
        prev_t = None
        gap = max((t1 - t0) / max(r.width(), 1) * 6, 1.5)
        for t, v in zip(times, values):
            if v is None:
                started = False
                continue
            pt = QPointF(self._x(t, t0, t1, r), y(v))
            if not started or (prev_t is not None and t - prev_t > gap):
                path.moveTo(pt)
                started = True
            else:
                path.lineTo(pt)
            prev_t = t
        if path.isEmpty():
            return
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(qc(C.TEXT), 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(path)
        if band is not None:  # выход за обычный диапазон - красным
            p.save()
            p.setPen(QPen(qc(C.ACCENT), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            top = y(band[1])
            p.setClipRect(QRectF(r.x(), r.y() - 2, r.width(), max(top - r.y() + 2, 0)))
            p.drawPath(path)
            if band[0] > lo:
                bottom = y(band[0])
                p.setClipRect(QRectF(r.x(), bottom, r.width(), r.bottom() - bottom + 2))
                p.drawPath(path)
            p.restore()

    def _paint_events(self, p, plot, t0, t1):
        cy = STRIP_H / 2
        for ev in self.timeline.events_between(t0, t1):
            x = self._x(ev.t, t0, t1, plot)
            if ev.kind == EV_MICRO:
                p.setPen(Qt.NoPen)
                p.setBrush(qc(C.ACCENT))
                p.drawPolygon(QPolygonF([QPointF(x, cy - 4.5), QPointF(x + 4, cy),
                                         QPointF(x, cy + 4.5), QPointF(x - 4, cy)]))
            elif ev.kind == EV_SPEECH:
                p.setPen(Qt.NoPen)
                p.setBrush(qc(C.TEXT_4))
                p.drawRoundedRect(QRectF(x - 1.5, cy - 5, 3, 10), 1.5, 1.5)
            elif ev.kind in (EV_CALIBRATION, EV_LABEL):
                p.setPen(QPen(qc(C.TEXT_3), 1, Qt.DashLine))
                p.drawLine(QPointF(x, cy + 6), QPointF(x, plot.bottom()))
                p.setPen(Qt.NoPen)
                p.setBrush(qc(C.TEXT_2))
                p.drawPolygon(QPolygonF([QPointF(x - 4, cy - 4), QPointF(x + 4, cy - 4), QPointF(x, cy + 3)]))
            elif ev.kind in (EV_QUALITY, EV_EYES):
                p.setPen(QPen(qc(C.TEXT_3), 1.2))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(x, cy), 3, 3)
            elif ev.kind == EV_SYSTEM and ev.level == "error":
                p.setPen(QPen(qc(C.ACCENT), 1.4))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QPointF(x, cy), 3.5, 3.5)

    def _paint_hover(self, p, plot, t0, t1, lane_h):
        tl = self.timeline
        x = self.hover_x
        t = t0 + (x - plot.x()) / max(plot.width(), 1) * (t1 - t0)
        p.setPen(QPen(qc(C.TEXT, 90), 1))
        p.drawLine(QPointF(x, STRIP_H - 2), QPointF(x, plot.bottom()))

        rows = [(lane.title, tl.value_at(lane.key, t), lane.fmt) for lane in LANES]
        events = tl.events_between(t - 1.5, t + 1.5)
        events = [e for e in events if e.kind in (EV_MICRO, EV_SPEECH, EV_CALIBRATION, EV_QUALITY,
                                                  EV_LABEL, EV_EYES, EV_SYSTEM)][:3]
        f_title, f_row = font(12, 600), font(12)
        fm = QFontMetrics(f_row)
        width = 236
        for e in events:
            width = max(width, min(fm.horizontalAdvance(e.text) + 40, 380))
        height = 34 + len(rows) * 19 + (8 + len(events) * 18 if events else 0)
        bx = x + 14 if x + 14 + width < self.width() else x - 14 - width
        by = max(4, min(STRIP_H, self.height() - height - 4))
        box = QRectF(bx, by, width, height)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(18, 18, 28, 26))
        p.drawRoundedRect(box.translated(0, 2), 10, 10)
        p.setBrush(qc(C.SURFACE))
        p.setPen(QPen(qc(C.BORDER_2), 1))
        p.drawRoundedRect(box, 10, 10)
        p.setFont(f_title)
        p.setPen(qc(C.TEXT))
        p.drawText(QRectF(bx + 12, by + 8, width - 24, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   f"Время {_fmt_time(t)}")
        yy = by + 30
        for title, value, fmt in rows:
            p.setFont(f_row)
            p.setPen(qc(C.TEXT_2))
            p.drawText(QRectF(bx + 12, yy, width - 24, 18), Qt.AlignLeft | Qt.AlignVCenter, title)
            p.setFont(numeric_font(13))
            p.setPen(qc(C.TEXT if value is not None else C.TEXT_4))
            p.drawText(QRectF(bx + 12, yy, width - 24, 18), Qt.AlignRight | Qt.AlignVCenter,
                       fmt(value) if value is not None else "нет данных")
            yy += 19
        if events:
            yy += 4
            p.setPen(QPen(qc(C.BORDER), 1))
            p.drawLine(QPointF(bx + 12, yy), QPointF(bx + width - 12, yy))
            yy += 4
            p.setFont(f_row)
            for e in events:
                p.setPen(qc(C.ACCENT if e.kind == EV_MICRO else C.TEXT_2))
                text = fm.elidedText(f"{_fmt_time(e.t)}  {e.text}", Qt.ElideRight, int(width - 24))
                p.drawText(QRectF(bx + 12, yy, width - 24, 18), Qt.AlignLeft | Qt.AlignVCenter, text)
                yy += 18


class TimelineCard(Card):
    WINDOWS = [("60", "1 мин", None), ("180", "3 мин", None), ("600", "10 мин", None), ("0", "Вся", None)]

    def __init__(self, timeline, parent=None):
        super().__init__(parent, margins=(18, 14, 18, 12))
        self.body.setSpacing(8)
        header = CardHeader("Динамика сессии",
                            "серая полоса — обычный диапазон, пунктир — база, красным — выход за диапазон")
        self.windows = SegmentedControl(self.WINDOWS, height=30, text_px=12)
        self.windows.set_key("180", animate=False)
        header.add_right(self.windows)
        self.body.addWidget(header)
        self.view = TimelineView(timeline)
        self.body.addWidget(self.view, 1)
        self.windows.changed.connect(lambda key: self.view.set_window(int(key)))

    def set_window(self, seconds):
        self.windows.set_key(str(seconds), animate=False)
        self.view.set_window(seconds)
