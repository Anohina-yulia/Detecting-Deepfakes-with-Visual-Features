# -*- coding: utf-8 -*-
"""Панель наблюдений: компактные строки-«приборы» с реально вычисляемыми показателями.

Каждая строка показывает значение, пояснение, мини-визуализацию и подсказку
с ограничениями метода. Если показатель не определён - «Нет данных».
"""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QWidget

from gui import icons
from gui.theme import C, font, numeric_font, qc
from gui.widgets import Card, CardHeader, Chip, label
from models import (EXPR_BROWS, EXPR_FROWN, EXPR_MOUTH, EXPR_NEUTRAL, EXPR_SMILE, EXPR_SQUINT,
                    RED_MILD, RED_NORMAL, RED_NOTICEABLE, RED_STRONG)
from suspicion_analyzer import (STATUS_CALIBRATION, STATUS_CALM, STATUS_HIGH, STATUS_MODERATE,
                                STATUS_RAISED)

NO_DATA = "Нет данных"
ROW_H = 48

# нормальные диапазоны (как на временной шкале)
BLINK_DELTA = 10.0       # отклонение частоты морганий от базы, как в индикаторе отклонения
ACTIVITY_HIGH = 0.6
ACTIVITY_MARGIN = 0.25
GAZE_HIGH = 0.5
GAZE_MARGIN = 0.25
REDNESS_HIGH = 0.16
DEVIATION_HIGH = 0.30

DEVIATION_TEXT = {
    STATUS_CALM: "в пределах базы",
    STATUS_MODERATE: "небольшое отклонение",
    STATUS_RAISED: "заметное отклонение",
    STATUS_HIGH: "сильное отклонение",
    STATUS_CALIBRATION: "сбор базы",
}
REDNESS_TEXT = {RED_NORMAL: "Норма", RED_MILD: "Лёгкое", RED_NOTICEABLE: "Заметное",
                RED_STRONG: "Сильное"}
EXPRESSION_ORDER = [("smile", EXPR_SMILE), ("mouth_open", EXPR_MOUTH), ("brow_raise", EXPR_BROWS),
                    ("brow_frown", EXPR_FROWN), ("eye_squint", EXPR_SQUINT)]

HINTS = {
    "face": ("Обнаружение лица",
             "Нейросеть MediaPipe Face Landmarker находит лицо и 478 точек. Качество кадра "
             "оценивается по размеру лица, яркости и резкости; при плохом свете показатели "
             "менее надёжны."),
    "gaze": ("Направление взгляда",
             "Положение радужки относительно уголков глаз — направление взгляда относительно "
             "камеры. Это не мера внимания или намерений. Точность падает при повороте головы, "
             "очках и бликах."),
    "blink": ("Частота морганий",
              "Моргания за последнюю минуту по степени раскрытия глаз (EAR) и длительность "
              "последнего закрытия глаз. На частоту влияют освещение, сухость глаз, усталость "
              "и экран — она не указывает на эмоции или правдивость."),
    "expression": ("Изменение выражения лица",
                   "Какое движение лица сейчас преобладает относительно нейтрального состояния, "
                   "зафиксированного при калибровке: улыбка, приоткрытый рот, поднятые или "
                   "сведённые брови, прищур. Это описание движения мышц, а не распознавание эмоций."),
    "activity": ("Активность мимики",
                 "Насколько быстро меняются мимические показатели (сглажено за ~1 с). Высокая "
                 "активность обычна при разговоре, смехе, артикуляции. «Быстрые изменения» — "
                 "резкие скачки мимики, отмеченные на временной шкале."),
    "redness": ("Динамика покраснения",
                "Изменение красного оттенка кожи щёк, носа и лба относительно калибровки. "
                "Сильно зависит от освещения, баланса белого и автоэкспозиции камеры; "
                "не указывает на смущение или другие чувства."),
    "speech": ("Распознавание речи",
               "Локальное распознавание речи (faster-whisper). Звук не отправляется в интернет. "
               "В режиме экрана нужен микрофон или «Стерео микшер», выбранный в настройках."),
    "deviation": ("Отклонение от базы",
                  "Сводный индекс: насколько мимика, моргание, взгляд и речь отличаются от "
                  "исходного уровня этого человека (личная база — первые 40 с с лицом в кадре; "
                  "до этого — типовые значения). Это не детектор лжи, стресса или эмоций, "
                  "а только мера изменения относительно базы."),
}


def _tooltip(key):
    title, text = HINTS[key]
    return (f"<div style='max-width:320px'><b>{title}</b><br>"
            f"<span style='color:{C.ON_DARK_2}'>{text}</span></div>")


def _pct(value):
    return f"{value * 100:.0f}"


# ---------------------------------------------------------------- строка

class ObservationRow(QWidget):
    def __init__(self, key, icon_name, title, parent=None):
        super().__init__(parent)
        self.key = key
        self.icon_name = icon_name
        self.title = title
        self.value = None
        self.unit = ""
        self.detail = "анализ не запущен"
        self.tone = "muted"           # normal | attention | muted
        self.viz = None               # (вид, данные)
        self.setFixedHeight(ROW_H)
        self.setToolTip(_tooltip(key))
        self.setMouseTracking(True)
        self._hover = False
        self._title_font = font(13, 600)
        self._detail_font = font(12)
        self._num_font = numeric_font(19)
        self._text_font = font(13, 600)
        self._unit_font = font(11)
        self._nodata_font = font(12)

    def set_state(self, value=None, unit="", detail="", tone="normal", viz=None):
        changed = (value, unit, detail, tone) != (self.value, self.unit, self.detail, self.tone)
        self.value, self.unit, self.detail, self.tone, self.viz = value, unit, detail, tone, viz
        if changed or viz is not None:
            self.update()

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    # ---------- рисование ----------

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        if self._hover:
            p.setPen(Qt.NoPen)
            p.setBrush(qc(C.SURFACE_2))
            p.drawRoundedRect(QRectF(-6, 2, w + 12, ROW_H - 4), 10, 10)

        attention = self.tone == "attention"
        muted = self.tone == "muted"
        box = QRectF(0, 8, 32, 32)
        p.setPen(Qt.NoPen)
        p.setBrush(qc(C.ACCENT_SOFT if attention else C.SURFACE_3))
        p.drawRoundedRect(box, 9, 9)
        icon_color = C.ACCENT if attention else (C.TEXT_3 if muted else C.TEXT)
        icons.draw_icon(p, self.icon_name, QRectF(7, 15, 18, 18), icon_color, 1.8)

        # строка 1: название и значение
        value_w = self._paint_value(p, w)
        p.setFont(self._title_font)
        p.setPen(qc(C.TEXT))
        p.drawText(QRectF(44, 5, w - 44 - value_w - 8, 20), Qt.AlignLeft | Qt.AlignVCenter, self.title)

        # строка 2: пояснение и мини-визуализация
        viz_w = 74 if self.viz else 0
        detail_w = w - 44 - viz_w - (10 if viz_w else 0)
        p.setFont(self._detail_font)
        p.setPen(qc(C.ACCENT_PRESSED if attention and self.detail else C.TEXT_3))
        fm = QFontMetrics(self._detail_font)
        p.drawText(QRectF(44, 25, detail_w, 18), Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(self.detail, Qt.ElideRight, int(detail_w)))
        if self.viz:
            self._paint_viz(p, QRectF(w - viz_w, 26, viz_w, 16))

    def _paint_value(self, p, w):
        if self.value is None:
            p.setFont(self._nodata_font)
            p.setPen(qc(C.TEXT_4))
            fm = QFontMetrics(self._nodata_font)
            tw = fm.horizontalAdvance(NO_DATA)
            p.drawText(QRectF(w - tw, 5, tw, 20), Qt.AlignRight | Qt.AlignVCenter, NO_DATA)
            return tw
        numeric = any(ch.isdigit() for ch in self.value) and len(self.value) <= 6
        value_font = self._num_font if numeric else self._text_font
        color = C.ACCENT if self.tone == "attention" else (C.TEXT_3 if self.tone == "muted" else C.TEXT)
        fm = QFontMetrics(value_font)
        vw = fm.horizontalAdvance(self.value)
        uw = 0
        if self.unit:
            uw = QFontMetrics(self._unit_font).horizontalAdvance(self.unit) + 4
        x = w - vw - uw
        p.setFont(value_font)
        p.setPen(qc(color))
        p.drawText(QRectF(x, 2, vw + 2, 24), Qt.AlignLeft | Qt.AlignVCenter, self.value)
        if self.unit:
            p.setFont(self._unit_font)
            p.setPen(qc(C.TEXT_3))
            p.drawText(QRectF(x + vw + 4, 6, uw, 20), Qt.AlignLeft | Qt.AlignVCenter, self.unit)
        return vw + uw

    def _paint_viz(self, p, r):
        kind, data = self.viz
        muted = self.tone == "muted"
        if kind == "spark":
            _paint_spark(p, r, data, muted)
        elif kind == "gaze":
            _paint_gaze(p, r, data, muted)
        elif kind == "bars":
            _paint_bars(p, r, data, muted)
        elif kind == "meter":
            _paint_meter(p, r, data, muted)
        elif kind == "level":
            _paint_level(p, r, data, muted)
        elif kind == "progress":
            _paint_progress(p, r, data, muted)


# ---------------------------------------------------------------- мини-визуализации

def _paint_spark(p, r, data, muted):
    values, baseline, band, lo, hi = data
    step = max(1, len(values) // int(r.width()))
    if step > 1:  # больше точек, чем пикселей, рисовать незачем
        values = values[::step] + ([values[-1]] if (len(values) - 1) % step else [])
    points = [(i, v) for i, v in enumerate(values) if v is not None]
    if hi - lo <= 1e-9:
        hi = lo + 1.0

    def y(v):
        v = max(lo, min(hi, v))
        return r.bottom() - (v - lo) / (hi - lo) * r.height()

    if band is not None:
        top, bottom = y(band[1]), y(band[0])
        p.setPen(Qt.NoPen)
        p.setBrush(qc(C.SURFACE_3))
        p.drawRect(QRectF(r.x(), top, r.width(), max(bottom - top, 1)))
    if baseline is not None:
        p.setPen(QPen(qc(C.TEXT_4), 1, Qt.DashLine))
        p.drawLine(QPointF(r.x(), y(baseline)), QPointF(r.right(), y(baseline)))
    if len(points) < 2:
        return
    n = max(len(values) - 1, 1)
    path = QPainterPath()
    started = False
    prev_i = None
    for i, v in points:
        pt = QPointF(r.x() + i / n * r.width(), y(v))
        if not started or (prev_i is not None and i - prev_i > 2):
            path.moveTo(pt)
            started = True
        else:
            path.lineTo(pt)
        prev_i = i
    p.setPen(QPen(qc(C.TEXT_4 if muted else C.TEXT_2), 1.4))
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)
    last_i, last_v = points[-1]
    out = band is not None and not (band[0] <= last_v <= band[1])
    p.setPen(Qt.NoPen)
    p.setBrush(qc(C.ACCENT if out and not muted else (C.TEXT_4 if muted else C.TEXT)))
    p.drawEllipse(QPointF(r.x() + last_i / n * r.width(), y(last_v)), 2.4, 2.4)


def _paint_gaze(p, r, data, muted):
    h, v, closed = data
    cx, cy = r.right() - 26, r.center().y()
    w, hh = 46, 18
    eye = QPainterPath()
    eye.moveTo(cx - w / 2, cy)
    eye.cubicTo(cx - w / 4, cy - hh * 0.75, cx + w / 4, cy - hh * 0.75, cx + w / 2, cy)
    eye.cubicTo(cx + w / 4, cy + hh * 0.75, cx - w / 4, cy + hh * 0.75, cx - w / 2, cy)
    p.setPen(QPen(qc(C.TEXT_4 if muted else C.TEXT_3), 1.2))
    p.setBrush(Qt.NoBrush)
    p.drawPath(eye)
    if closed or h is None or v is None:
        p.drawLine(QPointF(cx - w / 2 + 4, cy), QPointF(cx + w / 2 - 4, cy))
        return
    # h: 0 - левый край, 1 - правый; v: 0 - верх, 1 - низ
    x = cx + (h - 0.5) * (w - 14)
    y = cy + (v - 0.5) * (hh - 8)
    p.setPen(Qt.NoPen)
    p.setBrush(qc(C.TEXT_4 if muted else C.TEXT))
    p.drawEllipse(QPointF(x, y), 4.6, 4.6)
    p.setBrush(qc(C.SURFACE))
    p.drawEllipse(QPointF(x + 1.3, y - 1.3), 1.1, 1.1)


def _paint_bars(p, r, data, muted):
    values, dominant = data
    bw, gap = 7, 4
    total = len(values) * bw + (len(values) - 1) * gap
    x = r.right() - total
    for i, value in enumerate(values):
        full = QRectF(x, r.y(), bw, r.height())
        p.setPen(Qt.NoPen)
        p.setBrush(qc(C.SURFACE_3))
        p.drawRoundedRect(full, 2, 2)
        hgt = max(2.0, r.height() * max(0.0, min(1.0, value)))
        if i == dominant and not muted:
            color = C.TEXT
        else:
            color = C.TEXT_4 if muted else "#9C9CA6"
        p.setBrush(qc(color))
        p.drawRoundedRect(QRectF(x, r.bottom() - hgt, bw, hgt), 2, 2)
        x += bw + gap


def _paint_meter(p, r, value, muted):
    segments, sw, gap = 8, 6, 3
    total = segments * sw + (segments - 1) * gap
    x = r.right() - total
    lit = int(round(max(0.0, min(1.0, value)) * segments))
    for i in range(segments):
        p.setPen(Qt.NoPen)
        on = i < lit and not muted
        p.setBrush(qc(C.TEXT_2 if on else C.SURFACE_3))
        p.drawRoundedRect(QRectF(x, r.center().y() - 3, sw, 6), 1.5, 1.5)
        x += sw + gap


def _paint_level(p, r, value, muted):
    bars, bw, gap = 9, 3, 3
    total = bars * bw + (bars - 1) * gap
    x = r.right() - total
    level = max(0.0, min(1.0, value * 14.0))  # как индикатор громкости прежней версии
    shape = [0.35, 0.55, 0.8, 1.0, 0.85, 1.0, 0.7, 0.5, 0.3]
    for i in range(bars):
        hgt = r.height() * shape[i]
        active = (i + 0.5) / bars <= level and not muted
        p.setPen(Qt.NoPen)
        p.setBrush(qc(C.ACCENT if active and level > 0.92 else (C.TEXT_2 if active else C.SURFACE_3)))
        p.drawRoundedRect(QRectF(x, r.center().y() - hgt / 2, bw, hgt), 1.5, 1.5)
        x += bw + gap


def _paint_progress(p, r, value, muted=False):
    track = QRectF(r.x(), r.center().y() - 2, r.width(), 4)
    p.setPen(Qt.NoPen)
    p.setBrush(qc(C.SURFACE_3))
    p.drawRoundedRect(track, 2, 2)
    if value is None:
        return
    fill = QRectF(track)
    fill.setWidth(max(4.0, track.width() * max(0.0, min(1.0, value))))
    p.setBrush(qc(C.TEXT_4 if muted else C.ACCENT))
    p.drawRoundedRect(fill, 2, 2)


# ---------------------------------------------------------------- панель

class ObservationsPanel(Card):
    ROWS = [
        ("face", "face", "Лицо"),
        ("gaze", "gaze", "Направление взгляда"),
        ("blink", "blink", "Моргание"),
        ("expression", "expression", "Выражение лица"),
        ("activity", "activity", "Активность мимики"),
        ("redness", "redness", "Покраснение"),
        ("speech", "speech", "Речь"),
        ("deviation", "deviation", "Отклонение от базы"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent, margins=(18, 14, 18, 12))
        self.body.setSpacing(0)
        header = CardHeader("Наблюдения", "относительно исходного состояния")
        self.base_chip = Chip("", "user")
        self.base_chip.hide()
        header.add_right(self.base_chip)
        self.body.addWidget(header)
        self.body.addSpacing(6)

        self.rows = {}
        for i, (key, icon_name, title) in enumerate(self.ROWS):
            if i:
                line = QFrame()
                line.setFixedHeight(1)
                line.setStyleSheet(f"background: {C.BORDER};")
                self.body.addWidget(line)
            row = ObservationRow(key, icon_name, title)
            self.rows[key] = row
            self.body.addWidget(row)

        self.body.addSpacing(8)
        note = QWidget()
        note_lay = QHBoxLayout(note)
        note_lay.setContentsMargins(0, 0, 0, 0)
        note_lay.setSpacing(8)
        note_text = label("Сигналы — не выводы: показатели не определяют эмоции, мысли, "
                          "намерения или правдивость человека.", "Hint", wrap=True)
        note_lay.addWidget(note_text, 1)
        self.body.addWidget(note)

    # ---------- обновление ----------

    def reset(self, detail="анализ не запущен"):
        for key, row in self.rows.items():
            row.set_state(None, "", detail, "muted", None)
        self.base_chip.hide()

    def mute(self, detail):
        """Сессия завершена: последние значения остаются, но приглушены."""
        for row in self.rows.values():
            if row.key != "speech":
                row.set_state(row.value, row.unit, detail, "muted", row.viz)
        self.base_chip.hide()

    def set_speech_idle(self, value, detail, tone="muted"):
        self.rows["speech"].set_state(value, "", detail, tone, None)

    def update_from(self, tick, timeline, paused=False):
        s = tick.snapshot
        if s is None:
            return
        if paused:
            for row in self.rows.values():
                if row.key != "speech":
                    row.set_state(row.value, row.unit, "анализ на паузе", "muted", row.viz)
            self._update_speech(tick)
            return

        self._update_base_chip(s)
        self._update_face(s)
        if not s.face_detected:
            reason = "лицо не найдено" if s.face_model_ok else "анализ лица недоступен"
            for key in ("gaze", "blink", "expression", "activity", "redness", "deviation"):
                self.rows[key].set_state(None, "", reason, "muted", None)
        else:
            self._update_gaze(s)
            self._update_blink(s, timeline)
            self._update_expression(s)
            self._update_activity(s, timeline)
            self._update_redness(s, timeline)
            self._update_deviation(s, timeline)
        self._update_speech(tick)

    def _update_base_chip(self, s):
        if s.baseline_collecting:
            self.base_chip.set(f"Сбор базы {_pct(s.baseline_progress)}%", "accent", "user")
        elif s.baseline_personal:
            self.base_chip.set("База: личная", "neutral", "user")
        else:
            self.base_chip.set("База: типовая", "muted", "user")
        self.base_chip.show()

    def _update_face(self, s):
        row = self.rows["face"]
        if not s.face_model_ok:
            row.set_state("Недоступно", "", "модель лица не загружена", "attention", None)
            return
        if not s.face_detected:
            row.set_state("Не найдено", "", "лицо вне кадра или плохо видно", "attention",
                          ("meter", 0.0))
            return
        q = s.quality
        note = q.note if q else ""
        detail = s.face_position.lower()
        if note:
            detail += f" · кадр: {note}"
        row.set_state("В кадре", "", detail, "normal", ("meter", q.score if q else 0.0))

    def _update_gaze(self, s):
        row = self.rows["gaze"]
        if s.gaze_off_ratio is None:
            detail = "сбор данных…"
            tone = "normal"
        else:
            detail = f"вне центра {_pct(s.gaze_off_ratio)}% за 15 с"
            if s.baseline_gaze_off is not None:
                detail += f" · база {_pct(s.baseline_gaze_off)}%"
            high = (s.baseline_gaze_off + GAZE_MARGIN) if s.baseline_gaze_off is not None else GAZE_HIGH
            tone = "attention" if s.gaze_off_ratio > high else "normal"
        row.set_state(s.gaze, "", detail, tone, ("gaze", (s.gaze_h, s.gaze_v, s.eyes_closed)))

    def _update_blink(self, s, timeline):
        row = self.rows["blink"]
        base = s.baseline_blink_rate
        closure = f"закрытие {s.last_closure_ms:.0f} мс" if s.last_closure_ms else ""
        if s.blink_rate is None:
            detail = f"сбор данных {min(s.blink_visible_s, 15):.0f} из 15 с · морганий {s.blink_count}"
            row.set_state(None, "", detail, "normal", ("progress", min(s.blink_visible_s / 15.0, 1.0)))
            return
        parts = []
        if base is not None:
            parts.append(f"база {base:.0f}" + ("" if s.baseline_personal else " (типовая)"))
        if closure:
            parts.append(closure)
        tone = "attention" if base is not None and abs(s.blink_rate - base) > BLINK_DELTA else "normal"
        _, values = timeline.recent("blink_rate", 120)
        hi = max([v for v in values if v is not None] + [(base or 17.0) + BLINK_DELTA + 5, 30.0])
        band = (max(base - BLINK_DELTA, 0.0), base + BLINK_DELTA) if base is not None else None
        row.set_state(f"{s.blink_rate:.0f}", "/мин", " · ".join(parts), tone,
                      ("spark", (values, base, band, 0.0, hi)))

    def _update_expression(self, s):
        row = self.rows["expression"]
        m = s.mimic
        values = [getattr(m, attr) for attr, _ in EXPRESSION_ORDER] if m else [0.0] * 5
        dominant = next((i for i, (_, name) in enumerate(EXPRESSION_ORDER) if name == s.expression), -1)
        intensity = s.expression_intensity or 0.0
        if s.mimic_ready:
            detail = f"выраженность {_pct(intensity)}% · от нейтрального"
        else:
            detail = f"калибровка нейтрального {_pct(s.mimic_progress)}%"
        value = s.expression if s.expression else EXPR_NEUTRAL
        row.set_state(value, "", detail, "normal", ("bars", (values, dominant)))

    def _update_activity(self, s, timeline):
        row = self.rows["activity"]
        if s.activity is None:
            row.set_state(None, "", "сбор данных…", "normal", None)
            return
        base = s.baseline_activity
        high = (base + ACTIVITY_MARGIN) if base is not None else ACTIVITY_HIGH
        parts = [f"быстрых изменений: {s.micro_per_min} за мин"]
        if s.micro_event is not None:
            parts.append(s.micro_event.name)
        _, values = timeline.recent("activity", 120)
        tone = "attention" if s.activity > high else "normal"
        row.set_state(_pct(s.activity), "%", " · ".join(parts), tone,
                      ("spark", (values, base, (0.0, high), 0.0, 1.0)))

    def _update_redness(self, s, timeline):
        row = self.rows["redness"]
        if not s.redness_ready:
            row.set_state(None, "", f"калибровка цвета кожи {_pct(s.redness_progress)}%", "normal",
                          ("progress", s.redness_progress))
            return
        r = s.redness
        value = REDNESS_TEXT.get(r.status, r.status.capitalize() if r.status else None)
        if value is None:
            row.set_state(None, "", "нет данных о цвете кожи", "muted", None)
            return
        _, values = timeline.recent("redness", 120)
        hi = max([v for v in values if v is not None] + [0.35])
        tone = "attention" if r.status in (RED_NOTICEABLE, RED_STRONG) else "normal"
        row.set_state(value, "", f"индекс {r.overall:.2f} от исходного цвета", tone,
                      ("spark", (values, 0.0, (0.0, REDNESS_HIGH), 0.0, hi)))

    def _update_deviation(self, s, timeline):
        row = self.rows["deviation"]
        d = s.deviation
        if d is None:
            row.set_state(None, "", "сбор данных…", "normal", None)
            return
        if s.baseline_collecting:
            row.set_state(None, "", f"сбор личной базы {_pct(s.baseline_progress)}% · ведите себя "
                                    "как обычно", "normal", ("progress", s.baseline_progress))
            return
        if not s.quality_good:
            row.set_state(None, "", "кадр низкого качества — значение не обновляется", "muted", None)
            return
        _, values = timeline.recent("deviation", 120)
        tone = "attention" if d.status in (STATUS_RAISED, STATUS_HIGH) else "normal"
        detail = DEVIATION_TEXT.get(d.status, d.status)
        detail += " · база " + ("личная" if s.baseline_personal else "типовая")
        row.set_state(_pct(d.overall), "%", detail, tone,
                      ("spark", (values, 0.0, (0.0, DEVIATION_HIGH), 0.0, 1.0)))

    def _update_speech(self, tick):
        row = self.rows["speech"]
        state = tick.speech_state
        if state == "слушаю":
            detail = f"фраз: {tick.phrase_count}"
            if tick.speech_detail:
                detail += f" · {tick.speech_detail}"
            row.set_state("Слушаю", "", detail, "normal", ("level", tick.volume))
        elif state == "пауза":
            row.set_state("Пауза", "", f"фраз: {tick.phrase_count}", "muted", ("level", 0.0))
        elif state == "загрузка модели":
            row.set_state("Загрузка", "", "модель Whisper загружается…", "muted", ("progress", None))
        elif state == "выключено":
            row.set_state("Выключено", "", "можно включить в настройках", "muted", None)
        elif state == "нет микрофона":
            row.set_state("Нет микрофона", "", tick.speech_detail or "проверьте устройство ввода",
                          "attention", None)
        elif state == "нет звука в файле":
            row.set_state("Нет звука", "", "в видеофайле нет звуковой дорожки", "muted", None)
        else:
            row.set_state("Недоступно", "", tick.speech_detail or "модель речи не загружена",
                          "attention", None)
