# -*- coding: utf-8 -*-
"""Базовые виджеты интерфейса (рисуются вручную, без сторонних библиотек)."""
from PySide6.QtCore import (Property, QEasingCurve, QEvent, QPointF, QPropertyAnimation, QRect,
                            QRectF, QSize, Qt, QTimer, QVariantAnimation, Signal)
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (QAbstractButton, QFrame, QGraphicsOpacityEffect, QHBoxLayout,
                               QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from gui import icons
from gui.theme import C, font, qc, repolish


def make_button(text="", icon_name=None, variant=None, size=None, tooltip=None,
                icon_color=None, icon_size=18, parent=None):
    button = QPushButton(text, parent)
    if variant:
        button.setProperty("variant", variant)
    if size:
        button.setProperty("size", size)
    if not text:
        button.setProperty("iconOnly", "true")
    if icon_name:
        color = icon_color or {"primary": "#FFFFFF", "dark": "#FFFFFF",
                               "ghost": C.TEXT_2, "danger": C.ACCENT,
                               "onDark": C.ON_DARK}.get(variant, C.TEXT)
        button.setIcon(icons.icon(icon_name, color, icon_size))
        button.setIconSize(QSize(icon_size, icon_size))
    if tooltip:
        button.setToolTip(tooltip)
    button.setCursor(Qt.PointingHandCursor)
    button.setFocusPolicy(Qt.NoFocus)
    return button


def set_button_variant(button, variant, icon_name=None, text=None, icon_size=18):
    button.setProperty("variant", variant)
    if text is not None:
        button.setText(text)
    if icon_name:
        color = {"primary": "#FFFFFF", "dark": "#FFFFFF", "ghost": C.TEXT_2,
                 "danger": C.ACCENT}.get(variant, C.TEXT)
        button.setIcon(icons.icon(icon_name, color, icon_size))
    repolish(button)


def vsep(height=24):
    line = QFrame()
    line.setObjectName("VSep")
    line.setFixedHeight(height)
    return line


def label(text="", object_name=None, parent=None, wrap=False):
    widget = QLabel(text, parent)
    if object_name:
        widget.setObjectName(object_name)
    widget.setWordWrap(wrap)
    return widget


# ---------------------------------------------------------------- карточка

class Card(QFrame):
    """Белая карточка: тонкая рамка и мягкая тень, нарисованные вручную."""

    def __init__(self, parent=None, radius=16, margins=(18, 16, 18, 16)):
        super().__init__(parent)
        self.radius = radius
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(*margins)
        self.body.setSpacing(10)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1.5, 1.0, -1.5, -3.0)
        p.setPen(Qt.NoPen)
        for dy, alpha in ((2.2, 9), (1.1, 12)):
            p.setBrush(QColor(18, 18, 28, alpha))
            p.drawRoundedRect(r.translated(0, dy), self.radius, self.radius)
        p.setBrush(qc(C.SURFACE))
        p.setPen(QPen(qc(C.BORDER), 1))
        p.drawRoundedRect(r, self.radius, self.radius)


class CardHeader(QWidget):
    def __init__(self, title, subtitle=None, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        texts = QVBoxLayout()
        texts.setSpacing(1)
        self.title = label(title, "CardTitle")
        texts.addWidget(self.title)
        self.subtitle = None
        if subtitle:
            # перенос строк: подзаголовок не должен раздвигать карточку шире колонки
            self.subtitle = label(subtitle, "CardSubtitle", wrap=True)
            texts.addWidget(self.subtitle)
        lay.addLayout(texts, 1)
        self.right = QHBoxLayout()
        self.right.setSpacing(8)
        lay.addLayout(self.right)

    def add_right(self, widget):
        self.right.addWidget(widget, 0, Qt.AlignVCenter)


# ---------------------------------------------------------------- логотип

class LogoMark(QWidget):
    def __init__(self, size=28, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, event):
        p = QPainter(self)
        icons.paint_logo(p, QRectF(self.rect()))


# ---------------------------------------------------------------- статус

class StatusPill(QWidget):
    """Индикатор состояния: точка + текст. В режиме live точка мягко пульсирует."""

    STYLES = {
        "idle": (C.SURFACE_3, C.TEXT_4, C.TEXT_2),
        "busy": (C.SURFACE_3, C.TEXT_3, C.TEXT_2),
        "live": (C.ACCENT_SOFT, C.ACCENT, C.ACCENT_PRESSED),
        "paused": (C.SURFACE_3, C.TEXT, C.TEXT),
        "done": (C.SURFACE_3, C.TEXT_2, C.TEXT_2),
        "error": (C.ACCENT_SOFT, C.ACCENT, C.ACCENT_PRESSED),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = "idle"
        self.text = "Готово к работе"
        self._phase = 0.0
        self._font = font(12, 600)
        self._anim = QVariantAnimation(self, startValue=0.0, endValue=1.0, duration=1600)
        self._anim.setLoopCount(-1)
        self._anim.valueChanged.connect(self._on_phase)
        self.setFixedHeight(28)
        self._update_width()

    def _on_phase(self, value):
        self._phase = float(value)
        self.update()

    def _update_width(self):
        width = QFontMetrics(self._font).horizontalAdvance(self.text) + 36
        self.setFixedWidth(width)

    def set_state(self, state, text):
        self.state, self.text = state, text
        if state in ("live", "busy"):
            if self._anim.state() != QVariantAnimation.Running:
                self._anim.start()
        else:
            self._anim.stop()
            self._phase = 0.0
        self._update_width()
        self.update()

    def paintEvent(self, event):
        bg, dot, fg = self.STYLES.get(self.state, self.STYLES["idle"])
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(Qt.NoPen)
        p.setBrush(qc(bg))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        center = QPointF(15, r.center().y())
        if self.state in ("live", "busy"):
            halo = 3.5 + 5.0 * self._phase
            p.setBrush(qc(dot, int(110 * (1.0 - self._phase))))
            p.drawEllipse(center, halo, halo)
        p.setBrush(qc(dot))
        p.drawEllipse(center, 3.6, 3.6)
        p.setPen(qc(fg))
        p.setFont(self._font)
        p.drawText(QRectF(26, 0, r.width() - 30, r.height()), Qt.AlignVCenter | Qt.AlignLeft, self.text)


class Chip(QWidget):
    """Небольшая плашка с текстом (база, статус речи и т.п.)."""

    def __init__(self, text="", icon_name=None, parent=None):
        super().__init__(parent)
        self.text = text
        self.icon_name = icon_name
        self.tone = "neutral"   # neutral | accent | muted
        self._font = font(11, 600)
        self.setFixedHeight(24)
        self._fit()

    def _fit(self):
        width = QFontMetrics(self._font).horizontalAdvance(self.text) + 20
        if self.icon_name:
            width += 18
        self.setFixedWidth(width)

    def set(self, text, tone="neutral", icon_name=None):
        self.text, self.tone = text, tone
        if icon_name is not None:
            self.icon_name = icon_name
        self._fit()
        self.update()

    def paintEvent(self, event):
        bg, fg = {"accent": (C.ACCENT_SOFT, C.ACCENT_PRESSED),
                  "muted": (C.SURFACE_2, C.TEXT_3)}.get(self.tone, (C.SURFACE_3, C.TEXT_2))
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(Qt.NoPen)
        p.setBrush(qc(bg))
        p.drawRoundedRect(r, 8, 8)
        x = 10
        if self.icon_name:
            icons.draw_icon(p, self.icon_name, QRectF(8, (r.height() - 14) / 2 + 0.5, 14, 14), fg, 2.0)
            x = 26
        p.setPen(qc(fg))
        p.setFont(self._font)
        p.drawText(QRectF(x, 0, r.width() - x, r.height()), Qt.AlignVCenter | Qt.AlignLeft, self.text)


# ---------------------------------------------------------------- сегменты

class SegmentedControl(QWidget):
    """Переключатель из нескольких сегментов с плавно скользящей подложкой."""

    changed = Signal(str)

    def __init__(self, items, parent=None, height=34, text_px=12, min_segment=0):
        super().__init__(parent)
        self.items = list(items)        # [(key, текст, иконка или None)]
        self.index = 0
        self.hover = -1
        self._font = font(text_px, 600)
        self._min_segment = min_segment
        self._x = None
        self._anim = QVariantAnimation(self, duration=200)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)
        self.setFixedHeight(height)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)
        self._widths = self._measure()
        self.setFixedWidth(int(sum(self._widths) + 8))

    def _measure(self):
        fm = QFontMetrics(self._font)
        widths = []
        for _, text, icon_name in self.items:
            w = fm.horizontalAdvance(text) + 26
            if icon_name:
                w += 22
            widths.append(max(w, self._min_segment))
        return widths

    def _segment_rect(self, i):
        x = 4 + sum(self._widths[:i])
        return QRectF(x, 4, self._widths[i], self.height() - 8)

    def key(self):
        return self.items[self.index][0]

    def set_key(self, key, animate=True, emit=False):
        for i, (k, _, _) in enumerate(self.items):
            if k == key:
                self._select(i, animate, emit)
                return

    def _select(self, i, animate=True, emit=True):
        if i == self.index and self._x is not None:
            return
        start = self._segment_rect(self.index).x() if self._x is None else self._x
        self.index = i
        end = self._segment_rect(i).x()
        if animate and self.isVisible():
            self._anim.stop()
            self._anim.setStartValue(float(start))
            self._anim.setEndValue(float(end))
            self._anim.start()
        else:
            self._x = end
        self.update()
        if emit:
            self.changed.emit(self.items[i][0])

    def _on_anim(self, value):
        self._x = float(value)
        self.update()

    def _hit(self, pos):
        for i in range(len(self.items)):
            if self._segment_rect(i).contains(pos):
                return i
        return -1

    def mouseMoveEvent(self, event):
        hit = self._hit(event.position())
        if hit != self.hover:
            self.hover = hit
            self.update()

    def leaveEvent(self, event):
        self.hover = -1
        self.update()

    def mousePressEvent(self, event):
        if not self.isEnabled() or event.button() != Qt.LeftButton:
            return
        hit = self._hit(event.position())
        if hit >= 0:
            if hit == self.index:
                self.changed.emit(self.items[hit][0])  # повторный выбор (например, другой файл)
            else:
                self._select(hit)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        enabled = self.isEnabled()
        outer = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(Qt.NoPen)
        p.setBrush(qc(C.SURFACE_3))
        p.drawRoundedRect(outer, 10, 10)

        sel = self._segment_rect(self.index)
        if self._x is None:
            self._x = sel.x()
        sel.moveLeft(self._x)
        p.setBrush(QColor(18, 18, 28, 14))
        p.drawRoundedRect(sel.translated(0, 1), 8, 8)
        p.setBrush(qc(C.SURFACE))
        p.setPen(QPen(qc(C.BORDER), 1))
        p.drawRoundedRect(sel, 8, 8)

        p.setFont(self._font)
        for i, (_, text, icon_name) in enumerate(self.items):
            r = self._segment_rect(i)
            active = i == self.index
            if not enabled:
                color = C.TEXT_4
            elif active:
                color = C.TEXT
            elif i == self.hover:
                color = C.TEXT_2
            else:
                color = C.TEXT_3
            fm = QFontMetrics(self._font)
            content = fm.horizontalAdvance(text) + (22 if icon_name else 0)
            x = r.x() + (r.width() - content) / 2
            if icon_name:
                icons.draw_icon(p, icon_name, QRectF(x, r.center().y() - 8, 16, 16), color, 1.9)
                x += 22
            p.setPen(qc(color))
            p.drawText(QRectF(x, r.y(), r.width(), r.height()), Qt.AlignVCenter | Qt.AlignLeft, text)


# ---------------------------------------------------------------- приватность

class PrivacyIndicators(QWidget):
    """Камера, микрофон, анализ, запись на диск: тёмная иконка с красной точкой - активно."""

    ITEMS = [("camera", "camera", "Камера"), ("mic", "mic", "Микрофон"),
             ("analysis", "eye", "Анализ"), ("saving", "save", "Сохранение истории")]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = {key: False for key, _, _ in self.ITEMS}
        self.notes = {key: "" for key, _, _ in self.ITEMS}
        self.setFixedSize(len(self.ITEMS) * 32, 32)
        self.setMouseTracking(True)

    def set_state(self, key, active, note=""):
        if self.state.get(key) != active or self.notes.get(key) != note:
            self.state[key] = active
            self.notes[key] = note
            self.update()

    def event(self, ev):
        if ev.type() == QEvent.Type.ToolTip:
            i = int(ev.pos().x() // 32)
            if 0 <= i < len(self.ITEMS):
                key, _, title = self.ITEMS[i]
                status = "активно" if self.state[key] else "не активно"
                note = self.notes.get(key)
                self.setToolTip(f"<b>{title}</b>: {status}" + (f"<br>{note}" if note else ""))
        return super().event(ev)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        for i, (key, icon_name, _) in enumerate(self.ITEMS):
            active = self.state[key]
            x = i * 32
            if active:
                p.setPen(Qt.NoPen)
                p.setBrush(qc(C.SURFACE_3))
                p.drawRoundedRect(QRectF(x + 2, 2, 28, 28), 8, 8)
            icons.draw_icon(p, icon_name, QRectF(x + 7, 7, 18, 18),
                            C.TEXT if active else C.TEXT_4, 1.8)
            if active:
                p.setPen(QPen(qc(C.SURFACE), 1.6))
                p.setBrush(qc(C.ACCENT))
                p.drawEllipse(QPointF(x + 25, 7.5), 3.6, 3.6)


# ---------------------------------------------------------------- переключатель

class ToggleSwitch(QAbstractButton):
    def __init__(self, checked=False, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(42, 24)
        self._pos = 1.0 if checked else 0.0
        self._anim = QPropertyAnimation(self, b"knob", self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self.toggled.connect(self._animate)

    def _animate(self, checked):
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def get_knob(self):
        return self._pos

    def set_knob(self, value):
        self._pos = float(value)
        self.update()

    knob = Property(float, get_knob, set_knob)

    def sizeHint(self):
        return QSize(42, 24)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        off, on = qc(C.BORDER_2), qc(C.ACCENT if self.isEnabled() else C.ACCENT_DISABLED)
        t = self._pos
        track = QColor(int(off.red() + (on.red() - off.red()) * t),
                       int(off.green() + (on.green() - off.green()) * t),
                       int(off.blue() + (on.blue() - off.blue()) * t))
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        d = r.height() - 4
        x = r.x() + 2 + (r.width() - d - 4) * t
        p.setBrush(QColor(18, 18, 28, 30))
        p.drawEllipse(QRectF(x, r.y() + 2.6, d, d))
        p.setBrush(qc(C.SURFACE))
        p.drawEllipse(QRectF(x, r.y() + 2, d, d))


# ---------------------------------------------------------------- уведомления

class Toast(QWidget):
    """Короткое уведомление в углу окна: появляется и исчезает само."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.text = ""
        self.tone = "info"
        self._font = font(13, 500)
        self._effect = QGraphicsOpacityEffect(self)
        self._effect.setOpacity(0.0)
        self.setGraphicsEffect(self._effect)
        self._fade = QPropertyAnimation(self._effect, b"opacity", self)
        self._fade.setDuration(220)
        self._fade.finished.connect(self._on_fade_finished)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._hide)
        self._queue = []
        self.hide()

    def show_message(self, text, tone="info", ms=3200):
        if self.isVisible() and self._timer.isActive():
            if len(self._queue) < 4:
                self._queue.append((text, tone, ms))
            return
        self.text, self.tone = text, tone
        fm = QFontMetrics(self._font)
        text_w = min(fm.horizontalAdvance(text), 400)
        bound = fm.boundingRect(QRect(0, 0, 400, 400), Qt.TextWordWrap, text)
        self.resize(text_w + 60, max(44, bound.height() + 24))
        self.reposition()
        self.show()
        self.raise_()
        self._fade.stop()
        self._fade.setStartValue(self._effect.opacity())
        self._fade.setEndValue(1.0)
        self._fade.start()
        self._timer.start(ms)

    def reposition(self):
        parent = self.parentWidget()
        if parent is None:
            return
        margin_bottom = getattr(parent, "toast_bottom_margin", 96)
        self.move(parent.width() - self.width() - 24, parent.height() - self.height() - margin_bottom)

    def _hide(self):
        self._fade.stop()
        self._fade.setStartValue(self._effect.opacity())
        self._fade.setEndValue(0.0)
        self._fade.start()

    def _on_fade_finished(self):
        if self._effect.opacity() > 0.01:
            return
        self.hide()
        if self._queue:
            text, tone, ms = self._queue.pop(0)
            self.show_message(text, tone, ms)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -3)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 40))
        p.drawRoundedRect(r.translated(0, 2), 12, 12)
        p.setBrush(qc(C.GRAPHITE))
        p.drawRoundedRect(r, 12, 12)
        color = C.ACCENT if self.tone in ("error", "warning") else C.ON_DARK_2
        icon_name = {"error": "alert", "warning": "alert", "success": "check"}.get(self.tone, "info")
        icons.draw_icon(p, icon_name, QRectF(14, r.center().y() - 8, 16, 16), color, 2.0)
        p.setPen(qc(C.ON_DARK))
        p.setFont(self._font)
        p.drawText(QRectF(40, r.y(), r.width() - 52, r.height()),
                   Qt.AlignVCenter | Qt.AlignLeft | Qt.TextWordWrap, self.text)


class FieldRow(QWidget):
    """Строка настроек: название и пояснение слева, элемент управления справа."""

    def __init__(self, title, hint=None, control=None, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 8, 0, 8)
        lay.setSpacing(16)
        texts = QVBoxLayout()
        texts.setSpacing(2)
        texts.addWidget(label(title, "FieldTitle"))
        if hint:
            texts.addWidget(label(hint, "FieldHint", wrap=True))
        lay.addLayout(texts, 1)
        if control is not None:
            lay.addWidget(control, 0, Qt.AlignVCenter | Qt.AlignRight)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
