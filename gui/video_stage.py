# -*- coding: utf-8 -*-
"""Видеосцена: кадр, ненавязчивая подсветка, субтитры, статусы и пустые состояния."""
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QVariantAnimation, Signal
from PySide6.QtGui import (QColor, QFontMetrics, QPainter, QPainterPath, QPen, QPolygonF,
                           QRadialGradient)
from PySide6.QtWidgets import (QAbstractButton, QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget)

from gui import icons
from gui.theme import C, display_font, font, numeric_font, qc
from gui.widgets import LogoMark, make_button

RADIUS = 18


def _fit(rect, w, h):
    if w <= 0 or h <= 0:
        return QRectF(rect)
    scale = min(rect.width() / w, rect.height() / h)
    tw, th = w * scale, h * scale
    return QRectF(rect.x() + (rect.width() - tw) / 2, rect.y() + (rect.height() - th) / 2, tw, th)


# ---------------------------------------------------------------- стартовый экран

class SourceTile(QAbstractButton):
    def __init__(self, key, icon_name, title, description, parent=None):
        super().__init__(parent)
        self.key = key
        self.icon_name = icon_name
        self.title = title
        self.description = description
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)  # пробел всегда старт/пауза, а не нажатие плитки
        self.setFixedSize(176, 112)
        self.setMouseTracking(True)
        self._hover = False

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    def sizeHint(self):
        return QSize(176, 112)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        checked = self.isChecked()
        p.setBrush(qc("#1C1D22" if not checked else "#202127"))
        border = C.ACCENT if checked else ("#3A3B43" if self._hover else "#2B2C33")
        p.setPen(QPen(qc(border), 1.5 if checked else 1.0))
        p.drawRoundedRect(r, 14, 14)
        icons.draw_icon(p, self.icon_name, QRectF(16, 16, 22, 22),
                        C.ON_DARK if checked else C.ON_DARK_2, 1.7)
        if checked:
            p.setPen(Qt.NoPen)
            p.setBrush(qc(C.ACCENT))
            p.drawEllipse(QPointF(r.right() - 16, 24), 4, 4)
        p.setPen(qc(C.ON_DARK))
        p.setFont(font(14, 600))
        p.drawText(QRectF(16, 48, r.width() - 28, 22), Qt.AlignLeft | Qt.AlignVCenter, self.title)
        p.setPen(qc(C.ON_DARK_3))
        p.setFont(font(12))
        p.drawText(QRectF(16, 70, r.width() - 26, 34), Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap,
                   self.description)


class EmptyState(QWidget):
    start_requested = Signal()
    source_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.setAlignment(Qt.AlignHCenter)

        logo = LogoMark(48)
        lay.addWidget(logo, 0, Qt.AlignHCenter)
        lay.addSpacing(18)

        title = QLabel("Готово к наблюдению")
        title.setFont(display_font(24))
        title.setStyleSheet(f"color: {C.ON_DARK}; background: transparent;")
        title.setAlignment(Qt.AlignHCenter)
        lay.addWidget(title)
        lay.addSpacing(8)

        subtitle = QLabel("Выберите источник видео и начните анализ мимики, взгляда и речи.\n"
                          "Обработка выполняется на этом компьютере — видео и звук никуда не отправляются.")
        subtitle.setFont(font(13))
        subtitle.setStyleSheet(f"color: {C.ON_DARK_2}; background: transparent;")
        subtitle.setAlignment(Qt.AlignHCenter)
        lay.addWidget(subtitle)
        lay.addSpacing(26)

        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        self.tiles = {
            "camera": SourceTile("camera", "camera", "Веб-камера", "Живое изображение с камеры"),
            "file": SourceTile("file", "file", "Видеофайл", "MP4, AVI, MOV, MKV…"),
            "screen": SourceTile("screen", "screen", "Экран", "Видеозвонок или запись на экране"),
        }
        for tile in self.tiles.values():
            tile.clicked.connect(lambda _=False, k=tile.key: self.source_selected.emit(k))
            tiles.addWidget(tile)
        lay.addLayout(tiles)
        lay.addSpacing(14)

        self.source_note = QLabel("")
        self.source_note.setFont(font(12))
        self.source_note.setStyleSheet(f"color: {C.ON_DARK_2}; background: transparent;")
        self.source_note.setAlignment(Qt.AlignHCenter)
        lay.addWidget(self.source_note)
        lay.addSpacing(18)

        self.start_button = make_button("Начать анализ", "play", "primary", "lg")
        self.start_button.setFixedWidth(220)
        self.start_button.clicked.connect(self.start_requested.emit)
        lay.addWidget(self.start_button, 0, Qt.AlignHCenter)
        lay.addSpacing(22)

        self.models_line = QLabel("")
        self.models_line.setFont(font(12))
        self.models_line.setStyleSheet(f"color: {C.ON_DARK_3}; background: transparent;")
        self.models_line.setAlignment(Qt.AlignHCenter)
        self.models_line.setTextFormat(Qt.RichText)
        lay.addWidget(self.models_line)

    def set_source(self, key, note=""):
        for k, tile in self.tiles.items():
            tile.setChecked(k == key)
        self.source_note.setText(note)

    def set_models_line(self, html):
        self.models_line.setText(html)


# ---------------------------------------------------------------- карточка сообщения

class StatBlock(QWidget):
    def __init__(self, title, value, unit="", parent=None):
        super().__init__(parent)
        self.title, self.value, self.unit = title, value, unit
        self.setFixedSize(132, 60)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(qc(C.ON_DARK_3))
        p.setFont(font(11, 600))
        p.drawText(QRectF(0, 0, self.width(), 18), Qt.AlignLeft | Qt.AlignVCenter, self.title.upper())
        value_font = numeric_font(24) if any(ch.isdigit() for ch in self.value) else font(17, 600)
        p.setFont(value_font)
        p.setPen(qc(C.ON_DARK))
        fm = QFontMetrics(value_font)
        p.drawText(QRectF(0, 22, self.width(), 34), Qt.AlignLeft | Qt.AlignVCenter, self.value)
        if self.unit:
            x = fm.horizontalAdvance(self.value) + 5
            p.setPen(qc(C.ON_DARK_3))
            p.setFont(font(12))
            p.drawText(QRectF(x, 26, self.width() - x, 30), Qt.AlignLeft | Qt.AlignVCenter, self.unit)


class MessageCard(QFrame):
    """Тёмная карточка поверх сцены: ошибка или итоги сессии."""

    action = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tone = "info"
        self.icon_name = "info"
        self.setFixedWidth(600)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(28, 26, 28, 24)
        self.lay.setSpacing(0)

        self.title = QLabel()
        self.title.setFont(display_font(20))
        self.title.setStyleSheet(f"color: {C.ON_DARK}; background: transparent;")
        self.title.setContentsMargins(44, 0, 0, 0)
        self.lay.addWidget(self.title)
        self.lay.addSpacing(8)

        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setFont(font(13))
        self.text.setStyleSheet(f"color: {C.ON_DARK_2}; background: transparent;")
        self.text.setTextFormat(Qt.RichText)
        self.text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lay.addWidget(self.text)

        self.stats_host = QWidget()
        self.stats_host.setStyleSheet("background: transparent;")
        self.stats = QHBoxLayout(self.stats_host)
        self.stats.setContentsMargins(0, 18, 0, 0)
        self.stats.setSpacing(8)
        self.lay.addWidget(self.stats_host)

        self.lay.addSpacing(22)
        self.buttons = QHBoxLayout()
        self.buttons.setSpacing(10)
        self.lay.addLayout(self.buttons)

    def show_message(self, tone, icon_name, title, text, stats=None, actions=()):
        self.tone, self.icon_name = tone, icon_name
        self.title.setText(title)
        self.text.setText(text)
        self.text.setVisible(bool(text))
        while self.stats.count():
            item = self.stats.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for stat in stats or []:
            self.stats.addWidget(StatBlock(*stat))
        self.stats.addStretch(1)
        self.stats_host.setVisible(bool(stats))
        while self.buttons.count():
            item = self.buttons.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for i, (key, caption, icon_name_btn) in enumerate(actions):
            variant = "primary" if i == 0 else "onDark"
            button = make_button(caption, icon_name_btn, variant)
            button.clicked.connect(lambda _=False, k=key: self.action.emit(k))
            self.buttons.addWidget(button)
        self.buttons.addStretch(1)
        self.adjustSize()
        self.show()
        self.raise_()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setPen(QPen(qc("#2E2F36"), 1))
        p.setBrush(qc("#1B1C21", 245))
        p.drawRoundedRect(r, 18, 18)
        color = C.ACCENT if self.tone == "error" else C.ON_DARK
        p.setPen(Qt.NoPen)
        p.setBrush(qc(C.ACCENT, 40) if self.tone == "error" else qc("#2A2B31"))
        p.drawRoundedRect(QRectF(26, 24, 32, 32), 9, 9)
        icons.draw_icon(p, self.icon_name, QRectF(33, 31, 18, 18), color, 1.9)


# ---------------------------------------------------------------- сцена

class VideoStage(QWidget):
    start_requested = Signal()
    source_selected = Signal(str)
    action_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.image = None
        self.overlay = None
        self.mode = "empty"          # empty | connecting | live | paused | finished | error
        self.show_box = True
        self.show_eyes = True
        self.show_mesh = False
        self.show_captions = True
        self.caption = ""
        self.caption_live = False
        self.badge = ""              # LIVE / ПАУЗА / ФАЙЛ ...
        self.source_text = ""
        self.time_text = ""
        self.fps_text = ""
        self.warnings = []
        self.banner = None           # (заголовок, подзаголовок, прогресс 0..1 или None)
        self.progress = None
        self.connecting_text = ""
        self._spin = 0.0
        self._mask_key = None
        self._mask = None

        self.spinner = QVariantAnimation(self, startValue=0.0, endValue=360.0, duration=1100)
        self.spinner.setLoopCount(-1)
        self.spinner.valueChanged.connect(self._on_spin)

        self.empty = EmptyState(self)
        self.empty.start_requested.connect(self.start_requested.emit)
        self.empty.source_selected.connect(self.source_selected.emit)
        self.card = MessageCard(self)
        self.card.action.connect(self.action_requested.emit)
        self.card.hide()

    # ---------- API ----------

    def set_mode(self, mode, text=""):
        self.mode = mode
        self.connecting_text = text
        self.empty.setVisible(mode == "empty")
        if mode in ("empty", "connecting", "live", "paused"):
            self.card.hide()
        if mode == "connecting":
            if self.spinner.state() != QVariantAnimation.Running:
                self.spinner.start()
        else:
            self.spinner.stop()
        if mode == "empty":
            self.image = None
            self.overlay = None
            self.banner = None
            self.warnings = []
            self.caption = ""
            self.progress = None
        self._layout_children()
        self.update()

    def set_frame(self, image, overlay):
        self.image = image
        self.overlay = overlay
        self.update()

    def set_status(self, badge, source_text, time_text, fps_text):
        self.badge, self.source_text, self.time_text, self.fps_text = badge, source_text, time_text, fps_text

    def set_caption(self, text, live=False):
        self.caption, self.caption_live = text or "", live

    def set_warnings(self, warnings):
        self.warnings = list(warnings)

    def set_banner(self, banner):
        self.banner = banner

    def set_progress(self, value):
        self.progress = value

    def show_card(self, tone, icon_name, title, text, stats=None, actions=()):
        self.card.show_message(tone, icon_name, title, text, stats, actions)
        self._layout_children()

    # ---------- геометрия ----------

    def _on_spin(self, value):
        self._spin = float(value)
        if self.mode == "connecting":
            self.update()

    def resizeEvent(self, event):
        self._layout_children()
        super().resizeEvent(event)

    def _layout_children(self):
        hint = self.empty.sizeHint()
        self.empty.resize(hint)
        self.empty.move(int((self.width() - hint.width()) / 2), int((self.height() - hint.height()) / 2))
        if self.card.isVisible():
            self.card.adjustSize()
            w = min(600, self.width() - 48)
            self.card.setFixedWidth(max(w, 360))
            self.card.adjustSize()
            self.card.move(int((self.width() - self.card.width()) / 2),
                           int((self.height() - self.card.height()) / 2))

    def video_rect(self):
        if self.image is None:
            return QRectF(self.rect())
        return _fit(QRectF(self.rect()), self.image.width(), self.image.height())

    # ---------- рисование ----------

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        rect = QRectF(self.rect())
        p.setPen(Qt.NoPen)

        if self.mode == "empty" or self.image is None:
            gradient = QRadialGradient(rect.center(), max(rect.width(), rect.height()) * 0.65)
            gradient.setColorAt(0.0, qc("#1D1E23"))
            gradient.setColorAt(1.0, qc(C.STAGE))
            p.setBrush(gradient)
            p.drawRect(rect)
        else:
            p.setBrush(qc(C.STAGE))
            p.drawRect(rect)
            target = self.video_rect()
            p.drawImage(target, self.image)
            if self.mode in ("paused", "finished", "error"):
                p.fillRect(target, QColor(12, 12, 15, 110 if self.mode == "paused" else 170))
            elif self.mode == "live" and self.overlay is not None:
                self._paint_overlay(p, target)
            if self.show_captions and self.caption and self.mode in ("live", "paused"):
                self._paint_caption(p, rect)

        if self.mode == "connecting":
            self._paint_connecting(p, rect)
        if self.mode in ("live", "paused", "connecting"):
            self._paint_chips(p, rect)
        if self.mode == "live" and self.banner:
            self._paint_banner(p, rect)
        if self.mode in ("live", "paused") and self.progress is not None:
            self._paint_progress(p, rect)
        self._paint_corners(p, rect)

    def _paint_overlay(self, p, target):
        ov = self.overlay
        fw, fh = ov.frame_size
        if not fw or not fh:
            return
        sx, sy = target.width() / fw, target.height() / fh

        def pt(x, y):
            return QPointF(target.x() + x * sx, target.y() + y * sy)

        if self.show_mesh and ov.landmarks:
            p.setPen(QPen(QColor(255, 255, 255, 70), 1.6, Qt.SolidLine, Qt.RoundCap))
            p.drawPoints(QPolygonF([pt(x, y) for x, y in ov.landmarks]))

        if self.show_box and ov.bbox:
            x1, y1, x2, y2 = ov.bbox
            a, b = pt(x1, y1), pt(x2, y2)
            length = min(b.x() - a.x(), b.y() - a.y()) * 0.16
            p.setPen(QPen(QColor(255, 255, 255, 215), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)
            for cx, cy, dx, dy in ((a.x(), a.y(), 1, 1), (b.x(), a.y(), -1, 1),
                                   (a.x(), b.y(), 1, -1), (b.x(), b.y(), -1, -1)):
                path = QPainterPath(QPointF(cx + dx * length, cy))
                path.lineTo(cx, cy)
                path.lineTo(cx, cy + dy * length)
                p.drawPath(path)

        if self.show_eyes and ov.left_eye and ov.right_eye:
            eye_color = qc(C.ACCENT, 230) if ov.eyes_closed else QColor(255, 255, 255, 150)
            p.setPen(QPen(eye_color, 1.2))
            p.setBrush(Qt.NoBrush)
            for contour in (ov.left_eye, ov.right_eye):
                p.drawPolygon(QPolygonF([pt(x, y) for x, y in contour]))
            for iris in (ov.left_iris, ov.right_iris):
                if iris is None or ov.eyes_closed:
                    continue
                cx, cy, radius = iris
                c = pt(cx, cy)
                rr = max(radius * sx, 2.5)
                p.setPen(QPen(qc(C.ACCENT), 1.6))
                p.drawEllipse(c, rr, rr)
                p.setPen(Qt.NoPen)
                p.setBrush(qc(C.ACCENT))
                p.drawEllipse(c, 1.6, 1.6)
                p.setBrush(Qt.NoBrush)

    def _chip(self, p, x, y, text, dot=None, icon_name=None, align_right=False, accent=False):
        f = font(12, 600)
        fm = QFontMetrics(f)
        w = fm.horizontalAdvance(text) + 24 + (14 if dot else 0) + (20 if icon_name else 0)
        if align_right:
            x = x - w
        r = QRectF(x, y, w, 28)
        p.setPen(QPen(QColor(255, 255, 255, 22), 1))
        p.setBrush(QColor(16, 17, 20, 190))
        p.drawRoundedRect(r, 9, 9)
        tx = x + 12
        if dot:
            p.setPen(Qt.NoPen)
            p.setBrush(qc(dot))
            p.drawEllipse(QPointF(tx + 3.5, r.center().y()), 3.5, 3.5)
            tx += 14
        if icon_name:
            icons.draw_icon(p, icon_name, QRectF(tx, r.center().y() - 8, 16, 16),
                            C.ACCENT if accent else C.ON_DARK_2, 2.0)
            tx += 20
        p.setPen(qc(C.ON_DARK))
        p.setFont(f)
        p.drawText(QRectF(tx, y, w, 28), Qt.AlignVCenter | Qt.AlignLeft, text)
        return w

    def _paint_chips(self, p, rect):
        x, y = 16, 16
        if self.badge:
            dot = C.ACCENT if self.mode == "live" else C.ON_DARK_2
            x += self._chip(p, x, y, self.badge, dot=dot) + 8
        if self.source_text:
            x += self._chip(p, x, y, self.source_text) + 8
        if self.time_text:
            self._chip(p, x, y, self.time_text)
        if self.fps_text:
            self._chip(p, rect.width() - 16, y, self.fps_text, align_right=True)
        if self.mode == "live" and self.warnings:
            wy = 52
            for warning in self.warnings[:3]:
                f = font(12, 600)
                w = QFontMetrics(f).horizontalAdvance(warning) + 44
                self._chip(p, (rect.width() - w) / 2, wy, warning, icon_name="alert", accent=True)
                wy += 34

    def _paint_caption(self, p, rect):
        f = font(17, 600)
        fm = QFontMetrics(f)
        max_w = min(rect.width() * 0.74, 860)
        flags = Qt.AlignHCenter | Qt.TextWordWrap
        bound = fm.boundingRect(0, 0, int(max_w), 200, int(flags), self.caption)
        lines_h = min(bound.height(), fm.lineSpacing() * 2 + 2)
        w = min(bound.width(), max_w) + 32
        h = lines_h + 18
        bottom = rect.bottom() - (26 if self.progress is not None else 18)
        r = QRectF((rect.width() - w) / 2, bottom - h, w, h)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(10, 10, 12, 175))
        p.drawRoundedRect(r, 10, 10)
        p.setFont(f)
        p.setPen(QColor(255, 255, 255, 190 if self.caption_live else 245))
        p.drawText(r.adjusted(16, 9, -16, -9), int(flags) | Qt.AlignVCenter, self.caption)

    def _paint_banner(self, p, rect):
        title, subtitle, progress = self.banner
        w = min(520, rect.width() - 64)
        h = 70 if progress is not None else 58
        top = 56 + (34 * min(len(self.warnings), 3) if self.warnings else 0)
        r = QRectF((rect.width() - w) / 2, top, w, h)
        p.setPen(QPen(QColor(255, 255, 255, 24), 1))
        p.setBrush(QColor(20, 21, 25, 228))
        p.drawRoundedRect(r, 12, 12)
        p.setPen(qc(C.ON_DARK))
        p.setFont(font(13, 600))
        p.drawText(QRectF(r.x() + 16, r.y() + 10, w - 32, 20), Qt.AlignLeft | Qt.AlignVCenter, title)
        p.setPen(qc(C.ON_DARK_2))
        p.setFont(font(12))
        p.drawText(QRectF(r.x() + 16, r.y() + 30, w - 32, 18), Qt.AlignLeft | Qt.AlignVCenter, subtitle)
        if progress is not None:
            track = QRectF(r.x() + 16, r.bottom() - 16, w - 32, 4)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 255, 255, 30))
            p.drawRoundedRect(track, 2, 2)
            fill = QRectF(track)
            fill.setWidth(max(track.width() * max(0.0, min(1.0, progress)), 4))
            p.setBrush(qc(C.ACCENT))
            p.drawRoundedRect(fill, 2, 2)

    def _paint_progress(self, p, rect):
        track = QRectF(rect.x(), rect.bottom() - 4, rect.width(), 4)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, 28))
        p.drawRect(track)
        fill = QRectF(track)
        fill.setWidth(track.width() * max(0.0, min(1.0, self.progress)))
        p.setBrush(qc(C.ACCENT))
        p.drawRect(fill)

    def _paint_connecting(self, p, rect):
        center = rect.center()
        p.setPen(QPen(QColor(255, 255, 255, 30), 3))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(center, 18, 18)
        pen = QPen(qc(C.ACCENT), 3, Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(QRectF(center.x() - 18, center.y() - 18, 36, 36), int(-self._spin * 16), 90 * 16)
        p.setPen(qc(C.ON_DARK_2))
        p.setFont(font(13))
        p.drawText(QRectF(0, center.y() + 30, rect.width(), 24), Qt.AlignHCenter | Qt.AlignVCenter,
                   self.connecting_text or "Подготовка…")

    def _paint_corners(self, p, rect):
        key = (rect.width(), rect.height())
        if key != self._mask_key:
            outer = QPainterPath()
            outer.addRect(rect)
            inner = QPainterPath()
            inner.addRoundedRect(rect, RADIUS, RADIUS)
            self._mask = outer.subtracted(inner)
            self._mask_key = key
        p.fillPath(self._mask, qc(C.BG))
