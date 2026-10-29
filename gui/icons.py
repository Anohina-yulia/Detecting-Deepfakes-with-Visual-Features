# -*- coding: utf-8 -*-
"""Минималистичный набор линейных иконок (SVG 24x24, рисуются на лету) и логотип."""
from PySide6.QtCore import QByteArray, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer

from gui.theme import C

_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="{c}" stroke-width="{w}" stroke-linecap="round" stroke-linejoin="round">{body}</svg>')

_ICONS = {
    "play": '<path d="M8 5.9v12.2a.9.9 0 0 0 1.37.77l9.4-6.1a.9.9 0 0 0 0-1.52l-9.4-6.1A.9.9 0 0 0 8 5.9z" '
            'fill="{c}" stroke="none"/>',
    "pause": '<rect x="6.5" y="5" width="3.7" height="14" rx="1.1" fill="{c}" stroke="none"/>'
             '<rect x="13.8" y="5" width="3.7" height="14" rx="1.1" fill="{c}" stroke="none"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="2.2" fill="{c}" stroke="none"/>',
    "recalibrate": '<path d="M19.5 10.5A7.7 7.7 0 0 0 6 7.2L4.5 9"/><path d="M4.5 4.8V9h4.2"/>'
                   '<path d="M4.5 13.5A7.7 7.7 0 0 0 18 16.8l1.5-1.8"/><path d="M19.5 19.2V15h-4.2"/>',
    "history": '<path d="M4 12a8 8 0 1 0 2.4-5.7"/><path d="M4 4.2v3.6h3.6"/><path d="M12 8v4.2l2.8 1.8"/>',
    "settings": '<path d="M4 7h9"/><path d="M18 7h2"/><circle cx="15.5" cy="7" r="2.3"/>'
                '<path d="M4 17h3"/><path d="M12 17h8"/><circle cx="9.5" cy="17" r="2.3"/>',
    "journal": '<rect x="5" y="3.5" width="14" height="17" rx="2.5"/><path d="M9 8.5h6"/>'
               '<path d="M9 12h6"/><path d="M9 15.5h3.5"/>',
    "camera": '<circle cx="12" cy="10" r="6.3"/><circle cx="12" cy="10" r="2.4"/>'
              '<path d="M12 16.3v3.7"/><path d="M8.2 20.3h7.6"/>',
    "file": '<rect x="3.5" y="5" width="17" height="14" rx="2.6"/>'
            '<path d="M10.2 9.4v5.2l4.4-2.6z" fill="{c}" stroke-width="1.2"/>',
    "screen": '<rect x="3" y="4.5" width="18" height="12" rx="2.2"/><path d="M9 20h6"/><path d="M12 16.5V20"/>',
    "mic": '<rect x="9" y="3.5" width="6" height="11" rx="3"/><path d="M5.6 11.2a6.4 6.4 0 0 0 12.8 0"/>'
           '<path d="M12 17.6v2.9"/>',
    "eye": '<path d="M2.5 12s3.5-6.3 9.5-6.3S21.5 12 21.5 12 18 18.3 12 18.3 2.5 12 2.5 12z"/>'
           '<circle cx="12" cy="12" r="2.8"/>',
    "face": '<circle cx="12" cy="12" r="8.5"/><path d="M8.9 14.3a3.9 3.9 0 0 0 6.2 0"/>'
            '<path d="M9.3 9.8h.01"/><path d="M14.7 9.8h.01"/>',
    "blink": '<path d="M3 10.2s3.3 4.6 9 4.6 9-4.6 9-4.6"/><path d="M5.9 13.3l-1.4 2"/>'
             '<path d="M10 14.7l-.5 2.4"/><path d="M14 14.7l.5 2.4"/><path d="M18.1 13.3l1.4 2"/>',
    "gaze": '<circle cx="12" cy="12" r="7.3"/><circle cx="12" cy="12" r="2.2"/><path d="M12 2.5v2.6"/>'
            '<path d="M12 18.9v2.6"/><path d="M2.5 12h2.6"/><path d="M18.9 12h2.6"/>',
    "expression": '<circle cx="12" cy="12" r="8.5"/><path d="M8 9.1l2.4.7"/><path d="M16 9.1l-2.4.7"/>'
                  '<path d="M9 15.2h6"/>',
    "activity": '<path d="M3 12h3.6l2.4-5.8 4 11.6 2.4-5.8H21"/>',
    "redness": '<path d="M12 3.6s5.8 6.2 5.8 10.4a5.8 5.8 0 0 1-11.6 0c0-4.2 5.8-10.4 5.8-10.4z"/>'
               '<path d="M9.4 14.6a2.7 2.7 0 0 0 2.4 2.4"/>',
    "speech": '<path d="M4 10.2v3.6"/><path d="M8 7.2v9.6"/><path d="M12 4.5v15"/>'
              '<path d="M16 7.2v9.6"/><path d="M20 10.2v3.6"/>',
    "deviation": '<path d="M3.5 16.5 9 11l3.5 3.5 8-8"/><path d="M15 6.5h5.5V12"/>',
    "info": '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5"/><path d="M12 7.9h.01"/>',
    "folder": '<path d="M3.5 7.6a2 2 0 0 1 2-2h3.9l2.1 2.2h7a2 2 0 0 1 2 2v7.6a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>',
    "trash": '<path d="M4.5 7h15"/><path d="M9.5 7V4.8h5V7"/>'
             '<path d="M6.6 7l.8 12a1.6 1.6 0 0 0 1.6 1.5h6a1.6 1.6 0 0 0 1.6-1.5l.8-12"/>',
    "close": '<path d="M6.5 6.5l11 11"/><path d="M17.5 6.5l-11 11"/>',
    "chevron-down": '<path d="m6 9 6 6 6-6"/>',
    "check": '<path d="m5 12.5 4.4 4.4L19 7.4"/>',
    "alert": '<path d="M10.3 4.6 3 17.6a2 2 0 0 0 1.7 2.9h14.6a2 2 0 0 0 1.7-2.9L13.7 4.6a2 2 0 0 0-3.4 0z"/>'
             '<path d="M12 9.6v4.2"/><path d="M12 17h.01"/>',
    "user": '<circle cx="12" cy="8.4" r="3.8"/><path d="M4.6 20a7.4 7.4 0 0 1 14.8 0"/>',
    "save": '<path d="M5.5 4h10.2L19 7.3V19a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V4.5"/><path d="M8.5 4v4.6h6.5V4"/>'
            '<path d="M8.5 20v-5.5h7V20"/>',
    "shield": '<path d="M12 3.5 5 6.2v5.6c0 4.3 3 7.6 7 8.7 4-1.1 7-4.4 7-8.7V6.2z"/><path d="m9 12 2.1 2.1L15 10.2"/>',
    "tag": '<path d="M3.5 12.6V4.5a1 1 0 0 1 1-1h8.1l8 8a1.4 1.4 0 0 1 0 2l-7 7a1.4 1.4 0 0 1-2 0z"/>'
           '<circle cx="8" cy="8" r="1.3"/>',
    "external": '<path d="M14 4h6v6"/><path d="M20 4l-8.5 8.5"/>'
                '<path d="M18 13.5v5A1.5 1.5 0 0 1 16.5 20h-11A1.5 1.5 0 0 1 4 18.5v-11A1.5 1.5 0 0 1 5.5 6h5"/>',
    "layers": '<path d="M12 3.6 3 8.4l9 4.8 9-4.8z"/><path d="M3 12.6l9 4.8 9-4.8"/>',
    "captions": '<rect x="3" y="5" width="18" height="14" rx="3"/><path d="M10.6 10.3a2.2 2.2 0 1 0 0 3.4"/>'
                '<path d="M17 10.3a2.2 2.2 0 1 0 0 3.4"/>',
    "copy": '<rect x="8" y="8" width="12" height="12" rx="2.2"/><path d="M16 8V5.5A1.5 1.5 0 0 0 14.5 4h-9A1.5 1.5 0 0 0 4 5.5v9A1.5 1.5 0 0 0 5.5 16H8"/>',
    "maximize": '<path d="M4 9V4h5"/><path d="M20 9V4h-5"/><path d="M4 15v5h5"/><path d="M20 15v5h-5"/>',
}

_cache = {}


def _dpr():
    screen = QGuiApplication.primaryScreen()
    return screen.devicePixelRatio() if screen is not None else 1.0


def svg_data(name, color=C.TEXT, stroke=1.75):
    body = _ICONS[name].replace("{c}", color)
    return _SVG.format(c=color, w=stroke, body=body)


def pixmap(name, size=18, color=C.TEXT, stroke=1.75):
    dpr = _dpr()
    key = (name, size, color, stroke, dpr)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    renderer = QSvgRenderer(QByteArray(svg_data(name, color, stroke).encode("utf-8")))
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing)
    renderer.render(painter, QRectF(0, 0, size * dpr, size * dpr))
    painter.end()
    pm.setDevicePixelRatio(dpr)
    _cache[key] = pm
    return pm


def icon(name, color=C.TEXT, size=18, disabled_color=C.TEXT_4, stroke=1.75):
    result = QIcon()
    result.addPixmap(pixmap(name, size, color, stroke), QIcon.Normal)
    result.addPixmap(pixmap(name, size, color, stroke), QIcon.Active)
    result.addPixmap(pixmap(name, size, disabled_color, stroke), QIcon.Disabled)
    return result


def draw_icon(painter, name, rect, color=C.TEXT, stroke=1.75):
    """Нарисовать иконку в paintEvent (из кэша - перерисовка дешёвая)."""
    r = QRectF(rect)
    size = int(round(min(r.width(), r.height())))
    painter.drawPixmap(QPointF(r.x(), r.y()), pixmap(name, size, color, stroke))


# ---------------------------------------------------------------- логотип

def paint_logo(painter, rect, dark=C.GRAPHITE, accent=C.ACCENT, light="#FFFFFF"):
    """Знак: графитовая плитка, контур глаза-линзы и красная "радужка"."""
    r = QRectF(rect)
    s = min(r.width(), r.height())
    r = QRectF(r.center().x() - s / 2, r.center().y() - s / 2, s, s)
    painter.save()
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(dark))
    painter.drawRoundedRect(r, s * 0.28, s * 0.28)

    cx, cy = r.center().x(), r.center().y()
    w, h = s * 0.62, s * 0.36
    eye = QPainterPath()
    eye.moveTo(cx - w / 2, cy)
    eye.cubicTo(cx - w / 4, cy - h * 0.95, cx + w / 4, cy - h * 0.95, cx + w / 2, cy)
    eye.cubicTo(cx + w / 4, cy + h * 0.95, cx - w / 4, cy + h * 0.95, cx - w / 2, cy)
    pen = QPen(QColor(light), max(1.2, s * 0.065))
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    painter.drawPath(eye)

    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(accent))
    painter.drawEllipse(QPointF(cx, cy), s * 0.11, s * 0.11)
    painter.restore()


def app_icon():
    result = QIcon()
    for size in (16, 20, 24, 32, 40, 48, 64, 128, 256):
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        paint_logo(painter, QRectF(0, 0, size, size))
        painter.end()
        result.addPixmap(pm)
    return result
