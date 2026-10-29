# -*- coding: utf-8 -*-
"""Общие элементы интерфейса.

* load_font      - шрифт с кириллицей (Arial / Segoe / DejaVu)
* new_panel      - полупрозрачная панель (PIL, RGBA)
* draw_bar       - полоса значения
* PanelManager   - панели, которые можно двигать мышью (положение сохраняется)
* DisplayMapper  - окно можно растягивать, картинка сохраняет пропорции
"""
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WHITE = (255, 255, 255)
GRAY = (220, 220, 220)
GRAY_DARK = (170, 170, 170)
BLUE = (80, 190, 255)
GREEN = (110, 230, 140)
ORANGE = (255, 185, 90)
YELLOW = (245, 225, 100)
RED = (255, 120, 120)

_FONT_PATHS = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "arial.ttf",
    "DejaVuSans.ttf",
]
_font_cache = {}


def load_font(size=15):
    if size in _font_cache:
        return _font_cache[size]
    font = None
    for path in _FONT_PATHS:
        try:
            font = ImageFont.truetype(path, size)
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size)
        except TypeError:
            font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def clamp01(value):
    return float(max(0.0, min(1.0, value)))


def new_panel(width, height, outline=True):
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        (0, 0, width - 1, height - 1),
        radius=10,
        fill=(0, 0, 0, 215),
        outline=(255, 255, 255, 255) if outline else None,
        width=2,
    )
    return img, draw


def to_bgra(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def draw_bar(draw, x, y, value, label, color=BLUE, font=None, label_w=115,
             width=170, height=14, marker=False, end_labels=None,
             show_value=True, label_color=None):
    """Полоса значения. marker=True рисует ползунок с центральной меткой."""
    font = font or load_font(15)
    draw.text((x, y - 2), label, font=font, fill=label_color or WHITE)

    bx1, by1 = x + label_w, y + 2
    bx2, by2 = bx1 + width, by1 + height
    draw.rounded_rectangle((bx1, by1, bx2, by2), radius=5, fill=(45, 45, 45),
                           outline=(110, 110, 110), width=1)
    value = clamp01(value)

    if marker:
        cx = (bx1 + bx2) // 2
        draw.line((cx, by1 - 2, cx, by2 + 2), fill=GRAY_DARK, width=1)
        mx = int(bx1 + value * width)
        draw.rounded_rectangle((mx - 4, by1 - 1, mx + 4, by2 + 1), radius=3, fill=color)
        if end_labels:
            draw.text((bx1, by2 + 3), end_labels[0], font=font, fill=GRAY_DARK)
            draw.text((bx2 - 12, by2 + 3), end_labels[1], font=font, fill=GRAY_DARK)
    else:
        fill_w = int(width * value)
        if fill_w > 0:
            draw.rounded_rectangle((bx1, by1, bx1 + fill_w, by2), radius=5, fill=color)
        if show_value:
            draw.text((bx2 + 8, y - 2), f"{value:.2f}", font=font, fill=WHITE)


def blit_bgra(canvas, panel, x, y):
    """Накладывает BGRA-панель на BGR-кадр с учётом прозрачности."""
    ph, pw = panel.shape[:2]
    ch, cw = canvas.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + pw, cw), min(y + ph, ch)
    if x2 <= x1 or y2 <= y1:
        return
    sub = panel[y1 - y:y2 - y, x1 - x:x2 - x]
    alpha = sub[:, :, 3:4].astype(np.float32) / 255.0
    roi = canvas[y1:y2, x1:x2]
    roi[:] = (sub[:, :, :3] * alpha + roi * (1.0 - alpha)).astype(np.uint8)


class PanelManager:
    """Хранит панели и позволяет перетаскивать их мышью."""

    def __init__(self, canvas_w, canvas_h, layout_file="panel_layout.json"):
        self.cw, self.ch = canvas_w, canvas_h
        self.layout_file = layout_file
        self.panels = {}
        self.order = []  # последняя панель рисуется сверху
        self.visible = True
        self.drag_id = None
        self.dx = self.dy = 0
        self.saved = self._load()

    def _load(self):
        try:
            with open(self.layout_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            return {}

    def save(self):
        data = {pid: [p["x"], p["y"]] for pid, p in self.panels.items()}
        try:
            with open(self.layout_file, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False)
        except Exception as error:
            print("Не удалось сохранить расположение панелей:", error)

    def register(self, pid, default_xy):
        x, y = self.saved.get(pid, default_xy)
        self.panels[pid] = {"x": int(x), "y": int(y), "img": None,
                            "default": tuple(default_xy)}
        self.order.append(pid)

    def set_image(self, pid, bgra):
        self.panels[pid]["img"] = bgra

    def reset_layout(self):
        for p in self.panels.values():
            p["x"], p["y"] = p["default"]
        self.save()

    def toggle_visible(self):
        self.visible = not self.visible

    def draw(self, canvas):
        if not self.visible:
            return
        for pid in self.order:
            p = self.panels[pid]
            if p["img"] is not None:
                blit_bgra(canvas, p["img"], p["x"], p["y"])

    def handle_mouse(self, event, x, y):
        """x, y - координаты в системе кадра (1280x720)."""
        if event == cv2.EVENT_LBUTTONDOWN and self.visible:
            for pid in reversed(self.order):
                p = self.panels[pid]
                if p["img"] is None:
                    continue
                h, w = p["img"].shape[:2]
                if p["x"] <= x < p["x"] + w and p["y"] <= y < p["y"] + h:
                    self.drag_id = pid
                    self.dx, self.dy = x - p["x"], y - p["y"]
                    self.order.remove(pid)
                    self.order.append(pid)
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_id:
            p = self.panels[self.drag_id]
            h, w = p["img"].shape[:2]
            p["x"] = int(min(max(x - self.dx, 0), max(self.cw - w, 0)))
            p["y"] = int(min(max(y - self.dy, 0), max(self.ch - h, 0)))
        elif event == cv2.EVENT_LBUTTONUP and self.drag_id:
            self.drag_id = None
            self.save()


class DisplayMapper:
    """Подгоняет кадр под размер окна (с чёрными полями) и пересчитывает координаты мыши."""

    def __init__(self, window_title, canvas_w, canvas_h):
        self.title = window_title
        self.cw, self.ch = canvas_w, canvas_h
        self.scale = 1.0
        self.off_x = 0
        self.off_y = 0

    def render(self, canvas):
        try:
            _, _, ww, wh = cv2.getWindowImageRect(self.title)
        except Exception:
            ww = wh = 0
        if ww < 50 or wh < 50:
            self.scale, self.off_x, self.off_y = 1.0, 0, 0
            return canvas

        scale = min(ww / self.cw, wh / self.ch)
        nw, nh = max(1, int(self.cw * scale)), max(1, int(self.ch * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(canvas, (nw, nh), interpolation=interp)

        out = np.zeros((wh, ww, 3), dtype=np.uint8)
        ox, oy = (ww - nw) // 2, (wh - nh) // 2
        out[oy:oy + nh, ox:ox + nw] = resized
        self.scale, self.off_x, self.off_y = scale, ox, oy
        return out

    def to_canvas(self, x, y):
        return int((x - self.off_x) / self.scale), int((y - self.off_y) / self.scale)