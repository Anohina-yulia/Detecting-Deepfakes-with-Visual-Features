# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np
from PIL import ImageGrab


def get_screen_size():
    return ImageGrab.grab().size


def fit_frame(frame, width, height):
    """Вписывает кадр в width x height без растягивания (чёрные поля)."""
    h, w = frame.shape[:2]
    if (w, h) == (width, height):
        return frame

    scale = min(width / w, height / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (nw, nh), interpolation=interp)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    ox, oy = (width - nw) // 2, (height - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


class WebcamSource:
    def __init__(self, index=0, width=1280, height=720):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width, self.height = width, height

    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, fit_frame(frame, self.width, self.height)

    def release(self):
        self.cap.release()


class ScreenSource:
    def __init__(self, width=1280, height=720):
        self.width, self.height = width, height
        self._opened = True

    def isOpened(self):
        return self._opened

    def read(self):
        try:
            frame = cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_RGB2BGR)
            return True, fit_frame(frame, self.width, self.height)
        except Exception as error:
            print("Ошибка захвата экрана:", error)
            return False, None

    def release(self):
        self._opened = False


class VideoFileSource:
    def __init__(self, filepath, width=1280, height=720):
        self.filepath = filepath
        self.cap = cv2.VideoCapture(filepath)
        self.width, self.height = width, height

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 0:
            self.fps = 30.0

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        self.current_frame += 1
        return True, fit_frame(frame, self.width, self.height)

    def get_fps(self):
        return self.fps

    def get_frame_count(self):
        return self.frame_count

    def get_current_frame(self):
        return self.current_frame

    def get_time(self):
        """Время только что прочитанного кадра в секундах от начала видео."""
        return max(self.current_frame - 1, 0) / self.fps

    def release(self):
        self.cap.release()


def select_video_file():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    filepath = filedialog.askopenfilename(
        title="Выберите видеофайл",
        filetypes=[
            ("Видеофайлы", "*.mp4 *.avi *.mov *.mkv *.wmv *.webm"),
            ("Все файлы", "*.*"),
        ],
    )
    root.destroy()
    return filepath


def choose_video_source(width=1280, height=720):
    print()
    print("Выберите источник видео:")
    print("1 - Веб-камера")
    print("2 - Экран ноутбука")
    print("3 - Видеофайл")
    print()

    choice = input("Введите 1, 2 или 3: ").strip()

    if choice == "2":
        print("Источник видео: экран ноутбука")
        return ScreenSource(width=width, height=height)

    if choice == "3":
        print()
        print("Выберите видеофайл в открывшемся окне...")
        filepath = select_video_file()

        if not filepath:
            print("Файл не выбран, переключаюсь на веб-камеру.")
            return WebcamSource(width=width, height=height)

        print("Выбран файл:", filepath)
        return VideoFileSource(filepath=filepath, width=width, height=height)

    print("Источник видео: веб-камера")
    return WebcamSource(width=width, height=height)