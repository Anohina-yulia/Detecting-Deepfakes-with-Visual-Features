# -*- coding: utf-8 -*-
"""Исключение окна программы из захвата экрана (только Windows)."""
import ctypes
import os

WDA_NONE = 0x00000000
WDA_EXCLUDEFROMCAPTURE = 0x00000011


def _set_affinity(window_title, flag):
    if os.name != "nt":
        return False
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_title)
        if not hwnd:
            return False
        return bool(user32.SetWindowDisplayAffinity(hwnd, flag))
    except Exception as error:
        print("Ошибка защиты окна от захвата:", error)
        return False


def exclude_window_from_capture(window_title):
    """Окно перестаёт попадать на скриншоты и в запись экрана."""
    return _set_affinity(window_title, WDA_EXCLUDEFROMCAPTURE)


def include_window_in_capture(window_title):
    """Возвращает окно в захват экрана."""
    return _set_affinity(window_title, WDA_NONE)