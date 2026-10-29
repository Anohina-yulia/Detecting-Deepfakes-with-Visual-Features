# -*- coding: utf-8 -*-
"""Настройки приложения (settings.json в папке проекта).

Секретов здесь нет и быть не должно: только параметры интерфейса,
источника видео, распознавания речи и сохранения истории.
"""
import json
import os

from app_paths import SETTINGS_FILE

DEFAULTS = {
    # источник
    "source": "camera",            # camera | file | screen
    "camera_index": 0,
    "last_video_dir": "",
    "pace_video_file": True,       # видеофайл воспроизводится в реальном времени

    # сессия
    "participant": "",
    "personal_baseline": True,     # собирать личную базу (40 с) в начале сессии

    # речь
    "speech_enabled": True,
    "audio_device": None,          # None - устройство по умолчанию, иначе индекс sounddevice
    "whisper_model": "small",
    "text_correction": True,
    "allow_model_download": True,

    # отображение
    "overlay_face_box": True,
    "overlay_eyes": True,
    "overlay_mesh": False,
    "captions_on_video": True,
    "timeline_window": 180,        # секунд; 0 - вся сессия

    # приватность и хранение
    "save_log": True,              # числовой журнал analysis_log.csv
    "save_subtitles": True,        # текст субтитров
    "save_video": False,           # обработанное видео (только для видеофайлов)
    "protect_window": True,        # скрывать окно из захвата экрана в режиме "Экран"

    # ручная разметка для evaluate_lie_signals.py
    "markup_enabled": False,
}


class SettingsStore:
    def __init__(self, path=SETTINGS_FILE):
        self.path = path
        self.data = dict(DEFAULTS)
        self.load()

    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as file:
                saved = json.load(file)
            if isinstance(saved, dict):
                for key, value in saved.items():
                    if key in DEFAULTS:
                        self.data[key] = value
        except FileNotFoundError:
            pass
        except Exception as error:
            print("Не удалось прочитать настройки, использую значения по умолчанию:", error)

    def save(self):
        try:
            folder = os.path.dirname(self.path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as file:
                json.dump(self.data, file, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception as error:
            print("Не удалось сохранить настройки:", error)

    def get(self, key):
        return self.data.get(key, DEFAULTS.get(key))

    def set(self, key, value):
        self.data[key] = value

    def update(self, values):
        for key, value in values.items():
            if key in DEFAULTS:
                self.data[key] = value

    def __getitem__(self, key):
        return self.get(key)
