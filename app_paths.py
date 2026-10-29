# -*- coding: utf-8 -*-
"""Пути проекта.

Раньше results/ и models/ брались относительно текущей папки, поэтому запуск
не из каталога проекта (ярлык, IDE, другой терминал) ломал загрузку модели
и складывал результаты куда попало. Теперь всё считается от папки проекта.
"""
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ATG_RESULTS_DIR позволяет тестам писать результаты во временную папку
RESULTS_DIR = os.environ.get("ATG_RESULTS_DIR") or os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SETTINGS_FILE = os.environ.get("ATG_SETTINGS_FILE") or os.path.join(BASE_DIR, "settings.json")
PANEL_LAYOUT_FILE = os.path.join(BASE_DIR, "panel_layout.json")


def project_path(*parts):
    return os.path.join(BASE_DIR, *parts)


def make_session_dir(base, results_dir=None):
    """Каждый сеанс пишет результаты в свою папку и ничего не затирает."""
    root = results_dir or RESULTS_DIR
    path = os.path.join(root, f"{base}_{datetime.now():%Y%m%d_%H%M%S}")
    suffix = 1
    while os.path.exists(path):  # два сеанса в одну секунду
        suffix += 1
        path = os.path.join(root, f"{base}_{datetime.now():%Y%m%d_%H%M%S}_{suffix}")
    os.makedirs(path, exist_ok=True)
    return path
