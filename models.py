# -*- coding: utf-8 -*-
"""Общие структуры данных и названия состояний.

Все модули берут строки отсюда, поэтому расхождений вида
"Center" / "center" / "по центру" больше быть не может.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --- Положение лица в кадре ---
POS_LEFT = "Слева"
POS_CENTER = "По центру"
POS_RIGHT = "Справа"
POS_NONE = "Не найдено"

# --- Направление взгляда ---
GAZE_CENTER = "Центр"
GAZE_LEFT = "Влево"
GAZE_RIGHT = "Вправо"
GAZE_UP = "Вверх"
GAZE_DOWN = "Вниз"
GAZE_NONE = "Не найдено"

# --- Мимика ---
EXPR_NEUTRAL = "Нейтрально"
EXPR_SMILE = "Улыбка"
EXPR_MOUTH = "Рот открыт"
EXPR_FROWN = "Нахмурено"
EXPR_BROWS = "Брови вверх"
EXPR_SQUINT = "Прищур"
EXPR_NO_FACE = "Нет лица"

# --- Метки разметки (для оценки) ---
LABEL_TRUTH = "правда"
LABEL_LIE = "ложь"

# --- Покраснение ---
RED_NORMAL = "норма"
RED_MILD = "лёгкое"
RED_NOTICEABLE = "заметное"
RED_STRONG = "сильное"
RED_NO_FACE = "нет лица"
RED_CALIBRATION = "калибровка"
RED_WAIT = "ожидание"


@dataclass
class RednessData:
    overall: float = 0.0
    left_cheek: float = 0.0
    right_cheek: float = 0.0
    nose: float = 0.0
    forehead: float = 0.0
    status: str = RED_WAIT
    calibrated: bool = False
    progress: int = 0  # сколько кадров калибровки набрано


@dataclass
class FaceData:
    detected: bool = False
    landmarks: Optional[List[Tuple[int, int]]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    frame_width: int = 0
    frame_height: int = 0
    timestamp: float = 0.0  # время кадра в секундах (для файла - по таймлайну видео)
    face_position: str = POS_NONE
    blendshapes: Dict[str, float] = field(default_factory=dict)


@dataclass
class MimicData:
    smile: float = 0.0
    mouth_open: float = 0.0
    brow_raise: float = 0.0
    brow_frown: float = 0.0
    eye_squint: float = 0.0
    dominant_expression: str = EXPR_NEUTRAL


@dataclass
class MicroEvent:
    name: str
    strength: float
    duration_ms: float  # длительность нарастания (измеренная), а не константа
    timestamp: float


# --- События сессии (журнал и временная шкала) ---
EV_CALIBRATION = "калибровка"
EV_FACE = "лицо"
EV_MICRO = "мимика"
EV_EYES = "глаза"
EV_SPEECH = "речь"
EV_QUALITY = "качество"
EV_LABEL = "разметка"
EV_SYSTEM = "система"


@dataclass
class SessionEvent:
    t: float            # время сессии, с
    kind: str           # одна из констант EV_*
    text: str
    level: str = "info"  # info | notice | warning | error