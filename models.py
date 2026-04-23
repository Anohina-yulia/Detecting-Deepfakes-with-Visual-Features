from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class RednessData:
    overall: float = 0.0
    left_cheek: float = 0.0
    right_cheek: float = 0.0
    nose: float = 0.0
    forehead: float = 0.0
    status: str = "нет данных"


@dataclass
class FaceData:
    detected: bool = False
    landmarks: Optional[List[Tuple[int, int]]] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    frame_width: int = 0
    frame_height: int = 0
    timestamp: float = 0.0
    face_position: str = "не найдено"
    blendshapes: Dict[str, float] = field(default_factory=dict)


@dataclass
class MimicData:
    smile: float = 0.0
    mouth_open: float = 0.0
    brow_raise: float = 0.0
    brow_frown: float = 0.0
    eye_squint: float = 0.0
    dominant_expression: str = "нейтрально"


@dataclass
class MicroEvent:
    name: str
    strength: float
    duration_ms: float
    timestamp: float
