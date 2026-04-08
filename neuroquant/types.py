"""
Спільні типи даних для NeuroQuant.

Винесені в окремий файл для уникнення циклічних залежностей.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


@dataclass
class FrameComplexity:
    """Дані про складність одного кадру."""
    frame_idx: int
    spatial: float      # Просторова складність [0, 1]
    temporal: float     # Тимчасова складність [0, 1]
    is_scene_cut: bool  # Чи є зміна сцени
    complexity: float   # Фінальна зважена складність [0, 1]


class FrameType(Enum):
    """Тип кадру для GOP структури."""
    I = "I"  # Intra (keyframe)
    P = "P"  # Predictive
    B = "B"  # Bi-directional


@dataclass
class QPPlan:
    """QP-план для одного кадру."""
    frame_idx: int
    frame_type: FrameType
    qp: int
    complexity: float


@dataclass
class EncodingResult:
    """Результат кодування."""
    output_path: str
    actual_bitrate: int
    file_size: int
    duration: float
    codec: str
    success: bool
    error_message: Optional[str] = None
    encoding_time: float = 0.0


@dataclass
class SRResult:
    """Результат SR постпроцесингу."""
    output_path: str
    frames_processed: int
    frames_total: int
    processing_time: float
    success: bool
    error_message: Optional[str] = None
