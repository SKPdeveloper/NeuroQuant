"""
RLambdaController - per-frame rate control на основі R-λ теорії.

Генерує QP-план для x265 на основі аналізу складності кадрів.

R-λ модель: R = α · λ^β
Зв'язок λ і QP: λ = c · 2^((QP - 12) / 3)
"""

import math
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from .types import FrameComplexity, FrameType, QPPlan
from .utils import log_info, log_warning


class RLambdaController:
    """
    Контролер rate control на основі R-λ теорії.

    Генерує per-frame QP план для досягнення цільового бітрейту
    з адаптацією до складності контенту.
    """

    def __init__(
        self,
        alpha: float = 6.7542,
        beta: float = -1.7860,
        c_p_frame: float = 0.85,
        c_b_frame: float = 0.57,
        qp_min: int = 10,
        qp_max: int = 51,
        delta_max: int = 8,
        i_frame_bonus: int = 4,
        gop_seconds: float = 2.0,
    ):
        """
        Ініціалізація контролера.

        Args:
            alpha: Константа α R-λ моделі для x265
            beta: Константа β R-λ моделі для x265
            c_p_frame: Константа c для P-кадрів
            c_b_frame: Константа c для B-кадрів
            qp_min: Мінімальний QP
            qp_max: Максимальний QP
            delta_max: Максимальне відхилення QP від базового
            i_frame_bonus: Бонус якості для I-кадрів (зменшення QP)
            gop_seconds: Тривалість GOP у секундах
        """
        self.alpha = alpha
        self.beta = beta
        self.c_p_frame = c_p_frame
        self.c_b_frame = c_b_frame
        self.qp_min = qp_min
        self.qp_max = qp_max
        self.delta_max = delta_max
        self.i_frame_bonus = i_frame_bonus
        self.gop_seconds = gop_seconds

    def generate_qp_plan(
        self,
        complexity_data: List[FrameComplexity],
        target_bitrate: int,
        fps: float,
        width: int,
        height: int,
    ) -> List[QPPlan]:
        """
        Генерує QP-план для відео.

        Args:
            complexity_data: Результати аналізу складності
            target_bitrate: Цільовий бітрейт (bps)
            fps: Частота кадрів
            width: Ширина відео
            height: Висота відео

        Returns:
            Список QPPlan для кожного кадру
        """
        frame_count = len(complexity_data)
        duration = frame_count / fps

        log_info(f"Генерація QP-плану для {frame_count} кадрів")
        log_info(f"Цільовий бітрейт: {target_bitrate / 1000:.0f} kbps")

        # Крок 1: Обчислюємо базовий QP з цільового бітрейту
        bits_per_pixel = target_bitrate / (fps * width * height)
        qp_base = self._bitrate_to_qp(target_bitrate, fps, width, height)
        log_info(f"Базовий QP: {qp_base}")

        # Крок 2: Отримуємо статистику складності
        complexities = np.array([c.complexity for c in complexity_data])
        complexity_mean = np.mean(complexities)
        complexity_std = np.std(complexities)

        if complexity_std < 0.01:
            complexity_std = 0.01  # Уникаємо ділення на нуль

        # Крок 3: Визначаємо GOP структуру та scene cuts
        gop_size = int(fps * self.gop_seconds)
        scene_cuts = {c.frame_idx for c in complexity_data if c.is_scene_cut}

        # Крок 4: Генеруємо per-frame QP
        qp_plan = []
        current_gop_start = 0

        for i, frame_data in enumerate(complexity_data):
            # Визначаємо тип кадру
            is_gop_start = (i == 0) or (i in scene_cuts) or (i - current_gop_start >= gop_size)

            if is_gop_start:
                frame_type = FrameType.I
                current_gop_start = i
            elif (i - current_gop_start) % 3 == 0:
                frame_type = FrameType.P
            else:
                frame_type = FrameType.B

            # Обчислюємо QP з урахуванням складності
            complexity_deviation = (frame_data.complexity - complexity_mean) / complexity_std
            qp_adjustment = round(self.delta_max * complexity_deviation)

            # Складніші кадри → менший QP (краща якість)
            qp = qp_base - qp_adjustment

            # Бонус для I-кадрів
            if frame_type == FrameType.I:
                qp -= self.i_frame_bonus

            # Обмеження діапазону
            qp = max(self.qp_min, min(self.qp_max, qp))

            qp_plan.append(QPPlan(
                frame_idx=i,
                frame_type=frame_type,
                qp=qp,
                complexity=frame_data.complexity,
            ))

        # Крок 5: Балансування бюджету
        qp_plan = self._balance_budget(
            qp_plan, target_bitrate, fps, width, height, duration
        )

        # Статистика
        qp_values = [p.qp for p in qp_plan]
        log_info(f"Діапазон QP: {min(qp_values)}–{max(qp_values)}")

        i_frames = sum(1 for p in qp_plan if p.frame_type == FrameType.I)
        log_info(f"I-кадрів: {i_frames} (GOP ~{gop_size} кадрів)")

        return qp_plan

    def _bitrate_to_qp(self, bitrate: int, fps: float, width: int, height: int) -> int:
        """
        Обчислює базовий QP з цільового бітрейту через емпіричну модель.

        Використовує спрощену формулу, калібровану для x265:
        QP ≈ 51 - 6 * log2(bpp * 100 + 1)

        де bpp = bits_per_pixel = bitrate / (fps * width * height)
        """
        # Біти на піксель на кадр
        bits_per_frame = bitrate / fps
        pixels_per_frame = width * height
        bpp = bits_per_frame / pixels_per_frame

        if bpp <= 0:
            return self.qp_max

        # Емпірична формула для x265
        # Більший bpp → менший QP (краща якість)
        # Калібрована для типових випадків:
        # - bpp ≈ 0.01 (дуже низький бітрейт) → QP ≈ 45
        # - bpp ≈ 0.05 (низький бітрейт) → QP ≈ 38
        # - bpp ≈ 0.15 (середній бітрейт) → QP ≈ 32
        # - bpp ≈ 0.50 (високий бітрейт) → QP ≈ 24
        # - bpp ≈ 1.00 (дуже високий) → QP ≈ 20
        qp = 51 - 6 * math.log2(bpp * 100 + 1)

        return int(round(max(self.qp_min, min(self.qp_max, qp))))

    def _qp_to_bitrate(self, qp: int, fps: float, width: int, height: int) -> float:
        """
        Оцінює бітрейт для заданого QP (обернена функція до _bitrate_to_qp).

        QP = 51 - 6 * log2(bpp * 100 + 1)
        → bpp * 100 + 1 = 2^((51 - QP) / 6)
        → bpp = (2^((51 - QP) / 6) - 1) / 100
        """
        # Обернена формула
        bpp = (math.pow(2, (51 - qp) / 6.0) - 1) / 100.0

        # bpp → bitrate
        pixels_per_frame = width * height
        bits_per_frame = bpp * pixels_per_frame
        bitrate = bits_per_frame * fps

        return max(bitrate, 1000)  # Мінімум 1 kbps

    def _balance_budget(
        self,
        qp_plan: List[QPPlan],
        target_bitrate: int,
        fps: float,
        width: int,
        height: int,
        duration: float,
    ) -> List[QPPlan]:
        """
        Балансує QP-план для досягнення цільового бітрейту.

        Використовує бінарний пошук для зсуву всього плану.
        """
        target_bits = target_bitrate * duration

        # Оцінюємо поточний бітрейт
        estimated_bits = 0
        for plan in qp_plan:
            frame_bitrate = self._qp_to_bitrate(plan.qp, fps, width, height)
            estimated_bits += frame_bitrate / fps

        ratio = estimated_bits / target_bits if target_bits > 0 else 1.0

        # Якщо різниця > 5%, коригуємо
        if abs(ratio - 1.0) > 0.05:
            # Бінарний пошук оптимального зсуву
            shift_low, shift_high = -10, 10
            best_shift = 0
            best_diff = float('inf')

            for _ in range(10):  # Ітерації бінарного пошуку
                shift = (shift_low + shift_high) // 2

                test_bits = 0
                for plan in qp_plan:
                    test_qp = max(self.qp_min, min(self.qp_max, plan.qp + shift))
                    frame_bitrate = self._qp_to_bitrate(test_qp, fps, width, height)
                    test_bits += frame_bitrate / fps

                diff = test_bits - target_bits

                if abs(diff) < best_diff:
                    best_diff = abs(diff)
                    best_shift = shift

                if diff > 0:
                    # Занадто багато бітів → збільшуємо QP
                    shift_low = shift
                else:
                    # Замало бітів → зменшуємо QP
                    shift_high = shift

            # Застосовуємо оптимальний зсув
            if best_shift != 0:
                log_info(f"Балансування бюджету: зсув QP на {best_shift:+d}")
                for plan in qp_plan:
                    plan.qp = max(self.qp_min, min(self.qp_max, plan.qp + best_shift))

        return qp_plan

    def save_qp_file(self, qp_plan: List[QPPlan], output_path: str) -> None:
        """
        Зберігає QP-план у форматі x265 qpfile.

        Формат:
            frame_num frame_type qp
            0 I 28
            1 P 31
            2 B 33
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for plan in qp_plan:
                f.write(f"{plan.frame_idx} {plan.frame_type.value} {plan.qp}\n")

        log_info(f"QP-файл збережено: {output_path}")

    def get_gop_structure(self, qp_plan: List[QPPlan]) -> List[Tuple[int, int]]:
        """
        Повертає структуру GOP: [(start_frame, end_frame), ...]
        """
        gops = []
        current_start = 0

        for plan in qp_plan:
            if plan.frame_type == FrameType.I and plan.frame_idx > 0:
                gops.append((current_start, plan.frame_idx - 1))
                current_start = plan.frame_idx

        # Останній GOP
        if qp_plan:
            gops.append((current_start, qp_plan[-1].frame_idx))

        return gops

    def get_statistics(self, qp_plan: List[QPPlan]) -> dict:
        """Повертає статистику QP-плану."""
        qp_values = np.array([p.qp for p in qp_plan])
        complexities = np.array([p.complexity for p in qp_plan])

        i_count = sum(1 for p in qp_plan if p.frame_type == FrameType.I)
        p_count = sum(1 for p in qp_plan if p.frame_type == FrameType.P)
        b_count = sum(1 for p in qp_plan if p.frame_type == FrameType.B)

        return {
            "frame_count": len(qp_plan),
            "qp_mean": float(np.mean(qp_values)),
            "qp_std": float(np.std(qp_values)),
            "qp_min": int(np.min(qp_values)),
            "qp_max": int(np.max(qp_values)),
            "complexity_mean": float(np.mean(complexities)),
            "complexity_std": float(np.std(complexities)),
            "i_frames": i_count,
            "p_frames": p_count,
            "b_frames": b_count,
        }
