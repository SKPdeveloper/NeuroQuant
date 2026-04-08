"""Тести для MetricsCollector та BDRateCalculator."""

import sys
from pathlib import Path

# Додаємо шлях до проєкту
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

# Імпортуємо напряму
from neuroquant.metrics import BDRateCalculator


class TestBDRateCalculator:
    """Тести для BD-Rate калькулятора."""

    def setup_method(self):
        """Ініціалізація."""
        self.calculator = BDRateCalculator()

    def test_identical_curves(self):
        """BD-Rate для ідентичних кривих має бути ~0."""
        rates = [100, 200, 400, 800]
        quality = [30.0, 33.0, 36.0, 39.0]

        bd_rate = self.calculator.compute_bd_rate(
            rates, quality, rates, quality
        )

        assert abs(bd_rate) < 0.1  # Має бути близько до 0

    def test_better_curve_negative_bd_rate(self):
        """Краща крива (вища якість при тому ж бітрейті) → від'ємний BD-Rate."""
        rates_anchor = [100, 200, 400, 800]
        quality_anchor = [30.0, 33.0, 36.0, 39.0]

        # Test система: та сама якість при меншому бітрейті
        rates_test = [80, 160, 320, 640]  # На 20% менше
        quality_test = [30.0, 33.0, 36.0, 39.0]

        bd_rate = self.calculator.compute_bd_rate(
            rates_anchor, quality_anchor,
            rates_test, quality_test
        )

        # BD-Rate має бути від'ємним (test краще)
        assert bd_rate < 0

    def test_worse_curve_positive_bd_rate(self):
        """Гірша крива → додатний BD-Rate."""
        rates_anchor = [100, 200, 400, 800]
        quality_anchor = [30.0, 33.0, 36.0, 39.0]

        # Test система: та сама якість при більшому бітрейті
        rates_test = [120, 240, 480, 960]  # На 20% більше
        quality_test = [30.0, 33.0, 36.0, 39.0]

        bd_rate = self.calculator.compute_bd_rate(
            rates_anchor, quality_anchor,
            rates_test, quality_test
        )

        # BD-Rate має бути додатним (test гірше)
        assert bd_rate > 0

    def test_bd_psnr(self):
        """Тест BD-PSNR обчислення."""
        rates_anchor = [100, 200, 400, 800]
        psnr_anchor = [30.0, 33.0, 36.0, 39.0]

        # Test система: краща якість при тому ж бітрейті
        rates_test = [100, 200, 400, 800]
        psnr_test = [31.0, 34.0, 37.0, 40.0]  # На 1 dB краще

        bd_psnr = self.calculator.compute_bd_psnr(
            rates_anchor, psnr_anchor,
            rates_test, psnr_test
        )

        # BD-PSNR має бути додатним (test краще)
        assert bd_psnr > 0
        assert abs(bd_psnr - 1.0) < 0.5  # Приблизно 1 dB

    def test_insufficient_overlap(self):
        """Тест при недостатньому перекритті кривих."""
        rates_anchor = [100, 200, 400, 800]
        quality_anchor = [30.0, 33.0, 36.0, 39.0]

        # Test система з зовсім іншим діапазоном якості
        rates_test = [100, 200, 400, 800]
        quality_test = [50.0, 53.0, 56.0, 59.0]  # Немає перекриття

        bd_rate = self.calculator.compute_bd_rate(
            rates_anchor, quality_anchor,
            rates_test, quality_test
        )

        # Має повернути 0 при недостатньому перекритті
        assert bd_rate == 0.0

    def test_typical_codec_comparison(self):
        """Тест типового порівняння кодеків."""
        # H.264 (anchor)
        h264_rates = [500, 1000, 2000, 4000]
        h264_vmaf = [70.0, 80.0, 88.0, 93.0]

        # HEVC (зазвичай ~30-40% ефективніший)
        hevc_rates = [350, 700, 1400, 2800]  # ~30% менше
        hevc_vmaf = [70.0, 80.0, 88.0, 93.0]

        bd_rate = self.calculator.compute_bd_rate(
            h264_rates, h264_vmaf,
            hevc_rates, hevc_vmaf
        )

        # HEVC має показати ~-30% BD-Rate
        assert -50 < bd_rate < -20

    def test_edge_cases(self):
        """Тест граничних випадків."""
        # Порожні списки
        bd_rate = self.calculator.compute_bd_rate([], [], [], [])
        # Має не впасти

        # Один елемент
        bd_rate = self.calculator.compute_bd_rate([100], [30], [100], [30])
        # Має не впасти
