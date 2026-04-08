"""Тести для утиліт."""

import sys
from pathlib import Path

# Додаємо шлях до проєкту
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from neuroquant.utils import parse_bitrate, format_bitrate, format_time


class TestBitrateUtils:
    """Тести для функцій роботи з бітрейтом."""

    def test_parse_bitrate_megabits(self):
        """Парсинг мегабіт."""
        assert parse_bitrate("1M") == 1_000_000
        assert parse_bitrate("1.5M") == 1_500_000
        assert parse_bitrate("2m") == 2_000_000

    def test_parse_bitrate_kilobits(self):
        """Парсинг кілобіт."""
        assert parse_bitrate("500k") == 500_000
        assert parse_bitrate("500K") == 500_000
        assert parse_bitrate("1500k") == 1_500_000

    def test_parse_bitrate_raw(self):
        """Парсинг числового значення."""
        assert parse_bitrate("1000000") == 1_000_000
        assert parse_bitrate("500000") == 500_000

    def test_format_bitrate_megabits(self):
        """Форматування мегабіт."""
        assert format_bitrate(1_000_000) == "1.0M"
        assert format_bitrate(1_500_000) == "1.5M"
        assert format_bitrate(2_000_000) == "2.0M"

    def test_format_bitrate_kilobits(self):
        """Форматування кілобіт."""
        assert format_bitrate(500_000) == "500k"
        assert format_bitrate(300_000) == "300k"

    def test_format_bitrate_bits(self):
        """Форматування малих значень."""
        assert format_bitrate(500) == "500"


class TestTimeUtils:
    """Тести для функцій роботи з часом."""

    def test_format_time_seconds(self):
        """Форматування секунд."""
        assert format_time(30.5) == "30.5сек"
        assert format_time(5.0) == "5.0сек"

    def test_format_time_minutes(self):
        """Форматування хвилин."""
        assert format_time(90) == "1хв 30сек"
        assert format_time(120) == "2хв 0сек"
        assert format_time(3599) == "59хв 59сек"

    def test_format_time_hours(self):
        """Форматування годин."""
        assert format_time(3600) == "1год 0хв"
        assert format_time(3660) == "1год 1хв"
        assert format_time(7200) == "2год 0хв"
