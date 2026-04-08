"""
Допоміжні функції для NeuroQuant.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import yaml
from rich.console import Console

console = Console()


def load_config(config_path: Optional[str] = None) -> dict:
    """Завантажує конфігурацію з YAML файлу."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_video_info(video_path: str) -> dict:
    """
    Отримує інформацію про відео через ffprobe.

    Повертає:
        dict з ключами: width, height, fps, duration, frame_count, codec
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    import json
    data = json.loads(result.stdout)

    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise ValueError(f"Відео потік не знайдено у {video_path}")

    # Парсимо fps
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den > 0 else 30.0
    else:
        fps = float(fps_str)

    # Тривалість
    duration = float(data.get("format", {}).get("duration", 0))

    # Кількість кадрів
    frame_count = int(video_stream.get("nb_frames", 0))
    if frame_count == 0:
        frame_count = int(duration * fps)

    return {
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": fps,
        "duration": duration,
        "frame_count": frame_count,
        "codec": video_stream.get("codec_name", "unknown"),
        "bitrate": int(data.get("format", {}).get("bit_rate", 0)),
    }


def parse_bitrate(bitrate_str: str) -> int:
    """
    Парсить рядок бітрейту у біти/секунду.

    Приклади:
        "1M" -> 1_000_000
        "500k" -> 500_000
        "1500000" -> 1_500_000
    """
    bitrate_str = bitrate_str.strip().upper()

    if bitrate_str.endswith("M"):
        return int(float(bitrate_str[:-1]) * 1_000_000)
    elif bitrate_str.endswith("K"):
        return int(float(bitrate_str[:-1]) * 1_000)
    else:
        return int(bitrate_str)


def format_bitrate(bitrate: int) -> str:
    """Форматує бітрейт для відображення."""
    if bitrate >= 1_000_000:
        return f"{bitrate / 1_000_000:.1f}M"
    elif bitrate >= 1_000:
        return f"{bitrate / 1_000:.0f}k"
    else:
        return f"{bitrate}"


def format_time(seconds: float) -> str:
    """Форматує час у хв:сек або год:хв:сек."""
    if seconds < 60:
        return f"{seconds:.1f}сек"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}хв {secs}сек"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}год {mins}хв"


def ensure_dir(path: str) -> Path:
    """Створює директорію, якщо не існує."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_temp_dir() -> Path:
    """Повертає тимчасову директорію для проміжних файлів."""
    temp_dir = Path(tempfile.gettempdir()) / "neuroquant"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def check_ffmpeg() -> Tuple[bool, str]:
    """
    Перевіряє наявність ffmpeg та необхідних кодеків.

    Повертає:
        (success, message)
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True
        )
        version_line = result.stdout.split("\n")[0]

        # Перевіряємо кодеки
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True,
            text=True,
            check=True
        )
        encoders = result.stdout

        missing = []
        for codec in ["libx264", "libx265"]:
            if codec not in encoders:
                missing.append(codec)

        if missing:
            return False, f"Відсутні кодеки: {', '.join(missing)}"

        return True, version_line

    except FileNotFoundError:
        return False, "ffmpeg не знайдено. Встановіть ffmpeg."
    except subprocess.CalledProcessError as e:
        return False, f"Помилка ffmpeg: {e}"


def check_cuda() -> Tuple[bool, str]:
    """
    Перевіряє наявність CUDA для GPU-прискорення.

    Повертає:
        (available, device_info)
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return True, f"{device_name} ({vram:.1f} GB VRAM)"
        else:
            return False, "CUDA not available"
    except ImportError:
        return False, "PyTorch not installed"


def log_info(message: str) -> None:
    """Виводить інформаційне повідомлення."""
    console.print(f"[cyan][NeuroQuant][/cyan] {message}")


def log_success(message: str) -> None:
    """Виводить успішне повідомлення."""
    console.print(f"[green][NeuroQuant][/green] {message}")


def log_warning(message: str) -> None:
    """Виводить попередження."""
    console.print(f"[yellow][NeuroQuant][/yellow] {message}")


def log_error(message: str) -> None:
    """Виводить помилку."""
    console.print(f"[red][NeuroQuant][/red] {message}")
