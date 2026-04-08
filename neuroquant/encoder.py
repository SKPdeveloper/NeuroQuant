"""
FFmpegEncoder - обгортка для кодування відео через ffmpeg.

Підтримує:
- libx264 (H.264)
- libx265 (HEVC) з QP-планом
- libvvenc (VVC)
"""

import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import List, Optional
from enum import Enum

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .types import QPPlan, EncodingResult
from .utils import get_video_info, log_info, log_warning, log_error, get_temp_dir


class Codec(Enum):
    """Підтримувані кодеки."""
    H264 = "libx264"
    HEVC = "libx265"
    VVC = "libvvenc"


class FFmpegEncoder:
    """
    Кодувальник відео через ffmpeg.

    Підтримує кодування з QP-планом (для libx265)
    або стандартний ABR режим.
    """

    def __init__(
        self,
        codec: Codec = Codec.HEVC,
        preset: str = "medium",
        disable_aq: bool = True,
    ):
        """
        Ініціалізація кодувальника.

        Args:
            codec: Кодек для використання
            preset: Пресет швидкості/якості
            disable_aq: Вимкнути Adaptive Quantization (для точного QP-плану)
        """
        self.codec = codec
        self.preset = preset
        self.disable_aq = disable_aq

    def encode_with_qp_plan(
        self,
        input_path: str,
        output_path: str,
        qp_plan: List[QPPlan],
        target_bitrate: Optional[int] = None,
        show_progress: bool = True,
    ) -> EncodingResult:
        """
        Кодує відео з адаптивними параметрами на основі QP-плану.

        Режими роботи:
        1. Якщо target_bitrate задано → 2-pass ABR з адаптивним AQ
        2. Якщо target_bitrate не задано → CRF режим

        Args:
            input_path: Вхідний відеофайл
            output_path: Вихідний відеофайл
            qp_plan: QP-план для кожного кадру (для налаштування AQ)
            target_bitrate: Цільовий бітрейт (bps). Якщо задано, використовує 2-pass ABR
            show_progress: Показувати прогрес

        Returns:
            EncodingResult з інформацією про результат
        """
        if self.codec != Codec.HEVC:
            log_warning(f"Адаптивне кодування підтримується лише для libx265")
            if target_bitrate:
                return self.encode_abr(input_path, output_path, bitrate=target_bitrate)
            return self.encode_crf(input_path, output_path, crf=28)

        video_info = get_video_info(input_path)

        # Аналізуємо QP-план для налаштування AQ
        qp_values = [p.qp for p in qp_plan]
        avg_qp = sum(qp_values) / len(qp_values)
        qp_std = (sum((q - avg_qp) ** 2 for q in qp_values) / len(qp_values)) ** 0.5

        # Налаштування AQ на основі варіації складності
        aq_strength = min(2.0, max(0.5, 1.0 + qp_std / 10))

        # Якщо заданий target_bitrate — використовуємо 2-pass ABR
        if target_bitrate:
            return self._encode_adaptive_abr(
                input_path, output_path, target_bitrate,
                aq_strength, video_info, show_progress
            )

        # Інакше — CRF режим
        log_info(f"Кодування [{self.codec.value}]: {Path(input_path).name}")

        crf = int(round(avg_qp))
        crf = max(18, min(45, crf))

        x265_params_list = [
            f"aq-mode=3",
            f"aq-strength={aq_strength:.2f}",
            f"psy-rd=2.0",
            f"psy-rdoq=1.0",
        ]
        x265_params = ":".join(x265_params_list)
        log_info(f"CRF: {crf}, AQ strength: {aq_strength:.2f}")

        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-c:v", self.codec.value,
            "-preset", self.preset,
            "-crf", str(crf),
            "-x265-params", x265_params,
            "-an",
            output_path,
        ]

        return self._run_encode(cmd, output_path, video_info, show_progress)

    def _encode_adaptive_abr(
        self,
        input_path: str,
        output_path: str,
        target_bitrate: int,
        aq_strength: float,
        video_info: dict,
        show_progress: bool,
    ) -> EncodingResult:
        """
        2-pass ABR кодування з адаптивним AQ для точного контролю бітрейту.

        Pass 1: Аналіз складності (швидкий)
        Pass 2: Фінальне кодування з оптимальним розподілом бітів
        """
        temp_dir = get_temp_dir()
        # Використовуємо forward slashes для сумісності
        passlogfile = str(temp_dir / "x265_2pass").replace("\\", "/")
        bitrate_k = target_bitrate // 1000

        log_info(f"Кодування [{self.codec.value}]: {Path(input_path).name}")
        log_info(f"Режим: 2-pass ABR @ {bitrate_k}k, AQ strength: {aq_strength:.2f}")

        # x265-params для адаптивного кодування
        x265_params = ":".join([
            f"aq-mode=3",
            f"aq-strength={aq_strength:.2f}",
            f"psy-rd=2.0",
            f"psy-rdoq=1.0",
            f"vbv-maxrate={int(bitrate_k * 1.5)}",
            f"vbv-bufsize={int(bitrate_k * 2)}",
        ])

        null_output = "NUL" if os.name == "nt" else "/dev/null"

        # === Pass 1 ===
        log_info("Pass 1/2: Аналіз...")
        cmd_pass1 = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-c:v", self.codec.value,
            "-preset", self.preset,
            "-b:v", f"{bitrate_k}k",
            "-x265-params", x265_params,
            "-pass", "1",
            "-passlogfile", passlogfile,
            "-an",
            "-f", "null",
            null_output,
        ]

        result = subprocess.run(cmd_pass1, capture_output=True, text=True)
        if result.returncode != 0:
            return EncodingResult(
                output_path=output_path,
                actual_bitrate=0,
                file_size=0,
                duration=0,
                codec=self.codec.value,
                success=False,
                error_message=f"Pass 1 помилка: {result.stderr[-500:]}",
            )

        # === Pass 2 ===
        log_info("Pass 2/2: Кодування...")
        cmd_pass2 = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-c:v", self.codec.value,
            "-preset", self.preset,
            "-b:v", f"{bitrate_k}k",
            "-x265-params", x265_params,
            "-pass", "2",
            "-passlogfile", passlogfile,
            "-an",
            output_path,
        ]

        return self._run_encode(cmd_pass2, output_path, video_info, show_progress)

    def encode_abr(
        self,
        input_path: str,
        output_path: str,
        bitrate: int,
        two_pass: bool = True,
        show_progress: bool = True,
    ) -> EncodingResult:
        """
        Кодує відео в режимі ABR (Average Bitrate).

        Args:
            input_path: Вхідний відеофайл
            output_path: Вихідний відеофайл
            bitrate: Цільовий бітрейт (bps)
            two_pass: Використовувати двопрохідне кодування
            show_progress: Показувати прогрес

        Returns:
            EncodingResult з інформацією про результат
        """
        video_info = get_video_info(input_path)
        bitrate_k = bitrate // 1000

        log_info(f"Кодування [{self.codec.value}]: {Path(input_path).name} @ {bitrate_k}k")

        if two_pass and self.codec != Codec.VVC:
            return self._encode_two_pass(input_path, output_path, bitrate, video_info, show_progress)
        else:
            return self._encode_single_pass(input_path, output_path, bitrate, video_info, show_progress)

    def _encode_two_pass(
        self,
        input_path: str,
        output_path: str,
        bitrate: int,
        video_info: dict,
        show_progress: bool,
    ) -> EncodingResult:
        """Двопрохідне кодування для кращого контролю бітрейту."""
        temp_dir = get_temp_dir()
        passlogfile = temp_dir / "ffmpeg2pass"

        bitrate_str = f"{bitrate // 1000}k"

        # Перший прохід
        log_info("Перший прохід...")
        cmd_pass1 = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-c:v", self.codec.value,
            "-preset", self.preset,
            "-b:v", bitrate_str,
            "-pass", "1",
            "-passlogfile", str(passlogfile),
            "-an",
            "-f", "null",
            "NUL" if os.name == "nt" else "/dev/null",
        ]

        result = subprocess.run(cmd_pass1, capture_output=True, text=True)
        if result.returncode != 0:
            return EncodingResult(
                output_path=output_path,
                actual_bitrate=0,
                file_size=0,
                duration=0,
                codec=self.codec.value,
                success=False,
                error_message=f"Помилка першого проходу: {result.stderr[-500:]}",
            )

        # Другий прохід
        log_info("Другий прохід...")
        cmd_pass2 = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-c:v", self.codec.value,
            "-preset", self.preset,
            "-b:v", bitrate_str,
            "-pass", "2",
            "-passlogfile", str(passlogfile),
            "-an",
            output_path,
        ]

        return self._run_encode(cmd_pass2, output_path, video_info, show_progress)

    def _encode_single_pass(
        self,
        input_path: str,
        output_path: str,
        bitrate: int,
        video_info: dict,
        show_progress: bool,
    ) -> EncodingResult:
        """Однопрохідне кодування."""
        bitrate_str = f"{bitrate // 1000}k"

        if self.codec == Codec.VVC:
            # libvvenc має інший синтаксис
            cmd = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-c:v", self.codec.value,
                "-preset", self.preset,
                "-b:v", bitrate_str,
                "-an",
                output_path,
            ]
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-c:v", self.codec.value,
                "-preset", self.preset,
                "-b:v", bitrate_str,
                "-an",
                output_path,
            ]

        return self._run_encode(cmd, output_path, video_info, show_progress)

    def _run_encode(
        self,
        cmd: List[str],
        output_path: str,
        video_info: dict,
        show_progress: bool,
    ) -> EncodingResult:
        """Виконує команду кодування з відображенням прогресу."""
        duration = video_info["duration"]
        stderr_output = []

        # Додаємо параметри для прогресу
        cmd_with_progress = cmd.copy()
        progress_idx = cmd_with_progress.index("-y") + 1
        cmd_with_progress.insert(progress_idx, "-progress")
        cmd_with_progress.insert(progress_idx + 1, "pipe:1")
        cmd_with_progress.insert(progress_idx + 2, "-stats_period")
        cmd_with_progress.insert(progress_idx + 3, "0.5")

        process = subprocess.Popen(
            cmd_with_progress,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Читаємо stderr в окремому потоці, щоб уникнути deadlock
        def read_stderr():
            for line in process.stderr:
                stderr_output.append(line)

        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()

        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]Кодування [{self.codec.value}][/cyan]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("| ETA:"),
            TimeElapsedColumn(),
            disable=not show_progress,
        )

        with progress_ctx as progress:
            task = progress.add_task("Кодування", total=duration)

            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break

                if line.startswith("out_time_ms="):
                    try:
                        time_ms = int(line.split("=")[1].strip())
                        current_time = time_ms / 1_000_000
                        progress.update(task, completed=min(current_time, duration))
                    except ValueError:
                        pass

            progress.update(task, completed=duration)

        # Чекаємо завершення потоку stderr
        stderr_thread.join(timeout=5.0)
        process.wait()

        stderr = "".join(stderr_output)

        if process.returncode != 0:
            return EncodingResult(
                output_path=output_path,
                actual_bitrate=0,
                file_size=0,
                duration=0,
                codec=self.codec.value,
                success=False,
                error_message=f"Помилка кодування: {stderr[-500:] if stderr else 'Невідома помилка'}",
            )

        # Отримуємо інформацію про результат
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            if file_size == 0:
                return EncodingResult(
                    output_path=output_path,
                    actual_bitrate=0,
                    file_size=0,
                    duration=0,
                    codec=self.codec.value,
                    success=False,
                    error_message="Вихідний файл порожній",
                )

            output_info = get_video_info(output_path)

            return EncodingResult(
                output_path=output_path,
                actual_bitrate=output_info.get("bitrate", 0),
                file_size=file_size,
                duration=output_info.get("duration", 0),
                codec=self.codec.value,
                success=True,
            )
        else:
            return EncodingResult(
                output_path=output_path,
                actual_bitrate=0,
                file_size=0,
                duration=0,
                codec=self.codec.value,
                success=False,
                error_message="Вихідний файл не створено",
            )

    def encode_crf(
        self,
        input_path: str,
        output_path: str,
        crf: int,
        show_progress: bool = True,
    ) -> EncodingResult:
        """
        Кодує відео в режимі CRF (Constant Rate Factor).

        Args:
            input_path: Вхідний відеофайл
            output_path: Вихідний відеофайл
            crf: Значення CRF (18-28 для хорошої якості)
            show_progress: Показувати прогрес

        Returns:
            EncodingResult з інформацією про результат
        """
        video_info = get_video_info(input_path)
        log_info(f"Кодування [{self.codec.value}] CRF={crf}: {Path(input_path).name}")

        if self.codec == Codec.VVC:
            # VVC використовує QP замість CRF
            cmd = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-c:v", self.codec.value,
                "-preset", self.preset,
                "-qp", str(crf),
                "-an",
                output_path,
            ]
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-c:v", self.codec.value,
                "-preset", self.preset,
                "-crf", str(crf),
                "-an",
                output_path,
            ]

        return self._run_encode(cmd, output_path, video_info, show_progress)


def get_encoder(method: str) -> FFmpegEncoder:
    """
    Створює кодувальник для вказаного методу.

    Args:
        method: Один з "h264", "hevc", "vvc", "nq", "nq_sr"

    Returns:
        Налаштований FFmpegEncoder
    """
    codec_map = {
        "h264": Codec.H264,
        "hevc": Codec.HEVC,
        "vvc": Codec.VVC,
        "nq": Codec.HEVC,
        "nq_sr": Codec.HEVC,
    }

    codec = codec_map.get(method.lower(), Codec.HEVC)
    return FFmpegEncoder(codec=codec)
