"""
MetricsCollector - збір та обчислення метрик якості відео.

Підтримує:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- VMAF (Video Multi-Method Assessment Fusion)
- BD-Rate (Bjøntegaard Delta Rate)
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field

import numpy as np
from scipy import interpolate

from .utils import get_video_info, log_info, log_warning, get_temp_dir


@dataclass
class FrameMetrics:
    """Метрики для одного кадру."""
    frame_idx: int
    psnr: float
    ssim: float
    vmaf: float


@dataclass
class VideoMetrics:
    """Агреговані метрики для відео."""
    psnr_mean: float
    psnr_std: float
    ssim_mean: float
    ssim_std: float
    vmaf_mean: float
    vmaf_std: float
    bitrate: int
    file_size: int
    frame_metrics: List[FrameMetrics] = field(default_factory=list)


class MetricsCollector:
    """
    Збирач метрик якості відео.

    Обчислює PSNR, SSIM, VMAF для порівняння
    закодованого відео з оригіналом.
    """

    def __init__(self, n_threads: int = 4):
        """
        Ініціалізація колектора.

        Args:
            n_threads: Кількість потоків для обчислення
        """
        self.n_threads = n_threads

    def compute_metrics(
        self,
        distorted_path: str,
        reference_path: str,
        compute_per_frame: bool = False,
    ) -> VideoMetrics:
        """
        Обчислює всі метрики для відео.

        Args:
            distorted_path: Шлях до закодованого відео
            reference_path: Шлях до оригінального відео
            compute_per_frame: Обчислювати per-frame метрики

        Returns:
            VideoMetrics з агрегованими та per-frame метриками
        """
        log_info(f"Обчислення метрик: {Path(distorted_path).name}")

        # Отримуємо базову інформацію
        dist_info = get_video_info(distorted_path)
        bitrate = dist_info.get("bitrate", 0)
        file_size = Path(distorted_path).stat().st_size

        # Обчислюємо PSNR та SSIM через ffmpeg
        psnr_mean, psnr_std, ssim_mean, ssim_std = self._compute_psnr_ssim(
            distorted_path, reference_path
        )

        # Обчислюємо VMAF
        vmaf_mean, vmaf_std, vmaf_per_frame = self._compute_vmaf(
            distorted_path, reference_path, per_frame=compute_per_frame
        )

        # Формуємо per-frame метрики, якщо потрібно
        frame_metrics = []
        if compute_per_frame and vmaf_per_frame:
            for i, vmaf in enumerate(vmaf_per_frame):
                frame_metrics.append(FrameMetrics(
                    frame_idx=i,
                    psnr=0.0,  # ffmpeg не дає per-frame PSNR легко
                    ssim=0.0,
                    vmaf=vmaf,
                ))

        log_info(f"PSNR: {psnr_mean:.2f} dB | SSIM: {ssim_mean:.4f} | VMAF: {vmaf_mean:.2f}")

        return VideoMetrics(
            psnr_mean=psnr_mean,
            psnr_std=psnr_std,
            ssim_mean=ssim_mean,
            ssim_std=ssim_std,
            vmaf_mean=vmaf_mean,
            vmaf_std=vmaf_std,
            bitrate=bitrate,
            file_size=file_size,
            frame_metrics=frame_metrics,
        )

    def _compute_psnr_ssim(
        self,
        distorted_path: str,
        reference_path: str,
    ) -> Tuple[float, float, float, float]:
        """
        Обчислює PSNR та SSIM через ffmpeg.

        Returns:
            (psnr_mean, psnr_std, ssim_mean, ssim_std)
        """
        temp_dir = get_temp_dir()
        psnr_log = temp_dir / "psnr_log.txt"
        ssim_log = temp_dir / "ssim_log.txt"

        # PSNR
        cmd_psnr = [
            "ffmpeg",
            "-i", distorted_path,
            "-i", reference_path,
            "-lavfi", f"psnr=stats_file={psnr_log}",
            "-f", "null",
            "NUL" if os.name == "nt" else "/dev/null",
        ]

        result = subprocess.run(cmd_psnr, capture_output=True, text=True)

        psnr_mean, psnr_std = 0.0, 0.0
        # Парсимо вивід ffmpeg для середнього PSNR
        for line in result.stderr.split("\n"):
            if "average:" in line.lower():
                parts = line.split()
                for i, p in enumerate(parts):
                    if p.lower().startswith("average:"):
                        try:
                            psnr_mean = float(parts[i].split(":")[1])
                        except (ValueError, IndexError):
                            pass
            if "PSNR" in line and "average" in line:
                try:
                    # Формат: [Parsed_psnr_0 @ ...] PSNR y:XX.XX u:XX.XX v:XX.XX average:XX.XX
                    if "average:" in line:
                        psnr_mean = float(line.split("average:")[1].split()[0])
                except (ValueError, IndexError):
                    pass

        # SSIM
        cmd_ssim = [
            "ffmpeg",
            "-i", distorted_path,
            "-i", reference_path,
            "-lavfi", f"ssim=stats_file={ssim_log}",
            "-f", "null",
            "NUL" if os.name == "nt" else "/dev/null",
        ]

        result = subprocess.run(cmd_ssim, capture_output=True, text=True)

        ssim_mean, ssim_std = 0.0, 0.0
        for line in result.stderr.split("\n"):
            if "All:" in line:
                try:
                    # Формат: [Parsed_ssim_0 @ ...] SSIM Y:X.XXX ... All:X.XXX
                    ssim_mean = float(line.split("All:")[1].split()[0])
                except (ValueError, IndexError):
                    pass

        return psnr_mean, psnr_std, ssim_mean, ssim_std

    def _compute_vmaf(
        self,
        distorted_path: str,
        reference_path: str,
        per_frame: bool = False,
    ) -> Tuple[float, float, List[float]]:
        """
        Обчислює VMAF через ffmpeg libvmaf.

        Returns:
            (vmaf_mean, vmaf_std, per_frame_scores)
        """
        temp_dir = get_temp_dir()
        vmaf_log = temp_dir / "vmaf_log.json"

        log_fmt = "json"  # JSON потрібен для обох випадків (per-frame та aggregated)
        cmd = [
            "ffmpeg",
            "-i", distorted_path,
            "-i", reference_path,
            "-lavfi", f"libvmaf=log_path={vmaf_log}:log_fmt={log_fmt}:n_threads={self.n_threads}",
            "-f", "null",
            "NUL" if os.name == "nt" else "/dev/null",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            log_warning(f"VMAF помилка: {result.stderr[-300:]}")
            return 0.0, 0.0, []

        # Парсимо JSON
        import json
        try:
            with open(vmaf_log, "r") as f:
                vmaf_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return 0.0, 0.0, []

        # Агреговані метрики
        pooled = vmaf_data.get("pooled_metrics", {})
        vmaf_mean = pooled.get("vmaf", {}).get("mean", 0.0)
        vmaf_std = pooled.get("vmaf", {}).get("stdev", 0.0)  # Стандартне відхилення

        # Per-frame
        per_frame_scores = []
        if per_frame:
            for frame in vmaf_data.get("frames", []):
                metrics = frame.get("metrics", {})
                per_frame_scores.append(metrics.get("vmaf", 0.0))

        return vmaf_mean, vmaf_std, per_frame_scores

    def compute_vmaf_only(
        self,
        distorted_path: str,
        reference_path: str,
    ) -> float:
        """Обчислює лише середній VMAF."""
        vmaf_mean, _, _ = self._compute_vmaf(distorted_path, reference_path)
        return vmaf_mean


class BDRateCalculator:
    """
    Калькулятор BD-Rate (Bjøntegaard Delta Rate).

    BD-Rate показує різницю в бітрейті між двома RD-кривими
    при еквівалентній якості.
    """

    @staticmethod
    def compute_bd_rate(
        rate_anchor: List[float],
        quality_anchor: List[float],
        rate_test: List[float],
        quality_test: List[float],
    ) -> float:
        """
        Обчислює BD-Rate між двома RD-кривими.

        Args:
            rate_anchor: Бітрейти anchor системи
            quality_anchor: Якість (PSNR/VMAF) anchor системи
            rate_test: Бітрейти test системи
            quality_test: Якість test системи

        Returns:
            BD-Rate у відсотках (від'ємне = test краще)
        """
        # Перевірка на порожні списки
        if not rate_anchor or not rate_test or not quality_anchor or not quality_test:
            return 0.0

        if len(rate_anchor) < 2 or len(rate_test) < 2:
            return 0.0

        # Перетворюємо в log scale для бітрейту
        log_rate_anchor = np.log10(np.array(rate_anchor))
        log_rate_test = np.log10(np.array(rate_test))
        quality_anchor = np.array(quality_anchor)
        quality_test = np.array(quality_test)

        # Сортуємо за якістю
        idx_anchor = np.argsort(quality_anchor)
        idx_test = np.argsort(quality_test)

        log_rate_anchor = log_rate_anchor[idx_anchor]
        quality_anchor = quality_anchor[idx_anchor]
        log_rate_test = log_rate_test[idx_test]
        quality_test = quality_test[idx_test]

        # Знаходимо діапазон перекриття якості
        min_q = max(quality_anchor.min(), quality_test.min())
        max_q = min(quality_anchor.max(), quality_test.max())

        if min_q >= max_q:
            log_warning("Недостатнє перекриття RD-кривих для BD-Rate")
            return 0.0

        # Cubic spline інтерполяція
        try:
            spline_anchor = interpolate.PchipInterpolator(quality_anchor, log_rate_anchor)
            spline_test = interpolate.PchipInterpolator(quality_test, log_rate_test)
        except ValueError as e:
            log_warning(f"Помилка інтерполяції: {e}")
            return 0.0

        # Інтегруємо різницю
        n_points = 100
        q_range = np.linspace(min_q, max_q, n_points)

        integral_anchor = np.trapezoid(spline_anchor(q_range), q_range)
        integral_test = np.trapezoid(spline_test(q_range), q_range)

        # BD-Rate
        avg_diff = (integral_test - integral_anchor) / (max_q - min_q)
        bd_rate = (np.power(10, avg_diff) - 1) * 100

        return bd_rate

    @staticmethod
    def compute_bd_psnr(
        rate_anchor: List[float],
        psnr_anchor: List[float],
        rate_test: List[float],
        psnr_test: List[float],
    ) -> float:
        """
        Обчислює BD-PSNR між двома RD-кривими.

        Returns:
            BD-PSNR у dB (додатне = test краще)
        """
        # Подібно до BD-Rate, але інтегруємо по log(rate)
        log_rate_anchor = np.log10(np.array(rate_anchor))
        log_rate_test = np.log10(np.array(rate_test))
        psnr_anchor = np.array(psnr_anchor)
        psnr_test = np.array(psnr_test)

        # Сортуємо за бітрейтом
        idx_anchor = np.argsort(log_rate_anchor)
        idx_test = np.argsort(log_rate_test)

        log_rate_anchor = log_rate_anchor[idx_anchor]
        psnr_anchor = psnr_anchor[idx_anchor]
        log_rate_test = log_rate_test[idx_test]
        psnr_test = psnr_test[idx_test]

        # Діапазон перекриття
        min_r = max(log_rate_anchor.min(), log_rate_test.min())
        max_r = min(log_rate_anchor.max(), log_rate_test.max())

        if min_r >= max_r:
            return 0.0

        try:
            spline_anchor = interpolate.PchipInterpolator(log_rate_anchor, psnr_anchor)
            spline_test = interpolate.PchipInterpolator(log_rate_test, psnr_test)
        except ValueError:
            return 0.0

        n_points = 100
        r_range = np.linspace(min_r, max_r, n_points)

        integral_anchor = np.trapezoid(spline_anchor(r_range), r_range)
        integral_test = np.trapezoid(spline_test(r_range), r_range)

        bd_psnr = (integral_test - integral_anchor) / (max_r - min_r)

        return bd_psnr


def compute_rd_metrics(
    results: Dict[str, List[VideoMetrics]],
    anchor_method: str = "h264",
) -> Dict[str, Dict[str, float]]:
    """
    Обчислює BD-Rate для всіх методів відносно anchor.

    Args:
        results: {method: [VideoMetrics for each bitrate]}
        anchor_method: Базовий метод для порівняння

    Returns:
        {method: {"bd_rate_psnr": X, "bd_rate_vmaf": Y}}
    """
    if anchor_method not in results:
        log_warning(f"Anchor метод {anchor_method} не знайдено")
        return {}

    anchor_metrics = results[anchor_method]
    anchor_rates = [m.bitrate for m in anchor_metrics]
    anchor_psnr = [m.psnr_mean for m in anchor_metrics]
    anchor_vmaf = [m.vmaf_mean for m in anchor_metrics]

    bd_calculator = BDRateCalculator()
    rd_metrics = {}

    for method, metrics_list in results.items():
        if method == anchor_method:
            rd_metrics[method] = {"bd_rate_psnr": 0.0, "bd_rate_vmaf": 0.0}
            continue

        test_rates = [m.bitrate for m in metrics_list]
        test_psnr = [m.psnr_mean for m in metrics_list]
        test_vmaf = [m.vmaf_mean for m in metrics_list]

        bd_rate_psnr = bd_calculator.compute_bd_rate(
            anchor_rates, anchor_psnr, test_rates, test_psnr
        )
        bd_rate_vmaf = bd_calculator.compute_bd_rate(
            anchor_rates, anchor_vmaf, test_rates, test_vmaf
        )

        rd_metrics[method] = {
            "bd_rate_psnr": bd_rate_psnr,
            "bd_rate_vmaf": bd_rate_vmaf,
        }

    return rd_metrics
