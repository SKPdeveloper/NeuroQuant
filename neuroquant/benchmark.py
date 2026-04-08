"""
BenchmarkEngine - система порівняльного тестування.

Порівнює методи:
- h264: H.264 ABR (libx264)
- hevc: HEVC ABR (libx265)
- vvc: VVC ABR (libvvenc)
- nq: NeuroQuant (libx265 + R-λ QP план)
- nq_sr: NeuroQuant + Real-ESRGAN SR
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.console import Console

from .analyzer import ComplexityAnalyzer
from .controller import RLambdaController
from .encoder import FFmpegEncoder, Codec, get_encoder
from .sr_processor import SRPostProcessor
from .metrics import MetricsCollector, VideoMetrics, compute_rd_metrics
from .utils import (
    get_video_info, load_config, ensure_dir, log_info, log_warning,
    log_success, log_error, format_bitrate, get_temp_dir
)


@dataclass
class BenchmarkResult:
    """Результат бенчмарку для одного методу на одному бітрейті."""
    method: str
    bitrate_target: int
    bitrate_actual: int
    psnr: float
    ssim: float
    vmaf: float
    file_size: int
    encode_time: float
    output_path: str


@dataclass
class VideoBenchmark:
    """Результати бенчмарку для одного відео."""
    video_name: str
    video_path: str
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    results: List[BenchmarkResult] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Повний звіт бенчмарку."""
    timestamp: str
    methods: List[str]
    bitrates: List[int]
    videos: List[VideoBenchmark] = field(default_factory=list)
    bd_rates: Dict[str, Dict[str, float]] = field(default_factory=dict)


class BenchmarkEngine:
    """
    Двигун порівняльного тестування.

    Виконує кодування тестових відео всіма методами
    на всіх бітрейтах та збирає метрики.
    """

    def __init__(
        self,
        methods: List[str] = None,
        bitrates: List[int] = None,
        config_path: Optional[str] = None,
    ):
        """
        Ініціалізація двигуна.

        Args:
            methods: Методи для тестування
            bitrates: Бітрейти для тестування (bps)
            config_path: Шлях до конфігурації
        """
        config = load_config(config_path)

        self.methods = methods or config.get("benchmark", {}).get(
            "methods", ["h264", "hevc", "nq"]
        )
        self.bitrates = bitrates or config.get("benchmark", {}).get(
            "bitrates", [300000, 600000, 1200000]
        )

        # Ініціалізуємо компоненти
        self.analyzer = ComplexityAnalyzer(
            spatial_weight=config.get("complexity", {}).get("spatial_weight", 0.4),
            temporal_weight=config.get("complexity", {}).get("temporal_weight", 0.5),
            cut_weight=config.get("complexity", {}).get("cut_weight", 0.1),
            scene_threshold=config.get("complexity", {}).get("scene_threshold", 27.0),
            analysis_scale=config.get("complexity", {}).get("analysis_scale", 0.5),
        )

        self.controller = RLambdaController(
            qp_min=config.get("rate_control", {}).get("qp_min", 10),
            qp_max=config.get("rate_control", {}).get("qp_max", 51),
            delta_max=config.get("rate_control", {}).get("delta_max", 8),
            i_frame_bonus=config.get("rate_control", {}).get("i_frame_bonus", 4),
            gop_seconds=config.get("rate_control", {}).get("gop_seconds", 2.0),
        )

        self.sr_processor = SRPostProcessor(
            vmaf_threshold=config.get("sr", {}).get("vmaf_threshold", 70.0),
            model_name=config.get("sr", {}).get("model", "RealESRNet_x2plus"),
            tile_size=config.get("sr", {}).get("tile_size", 512),
        )

        self.metrics_collector = MetricsCollector(
            n_threads=config.get("metrics", {}).get("n_threads", 4)
        )

        self.console = Console()

    def run(
        self,
        video_paths: List[str],
        output_dir: str,
        show_progress: bool = True,
    ) -> BenchmarkReport:
        """
        Запускає повний бенчмарк.

        Args:
            video_paths: Список відео для тестування
            output_dir: Директорія для результатів
            show_progress: Показувати прогрес

        Returns:
            BenchmarkReport з усіма результатами
        """
        output_dir = ensure_dir(output_dir)

        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            methods=self.methods,
            bitrates=self.bitrates,
        )

        total_tasks = len(video_paths) * len(self.methods) * len(self.bitrates)
        log_info(f"Початок бенчмарку: {len(video_paths)} відео x {len(self.methods)} методів x {len(self.bitrates)} бітрейтів = {total_tasks} задач")

        for video_path in video_paths:
            video_benchmark = self._benchmark_video(
                video_path, output_dir, show_progress
            )
            report.videos.append(video_benchmark)

        # Обчислюємо BD-Rate
        report.bd_rates = self._compute_bd_rates(report)

        # Зберігаємо звіт
        report_path = output_dir / "benchmark_report.json"
        self._save_report(report, str(report_path))

        # Виводимо таблицю результатів
        self._print_summary(report)

        return report

    def _benchmark_video(
        self,
        video_path: str,
        output_dir: Path,
        show_progress: bool,
    ) -> VideoBenchmark:
        """Виконує бенчмарк для одного відео."""
        video_path = Path(video_path)
        video_info = get_video_info(str(video_path))

        log_info(f"\n{'='*60}")
        log_info(f"Відео: {video_path.name}")
        log_info(f"Роздільність: {video_info['width']}×{video_info['height']} @ {video_info['fps']:.0f}fps")
        log_info(f"Тривалість: {video_info['duration']:.1f} сек ({video_info['frame_count']} кадрів)")
        log_info(f"{'='*60}")

        benchmark = VideoBenchmark(
            video_name=video_path.stem,
            video_path=str(video_path),
            width=video_info["width"],
            height=video_info["height"],
            fps=video_info["fps"],
            duration=video_info["duration"],
            frame_count=video_info["frame_count"],
        )

        # Попередній аналіз для NeuroQuant методів
        complexity_data = None
        if any(m in self.methods for m in ["nq", "nq_sr"]):
            complexity_data = self.analyzer.analyze(str(video_path), show_progress)

        # Тестуємо кожен метод на кожному бітрейті
        video_output_dir = output_dir / video_path.stem
        video_output_dir.mkdir(parents=True, exist_ok=True)

        for method in self.methods:
            for bitrate in self.bitrates:
                result = self._encode_and_measure(
                    video_path=str(video_path),
                    method=method,
                    bitrate=bitrate,
                    output_dir=video_output_dir,
                    video_info=video_info,
                    complexity_data=complexity_data,
                    show_progress=show_progress,
                )
                benchmark.results.append(result)

        return benchmark

    def _encode_and_measure(
        self,
        video_path: str,
        method: str,
        bitrate: int,
        output_dir: Path,
        video_info: dict,
        complexity_data,
        show_progress: bool,
    ) -> BenchmarkResult:
        """Кодує відео та вимірює метрики."""
        bitrate_str = format_bitrate(bitrate)
        output_name = f"{Path(video_path).stem}_{method}_{bitrate_str}.mp4"
        output_path = output_dir / output_name

        log_info(f"\n[{method.upper()}] @ {bitrate_str}")

        start_time = time.time()

        try:
            if method in ["h264", "hevc", "vvc"]:
                # Стандартне ABR кодування
                encoder = get_encoder(method)
                encode_result = encoder.encode_abr(
                    video_path, str(output_path), bitrate, show_progress=show_progress
                )
            elif method == "nq":
                # NeuroQuant без SR
                encode_result = self._encode_neuroquant(
                    video_path, str(output_path), bitrate,
                    video_info, complexity_data, show_progress
                )
            elif method == "nq_sr":
                # NeuroQuant з SR
                temp_output = output_dir / f"temp_{output_name}"
                encode_result = self._encode_neuroquant(
                    video_path, str(temp_output), bitrate,
                    video_info, complexity_data, show_progress
                )

                if encode_result.success:
                    # Застосовуємо SR
                    sr_result = self.sr_processor.process_video(
                        str(temp_output), video_path, str(output_path),
                        show_progress=show_progress
                    )
                    # Видаляємо тимчасовий файл
                    temp_output.unlink(missing_ok=True)
            else:
                log_warning(f"Невідомий метод: {method}")
                return self._empty_result(method, bitrate, str(output_path))

            encode_time = time.time() - start_time

            if not encode_result.success:
                log_error(f"Помилка кодування: {encode_result.error_message}")
                return self._empty_result(method, bitrate, str(output_path))

            # Вимірюємо метрики
            metrics = self.metrics_collector.compute_metrics(
                str(output_path), video_path
            )

            return BenchmarkResult(
                method=method,
                bitrate_target=bitrate,
                bitrate_actual=metrics.bitrate,
                psnr=metrics.psnr_mean,
                ssim=metrics.ssim_mean,
                vmaf=metrics.vmaf_mean,
                file_size=metrics.file_size,
                encode_time=encode_time,
                output_path=str(output_path),
            )

        except Exception as e:
            log_error(f"Виняток: {e}")
            return self._empty_result(method, bitrate, str(output_path))

    def _encode_neuroquant(
        self,
        video_path: str,
        output_path: str,
        bitrate: int,
        video_info: dict,
        complexity_data,
        show_progress: bool,
    ):
        """Кодування методом NeuroQuant."""
        # Генеруємо QP-план
        qp_plan = self.controller.generate_qp_plan(
            complexity_data,
            target_bitrate=bitrate,
            fps=video_info["fps"],
            width=video_info["width"],
            height=video_info["height"],
        )

        # Кодуємо з QP-планом та цільовим бітрейтом (2-pass ABR)
        encoder = FFmpegEncoder(codec=Codec.HEVC)
        return encoder.encode_with_qp_plan(
            video_path, output_path, qp_plan,
            target_bitrate=bitrate, show_progress=show_progress
        )

    def _empty_result(self, method: str, bitrate: int, output_path: str) -> BenchmarkResult:
        """Повертає порожній результат при помилці."""
        return BenchmarkResult(
            method=method,
            bitrate_target=bitrate,
            bitrate_actual=0,
            psnr=0.0,
            ssim=0.0,
            vmaf=0.0,
            file_size=0,
            encode_time=0.0,
            output_path=output_path,
        )

    def _compute_bd_rates(self, report: BenchmarkReport) -> Dict[str, Dict[str, float]]:
        """Обчислює BD-Rate для всіх методів."""
        from .metrics import BDRateCalculator

        bd_calculator = BDRateCalculator()
        bd_rates = {}

        # Агрегуємо результати по методах
        method_results = {m: [] for m in self.methods}

        for video in report.videos:
            for result in video.results:
                method_results[result.method].append(result)

        # H.264 як anchor
        anchor = "h264"
        if anchor not in method_results or not method_results[anchor]:
            anchor = self.methods[0]

        anchor_data = method_results[anchor]
        anchor_rates = [r.bitrate_actual or r.bitrate_target for r in anchor_data]
        anchor_psnr = [r.psnr for r in anchor_data]
        anchor_vmaf = [r.vmaf for r in anchor_data]

        for method in self.methods:
            if method == anchor:
                bd_rates[method] = {"bd_rate_psnr": 0.0, "bd_rate_vmaf": 0.0}
                continue

            test_data = method_results[method]
            if not test_data:
                continue

            test_rates = [r.bitrate_actual or r.bitrate_target for r in test_data]
            test_psnr = [r.psnr for r in test_data]
            test_vmaf = [r.vmaf for r in test_data]

            try:
                bd_psnr = bd_calculator.compute_bd_rate(
                    anchor_rates, anchor_psnr, test_rates, test_psnr
                )
                bd_vmaf = bd_calculator.compute_bd_rate(
                    anchor_rates, anchor_vmaf, test_rates, test_vmaf
                )
                bd_rates[method] = {
                    "bd_rate_psnr": bd_psnr,
                    "bd_rate_vmaf": bd_vmaf,
                }
            except Exception as e:
                log_warning(f"Помилка BD-Rate для {method}: {e}")
                bd_rates[method] = {"bd_rate_psnr": 0.0, "bd_rate_vmaf": 0.0}

        return bd_rates

    def _save_report(self, report: BenchmarkReport, path: str) -> None:
        """Зберігає звіт у JSON."""
        # Конвертуємо dataclass у dict
        report_dict = {
            "timestamp": report.timestamp,
            "methods": report.methods,
            "bitrates": report.bitrates,
            "bd_rates": report.bd_rates,
            "videos": [],
        }

        for video in report.videos:
            video_dict = {
                "video_name": video.video_name,
                "video_path": video.video_path,
                "width": video.width,
                "height": video.height,
                "fps": video.fps,
                "duration": video.duration,
                "frame_count": video.frame_count,
                "results": [asdict(r) for r in video.results],
            }
            report_dict["videos"].append(video_dict)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        log_success(f"Звіт збережено: {path}")

    def _print_summary(self, report: BenchmarkReport) -> None:
        """Виводить таблицю результатів."""
        self.console.print("\n")

        # Таблиця BD-Rate
        table = Table(title="BD-Rate відносно H.264")
        table.add_column("Метод", style="cyan")
        table.add_column("BD-Rate (PSNR)", justify="right")
        table.add_column("BD-Rate (VMAF)", justify="right")

        for method, rates in report.bd_rates.items():
            psnr_str = f"{rates['bd_rate_psnr']:+.1f}%"
            vmaf_str = f"{rates['bd_rate_vmaf']:+.1f}%"

            # Кольорове кодування
            psnr_style = "green" if rates["bd_rate_psnr"] < 0 else "red"
            vmaf_style = "green" if rates["bd_rate_vmaf"] < 0 else "red"

            table.add_row(
                method.upper(),
                f"[{psnr_style}]{psnr_str}[/{psnr_style}]",
                f"[{vmaf_style}]{vmaf_str}[/{vmaf_style}]",
            )

        self.console.print(table)

        # Детальна таблиця по відео
        for video in report.videos:
            video_table = Table(title=f"\n{video.video_name}")
            video_table.add_column("Метод", style="cyan")
            video_table.add_column("Бітрейт", justify="right")
            video_table.add_column("PSNR", justify="right")
            video_table.add_column("SSIM", justify="right")
            video_table.add_column("VMAF", justify="right")
            video_table.add_column("Час", justify="right")

            for result in video.results:
                video_table.add_row(
                    result.method.upper(),
                    format_bitrate(result.bitrate_actual or result.bitrate_target),
                    f"{result.psnr:.2f} dB",
                    f"{result.ssim:.4f}",
                    f"{result.vmaf:.1f}",
                    f"{result.encode_time:.1f}s",
                )

            self.console.print(video_table)
