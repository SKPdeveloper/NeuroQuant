"""
NeuroQuant GUI - повний інтерфейс з порівнянням, графіками і бенчмарком.
"""

import sys
import json
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox,
    QCheckBox, QProgressBar, QTextEdit, QGroupBox, QSpinBox,
    QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QSplitter, QFrame, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QImage, QPainter, QColor

import numpy as np

from neuroquant.utils import (
    get_video_info, parse_bitrate, format_bitrate,
    check_ffmpeg, load_config
)

# Перевірка torch
TORCH_OK = False
CUDA_OK = False
TORCH_MSG = ""
try:
    import torch
    TORCH_OK = True
    CUDA_OK = torch.cuda.is_available()
    if CUDA_OK:
        TORCH_MSG = f"torch {torch.__version__}, CUDA: {torch.cuda.get_device_name(0)}"
    else:
        TORCH_MSG = f"torch {torch.__version__}, CPU only"
except Exception as e:
    TORCH_MSG = str(e)[:50]

# Matplotlib для графіків
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MPL_OK = True
except:
    MPL_OK = False


class EncodeWorker(QThread):
    """Фоновий потік для кодування."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, dict)  # success, message, stats
    log = pyqtSignal(str)

    def __init__(self, input_path: str, output_path: str, bitrate: int,
                 use_sr: bool, sr_threshold: float):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.bitrate = bitrate
        self.use_sr = use_sr
        self.sr_threshold = sr_threshold
        self.stats = {}

    def run(self):
        try:
            cfg = load_config(None)
            video_info = get_video_info(self.input_path)

            # Крок 1: Аналіз
            self.progress.emit(5, "Аналіз складності...")
            self.log.emit("=" * 50)
            self.log.emit("КРОК 1/4: Аналіз складності кадрів")

            from neuroquant.analyzer import ComplexityAnalyzer
            analyzer = ComplexityAnalyzer()
            complexity_data = analyzer.analyze(self.input_path, show_progress=False)

            complexities = [r.complexity for r in complexity_data]
            spatials = [r.spatial for r in complexity_data]
            temporals = [r.temporal for r in complexity_data]
            scene_cuts = [i for i, r in enumerate(complexity_data) if r.is_scene_cut]

            self.stats['complexity'] = complexities
            self.stats['spatial'] = spatials
            self.stats['temporal'] = temporals
            self.stats['scene_cuts'] = scene_cuts
            self.stats['frame_count'] = len(complexity_data)

            self.log.emit(f"  Кадрів: {len(complexity_data)}")
            self.log.emit(f"  Сцен: {len(scene_cuts)}")
            self.log.emit(f"  Складність: min={min(complexities):.3f}, max={max(complexities):.3f}, avg={np.mean(complexities):.3f}")

            # Крок 2: QP план
            self.progress.emit(25, "QP план (R-λ модель)...")
            self.log.emit("=" * 50)
            self.log.emit("КРОК 2/4: Генерація QP плану")
            self.log.emit(f"  R-λ модель: R = α·λ^β")
            self.log.emit(f"  α = 6.7542, β = -1.7860")

            from neuroquant.controller import RLambdaController
            controller = RLambdaController()
            qp_plan = controller.generate_qp_plan(
                complexity_data,
                target_bitrate=self.bitrate,
                fps=video_info["fps"],
                width=video_info["width"],
                height=video_info["height"],
            )

            qp_values = [f.qp for f in qp_plan]
            self.stats['qp_values'] = qp_values
            self.stats['qp_min'] = min(qp_values)
            self.stats['qp_max'] = max(qp_values)
            self.stats['qp_avg'] = np.mean(qp_values)

            self.log.emit(f"  QP: min={min(qp_values)}, max={max(qp_values)}, avg={np.mean(qp_values):.1f}")

            # Крок 3: Кодування
            self.progress.emit(35, "Кодування x265...")
            self.log.emit("=" * 50)
            self.log.emit("КРОК 3/4: Кодування HEVC (x265)")

            from neuroquant.encoder import FFmpegEncoder, Codec
            encoder = FFmpegEncoder(codec=Codec.HEVC)

            output_target = self.output_path
            if self.use_sr:
                output_target = str(Path(self.output_path).with_suffix(".temp.mp4"))

            encode_result = encoder.encode_with_qp_plan(
                self.input_path, output_target, qp_plan,
                target_bitrate=self.bitrate, show_progress=False
            )

            if not encode_result.success:
                self.finished.emit(False, f"Помилка: {encode_result.error_message}", {})
                return

            enc_time = getattr(encode_result, 'encoding_time', 0) or encode_result.duration
            self.stats['encoding_time'] = enc_time
            self.log.emit(f"  Тривалість: {encode_result.duration:.1f} сек")
            self.progress.emit(75, "Кодування завершено")

            # Крок 4: SR
            if self.use_sr:
                self.progress.emit(80, "Real-ESRGAN SR...")
                self.log.emit("=" * 50)
                self.log.emit("КРОК 4/4: Real-ESRGAN Super-Resolution")

                from neuroquant.sr_processor import SRPostProcessor
                sr_processor = SRPostProcessor(vmaf_threshold=self.sr_threshold)
                sr_result = sr_processor.process_video(
                    output_target, self.input_path, self.output_path, show_progress=False
                )

                Path(output_target).unlink(missing_ok=True)

                if not sr_result.success:
                    self.finished.emit(False, f"SR помилка: {sr_result.error_message}", {})
                    return

                self.stats['sr_frames'] = sr_result.frames_processed
                self.stats['sr_total'] = sr_result.frames_total
                self.log.emit(f"  Оброблено SR: {sr_result.frames_processed}/{sr_result.frames_total}")
            else:
                self.log.emit("=" * 50)
                self.log.emit("КРОК 4/4: SR пропущено")

            self.progress.emit(100, "Готово!")

            # Фінальна статистика
            output_size = Path(self.output_path).stat().st_size
            input_size = Path(self.input_path).stat().st_size
            output_info = get_video_info(self.output_path)

            self.stats['input_size'] = input_size
            self.stats['output_size'] = output_size
            self.stats['compression_ratio'] = input_size / output_size if output_size > 0 else 0
            self.stats['actual_bitrate'] = output_info.get('bitrate', 0)
            self.stats['target_bitrate'] = self.bitrate

            self.log.emit("=" * 50)
            self.log.emit("РЕЗУЛЬТАТ:")
            self.log.emit(f"  Вхід: {input_size / (1024*1024):.2f} MB")
            self.log.emit(f"  Вихід: {output_size / (1024*1024):.2f} MB")
            self.log.emit(f"  Стиснення: {self.stats['compression_ratio']:.1f}x")
            self.log.emit(f"  Бітрейт: {format_bitrate(self.stats['actual_bitrate'])}")

            self.finished.emit(True, f"Готово! {output_size / (1024*1024):.2f} MB", self.stats)

        except Exception as e:
            import traceback
            self.log.emit(f"ПОМИЛКА: {e}")
            self.log.emit(traceback.format_exc())
            self.finished.emit(False, str(e), {})


class BenchmarkWorker(QThread):
    """Потік для бенчмарку порівняння кодеків."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str, list)
    log = pyqtSignal(str)

    def __init__(self, input_path: str, output_dir: str, bitrates: List[int], methods: List[str]):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.bitrates = bitrates
        self.methods = methods

    def run(self):
        try:
            from neuroquant.benchmark import BenchmarkEngine

            results = []
            total_steps = len(self.bitrates) * len(self.methods)
            step = 0

            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

            for method in self.methods:
                for bitrate in self.bitrates:
                    step += 1
                    pct = int(step / total_steps * 100)
                    self.progress.emit(pct, f"{method.upper()} @ {format_bitrate(bitrate)}")
                    self.log.emit(f"\n[{step}/{total_steps}] {method.upper()} @ {format_bitrate(bitrate)}")

                    output_path = Path(self.output_dir) / f"{Path(self.input_path).stem}_{method}_{bitrate}.mp4"

                    try:
                        from neuroquant.encoder import FFmpegEncoder, Codec

                        if method == 'h264':
                            encoder = FFmpegEncoder(codec=Codec.H264)
                            result = encoder.encode(self.input_path, str(output_path), bitrate)
                        elif method == 'hevc':
                            encoder = FFmpegEncoder(codec=Codec.HEVC)
                            result = encoder.encode(self.input_path, str(output_path), bitrate)
                        elif method == 'vvc':
                            encoder = FFmpegEncoder(codec=Codec.VVC)
                            result = encoder.encode(self.input_path, str(output_path), bitrate)
                        elif method == 'nq':
                            # NeuroQuant
                            from neuroquant.analyzer import ComplexityAnalyzer
                            from neuroquant.controller import RLambdaController

                            analyzer = ComplexityAnalyzer()
                            complexity = analyzer.analyze(self.input_path, show_progress=False)

                            video_info = get_video_info(self.input_path)
                            controller = RLambdaController()
                            qp_plan = controller.generate_qp_plan(
                                complexity, bitrate, video_info['fps'],
                                video_info['width'], video_info['height']
                            )

                            encoder = FFmpegEncoder(codec=Codec.HEVC)
                            result = encoder.encode_with_qp_plan(
                                self.input_path, str(output_path), qp_plan,
                                target_bitrate=bitrate, show_progress=False
                            )
                        else:
                            continue

                        if result.success and output_path.exists():
                            # Обчислення метрик
                            from neuroquant.metrics import calculate_psnr, calculate_ssim

                            out_info = get_video_info(str(output_path))
                            out_size = output_path.stat().st_size

                            psnr = calculate_psnr(self.input_path, str(output_path))
                            ssim = calculate_ssim(self.input_path, str(output_path))

                            results.append({
                                'method': method,
                                'bitrate': bitrate,
                                'actual_bitrate': out_info.get('bitrate', bitrate),
                                'size_mb': out_size / (1024 * 1024),
                                'psnr': psnr,
                                'ssim': ssim,
                                'encoding_time': getattr(result, 'encoding_time', 0) or result.duration
                            })

                            self.log.emit(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
                        else:
                            self.log.emit(f"  ПОМИЛКА: {result.error_message}")

                    except Exception as e:
                        self.log.emit(f"  ПОМИЛКА: {e}")

            self.finished.emit(True, "Бенчмарк завершено", results)

        except Exception as e:
            self.log.emit(f"ПОМИЛКА: {e}")
            self.finished.emit(False, str(e), [])


class PlotWidget(QWidget):
    """Віджет для відображення matplotlib графіків."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def clear(self):
        self.figure.clear()
        self.canvas.draw()

    def plot_complexity(self, complexity: List[float], spatial: List[float],
                        temporal: List[float], scene_cuts: List[int]):
        """Графік складності кадрів."""
        self.figure.clear()

        ax1 = self.figure.add_subplot(211)
        frames = np.arange(len(complexity))

        ax1.fill_between(frames, complexity, alpha=0.3, color='blue')
        ax1.plot(frames, complexity, linewidth=0.8, color='blue', label='Complexity')

        for cut in scene_cuts:
            ax1.axvline(cut, color='red', linestyle='--', alpha=0.5, linewidth=0.5)

        ax1.set_ylabel('Складність')
        ax1.set_title('Аналіз складності кадрів')
        ax1.set_xlim(0, len(complexity))
        ax1.set_ylim(0, 1)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        ax2 = self.figure.add_subplot(212)
        ax2.plot(frames, spatial, linewidth=0.8, label='Spatial', color='green')
        ax2.plot(frames, temporal, linewidth=0.8, label='Temporal', color='orange')
        ax2.set_xlabel('Кадр')
        ax2.set_ylabel('Значення')
        ax2.set_xlim(0, len(complexity))
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_qp(self, qp_values: List[int], complexity: List[float]):
        """Графік QP плану."""
        self.figure.clear()

        ax1 = self.figure.add_subplot(111)
        frames = np.arange(len(qp_values))

        ax1.plot(frames, qp_values, linewidth=0.8, color='purple', label='QP')
        ax1.set_xlabel('Кадр')
        ax1.set_ylabel('QP', color='purple')
        ax1.tick_params(axis='y', labelcolor='purple')
        ax1.set_xlim(0, len(qp_values))
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.fill_between(frames, complexity[:len(qp_values)], alpha=0.2, color='blue')
        ax2.set_ylabel('Складність', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.set_ylim(0, 1)

        ax1.set_title('QP план vs Складність (R-λ модель)')
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_rd_curve(self, results: List[Dict]):
        """RD-криві для порівняння методів."""
        self.figure.clear()

        ax = self.figure.add_subplot(111)

        methods = list(set(r['method'] for r in results))
        colors = {'h264': 'red', 'hevc': 'green', 'vvc': 'blue', 'nq': 'purple'}

        for method in methods:
            data = [r for r in results if r['method'] == method]
            data.sort(key=lambda x: x['bitrate'])

            bitrates = [r['actual_bitrate'] / 1000 for r in data]  # kbps
            psnrs = [r['psnr'] for r in data]

            ax.plot(bitrates, psnrs, 'o-', label=method.upper(),
                    color=colors.get(method, 'gray'), linewidth=2, markersize=6)

        ax.set_xlabel('Бітрейт (kbps)')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('RD-криві: Порівняння методів кодування')
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_comparison_bars(self, results: List[Dict], metric: str = 'psnr'):
        """Стовпчикова діаграма порівняння."""
        self.figure.clear()

        ax = self.figure.add_subplot(111)

        methods = list(set(r['method'] for r in results))
        bitrates = sorted(list(set(r['bitrate'] for r in results)))

        x = np.arange(len(bitrates))
        width = 0.2
        colors = {'h264': 'red', 'hevc': 'green', 'vvc': 'blue', 'nq': 'purple'}

        for i, method in enumerate(methods):
            values = []
            for br in bitrates:
                data = [r for r in results if r['method'] == method and r['bitrate'] == br]
                if data:
                    values.append(data[0][metric])
                else:
                    values.append(0)

            ax.bar(x + i * width, values, width, label=method.upper(),
                   color=colors.get(method, 'gray'))

        ax.set_xlabel('Бітрейт')
        ax.set_ylabel('PSNR (dB)' if metric == 'psnr' else 'SSIM')
        ax.set_title(f'Порівняння {metric.upper()} при різних бітрейтах')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([format_bitrate(b) for b in bitrates])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        self.figure.tight_layout()
        self.canvas.draw()


class NeuroQuantGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker: Optional[EncodeWorker] = None
        self.benchmark_worker: Optional[BenchmarkWorker] = None
        self.last_stats = {}
        self.benchmark_results = []
        self.init_ui()
        self.check_system()

    def init_ui(self):
        self.setWindowTitle("NeuroQuant - Інтелектуальне стиснення відео")
        self.setMinimumSize(1000, 700)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Заголовок
        header = QHBoxLayout()
        title = QLabel("NeuroQuant")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: #2196F3;")
        header.addWidget(title)

        subtitle = QLabel("R-λ Rate Control + Real-ESRGAN Super-Resolution")
        subtitle.setStyleSheet("color: #666; padding-top: 10px;")
        header.addWidget(subtitle)
        header.addStretch()
        layout.addLayout(header)

        # Вкладки
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Вкладка 1: Кодування
        self.tabs.addTab(self.create_encode_tab(), "Кодування")

        # Вкладка 2: Аналіз
        self.tabs.addTab(self.create_analysis_tab(), "Аналіз")

        # Вкладка 3: Бенчмарк
        self.tabs.addTab(self.create_benchmark_tab(), "Бенчмарк")

        # Вкладка 4: Математика
        self.tabs.addTab(self.create_math_tab(), "Теорія")

        # В��ладка 5: Довідка
        self.tabs.addTab(self.create_help_tab(), "Довідка")

    def create_encode_tab(self) -> QWidget:
        """Вкладка кодування."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Вхідний файл
        input_group = QGroupBox("Вхідне відео")
        input_layout = QHBoxLayout(input_group)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Оберіть відеофайл...")
        self.input_edit.textChanged.connect(self.on_input_changed)
        input_layout.addWidget(self.input_edit)
        input_btn = QPushButton("Огляд...")
        input_btn.clicked.connect(self.browse_input)
        input_layout.addWidget(input_btn)
        layout.addWidget(input_group)

        self.video_info = QLabel("")
        self.video_info.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(self.video_info)

        # Вихідний файл
        output_group = QGroupBox("Вихідний файл")
        output_layout = QHBoxLayout(output_group)
        self.output_edit = QLineEdit()
        output_layout.addWidget(self.output_edit)
        output_btn = QPushButton("Огляд...")
        output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(output_btn)
        layout.addWidget(output_group)

        # Налаштування
        settings_group = QGroupBox("Параметри")
        settings_layout = QHBoxLayout(settings_group)

        settings_layout.addWidget(QLabel("Бітрейт:"))
        self.bitrate_combo = QComboBox()
        self.bitrate_combo.addItems(["150k", "300k", "600k", "1M", "1.5M", "2M", "2.5M", "4M"])
        self.bitrate_combo.setCurrentText("1M")
        self.bitrate_combo.setEditable(True)
        settings_layout.addWidget(self.bitrate_combo)

        settings_layout.addSpacing(30)

        self.sr_check = QCheckBox("Real-ESRGAN SR")
        self.sr_check.toggled.connect(self.on_sr_toggled)
        settings_layout.addWidget(self.sr_check)

        settings_layout.addWidget(QLabel("VMAF поріг:"))
        self.sr_threshold = QSpinBox()
        self.sr_threshold.setRange(50, 95)
        self.sr_threshold.setValue(70)
        self.sr_threshold.setEnabled(False)
        settings_layout.addWidget(self.sr_threshold)

        settings_layout.addStretch()
        layout.addWidget(settings_group)

        # Прогрес
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Готово до роботи")
        layout.addWidget(self.status_label)

        # Лог
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #ddd;")
        layout.addWidget(self.log_text)

        # Кнопки
        btn_layout = QHBoxLayout()
        self.encode_btn = QPushButton("▶ Кодувати")
        self.encode_btn.setMinimumHeight(45)
        self.encode_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.encode_btn.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; border-radius: 5px; }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #888; }
        """)
        self.encode_btn.clicked.connect(self.start_encode)
        btn_layout.addWidget(self.encode_btn)

        self.cancel_btn = QPushButton("Скасувати")
        self.cancel_btn.setMinimumHeight(45)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_encode)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

        return tab

    def create_analysis_tab(self) -> QWidget:
        """Вкладка аналізу з графіками."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        if not MPL_OK:
            layout.addWidget(QLabel("Matplotlib не встановлено. Графіки недоступні."))
            return tab

        # Графік складності
        self.complexity_plot = PlotWidget()
        layout.addWidget(QLabel("Складність кадрів:"))
        layout.addWidget(self.complexity_plot)

        # Графік QP
        self.qp_plot = PlotWidget()
        layout.addWidget(QLabel("QP план:"))
        layout.addWidget(self.qp_plot)

        # Статистика
        stats_group = QGroupBox("Статистика")
        stats_layout = QHBoxLayout(stats_group)

        self.stats_table = QTableWidget(6, 2)
        self.stats_table.setHorizontalHeaderLabels(["Параметр", "Значення"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        stats_layout.addWidget(self.stats_table)

        layout.addWidget(stats_group)

        return tab

    def create_benchmark_tab(self) -> QWidget:
        """Вкладка бенчмарку."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Налаштування бенчмарку
        settings_group = QGroupBox("Налаштування бенчмарку")
        settings_layout = QVBoxLayout(settings_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Відео:"))
        self.bench_input = QLineEdit()
        self.bench_input.setPlaceholderText("Оберіть тестове відео...")
        row1.addWidget(self.bench_input)
        bench_btn = QPushButton("...")
        bench_btn.setMaximumWidth(40)
        bench_btn.clicked.connect(lambda: self.browse_for_edit(self.bench_input))
        row1.addWidget(bench_btn)
        settings_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Методи:"))
        self.method_h264 = QCheckBox("H.264")
        self.method_h264.setChecked(True)
        row2.addWidget(self.method_h264)
        self.method_hevc = QCheckBox("HEVC")
        self.method_hevc.setChecked(True)
        row2.addWidget(self.method_hevc)
        self.method_vvc = QCheckBox("VVC")
        row2.addWidget(self.method_vvc)
        self.method_nq = QCheckBox("NeuroQuant")
        self.method_nq.setChecked(True)
        row2.addWidget(self.method_nq)
        row2.addStretch()
        settings_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Бітрейти:"))
        self.bench_bitrates = QLineEdit("300k, 600k, 1M, 2M")
        row3.addWidget(self.bench_bitrates)
        row3.addStretch()
        settings_layout.addLayout(row3)

        layout.addWidget(settings_group)

        # Прогрес
        self.bench_progress = QProgressBar()
        layout.addWidget(self.bench_progress)

        self.bench_status = QLabel("Готово")
        layout.addWidget(self.bench_status)

        # Кнопка запуску
        self.bench_btn = QPushButton("▶ Запустити бенчмарк")
        self.bench_btn.setMinimumHeight(40)
        self.bench_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.bench_btn.clicked.connect(self.start_benchmark)
        layout.addWidget(self.bench_btn)

        # Результати
        results_group = QGroupBox("Результати")
        results_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "Метод", "Бітрейт", "Розмір (MB)", "PSNR (dB)", "SSIM", "Час (сек)", "Ефективність"
        ])
        results_layout.addWidget(self.results_table)

        layout.addWidget(results_group)

        # RD-крива
        if MPL_OK:
            self.rd_plot = PlotWidget()
            layout.addWidget(QLabel("RD-крива:"))
            layout.addWidget(self.rd_plot)

        return tab

    def create_math_tab(self) -> QWidget:
        """Вкладка з теорією R-λ моделі."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)

        # Текст теорії
        theory = QTextEdit()
        theory.setReadOnly(True)
        theory.setFont(QFont("Segoe UI", 10))
        theory.setHtml("""
        <h2 style="color: #2196F3;">R-λ модель Rate Control</h2>

        <h3>1. Базова залежність бітрейт-лямбда</h3>
        <p>Зв'язок між бітрейтом R та параметром Лагранжа λ описується степеневою функцією:</p>
        <p style="font-family: monospace; font-size: 14px; background: #f5f5f5; padding: 10px;">
        <b>R = α · λ<sup>β</sup></b>
        </p>
        <p>Де для x265:</p>
        <ul>
            <li><b>α = 6.7542</b> — масштабний коефіцієнт</li>
            <li><b>β = -1.7860</b> — показник степеня</li>
        </ul>

        <h3>2. Зв'язок λ та QP</h3>
        <p>Параметр λ визначає компроміс між якістю та бітрейтом. Зв'язок з QP:</p>
        <p style="font-family: monospace; font-size: 14px; background: #f5f5f5; padding: 10px;">
        <b>λ = c · 2<sup>(QP - 12) / 3</sup></b>
        </p>
        <p>Де коефіцієнт c залежить від типу кадру:</p>
        <ul>
            <li><b>c = 0.85</b> для P-кадрів</li>
            <li><b>c = 0.57</b> для B-кадрів</li>
            <li><b>c = 0.65</b> для I-кадрів</li>
        </ul>

        <h3>3. Формула складності кадру</h3>
        <p>Складність кадру обчислюється як зважена сума компонент:</p>
        <p style="font-family: monospace; font-size: 14px; background: #f5f5f5; padding: 10px;">
        <b>complexity<sub>i</sub> = 0.4·spatial<sub>i</sub> + 0.5·temporal<sub>i</sub> + 0.1·cut<sub>i</sub></b>
        </p>
        <ul>
            <li><b>spatial</b> — нормований градієнт Sobel (деталізація)</li>
            <li><b>temporal</b> — SAD між сусідніми кадрами (рух)</li>
            <li><b>cut</b> — індикатор зміни сцени (0 або 1)</li>
        </ul>

        <h3>4. Адаптивний QP</h3>
        <p>Per-frame QP обчислюється на основі відхилення складності від середнього:</p>
        <p style="font-family: monospace; font-size: 14px; background: #f5f5f5; padding: 10px;">
        <b>QP<sub>i</sub> = QP<sub>base</sub> - round(Δ<sub>max</sub> · (complexity<sub>i</sub> - μ) / σ)</b>
        </p>
        <ul>
            <li><b>Δ<sub>max</sub> = 6</b> — максимальне відхилення QP</li>
            <li><b>μ</b> — середня складність</li>
            <li><b>σ</b> — стандартне відхилення складності</li>
            <li>Результат обмежується діапазоном [18, 45]</li>
        </ul>

        <h3>5. Принцип роботи</h3>
        <p>
        <b>Складний кадр</b> (вибух, швидкий рух) → нижчий QP → більше біт → краща якість<br>
        <b>Простий кадр</b> (статика, небо) → вищий QP → менше біт → економія
        </p>

        <h3>6. Real-ESRGAN постобробка</h3>
        <p>Кадри з VMAF &lt; threshold проходять через Real-ESRGAN x2 для відновлення деталей,
        втрачених при агресивному стисненні.</p>
        """)
        content_layout.addWidget(theory)

        scroll.setWidget(content)
        layout.addWidget(scroll)

        return tab

    def create_help_tab(self) -> QWidget:
        """Вкладка довідки."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)

        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setFont(QFont("Segoe UI", 10))
        help_text.setHtml("""
        <h2 style="color: #4CAF50;">Довідка NeuroQuant</h2>

        <h3>🎯 Що це?</h3>
        <p>NeuroQuant — система інтелектуального стиснення відео, що поєднує:</p>
        <ul>
            <li><b>R-λ Rate Control</b> — адаптивний розподіл бітів на основі складності кадрів</li>
            <li><b>Real-ESRGAN SR</b> — відновлення якості низькобітрейтних кадрів через суперроздільність</li>
        </ul>

        <h3>📹 Вкладка "Кодування"</h3>
        <ol>
            <li>Оберіть вхідне відео (MP4, MKV, AVI, MOV)</li>
            <li>Вкажіть шлях для вихідного файлу</li>
            <li>Встановіть цільовий бітрейт (рекомендовано: 600k-2M для 1080p)</li>
            <li>Увімкніть SR якщо потрібно відновлення деталей (потребує CUDA)</li>
            <li>Натисніть "Кодувати"</li>
        </ol>

        <h3>📊 Вкладка "Аналіз"</h3>
        <p>Після кодування тут з'являться:</p>
        <ul>
            <li><b>Графік складності</b> — spatial (деталі) + temporal (рух) по кадрах</li>
            <li><b>QP план</b> — як QP адаптується до складності</li>
            <li><b>Статистика</b> — стиснення, час, діапазон QP</li>
        </ul>

        <h3>⚡ Вкладка "Бенчмарк"</h3>
        <p>Порівняння методів кодування:</p>
        <ul>
            <li><b>H.264</b> — найпоширеніший, сумісний</li>
            <li><b>HEVC (H.265)</b> — на 30-50% ефективніший за H.264</li>
            <li><b>VVC (H.266)</b> — найновіший стандарт</li>
            <li><b>NeuroQuant</b> — HEVC + адаптивний QP</li>
        </ul>
        <p>Результат: RD-криві (Rate-Distortion) для об'єктивного порівняння.</p>

        <h3>📐 Вкладка "Теорія"</h3>
        <p>Математичний опис R-λ моделі та формули розрахунку QP.</p>

        <h3>⚙️ Системні вимоги</h3>
        <ul>
            <li><b>FFmpeg</b> з libx265 (обов'язково)</li>
            <li><b>Python 3.9-3.11</b> (для torch з CUDA)</li>
            <li><b>NVIDIA GPU</b> (для SR, опційно)</li>
        </ul>

        <h3>🔧 CLI команди</h3>
        <pre style="background: #2d2d2d; padding: 10px; color: #ddd;">
# Кодування
py -3.9 -m neuroquant encode input.mp4 output.mp4 --bitrate 1M

# З SR
py -3.9 -m neuroquant encode input.mp4 output.mp4 --bitrate 500k --sr

# Аналіз складності
py -3.9 -m neuroquant analyze input.mp4 --output complexity.json --plot

# Бенчмарк
py -3.9 -m neuroquant benchmark ./videos -m h264,hevc,nq -b 300k,600k,1M
        </pre>

        <h3>📧 Контакти</h3>
        <p>Дипломний проєкт 2026. Інтелектуальне стиснення відео.</p>
        """)
        content_layout.addWidget(help_text)

        scroll.setWidget(content)
        layout.addWidget(scroll)

        return tab

    def check_system(self):
        ok, msg = check_ffmpeg()
        self.log_text.append(f"FFmpeg: {'OK' if ok else 'ПОМИЛКА'} - {msg[:50]}")
        self.log_text.append(f"Torch: {'OK' if TORCH_OK else 'ПОМИЛКА'} - {TORCH_MSG}")
        self.log_text.append(f"CUDA: {'Доступна' if CUDA_OK else 'Недоступна'}")

        if not TORCH_OK:
            self.sr_check.setEnabled(False)
            self.sr_check.setToolTip(f"Torch недоступний: {TORCH_MSG}")

        if not ok:
            QMessageBox.critical(self, "Помилка", "FFmpeg не знайдено!")

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Оберіть відео", "",
            "Відео (*.mp4 *.mkv *.avi *.mov *.webm);;Всі файли (*)"
        )
        if path:
            self.input_edit.setText(path)

    def browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Зберегти як", "",
            "MP4 (*.mp4);;MKV (*.mkv)"
        )
        if path:
            self.output_edit.setText(path)

    def browse_for_edit(self, edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Оберіть відео", "",
            "Відео (*.mp4 *.mkv *.avi *.mov)"
        )
        if path:
            edit.setText(path)

    def on_input_changed(self, path: str):
        if path and Path(path).exists():
            try:
                info = get_video_info(path)
                self.video_info.setText(
                    f"📹 {info['width']}×{info['height']} | "
                    f"{info['fps']:.1f} fps | "
                    f"{info['duration']:.1f} сек | "
                    f"{info['frame_count']} кадрів | "
                    f"{Path(path).stat().st_size / (1024*1024):.1f} MB"
                )
                if not self.output_edit.text():
                    p = Path(path)
                    self.output_edit.setText(str(p.parent / f"{p.stem}_nq.mp4"))
            except Exception as e:
                self.video_info.setText(f"Помилка: {e}")

    def on_sr_toggled(self, checked: bool):
        self.sr_threshold.setEnabled(checked)

    def start_encode(self):
        inp = self.input_edit.text()
        out = self.output_edit.text()

        if not inp or not Path(inp).exists():
            QMessageBox.warning(self, "Помилка", "Оберіть вхідний файл")
            return
        if not out:
            QMessageBox.warning(self, "Помилка", "Вкажіть вихідний файл")
            return

        self.log_text.clear()
        self.encode_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        bitrate = parse_bitrate(self.bitrate_combo.currentText())

        self.worker = EncodeWorker(
            inp, out, bitrate,
            self.sr_check.isChecked(), self.sr_threshold.value()
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(lambda m: self.log_text.append(m))
        self.worker.finished.connect(self.on_encode_finished)
        self.worker.start()

    def on_progress(self, percent: int, msg: str):
        self.progress_bar.setValue(percent)
        self.status_label.setText(msg)

    def cancel_encode(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.on_encode_finished(False, "Скасовано", {})

    def on_encode_finished(self, ok: bool, msg: str, stats: dict):
        self.encode_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText(msg)

        if ok and stats:
            self.last_stats = stats
            self.update_analysis_tab(stats)

        if ok:
            QMessageBox.information(self, "Готово", msg)
        else:
            QMessageBox.critical(self, "Помилка", msg)

    def update_analysis_tab(self, stats: dict):
        """Оновлення графіків після кодування."""
        if not MPL_OK:
            return

        if 'complexity' in stats:
            self.complexity_plot.plot_complexity(
                stats['complexity'],
                stats.get('spatial', stats['complexity']),
                stats.get('temporal', stats['complexity']),
                stats.get('scene_cuts', [])
            )

        if 'qp_values' in stats and 'complexity' in stats:
            self.qp_plot.plot_qp(stats['qp_values'], stats['complexity'])

        # Таблиця статистики
        rows = [
            ("Кадрів", str(stats.get('frame_count', '-'))),
            ("QP середній", f"{stats.get('qp_avg', 0):.1f}"),
            ("QP діапазон", f"{stats.get('qp_min', 0)} - {stats.get('qp_max', 0)}"),
            ("Стиснення", f"{stats.get('compression_ratio', 0):.1f}x"),
            ("Час кодування", f"{stats.get('encoding_time', 0):.1f} сек"),
            ("Бітрейт", format_bitrate(stats.get('actual_bitrate', 0))),
        ]

        self.stats_table.setRowCount(len(rows))
        for i, (param, value) in enumerate(rows):
            self.stats_table.setItem(i, 0, QTableWidgetItem(param))
            self.stats_table.setItem(i, 1, QTableWidgetItem(value))

        # Переключитись на вкладку аналізу
        self.tabs.setCurrentIndex(1)

    def start_benchmark(self):
        inp = self.bench_input.text()
        if not inp or not Path(inp).exists():
            QMessageBox.warning(self, "Помилка", "Оберіть тестове відео")
            return

        methods = []
        if self.method_h264.isChecked(): methods.append('h264')
        if self.method_hevc.isChecked(): methods.append('hevc')
        if self.method_vvc.isChecked(): methods.append('vvc')
        if self.method_nq.isChecked(): methods.append('nq')

        if not methods:
            QMessageBox.warning(self, "Помилка", "Оберіть хоча б один метод")
            return

        bitrates = [parse_bitrate(b.strip()) for b in self.bench_bitrates.text().split(',')]

        output_dir = str(Path(inp).parent / "benchmark_results")

        self.bench_btn.setEnabled(False)
        self.bench_progress.setValue(0)
        self.results_table.setRowCount(0)

        self.benchmark_worker = BenchmarkWorker(inp, output_dir, bitrates, methods)
        self.benchmark_worker.progress.connect(lambda p, m: (
            self.bench_progress.setValue(p),
            self.bench_status.setText(m)
        ))
        self.benchmark_worker.log.connect(lambda m: self.log_text.append(m))
        self.benchmark_worker.finished.connect(self.on_benchmark_finished)
        self.benchmark_worker.start()

    def on_benchmark_finished(self, ok: bool, msg: str, results: List[Dict]):
        self.bench_btn.setEnabled(True)
        self.bench_status.setText(msg)

        if ok and results:
            self.benchmark_results = results
            self.update_benchmark_results(results)

    def update_benchmark_results(self, results: List[Dict]):
        """Оновлення таблиці та графіків бенчмарку."""
        self.results_table.setRowCount(len(results))

        for i, r in enumerate(results):
            self.results_table.setItem(i, 0, QTableWidgetItem(r['method'].upper()))
            self.results_table.setItem(i, 1, QTableWidgetItem(format_bitrate(r['bitrate'])))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{r['size_mb']:.2f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{r['psnr']:.2f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{r['ssim']:.4f}"))
            self.results_table.setItem(i, 5, QTableWidgetItem(f"{r['encoding_time']:.1f}"))

            # Ефективність = PSNR / bitrate
            eff = r['psnr'] / (r['actual_bitrate'] / 1000) if r['actual_bitrate'] > 0 else 0
            self.results_table.setItem(i, 6, QTableWidgetItem(f"{eff:.3f}"))

        if MPL_OK and hasattr(self, 'rd_plot'):
            self.rd_plot.plot_rd_curve(results)


def apply_dark_theme(app):
    """Застосувати темну тему."""
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(palette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(palette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(palette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(palette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)


def apply_light_theme(app):
    """Застосувати світлу тему."""
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, QColor(240, 240, 240))
    palette.setColor(palette.ColorRole.WindowText, QColor(0, 0, 0))
    palette.setColor(palette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(palette.ColorRole.AlternateBase, QColor(245, 245, 245))
    palette.setColor(palette.ColorRole.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(palette.ColorRole.ToolTipText, QColor(0, 0, 0))
    palette.setColor(palette.ColorRole.Text, QColor(0, 0, 0))
    palette.setColor(palette.ColorRole.Button, QColor(240, 240, 240))
    palette.setColor(palette.ColorRole.ButtonText, QColor(0, 0, 0))
    palette.setColor(palette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(palette.ColorRole.Link, QColor(0, 100, 200))
    palette.setColor(palette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(palette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dark', action='store_true', help='Темна тема')
    parser.add_argument('--light', action='store_true', help='Світла тема')
    args, _ = parser.parse_known_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # За замовчуванням світла тема
    if args.dark:
        apply_dark_theme(app)
    else:
        apply_light_theme(app)

    w = NeuroQuantGUI()
    w.show()

    sys.exit(app.exec())
