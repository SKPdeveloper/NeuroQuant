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
    QSplitter, QFrame, QScrollArea, QSizePolicy, QSlider
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPixmap, QImage, QPainter, QColor

import cv2

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

            enc_time = encode_result.encoding_time if encode_result.encoding_time > 0 else encode_result.duration
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

                    output_path = (Path(self.output_dir) / f"{Path(self.input_path).stem}_{method}_{bitrate}.mp4").resolve()

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
                        elif method == 'nq_sr':
                            # NeuroQuant + Real-ESRGAN SR
                            from neuroquant.analyzer import ComplexityAnalyzer
                            from neuroquant.controller import RLambdaController
                            from neuroquant.sr_processor import SRPostProcessor

                            analyzer = ComplexityAnalyzer()
                            complexity = analyzer.analyze(self.input_path, show_progress=False)

                            video_info = get_video_info(self.input_path)
                            controller = RLambdaController()
                            qp_plan = controller.generate_qp_plan(
                                complexity, bitrate, video_info['fps'],
                                video_info['width'], video_info['height']
                            )

                            # Спочатку кодуємо
                            temp_path = (Path(self.output_dir) / f"temp_{Path(self.input_path).stem}_{bitrate}.mp4").resolve()
                            encoder = FFmpegEncoder(codec=Codec.HEVC)
                            result = encoder.encode_with_qp_plan(
                                self.input_path, str(temp_path), qp_plan,
                                target_bitrate=bitrate, show_progress=False
                            )

                            if result.success:
                                # Застосовуємо SR
                                sr_processor = SRPostProcessor(vmaf_threshold=70.0)
                                sr_result = sr_processor.process_video(
                                    str(temp_path), self.input_path, str(output_path),
                                    show_progress=False
                                )
                                temp_path.unlink(missing_ok=True)
                                if not sr_result.success:
                                    self.log.emit(f"  ПОМИЛКА SR: {sr_result.error_message}")
                                    continue
                            else:
                                temp_path.unlink(missing_ok=True)
                                self.log.emit(f"  ПОМИЛКА кодування: {result.error_message}")
                                continue
                        else:
                            continue

                        if output_path.exists():
                            # Обчислення метрик
                            from neuroquant.metrics import calculate_psnr, calculate_ssim

                            out_info = get_video_info(str(output_path))
                            out_size = output_path.stat().st_size

                            psnr = calculate_psnr(self.input_path, str(output_path))
                            ssim = calculate_ssim(self.input_path, str(output_path))

                            # Отримуємо час кодування (для nq_sr беремо з result)
                            enc_time = 0.0
                            if hasattr(result, 'encoding_time') and result.encoding_time > 0:
                                enc_time = result.encoding_time
                            elif hasattr(result, 'duration') and result.duration > 0:
                                enc_time = result.duration

                            results.append({
                                'method': method,
                                'bitrate': bitrate,
                                'actual_bitrate': out_info.get('bitrate', bitrate),
                                'size_mb': out_size / (1024 * 1024),
                                'psnr': psnr,
                                'ssim': ssim,
                                'encoding_time': enc_time
                            })

                            self.log.emit(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
                        else:
                            self.log.emit(f"  ПОМИЛКА: файл не створено")

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
        self.setMinimumSize(1100, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 8, 12, 8)

        # Заголовок
        header = QHBoxLayout()
        title = QLabel("NeuroQuant")
        title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        title.setStyleSheet("color: #333;")
        header.addWidget(title)

        subtitle = QLabel("R-λ Rate Control + Real-ESRGAN Super-Resolution")
        subtitle.setStyleSheet("color: #888; padding-top: 8px; font-size: 11px;")
        header.addWidget(subtitle)
        header.addStretch()

        # Версія
        version_label = QLabel("v1.0")
        version_label.setStyleSheet("color: #555; font-size: 10px;")
        header.addWidget(version_label)

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

        # Вкладка 5: Довідка
        self.tabs.addTab(self.create_help_tab(), "Довідка")

        # Вкладка 6: Порівняння (два екрани)
        self.tabs.addTab(self.create_compare_tab(), "Порівняння")

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
        self.encode_btn = QPushButton("Кодувати")
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
        """Вкладка аналізу з графіками та поясненнями."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        if not MPL_OK:
            layout.addWidget(QLabel("Matplotlib не встановлено. Графіки недоступні."))
            return tab

        # Графік складності з поясненням
        complexity_group = QGroupBox("Аналіз складності кадрів")
        complexity_layout = QVBoxLayout(complexity_group)

        complexity_explain = QLabel(
            "📊 <b>Синя лінія</b> — загальна складність кадру (0-1). "
            "<b>Червоні пунктири</b> — зміни сцен. "
            "Чим вище значення, тим більше деталей/руху в кадрі."
        )
        complexity_explain.setWordWrap(True)
        complexity_layout.addWidget(complexity_explain)

        self.complexity_plot = PlotWidget()
        complexity_layout.addWidget(self.complexity_plot)
        layout.addWidget(complexity_group)

        # Графік QP з поясненням
        qp_group = QGroupBox("QP план (адаптивний rate control)")
        qp_layout = QVBoxLayout(qp_group)

        qp_explain = QLabel(
            "📉 <b>Фіолетова лінія</b> — QP для кожного кадру. "
            "<b>Низький QP</b> = більше біт = краща якість (для складних сцен). "
            "<b>Високий QP</b> = менше біт = економія (для простих сцен)."
        )
        qp_explain.setWordWrap(True)
        qp_layout.addWidget(qp_explain)

        self.qp_plot = PlotWidget()
        qp_layout.addWidget(self.qp_plot)
        layout.addWidget(qp_group)

        # Статистика та висновки
        results_layout = QHBoxLayout()

        # Таблиця статистики
        stats_group = QGroupBox("Статистика")
        stats_layout_inner = QVBoxLayout(stats_group)
        self.stats_table = QTableWidget(6, 2)
        self.stats_table.setHorizontalHeaderLabels(["Параметр", "Значення"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        stats_layout_inner.addWidget(self.stats_table)
        results_layout.addWidget(stats_group)

        # Висновки
        conclusions_group = QGroupBox("Висновки")
        conclusions_layout = QVBoxLayout(conclusions_group)
        self.conclusions_label = QLabel(
            "Після кодування тут з'являться автоматичні висновки:\n"
            "• Характеристика контенту\n"
            "• Ефективність адаптації QP\n"
            "• Рекомендації"
        )
        self.conclusions_label.setWordWrap(True)
        self.conclusions_label.setFont(QFont("Segoe UI", 10))
        conclusions_layout.addWidget(self.conclusions_label)
        results_layout.addWidget(conclusions_group)

        layout.addLayout(results_layout)

        return tab

    def create_benchmark_tab(self) -> QWidget:
        """Вкладка бенчмарку з 4 екранами для порівняння."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Приховані поля для сумісності
        self.bench_input = QLineEdit()
        self.bench_input.hide()
        self.bench_bitrate = QComboBox()
        self.bench_bitrate.addItems(["300k", "600k", "1M", "2M"])
        self.bench_bitrate.hide()

        # Прогрес
        self.bench_progress = QProgressBar()
        layout.addWidget(self.bench_progress)
        self.bench_status = QLabel("Оберіть відео на вкладці 'Кодування' і натисніть 'Кодувати'")
        layout.addWidget(self.bench_status)

        # Таблиця порівняння (4 методи)
        self.bench_table = QTableWidget(4, 5)
        self.bench_table.setHorizontalHeaderLabels(["Кодек", "Розмір", "Бітрейт", "PSNR", "SSIM"])
        self.bench_table.verticalHeader().setVisible(False)
        self.bench_table.setMaximumHeight(160)
        self.bench_table.horizontalHeader().setStretchLastSection(True)

        # Заповнюємо назви кодеків
        codecs = ["H.264 (AVC)", "HEVC (H.265)", "NeuroQuant", "NeuroQuant + SR"]
        for i, codec in enumerate(codecs):
            self.bench_table.setItem(i, 0, QTableWidgetItem(codec))
            for j in range(1, 5):
                self.bench_table.setItem(i, j, QTableWidgetItem("—"))

        layout.addWidget(self.bench_table)

        # 4 екрани (2x2) - без SR, SR показуємо тільки в таблиці
        screens_widget = QWidget()
        screens_layout = QVBoxLayout(screens_widget)

        # Верхній ряд: H.264 | HEVC
        top_row = QHBoxLayout()

        # H.264
        h264_group = QGroupBox("H.264 (AVC)")
        h264_layout = QVBoxLayout(h264_group)
        self.bench_video_h264 = QLabel()
        self.bench_video_h264.setMinimumSize(350, 197)
        self.bench_video_h264.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bench_video_h264.setStyleSheet("background-color: #222;")
        h264_layout.addWidget(self.bench_video_h264)
        self.bench_params_h264 = QLabel("—")
        self.bench_params_h264.setFont(QFont("Consolas", 8))
        h264_layout.addWidget(self.bench_params_h264)
        top_row.addWidget(h264_group)

        # HEVC
        hevc_group = QGroupBox("HEVC (H.265)")
        hevc_layout = QVBoxLayout(hevc_group)
        self.bench_video_hevc = QLabel()
        self.bench_video_hevc.setMinimumSize(350, 197)
        self.bench_video_hevc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bench_video_hevc.setStyleSheet("background-color: #222;")
        hevc_layout.addWidget(self.bench_video_hevc)
        self.bench_params_hevc = QLabel("—")
        self.bench_params_hevc.setFont(QFont("Consolas", 8))
        hevc_layout.addWidget(self.bench_params_hevc)
        top_row.addWidget(hevc_group)

        screens_layout.addLayout(top_row)

        # Нижній ряд: VVC | NeuroQuant
        bottom_row = QHBoxLayout()

        # NeuroQuant
        vvc_group = QGroupBox("NeuroQuant (HEVC + R-λ)")
        vvc_layout = QVBoxLayout(vvc_group)
        self.bench_video_vvc = QLabel()
        self.bench_video_vvc.setMinimumSize(350, 197)
        self.bench_video_vvc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bench_video_vvc.setStyleSheet("background-color: #222;")
        vvc_layout.addWidget(self.bench_video_vvc)
        self.bench_params_vvc = QLabel("—")
        self.bench_params_vvc.setFont(QFont("Consolas", 8))
        vvc_layout.addWidget(self.bench_params_vvc)
        bottom_row.addWidget(vvc_group)

        # NeuroQuant + SR
        nq_group = QGroupBox("NeuroQuant + Real-ESRGAN")
        nq_layout = QVBoxLayout(nq_group)
        self.bench_video_nq = QLabel()
        self.bench_video_nq.setMinimumSize(350, 197)
        self.bench_video_nq.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bench_video_nq.setStyleSheet("background-color: #222;")
        nq_layout.addWidget(self.bench_video_nq)
        self.bench_params_nq = QLabel("—")
        self.bench_params_nq.setFont(QFont("Consolas", 9))
        nq_layout.addWidget(self.bench_params_nq)
        bottom_row.addWidget(nq_group)

        screens_layout.addLayout(bottom_row)
        layout.addWidget(screens_widget)

        # Слайдер
        slider_row = QHBoxLayout()
        self.bench_frame_label = QLabel("Кадр: 0 / 0")
        slider_row.addWidget(self.bench_frame_label)
        self.bench_slider = QSlider(Qt.Orientation.Horizontal)
        self.bench_slider.setEnabled(False)
        self.bench_slider.valueChanged.connect(self.on_bench_slider)
        slider_row.addWidget(self.bench_slider)
        self.bench_play_btn = QPushButton("Play")
        self.bench_play_btn.setEnabled(False)
        self.bench_play_btn.clicked.connect(self.toggle_bench_play)
        slider_row.addWidget(self.bench_play_btn)
        layout.addLayout(slider_row)

        # Підсумок
        self.bench_summary = QLabel("—")
        self.bench_summary.setFont(QFont("Consolas", 10))
        layout.addWidget(self.bench_summary)

        # Дані для відтворення
        self.bench_caps = {}
        self.bench_paths = {}
        self.bench_frame_count = 0
        self.bench_timer = QTimer()
        self.bench_timer.timeout.connect(self.bench_next_frame)
        self.bench_playing = False

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

    def create_compare_tab(self) -> QWidget:
        """Вкладка порівняння — два екрани поруч."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Інфо
        info_label = QLabel("Після кодування тут з'явиться порівняння оригіналу і результату")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("color: #888; padding: 20px;")
        layout.addWidget(info_label)
        self.compare_info_label = info_label

        # Два екрани поруч
        screens = QHBoxLayout()

        # Лівий — оригінал
        left_group = QGroupBox("ОРИГІНАЛ")
        left_layout = QVBoxLayout(left_group)
        self.left_video_label = QLabel()
        self.left_video_label.setMinimumSize(480, 270)
        self.left_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_video_label.setStyleSheet("background-color: #222;")
        left_layout.addWidget(self.left_video_label)

        self.left_params = QLabel("—")
        self.left_params.setFont(QFont("Consolas", 9))
        left_layout.addWidget(self.left_params)
        screens.addWidget(left_group)

        # Правий — стиснене
        right_group = QGroupBox("СТИСНЕНЕ (NeuroQuant)")
        right_layout = QVBoxLayout(right_group)
        self.right_video_label = QLabel()
        self.right_video_label.setMinimumSize(480, 270)
        self.right_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_video_label.setStyleSheet("background-color: #222;")
        right_layout.addWidget(self.right_video_label)

        self.right_params = QLabel("—")
        self.right_params.setFont(QFont("Consolas", 9))
        right_layout.addWidget(self.right_params)
        screens.addWidget(right_group)

        layout.addLayout(screens)

        # Слайдер
        slider_row = QHBoxLayout()
        self.compare_frame_label = QLabel("Кадр: 0 / 0")
        slider_row.addWidget(self.compare_frame_label)

        self.compare_slider = QSlider(Qt.Orientation.Horizontal)
        self.compare_slider.setEnabled(False)
        self.compare_slider.valueChanged.connect(self.on_compare_slider)
        slider_row.addWidget(self.compare_slider)

        self.compare_play_btn = QPushButton("Play")
        self.compare_play_btn.setEnabled(False)
        self.compare_play_btn.clicked.connect(self.toggle_compare_play)
        slider_row.addWidget(self.compare_play_btn)

        layout.addLayout(slider_row)

        # Підсумок порівняння
        summary_group = QGroupBox("Результат порівняння")
        summary_layout = QVBoxLayout(summary_group)
        self.compare_summary = QLabel("—")
        self.compare_summary.setFont(QFont("Consolas", 11))
        summary_layout.addWidget(self.compare_summary)
        layout.addWidget(summary_group)

        # Таймер для відтворення
        self.compare_timer = QTimer()
        self.compare_timer.timeout.connect(self.compare_next_frame)
        self.compare_playing = False

        # Відео дані
        self.compare_original_path = None
        self.compare_output_path = None
        self.compare_cap_orig = None
        self.compare_cap_comp = None
        self.compare_frame_count = 0

        return tab

    def load_compare_videos(self, original_path: str, output_path: str):
        """Завантажити відео для порівняння після кодування."""
        self.compare_original_path = original_path
        self.compare_output_path = output_path

        # Відкриваємо відео
        self.compare_cap_orig = cv2.VideoCapture(original_path)
        self.compare_cap_comp = cv2.VideoCapture(output_path)

        orig_info = get_video_info(original_path)
        comp_info = get_video_info(output_path)

        orig_size = Path(original_path).stat().st_size / (1024 * 1024)
        comp_size = Path(output_path).stat().st_size / (1024 * 1024)

        # Параметри
        self.left_params.setText(
            f"Файл: {Path(original_path).name}\n"
            f"Роздільність: {orig_info['width']} x {orig_info['height']}\n"
            f"FPS: {orig_info['fps']:.1f}\n"
            f"Бітрейт: {format_bitrate(orig_info.get('bitrate', 0))}\n"
            f"Розмір: {orig_size:.2f} MB"
        )

        self.right_params.setText(
            f"Файл: {Path(output_path).name}\n"
            f"Роздільність: {comp_info['width']} x {comp_info['height']}\n"
            f"FPS: {comp_info['fps']:.1f}\n"
            f"Бітрейт: {format_bitrate(comp_info.get('bitrate', 0))}\n"
            f"Розмір: {comp_size:.2f} MB"
        )

        # Підсумок
        ratio = orig_size / comp_size if comp_size > 0 else 0
        savings = (1 - comp_size / orig_size) * 100 if orig_size > 0 else 0
        orig_br = orig_info.get('bitrate', 0)
        comp_br = comp_info.get('bitrate', 0)

        self.compare_summary.setText(
            f"Стиснення: {ratio:.1f}x   |   "
            f"Економія: {savings:.1f}%   |   "
            f"Бітрейт: {format_bitrate(orig_br)} → {format_bitrate(comp_br)}"
        )

        # Слайдер
        self.compare_frame_count = min(
            int(self.compare_cap_orig.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(self.compare_cap_comp.get(cv2.CAP_PROP_FRAME_COUNT))
        )
        self.compare_slider.setMaximum(self.compare_frame_count - 1)
        self.compare_slider.setEnabled(True)
        self.compare_play_btn.setEnabled(True)

        self.compare_info_label.hide()

        # Показати перший кадр
        self.show_compare_frame(0)

        # Переключитись на вкладку порівняння
        self.tabs.setCurrentIndex(5)

    def show_compare_frame(self, idx: int):
        """Показати кадр на обох екранах."""
        if self.compare_cap_orig and self.compare_cap_comp:
            # Оригінал
            self.compare_cap_orig.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret1, frame1 = self.compare_cap_orig.read()
            if ret1:
                self.display_frame_on_label(self.left_video_label, frame1)

            # Стиснене
            self.compare_cap_comp.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret2, frame2 = self.compare_cap_comp.read()
            if ret2:
                self.display_frame_on_label(self.right_video_label, frame2)

            self.compare_frame_label.setText(f"Кадр: {idx + 1} / {self.compare_frame_count}")

    def display_frame_on_label(self, label: QLabel, frame):
        """Відобразити кадр на QLabel."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled)

    def on_compare_slider(self, value: int):
        self.show_compare_frame(value)

    def toggle_compare_play(self):
        if self.compare_playing:
            self.compare_timer.stop()
            self.compare_play_btn.setText("Play")
            self.compare_playing = False
        else:
            fps = 25
            if self.compare_cap_orig:
                fps = self.compare_cap_orig.get(cv2.CAP_PROP_FPS) or 25
            self.compare_timer.start(int(1000 / fps))
            self.compare_play_btn.setText("Pause")
            self.compare_playing = True

    def compare_next_frame(self):
        current = self.compare_slider.value()
        if current < self.compare_slider.maximum():
            self.compare_slider.setValue(current + 1)
        else:
            self.compare_slider.setValue(0)

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
        """Запуск основного кодування + бенчмарк паралельно."""
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
        self.bench_progress.setValue(0)

        bitrate = parse_bitrate(self.bitrate_combo.currentText())
        use_sr = self.sr_check.isChecked()
        sr_threshold = self.sr_threshold.value()

        # 1. Основне кодування NeuroQuant → вказаний вихід
        self.worker = EncodeWorker(inp, out, bitrate, use_sr, sr_threshold)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(lambda m: self.log_text.append(m))
        self.worker.finished.connect(self.on_encode_finished)
        self.worker.start()

        # 2. Паралельний бенчмарк для порівняння кодеків
        output_dir = Path(inp).parent / "benchmark_results"
        self.bench_input.setText(inp)
        self.bench_bitrate.setCurrentText(self.bitrate_combo.currentText())

        methods = ['h264', 'hevc', 'nq', 'nq_sr']
        self.benchmark_worker = BenchmarkWorker(inp, str(output_dir), [bitrate], methods)
        self.benchmark_worker.progress.connect(self.on_benchmark_progress)
        self.benchmark_worker.log.connect(lambda m: self.log_text.append(f"[Bench] {m}"))
        self.benchmark_worker.finished.connect(self.on_benchmark_finished)
        self.benchmark_worker.start()

    def on_benchmark_progress(self, percent: int, msg: str):
        """Оновлення прогресу на обох вкладках."""
        self.progress_bar.setValue(percent)
        self.bench_progress.setValue(percent)
        self.status_label.setText(msg)
        self.bench_status.setText(msg)

    def on_progress(self, percent: int, msg: str):
        self.progress_bar.setValue(percent)
        self.status_label.setText(msg)

    def cancel_encode(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.on_encode_finished(False, "Скасовано", {})
        if self.benchmark_worker and self.benchmark_worker.isRunning():
            self.benchmark_worker.terminate()
            self.benchmark_worker.wait()
            self.on_benchmark_finished(False, "Скасовано", [])

    def on_encode_finished(self, ok: bool, msg: str, stats: dict):
        self.encode_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText(msg)

        if ok and stats:
            self.last_stats = stats
            self.update_analysis_tab(stats)

            # Завантажити порівняння
            input_path = self.input_edit.text()
            output_path = self.output_edit.text()
            if input_path and output_path and Path(output_path).exists():
                self.load_compare_videos(input_path, output_path)

        if ok:
            QMessageBox.information(self, "Готово", msg)
        else:
            QMessageBox.critical(self, "Помилка", msg)

    def on_encode_benchmark_finished(self, ok: bool, msg: str, results: List[Dict]):
        """Обробка завершення бенчмарку з вкладки кодування."""
        self.encode_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText(msg)
        self.bench_status.setText(msg)

        if ok and results:
            self.benchmark_results = results
            self.load_benchmark_videos(results)

        if ok:
            QMessageBox.information(self, "Готово", f"Порівняння завершено. {msg}")
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

        # Генерація висновків
        conclusions = self.generate_conclusions(stats)
        self.conclusions_label.setText(conclusions)

    def generate_conclusions(self, stats: dict) -> str:
        """Генерація автоматичних висновків на основі аналізу."""
        conclusions = []

        # Аналіз складності
        if 'complexity' in stats:
            complexity = stats['complexity']
            avg_c = np.mean(complexity)
            std_c = np.std(complexity)

            if avg_c < 0.3:
                conclusions.append("• Контент: <b>простий</b> (статичні сцени, мало деталей)")
            elif avg_c < 0.6:
                conclusions.append("• Контент: <b>середньої складності</b> (помірний рух)")
            else:
                conclusions.append("• Контент: <b>складний</b> (багато руху, деталей)")

            if std_c > 0.2:
                conclusions.append("• Варіативність: <b>висока</b> — адаптивний QP дуже ефективний")
            else:
                conclusions.append("• Варіативність: <b>низька</b> — контент однорідний")

        # Аналіз QP
        qp_range = stats.get('qp_max', 0) - stats.get('qp_min', 0)
        if qp_range > 15:
            conclusions.append(f"• QP адаптація: <b>агресивна</b> (діапазон {qp_range})")
        elif qp_range > 8:
            conclusions.append(f"• QP адаптація: <b>помірна</b> (діапазон {qp_range})")
        else:
            conclusions.append(f"• QP адаптація: <b>мінімальна</b> (діапазон {qp_range})")

        # Стиснення
        ratio = stats.get('compression_ratio', 0)
        if ratio > 10:
            conclusions.append(f"• Стиснення: <b>дуже агресивне</b> ({ratio:.1f}x)")
        elif ratio > 5:
            conclusions.append(f"• Стиснення: <b>хороше</b> ({ratio:.1f}x)")
        else:
            conclusions.append(f"• Стиснення: <b>помірне</b> ({ratio:.1f}x)")

        # Рекомендації
        scene_cuts = len(stats.get('scene_cuts', []))
        if scene_cuts > 10:
            conclusions.append(f"• Рекомендація: багато змін сцен ({scene_cuts}) — підходить для адаптивного кодування")

        return "\n".join(conclusions) if conclusions else "Немає даних для аналізу"

    def on_benchmark_finished(self, ok: bool, msg: str, results: List[Dict]):
        self.bench_status.setText(msg)

        if ok and results:
            self.benchmark_results = results
            self.load_benchmark_videos(results)

    def load_benchmark_videos(self, results: List[Dict]):
        """Завантажити відео для 4 екранів бенчмарку."""
        # Закриваємо попередні
        for cap in self.bench_caps.values():
            if cap:
                cap.release()
        self.bench_caps = {}
        self.bench_paths = {}

        # Знаходимо шляхи до відео
        input_path = self.bench_input.text()
        output_dir = Path(input_path).parent / "benchmark_results"
        bitrate = parse_bitrate(self.bench_bitrate.currentText())
        stem = Path(input_path).stem

        method_labels = {
            'h264': (self.bench_video_h264, self.bench_params_h264, 0),
            'hevc': (self.bench_video_hevc, self.bench_params_hevc, 1),
            'nq': (self.bench_video_vvc, self.bench_params_vvc, 2),  # NQ на місці VVC
            'nq_sr': (self.bench_video_nq, self.bench_params_nq, 3),  # NQ+SR на місці NQ
        }

        best_psnr = 0
        best_method = ""

        for r in results:
            method = r['method']
            if method not in method_labels:
                continue

            video_label, params_label, table_row = method_labels[method]
            video_path = output_dir / f"{stem}_{method}_{bitrate}.mp4"

            if video_path.exists():
                # Відео екран (якщо є)
                if video_label is not None:
                    self.bench_caps[method] = cv2.VideoCapture(str(video_path))
                    self.bench_paths[method] = str(video_path)
                    params_label.setText(
                        f"Розмір: {r['size_mb']:.2f} MB\n"
                        f"Бітрейт: {format_bitrate(r['actual_bitrate'])}\n"
                        f"PSNR: {r['psnr']:.2f} dB\n"
                        f"SSIM: {r['ssim']:.4f}"
                    )

                # Заповнюємо таблицю
                self.bench_table.setItem(table_row, 1, QTableWidgetItem(f"{r['size_mb']:.2f} MB"))
                self.bench_table.setItem(table_row, 2, QTableWidgetItem(format_bitrate(r['actual_bitrate'])))
                self.bench_table.setItem(table_row, 3, QTableWidgetItem(f"{r['psnr']:.2f} dB"))
                self.bench_table.setItem(table_row, 4, QTableWidgetItem(f"{r['ssim']:.4f}"))

                if r['psnr'] > best_psnr:
                    best_psnr = r['psnr']
                    best_method = method
            else:
                if params_label is not None:
                    params_label.setText("Помилка кодування")

        # Виділяємо найкращий кодек в таблиці
        if best_method:
            row = method_labels[best_method][2]
            for col in range(5):
                item = self.bench_table.item(row, col)
                if item:
                    item.setBackground(QColor(200, 255, 200))

            codec_names = {'h264': 'H.264', 'hevc': 'HEVC', 'vvc': 'VVC', 'nq': 'NeuroQuant', 'nq_sr': 'NeuroQuant+SR'}
            self.bench_summary.setText(
                f"НАЙКРАЩИЙ: {codec_names.get(best_method, best_method)} — PSNR {best_psnr:.2f} dB"
            )

        # Налаштування слайдера
        if self.bench_caps:
            first_cap = list(self.bench_caps.values())[0]
            self.bench_frame_count = int(first_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.bench_slider.setMaximum(self.bench_frame_count - 1)
            self.bench_slider.setEnabled(True)
            self.bench_play_btn.setEnabled(True)
            self.show_bench_frame(0)

    def show_bench_frame(self, idx: int):
        """Показати кадр на всіх 4 екранах."""
        method_labels = {
            'h264': self.bench_video_h264,
            'hevc': self.bench_video_hevc,
            'nq': self.bench_video_vvc,      # NQ на 3-му екрані
            'nq_sr': self.bench_video_nq,    # NQ+SR на 4-му екрані
        }

        for method, cap in self.bench_caps.items():
            if cap and method in method_labels:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    self.display_frame_on_label(method_labels[method], frame)

        self.bench_frame_label.setText(f"Кадр: {idx + 1} / {self.bench_frame_count}")

    def on_bench_slider(self, value: int):
        self.show_bench_frame(value)

    def toggle_bench_play(self):
        if self.bench_playing:
            self.bench_timer.stop()
            self.bench_play_btn.setText("Play")
            self.bench_playing = False
        else:
            fps = 25
            if self.bench_caps:
                first_cap = list(self.bench_caps.values())[0]
                fps = first_cap.get(cv2.CAP_PROP_FPS) or 25
            self.bench_timer.start(int(1000 / fps))
            self.bench_play_btn.setText("Pause")
            self.bench_playing = True

    def bench_next_frame(self):
        current = self.bench_slider.value()
        if current < self.bench_slider.maximum():
            self.bench_slider.setValue(current + 1)
        else:
            self.bench_slider.setValue(0)


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
