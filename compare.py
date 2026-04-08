"""
NeuroQuant Compare - простий GUI для порівняння оригіналу і стисненого відео.
Два екрани поруч з технічними параметрами.
"""

import sys
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QSlider, QGroupBox, QFrame
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont

from neuroquant.utils import get_video_info, format_bitrate


class VideoPlayer:
    """Простий плеєр для відео через OpenCV."""

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.info = get_video_info(path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.current_frame = 0

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Отримати кадр за індексом."""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame_idx
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def close(self):
        self.cap.release()


class CompareWindow(QMainWindow):
    """Вікно порівняння двох відео."""

    def __init__(self):
        super().__init__()
        self.original: Optional[VideoPlayer] = None
        self.compressed: Optional[VideoPlayer] = None
        self.playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("NeuroQuant Compare")
        self.setMinimumSize(1200, 700)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Верхня панель з кнопками
        top_bar = QHBoxLayout()

        btn_original = QPushButton("Оригінал...")
        btn_original.clicked.connect(self.load_original)
        top_bar.addWidget(btn_original)

        btn_compressed = QPushButton("Стиснене...")
        btn_compressed.clicked.connect(self.load_compressed)
        top_bar.addWidget(btn_compressed)

        top_bar.addStretch()

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        top_bar.addWidget(self.play_btn)

        main_layout.addLayout(top_bar)

        # Два екрани поруч
        screens_layout = QHBoxLayout()

        # Лівий екран — оригінал
        left_group = QGroupBox("ОРИГІНАЛ")
        left_layout = QVBoxLayout(left_group)

        self.left_video = QLabel()
        self.left_video.setMinimumSize(560, 315)
        self.left_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_video.setStyleSheet("background-color: #1a1a1a;")
        left_layout.addWidget(self.left_video)

        self.left_info = QLabel("Завантажте відео")
        self.left_info.setFont(QFont("Consolas", 9))
        self.left_info.setStyleSheet("color: #888; padding: 5px;")
        left_layout.addWidget(self.left_info)

        screens_layout.addWidget(left_group)

        # Правий екран — стиснене
        right_group = QGroupBox("СТИСНЕНЕ (NeuroQuant)")
        right_layout = QVBoxLayout(right_group)

        self.right_video = QLabel()
        self.right_video.setMinimumSize(560, 315)
        self.right_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_video.setStyleSheet("background-color: #1a1a1a;")
        right_layout.addWidget(self.right_video)

        self.right_info = QLabel("Завантажте відео")
        self.right_info.setFont(QFont("Consolas", 9))
        self.right_info.setStyleSheet("color: #888; padding: 5px;")
        right_layout.addWidget(self.right_info)

        screens_layout.addWidget(right_group)

        main_layout.addLayout(screens_layout)

        # Слайдер
        slider_layout = QHBoxLayout()
        self.frame_label = QLabel("Кадр: 0 / 0")
        slider_layout.addWidget(self.frame_label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider_change)
        slider_layout.addWidget(self.slider)

        main_layout.addLayout(slider_layout)

        # Порівняння
        compare_group = QGroupBox("Порівняння")
        compare_layout = QHBoxLayout(compare_group)

        self.compare_label = QLabel("Завантажте обидва відео для порівняння")
        self.compare_label.setFont(QFont("Consolas", 10))
        compare_layout.addWidget(self.compare_label)

        main_layout.addWidget(compare_group)

    def load_original(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Оберіть оригінал", "",
            "Відео (*.mp4 *.mkv *.avi *.mov)"
        )
        if path:
            if self.original:
                self.original.close()
            self.original = VideoPlayer(path)
            self.update_info()
            self.show_frame(0)

    def load_compressed(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Оберіть стиснене відео", "",
            "Відео (*.mp4 *.mkv *.avi *.mov)"
        )
        if path:
            if self.compressed:
                self.compressed.close()
            self.compressed = VideoPlayer(path)
            self.update_info()
            self.show_frame(0)

    def update_info(self):
        """Оновити інформацію про відео."""
        if self.original:
            info = self.original.info
            size_mb = Path(self.original.path).stat().st_size / (1024 * 1024)
            self.left_info.setText(
                f"Файл: {Path(self.original.path).name}\n"
                f"Роздільність: {info['width']} x {info['height']}\n"
                f"FPS: {info['fps']:.2f}\n"
                f"Тривалість: {info['duration']:.1f} сек\n"
                f"Кадрів: {info['frame_count']}\n"
                f"Бітрейт: {format_bitrate(info.get('bitrate', 0))}\n"
                f"Розмір: {size_mb:.2f} MB"
            )

        if self.compressed:
            info = self.compressed.info
            size_mb = Path(self.compressed.path).stat().st_size / (1024 * 1024)
            self.right_info.setText(
                f"Файл: {Path(self.compressed.path).name}\n"
                f"Роздільність: {info['width']} x {info['height']}\n"
                f"FPS: {info['fps']:.2f}\n"
                f"Тривалість: {info['duration']:.1f} сек\n"
                f"Кадрів: {info['frame_count']}\n"
                f"Бітрейт: {format_bitrate(info.get('bitrate', 0))}\n"
                f"Розмір: {size_mb:.2f} MB"
            )

        # Порівняння
        if self.original and self.compressed:
            orig_size = Path(self.original.path).stat().st_size
            comp_size = Path(self.compressed.path).stat().st_size
            ratio = orig_size / comp_size if comp_size > 0 else 0
            savings = (1 - comp_size / orig_size) * 100 if orig_size > 0 else 0

            orig_br = self.original.info.get('bitrate', 0)
            comp_br = self.compressed.info.get('bitrate', 0)
            br_reduction = (1 - comp_br / orig_br) * 100 if orig_br > 0 else 0

            self.compare_label.setText(
                f"Стиснення: {ratio:.1f}x  |  "
                f"Економія: {savings:.1f}%  |  "
                f"Бітрейт: {format_bitrate(orig_br)} → {format_bitrate(comp_br)} ({br_reduction:.0f}% менше)"
            )

            # Налаштування слайдера
            max_frames = min(self.original.frame_count, self.compressed.frame_count)
            self.slider.setMaximum(max_frames - 1)
            self.slider.setEnabled(True)
            self.play_btn.setEnabled(True)

    def show_frame(self, idx: int):
        """Показати кадр на обох екранах."""
        if self.original:
            frame = self.original.get_frame(idx)
            if frame is not None:
                self.display_frame(self.left_video, frame)

        if self.compressed:
            frame = self.compressed.get_frame(idx)
            if frame is not None:
                self.display_frame(self.right_video, frame)

        max_frames = 0
        if self.original:
            max_frames = self.original.frame_count
        if self.compressed:
            max_frames = max(max_frames, self.compressed.frame_count)

        self.frame_label.setText(f"Кадр: {idx + 1} / {max_frames}")

    def display_frame(self, label: QLabel, frame: np.ndarray):
        """Відобразити кадр на QLabel."""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Масштабуємо під розмір label
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled)

    def on_slider_change(self, value: int):
        """Обробка зміни слайдера."""
        self.show_frame(value)

    def toggle_play(self):
        """Play/Pause."""
        if self.playing:
            self.timer.stop()
            self.play_btn.setText("Play")
            self.playing = False
        else:
            fps = 25
            if self.original:
                fps = self.original.fps
            self.timer.start(int(1000 / fps))
            self.play_btn.setText("Pause")
            self.playing = True

    def next_frame(self):
        """Наступний кадр при відтворенні."""
        current = self.slider.value()
        if current < self.slider.maximum():
            self.slider.setValue(current + 1)
        else:
            self.slider.setValue(0)

    def closeEvent(self, event):
        """Закриття вікна."""
        if self.original:
            self.original.close()
        if self.compressed:
            self.compressed.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = CompareWindow()
    window.show()

    sys.exit(app.exec())
