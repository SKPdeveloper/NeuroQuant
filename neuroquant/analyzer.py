"""
ComplexityAnalyzer - аналіз складності кадрів відео.

Обчислює per-frame complexity на основі:
- Просторової складності (Sobel градієнти)
- Тимчасової складності (SAD між кадрами)
- Scene cuts (PySceneDetect)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from scenedetect import detect, ContentDetector
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .types import FrameComplexity
from .utils import get_video_info, log_info, log_warning


class ComplexityAnalyzer:
    """
    Аналізатор складності відеокадрів.

    Обчислює complexity_i = α·spatial + β·temporal + γ·cut
    для кожного кадру відео.
    """

    def __init__(
        self,
        spatial_weight: float = 0.4,
        temporal_weight: float = 0.5,
        cut_weight: float = 0.1,
        scene_threshold: float = 27.0,
        analysis_scale: float = 0.5,
    ):
        """
        Ініціалізація аналізатора.

        Args:
            spatial_weight: Вага просторової складності (α)
            temporal_weight: Вага тимчасової складності (β)
            cut_weight: Вага scene cuts (γ)
            scene_threshold: Поріг для PySceneDetect ContentDetector
            analysis_scale: Масштаб для аналізу (0.5 = половина роздільної здатності)
        """
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.cut_weight = cut_weight
        self.scene_threshold = scene_threshold
        self.analysis_scale = analysis_scale

        # Валідація ваг
        total_weight = spatial_weight + temporal_weight + cut_weight
        if abs(total_weight - 1.0) > 0.01:
            log_warning(f"Сума ваг = {total_weight:.2f}, очікується 1.0")

    def analyze(self, video_path: str, show_progress: bool = True) -> List[FrameComplexity]:
        """
        Аналізує відео та повертає список складностей для кожного кадру.

        Args:
            video_path: Шлях до відеофайлу
            show_progress: Показувати прогрес-бар

        Returns:
            Список FrameComplexity для кожного кадру
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Відео не знайдено: {video_path}")

        video_info = get_video_info(str(video_path))
        frame_count = video_info["frame_count"]

        log_info(f"Аналіз складності відео: {video_path.name}")
        log_info(f"Кадрів для обробки: {frame_count} ({video_info['duration']:.1f} сек @ {video_info['fps']:.0f}fps)")

        # Крок 1: Детекція scene cuts
        scene_cuts = self._detect_scene_cuts(str(video_path))
        log_info(f"Виявлено змін сцени: {len(scene_cuts)}")

        # Крок 2: Обчислення spatial та temporal складності
        spatial_values, temporal_values = self._compute_complexities(
            str(video_path), frame_count, show_progress
        )

        # Крок 3: Нормалізація
        spatial_norm = self._normalize(spatial_values)
        temporal_norm = self._normalize(temporal_values)

        # Крок 4: Формування результату
        results = []
        scene_cut_set = set(scene_cuts)

        for i in range(frame_count):
            is_cut = i in scene_cut_set
            cut_value = 1.0 if is_cut else 0.0

            complexity = (
                self.spatial_weight * spatial_norm[i] +
                self.temporal_weight * temporal_norm[i] +
                self.cut_weight * cut_value
            )

            results.append(FrameComplexity(
                frame_idx=i,
                spatial=spatial_norm[i],
                temporal=temporal_norm[i],
                is_scene_cut=is_cut,
                complexity=complexity,
            ))

        # Статистика
        complexities = [r.complexity for r in results]
        log_info(
            f"Середня складність: {np.mean(complexities):.3f} | "
            f"Макс: {np.max(complexities):.3f} | "
            f"Мін: {np.min(complexities):.3f}"
        )

        return results

    def _detect_scene_cuts(self, video_path: str) -> List[int]:
        """Детектує scene cuts через PySceneDetect."""
        scene_list = detect(video_path, ContentDetector(threshold=self.scene_threshold))

        # Повертаємо номери кадрів початку кожної сцени (крім першої)
        cuts = []
        for scene in scene_list[1:]:  # Пропускаємо першу сцену
            cuts.append(scene[0].frame_num)

        return cuts

    def _compute_complexities(
        self,
        video_path: str,
        frame_count: int,
        show_progress: bool
    ) -> Tuple[List[float], List[float]]:
        """
        Обчислює просторову та тимчасову складність для кожного кадру.

        Returns:
            (spatial_values, temporal_values) - ненормалізовані значення
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Не вдалося відкрити відео: {video_path}")

        spatial_values = []
        temporal_values = []
        prev_gray = None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.analysis_scale)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.analysis_scale)

        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[cyan]Аналіз складності[/cyan]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            disable=not show_progress,
        )

        with progress_ctx as progress:
            task = progress.add_task("Аналіз", total=frame_count)

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Зменшуємо для швидкості
                if self.analysis_scale != 1.0:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Просторова складність: градієнт Sobel
                spatial = self._compute_spatial_complexity(gray)
                spatial_values.append(spatial)

                # Тимчасова складність: SAD з попереднім кадром
                if prev_gray is not None:
                    temporal = self._compute_temporal_complexity(gray, prev_gray)
                else:
                    temporal = 0.0
                temporal_values.append(temporal)

                prev_gray = gray.copy()
                frame_idx += 1
                progress.update(task, advance=1)

        cap.release()

        # Доповнюємо, якщо зчитали менше кадрів
        while len(spatial_values) < frame_count:
            spatial_values.append(spatial_values[-1] if spatial_values else 0.0)
            temporal_values.append(temporal_values[-1] if temporal_values else 0.0)

        return spatial_values, temporal_values

    def _compute_spatial_complexity(self, gray: np.ndarray) -> float:
        """
        Обчислює просторову складність через градієнт Sobel.

        spatial = mean(|∇I|) = mean(sqrt(Gx² + Gy²))
        """
        # Sobel градієнти
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Магнітуда градієнта
        magnitude = np.sqrt(gx**2 + gy**2)

        return float(np.mean(magnitude))

    def _compute_temporal_complexity(self, current: np.ndarray, previous: np.ndarray) -> float:
        """
        Обчислює тимчасову складність через SAD.

        temporal = SAD(current, previous) / (width × height × 255)
        """
        diff = np.abs(current.astype(np.float32) - previous.astype(np.float32))
        sad = np.sum(diff)

        # Нормалізація на розмір і максимальне значення
        max_sad = current.shape[0] * current.shape[1] * 255.0

        return float(sad / max_sad)

    def _normalize(self, values: List[float]) -> List[float]:
        """
        Нормалізує значення до діапазону [0, 1] за percentile-based підходом.

        Використовує P5 і P95 для робастності до викидів.
        """
        if not values:
            return []

        arr = np.array(values)

        # Percentile-based нормалізація
        p5 = np.percentile(arr, 5)
        p95 = np.percentile(arr, 95)

        if p95 - p5 < 1e-6:
            # Усі значення майже однакові
            return [0.5] * len(values)

        normalized = (arr - p5) / (p95 - p5)
        normalized = np.clip(normalized, 0.0, 1.0)

        return normalized.tolist()

    def get_complexity_map(self, results: List[FrameComplexity]) -> np.ndarray:
        """Повертає numpy масив складностей."""
        return np.array([r.complexity for r in results])

    def get_scene_cut_frames(self, results: List[FrameComplexity]) -> List[int]:
        """Повертає номери кадрів зі зміною сцени."""
        return [r.frame_idx for r in results if r.is_scene_cut]

    def save_to_json(self, results: List[FrameComplexity], output_path: str) -> None:
        """Зберігає результати аналізу у JSON файл."""
        import json

        data = {
            "frame_count": len(results),
            "weights": {
                "spatial": self.spatial_weight,
                "temporal": self.temporal_weight,
                "cut": self.cut_weight,
            },
            "statistics": {
                "mean": float(np.mean([r.complexity for r in results])),
                "std": float(np.std([r.complexity for r in results])),
                "min": float(np.min([r.complexity for r in results])),
                "max": float(np.max([r.complexity for r in results])),
            },
            "frames": [
                {
                    "idx": r.frame_idx,
                    "spatial": round(r.spatial, 4),
                    "temporal": round(r.temporal, 4),
                    "is_cut": r.is_scene_cut,
                    "complexity": round(r.complexity, 4),
                }
                for r in results
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        log_info(f"Результати збережено: {output_path}")
