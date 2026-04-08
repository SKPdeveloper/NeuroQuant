"""
SRPostProcessor - селективний Real-ESRGAN постпроцесинг.

Застосовує Super Resolution лише до кадрів з низькою якістю (VMAF < threshold).
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .types import SRResult
from .utils import get_video_info, log_info, log_warning, log_error, get_temp_dir, check_cuda


class SRPostProcessor:
    """
    Постпроцесор з Real-ESRGAN для відновлення якості.

    Селективно застосовує SR до кадрів з VMAF нижче порогу.
    """

    def __init__(
        self,
        vmaf_threshold: float = 70.0,
        model_name: str = "RealESRNet_x2plus",
        tile_size: int = 512,
        tile_pad: int = 10,
        device: Optional[str] = None,
    ):
        """
        Ініціалізація постпроцесора.

        Args:
            vmaf_threshold: Поріг VMAF для активації SR
            model_name: Назва моделі Real-ESRGAN
            tile_size: Розмір тайлу для inference
            tile_pad: Padding для тайлів
            device: Пристрій для inference ("cuda" або "cpu")
        """
        self.vmaf_threshold = vmaf_threshold
        self.model_name = model_name
        self.tile_size = tile_size
        self.tile_pad = tile_pad

        # Визначаємо пристрій
        if device is None:
            cuda_available, cuda_info = check_cuda()
            self.device = "cuda" if cuda_available else "cpu"
            if cuda_available:
                log_info(f"GPU: {cuda_info}")
            else:
                log_warning("CUDA недоступна, використовую CPU (повільно)")
        else:
            self.device = device

        self.model = None
        self.scale = 2 if "x2" in model_name else 4

    def _load_model(self):
        """Завантажує модель Real-ESRGAN через spandrel."""
        if self.model is not None:
            return

        log_info(f"Завантаження моделі {self.model_name}...")

        try:
            import spandrel
            from pathlib import Path

            # Визначаємо scale
            self.scale = 2 if "x2" in self.model_name else 4

            # Шукаємо модель локально
            project_root = Path(__file__).parent.parent
            local_model = project_root / "models" / f"RealESRGAN_x{self.scale}plus.pth"

            if local_model.exists():
                model_path = str(local_model)
                log_info(f"Використовую локальну модель: {model_path}")
            else:
                # Fallback на HuggingFace
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(
                    repo_id="xinntao/Real-ESRGAN",
                    filename=f"RealESRGAN_x{self.scale}plus.pth",
                )

            # Завантажуємо модель через spandrel
            self.model = spandrel.ModelLoader().load_from_file(model_path)
            self.model = self.model.to(self.device)
            if self.device == "cuda":
                self.model = self.model.half()
            self.model.eval()

            log_info(f"Модель завантажено: {self.model_name} (×{self.scale})")

        except Exception as e:
            self.model = None
            log_error(f"Помилка завантаження моделі: {e}")
            raise

    def process_video(
        self,
        input_path: str,
        reference_path: str,
        output_path: str,
        vmaf_scores: Optional[List[float]] = None,
        show_progress: bool = True,
    ) -> SRResult:
        """
        Обробляє відео з селективним SR.

        Args:
            input_path: Закодоване відео (для обробки)
            reference_path: Оригінальне відео (для VMAF)
            output_path: Вихідний файл
            vmaf_scores: Per-frame VMAF (якщо вже обчислені)
            show_progress: Показувати прогрес

        Returns:
            SRResult з інформацією про результат
        """
        import time
        start_time = time.time()

        video_info = get_video_info(input_path)
        frame_count = video_info["frame_count"]
        fps = video_info["fps"]
        width = video_info["width"]
        height = video_info["height"]

        log_info(f"SR постпроцесинг: {Path(input_path).name}")

        # Крок 1: Обчислюємо per-frame VMAF, якщо не передано
        if vmaf_scores is None:
            log_info("Обчислення per-frame VMAF...")
            vmaf_scores = self._compute_vmaf_per_frame(input_path, reference_path)

        # Крок 2: Визначаємо кадри для SR
        frames_to_process = []
        for i, score in enumerate(vmaf_scores):
            if score < self.vmaf_threshold:
                frames_to_process.append(i)

        sr_ratio = len(frames_to_process) / frame_count * 100
        log_info(f"Кадрів для SR (VMAF < {self.vmaf_threshold}): {len(frames_to_process)} / {frame_count} ({sr_ratio:.1f}%)")

        if not frames_to_process:
            log_info("Всі кадри мають достатню якість, SR не потрібен")
            # Просто копіюємо файл
            import shutil
            shutil.copy(input_path, output_path)
            return SRResult(
                output_path=output_path,
                frames_processed=0,
                frames_total=frame_count,
                processing_time=time.time() - start_time,
                success=True,
            )

        # Крок 3: Завантажуємо модель
        self._load_model()

        # Крок 4: Обробляємо кадри
        temp_dir = get_temp_dir() / "sr_frames"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Декодуємо всі кадри
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return SRResult(
                output_path=output_path,
                frames_processed=0,
                frames_total=frame_count,
                processing_time=time.time() - start_time,
                success=False,
                error_message=f"Не вдалося відкрити відео: {input_path}",
            )

        # Створюємо відео writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output_path = str(temp_dir / "temp_output.mp4")
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            return SRResult(
                output_path=output_path,
                frames_processed=0,
                frames_total=frame_count,
                processing_time=time.time() - start_time,
                success=False,
                error_message=f"Не вдалося створити VideoWriter",
            )

        frames_to_process_set = set(frames_to_process)
        processed_count = 0

        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[cyan]SR постпроцесинг[/cyan]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            disable=not show_progress,
        )

        try:
            with progress_ctx as progress:
                task = progress.add_task("Обробка", total=frame_count)

                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx in frames_to_process_set:
                        # Застосовуємо SR
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            sr_frame = self._enhance_with_spandrel(frame_rgb)
                            frame = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)

                            # Resize до оригінального розміру, якщо потрібно
                            if frame.shape[:2] != (height, width):
                                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)

                            processed_count += 1
                        except Exception as e:
                            log_warning(f"Помилка SR для кадру {frame_idx}: {e}")

                    out.write(frame)
                    frame_idx += 1
                    progress.update(task, advance=1)
        finally:
            cap.release()
            out.release()

        # Крок 5: Перекодовуємо у фінальний формат
        temp_output = temp_dir / "temp_output.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(temp_output),
            "-c:v", "libx265",
            "-crf", "18",
            "-preset", "fast",
            output_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                log_error(f"Помилка перекодування: {result.stderr[-300:]}")
                return SRResult(
                    output_path=output_path,
                    frames_processed=processed_count,
                    frames_total=frame_count,
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message=f"Помилка ffmpeg: {result.stderr[-300:]}",
                )
        except Exception as e:
            log_error(f"Помилка виклику ffmpeg: {e}")
            return SRResult(
                output_path=output_path,
                frames_processed=processed_count,
                frames_total=frame_count,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )
        finally:
            # Очищуємо тимчасові файли
            temp_output.unlink(missing_ok=True)

        processing_time = time.time() - start_time
        log_info(f"SR завершено за {processing_time:.1f} сек")

        return SRResult(
            output_path=output_path,
            frames_processed=processed_count,
            frames_total=frame_count,
            processing_time=processing_time,
            success=True,
        )

    def _compute_vmaf_per_frame(
        self,
        distorted_path: str,
        reference_path: str,
    ) -> List[float]:
        """
        Обчислює per-frame VMAF через ffmpeg libvmaf.

        Returns:
            Список VMAF scores для кожного кадру
        """
        temp_dir = get_temp_dir()
        vmaf_log = temp_dir / "vmaf_log.json"

        # Екрануємо шлях для Windows (замінюємо \ на / або \\)
        vmaf_log_escaped = str(vmaf_log).replace("\\", "/")

        cmd = [
            "ffmpeg",
            "-i", distorted_path,
            "-i", reference_path,
            "-lavfi", f"libvmaf=log_path={vmaf_log_escaped}:log_fmt=json:n_threads=4",
            "-f", "null",
            "NUL" if os.name == "nt" else "/dev/null",
        ]

        log_info(f"VMAF команда: {' '.join(cmd[:6])}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            log_warning(f"VMAF помилка: {result.stderr[-500:] if result.stderr else 'unknown'}")
            # Fallback: повертаємо середнє значення
            video_info = get_video_info(distorted_path)
            return [75.0] * video_info["frame_count"]

        # Парсимо JSON лог
        import json
        with open(vmaf_log, "r") as f:
            vmaf_data = json.load(f)

        scores = []
        for frame in vmaf_data.get("frames", []):
            metrics = frame.get("metrics", {})
            vmaf_score = metrics.get("vmaf", 75.0)
            scores.append(vmaf_score)

        return scores

    def _enhance_with_spandrel(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Покращує кадр через spandrel (Real-ESRGAN).

        Args:
            frame_rgb: RGB кадр (numpy array, uint8)

        Returns:
            Покращений RGB кадр (numpy array, uint8)
        """
        # Конвертуємо в тензор
        img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        if self.device == "cuda":
            img_tensor = img_tensor.half()

        # Tile-based inference для великих зображень
        _, _, h, w = img_tensor.shape
        if h * w > self.tile_size * self.tile_size:
            output = self._tile_inference(img_tensor, h, w)
        else:
            with torch.no_grad():
                output = self.model(img_tensor)

        # Конвертуємо назад у numpy
        output = output.squeeze(0).float().clamp(0, 1)
        output = (output.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        return output

    def _tile_inference(self, img_tensor: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Tile-based inference для економії VRAM."""
        tile = self.tile_size
        tile_pad = self.tile_pad
        scale = self.scale

        # Вихідний тензор
        output_h = h * scale
        output_w = w * scale
        output = torch.zeros((1, 3, output_h, output_w), dtype=img_tensor.dtype, device=img_tensor.device)

        for y in range(0, h, tile):
            for x in range(0, w, tile):
                # Визначаємо границі тайлу з padding
                y1 = max(0, y - tile_pad)
                x1 = max(0, x - tile_pad)
                y2 = min(h, y + tile + tile_pad)
                x2 = min(w, x + tile + tile_pad)

                # Вирізаємо тайл
                tile_input = img_tensor[:, :, y1:y2, x1:x2]

                with torch.no_grad():
                    tile_output = self.model(tile_input)

                # Визначаємо границі виходу
                out_y1 = (y - y1) * scale
                out_x1 = (x - x1) * scale
                out_y2 = out_y1 + min(tile, h - y) * scale
                out_x2 = out_x1 + min(tile, w - x) * scale

                # Копіюємо результат
                output[:, :, y*scale:(y+min(tile,h-y))*scale, x*scale:(x+min(tile,w-x))*scale] = \
                    tile_output[:, :, out_y1:out_y2, out_x1:out_x2]

        return output

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Покращує один кадр через Real-ESRGAN.

        Args:
            frame: BGR кадр (numpy array)

        Returns:
            Покращений BGR кадр
        """
        self._load_model()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sr_frame = self._enhance_with_spandrel(frame_rgb)
        return cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)

    def get_model_info(self) -> dict:
        """Повертає інформацію про модель."""
        return {
            "model_name": self.model_name,
            "scale": self.scale,
            "device": self.device,
            "tile_size": self.tile_size,
            "vmaf_threshold": self.vmaf_threshold,
        }
