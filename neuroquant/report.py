"""
ReportGenerator - генерація звітів та візуалізацій.

Створює:
- RD-криві (PSNR, SSIM, VMAF vs bitrate)
- Per-frame VMAF heatmaps
- Таблиці порівняння
- HTML звіти
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Для роботи без GUI
import seaborn as sns

from .utils import log_info, log_success, ensure_dir, format_bitrate


@dataclass
class RDPoint:
    """Точка на RD-кривій."""
    bitrate: int
    psnr: float
    ssim: float
    vmaf: float
    method: str


class ReportGenerator:
    """
    Генератор звітів та візуалізацій.

    Створює графіки RD-кривих, heatmaps якості
    та HTML звіти з результатами бенчмарку.
    """

    # Кольори для методів
    METHOD_COLORS = {
        "h264": "#1f77b4",   # синій
        "hevc": "#ff7f0e",   # оранжевий
        "vvc": "#2ca02c",    # зелений
        "nq": "#d62728",     # червоний
        "nq_sr": "#9467bd",  # фіолетовий
    }

    METHOD_LABELS = {
        "h264": "H.264 (libx264)",
        "hevc": "HEVC (libx265)",
        "vvc": "VVC (libvvenc)",
        "nq": "NeuroQuant",
        "nq_sr": "NeuroQuant + SR",
    }

    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        """
        Ініціалізація генератора.

        Args:
            style: Стиль matplotlib
        """
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("seaborn-whitegrid")

        # Налаштування для українського тексту
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']

    def generate_from_json(
        self,
        json_path: str,
        output_dir: str,
        formats: List[str] = None,
    ) -> List[str]:
        """
        Генерує звіт з JSON файлу бенчмарку.

        Args:
            json_path: Шлях до JSON з результатами
            output_dir: Директорія для виводу
            formats: Формати виводу ("png", "pdf", "html")

        Returns:
            Список створених файлів
        """
        formats = formats or ["png", "html"]
        output_dir = ensure_dir(output_dir)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        created_files = []

        # Генеруємо RD-криві
        for metric in ["psnr", "ssim", "vmaf"]:
            for fmt in formats:
                if fmt in ["png", "pdf", "svg"]:
                    path = self._plot_rd_curve(data, metric, output_dir, fmt)
                    if path:
                        created_files.append(path)

        # Генеруємо порівняльну таблицю
        table_path = self._generate_comparison_table(data, output_dir)
        if table_path:
            created_files.append(table_path)

        # Генеруємо HTML звіт
        if "html" in formats:
            html_path = self._generate_html_report(data, output_dir)
            if html_path:
                created_files.append(html_path)

        log_success(f"Створено {len(created_files)} файлів звіту")
        return created_files

    def _plot_rd_curve(
        self,
        data: dict,
        metric: str,
        output_dir: Path,
        fmt: str = "png",
    ) -> Optional[str]:
        """Створює RD-криву для вказаної метрики."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Збираємо дані по методах
        method_data = {}

        for video in data.get("videos", []):
            for result in video.get("results", []):
                method = result["method"]
                if method not in method_data:
                    method_data[method] = {"bitrates": [], "values": []}

                bitrate = result.get("bitrate_actual") or result.get("bitrate_target", 0)
                value = result.get(metric, 0)

                method_data[method]["bitrates"].append(bitrate / 1000)  # kbps
                method_data[method]["values"].append(value)

        # Будуємо криві
        for method, values in method_data.items():
            if not values["bitrates"]:
                continue

            # Сортуємо за бітрейтом
            sorted_pairs = sorted(zip(values["bitrates"], values["values"]))
            bitrates, metrics = zip(*sorted_pairs)

            color = self.METHOD_COLORS.get(method, "#333333")
            label = self.METHOD_LABELS.get(method, method.upper())

            ax.plot(
                bitrates, metrics,
                marker='o',
                linewidth=2,
                markersize=8,
                color=color,
                label=label,
            )

        # Налаштування графіка
        metric_labels = {
            "psnr": ("PSNR (dB)", "Rate-Distortion: PSNR"),
            "ssim": ("SSIM", "Rate-Distortion: SSIM"),
            "vmaf": ("VMAF", "Rate-Distortion: VMAF"),
        }

        ylabel, title = metric_labels.get(metric, (metric.upper(), f"Rate-Distortion: {metric}"))

        ax.set_xlabel("Бітрейт (kbps)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        # Логарифмічна шкала для бітрейту
        ax.set_xscale('log')

        plt.tight_layout()

        # Зберігаємо
        output_path = output_dir / f"rd_curve_{metric}.{fmt}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        log_info(f"Створено: {output_path.name}")
        return str(output_path)

    def _generate_comparison_table(
        self,
        data: dict,
        output_dir: Path,
    ) -> Optional[str]:
        """Генерує CSV таблицю порівняння."""
        import csv

        output_path = output_dir / "comparison_table.csv"

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Заголовок
            writer.writerow([
                "Відео", "Метод", "Цільовий бітрейт (kbps)",
                "Фактичний бітрейт (kbps)", "PSNR (dB)", "SSIM", "VMAF",
                "Розмір (MB)", "Час кодування (с)"
            ])

            for video in data.get("videos", []):
                video_name = video.get("video_name", "unknown")

                for result in video.get("results", []):
                    writer.writerow([
                        video_name,
                        result["method"].upper(),
                        result.get("bitrate_target", 0) // 1000,
                        (result.get("bitrate_actual") or result.get("bitrate_target", 0)) // 1000,
                        f"{result.get('psnr', 0):.2f}",
                        f"{result.get('ssim', 0):.4f}",
                        f"{result.get('vmaf', 0):.1f}",
                        f"{result.get('file_size', 0) / 1024 / 1024:.2f}",
                        f"{result.get('encode_time', 0):.1f}",
                    ])

        log_info(f"Створено: {output_path.name}")
        return str(output_path)

    def _generate_html_report(
        self,
        data: dict,
        output_dir: Path,
    ) -> Optional[str]:
        """Генерує HTML звіт."""
        output_path = output_dir / "report.html"

        # Генеруємо графіки у base64
        import base64
        from io import BytesIO

        def fig_to_base64(fig) -> str:
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')

        # Створюємо графіки
        charts = {}
        for metric in ["psnr", "ssim", "vmaf"]:
            fig = self._create_rd_figure(data, metric)
            if fig:
                charts[metric] = fig_to_base64(fig)
                plt.close(fig)

        # BD-Rate таблиця
        bd_rates = data.get("bd_rates", {})

        # HTML шаблон
        html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroQuant - Звіт бенчмарку</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #3498db;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .positive {{
            color: #27ae60;
            font-weight: bold;
        }}
        .negative {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .chart {{
            text-align: center;
            margin: 20px 0;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .stat-box {{
            background: #3498db;
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-box h3 {{
            margin: 0;
            font-size: 24px;
        }}
        .stat-box p {{
            margin: 5px 0 0;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <h1>🎬 NeuroQuant - Звіт бенчмарку</h1>

    <div class="card">
        <h2>📊 Загальна інформація</h2>
        <p><strong>Дата:</strong> {data.get('timestamp', 'N/A')}</p>
        <p><strong>Методи:</strong> {', '.join(m.upper() for m in data.get('methods', []))}</p>
        <p><strong>Бітрейти:</strong> {', '.join(format_bitrate(b) for b in data.get('bitrates', []))}</p>
        <p><strong>Відео:</strong> {len(data.get('videos', []))}</p>
    </div>

    <div class="card">
        <h2>📈 BD-Rate відносно H.264</h2>
        <table>
            <tr>
                <th>Метод</th>
                <th>BD-Rate (PSNR)</th>
                <th>BD-Rate (VMAF)</th>
            </tr>
            {''.join(self._bd_rate_row(m, r) for m, r in bd_rates.items())}
        </table>
        <p><em>Від'ємне значення = краще за H.264</em></p>
    </div>
"""

        # Додаємо графіки
        for metric, chart_data in charts.items():
            metric_name = {"psnr": "PSNR", "ssim": "SSIM", "vmaf": "VMAF"}.get(metric, metric)
            html += f"""
    <div class="card">
        <h2>📉 Rate-Distortion: {metric_name}</h2>
        <div class="chart">
            <img src="data:image/png;base64,{chart_data}" alt="RD-curve {metric_name}">
        </div>
    </div>
"""

        # Детальні результати по відео
        for video in data.get("videos", []):
            html += f"""
    <div class="card">
        <h2>🎥 {video.get('video_name', 'Unknown')}</h2>
        <p>
            <strong>Роздільність:</strong> {video.get('width', 0)}×{video.get('height', 0)} |
            <strong>FPS:</strong> {video.get('fps', 0):.0f} |
            <strong>Тривалість:</strong> {video.get('duration', 0):.1f} сек
        </p>
        <table>
            <tr>
                <th>Метод</th>
                <th>Бітрейт</th>
                <th>PSNR</th>
                <th>SSIM</th>
                <th>VMAF</th>
                <th>Час</th>
            </tr>
            {''.join(self._result_row(r) for r in video.get('results', []))}
        </table>
    </div>
"""

        html += """
    <footer style="text-align: center; padding: 20px; color: #666;">
        <p>Згенеровано NeuroQuant Benchmark Engine</p>
    </footer>
</body>
</html>
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        log_info(f"Створено: {output_path.name}")
        return str(output_path)

    def _bd_rate_row(self, method: str, rates: dict) -> str:
        """Генерує рядок таблиці BD-Rate."""
        psnr = rates.get("bd_rate_psnr", 0)
        vmaf = rates.get("bd_rate_vmaf", 0)

        psnr_class = "positive" if psnr < 0 else "negative"
        vmaf_class = "positive" if vmaf < 0 else "negative"

        return f"""
            <tr>
                <td>{self.METHOD_LABELS.get(method, method.upper())}</td>
                <td class="{psnr_class}">{psnr:+.1f}%</td>
                <td class="{vmaf_class}">{vmaf:+.1f}%</td>
            </tr>
"""

    def _result_row(self, result: dict) -> str:
        """Генерує рядок таблиці результатів."""
        bitrate = (result.get("bitrate_actual") or result.get("bitrate_target", 0)) // 1000
        return f"""
            <tr>
                <td>{result.get('method', 'N/A').upper()}</td>
                <td>{bitrate} kbps</td>
                <td>{result.get('psnr', 0):.2f} dB</td>
                <td>{result.get('ssim', 0):.4f}</td>
                <td>{result.get('vmaf', 0):.1f}</td>
                <td>{result.get('encode_time', 0):.1f}s</td>
            </tr>
"""

    def _create_rd_figure(self, data: dict, metric: str):
        """Створює matplotlib figure для RD-кривої."""
        fig, ax = plt.subplots(figsize=(8, 5))

        method_data = {}
        for video in data.get("videos", []):
            for result in video.get("results", []):
                method = result["method"]
                if method not in method_data:
                    method_data[method] = {"bitrates": [], "values": []}

                bitrate = result.get("bitrate_actual") or result.get("bitrate_target", 0)
                value = result.get(metric, 0)

                method_data[method]["bitrates"].append(bitrate / 1000)
                method_data[method]["values"].append(value)

        for method, values in method_data.items():
            if not values["bitrates"]:
                continue

            sorted_pairs = sorted(zip(values["bitrates"], values["values"]))
            bitrates, metrics = zip(*sorted_pairs)

            color = self.METHOD_COLORS.get(method, "#333333")
            label = self.METHOD_LABELS.get(method, method.upper())

            ax.plot(bitrates, metrics, marker='o', linewidth=2,
                    markersize=6, color=color, label=label)

        metric_labels = {
            "psnr": "PSNR (dB)",
            "ssim": "SSIM",
            "vmaf": "VMAF",
        }

        ax.set_xlabel("Бітрейт (kbps)", fontsize=10)
        ax.set_ylabel(metric_labels.get(metric, metric), fontsize=10)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        plt.tight_layout()
        return fig

    def plot_vmaf_heatmap(
        self,
        vmaf_scores: List[float],
        output_path: str,
        title: str = "Per-frame VMAF",
    ) -> str:
        """
        Створює heatmap per-frame VMAF.

        Args:
            vmaf_scores: Список VMAF для кожного кадру
            output_path: Шлях для збереження
            title: Заголовок графіка

        Returns:
            Шлях до створеного файлу
        """
        # Reshape у 2D для heatmap
        n_frames = len(vmaf_scores)
        cols = min(100, n_frames)
        rows = (n_frames + cols - 1) // cols

        # Padding до повного прямокутника
        padded = vmaf_scores + [np.nan] * (rows * cols - n_frames)
        data = np.array(padded).reshape(rows, cols)

        fig, ax = plt.subplots(figsize=(14, 6))

        sns.heatmap(
            data,
            ax=ax,
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
            cbar_kws={"label": "VMAF"},
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Кадр (mod 100)", fontsize=10)
        ax.set_ylabel("Секція", fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        log_info(f"Створено: {Path(output_path).name}")
        return output_path
