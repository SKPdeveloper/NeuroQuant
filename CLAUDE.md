# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ВАЖЛИВО: Python версія

**Завжди використовувати `py -3.9` замість `python`!**

```bash
py -3.9 gui.py              # GUI
py -3.9 -m neuroquant ...   # CLI
```

## Project Overview

**NeuroQuant** — інтелектуальна система аналізу та стиснення відео, що поєднує per-frame rate control на основі R-λ теорії з Real-ESRGAN постпроцесингом. Дипломний проєкт на Python 3.11+.

Ключова ідея: адаптивний QP-план для x265 + селективний SR для відновлення якості агресивно стиснених кадрів.

## Build & Run Commands

```bash
# Встановлення залежностей
pip install -r requirements.txt

# Перевірка ffmpeg (потрібен libx264, libx265, libvvenc, libvmaf)
ffmpeg -version

# Кодування одного відео
python -m neuroquant encode input.mp4 output.mp4 --bitrate 1M --sr --sr-threshold 70

# Повний бенчмарк
python -m neuroquant benchmark test_videos/ \
    --methods h264,hevc,vvc,nq,nq_sr \
    --bitrates 150k,300k,600k,1200k,2500k \
    --output results/

# Генерація звіту
python -m neuroquant report results/results.json --format html --output results/report.html

# Аналіз складності (без кодування)
python -m neuroquant analyze input.mp4 --output complexity_map.json --plot
```

## Architecture

```
input.mp4 → ComplexityAnalyzer → RLambdaController → FFmpegEncoder → SRPostProcessor → output.mp4
                  ↓                      ↓                  ↓                ↓
           complexity_map[]        qp_plan[]         encoded.mp4      +Real-ESRGAN
```

### Core Modules (neuroquant/)

| Модуль | Призначення |
|--------|-------------|
| `analyzer.py` | **ComplexityAnalyzer** — Sobel градієнти (spatial), SAD між кадрами (temporal), PySceneDetect (cuts). Вихід: `complexity_i ∈ [0,1]` |
| `controller.py` | **RLambdaController** — R-λ модель rate control. Формує per-frame QP план для x265 |
| `encoder.py` | **FFmpegEncoder** — обгортка ffmpeg з libx265, застосовує QP-план через `qpfile` |
| `sr_processor.py` | **SRPostProcessor** — Real-ESRGAN x2/x4 для кадрів з VMAF < threshold |
| `metrics.py` | PSNR, SSIM (scikit-image), VMAF (libvmaf), BD-Rate (scipy spline) |
| `benchmark.py` | **BenchmarkEngine** — порівняння h264/hevc/vvc/nq/nq_sr |
| `report.py` | RD-криві (matplotlib), heatmaps (seaborn), JSON-експорт |
| `cli.py` | Click CLI — точка входу |

### R-λ Model (key formulas)

```python
# Залежність бітрейт-лямбда
R = α · λ^β  # α ≈ 6.7542, β ≈ −1.7860 для x265

# Зв'язок λ і QP
λ = c · 2^((QP - 12) / 3)  # c = 0.85 (P-frame), 0.57 (B-frame)

# Per-frame QP адаптація
QP_i = QP_base - round(Δmax · (complexity_i - mean) / std)
# Δmax = 6, clip до [18, 45]
```

### Complexity Formula

```python
complexity_i = 0.4 * spatial_i + 0.5 * temporal_i + 0.1 * cut_i
```

- `spatial_i`: нормований градієнт Sobel
- `temporal_i`: SAD(frame_i, frame_{i-1}) / (w × h × 255)
- `cut_i`: 1 при scene cut (PySceneDetect), інакше 0

## QP File Format (x265)

```
# frames.qp
0 I 28
1 P 31
2 B 33
```

ffmpeg: `ffmpeg -i in.mp4 -c:v libx265 -x265-params "qpfile=frames.qp:aq-mode=0" out.mp4`

## Key Dependencies

| Пакет | Версія | Для чого |
|-------|--------|----------|
| torch | ≥2.1 | Real-ESRGAN inference (CUDA) |
| basicsr | ≥1.4 | Real-ESRGAN модель |
| scenedetect | ≥0.6 | PySceneDetect |
| opencv-python | ≥4.8 | Sobel, читання кадрів |
| click | ≥8.1 | CLI |
| rich | ≥13.0 | Прогрес-бари, terminal UI |

Системні: `ffmpeg` з libx264, libx265, libvvenc, --enable-libvmaf

## Test Bitrates

150k, 300k, 600k, 1200k, 2500k bps (для 1080p)

## SR Activation Logic

1. Обчислити per-frame VMAF декодованого відео
2. Позначити кадри з VMAF < threshold (default: 70)
3. Застосувати Real-ESRGAN лише до позначених
4. x2: 1080p→2160p→resize 1080p; x4: для bpp < 0.05

## Testing

```bash
# Запуск усіх тестів
pytest tests/ -v

# Тест конкретного модуля
pytest tests/test_controller.py -v

# З покриттям
pytest tests/ --cov=neuroquant --cov-report=html
```

## Output Locale

Всі повідомлення CLI — **українською мовою**.
