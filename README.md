# NeuroQuant

**Інтелектуальна система аналізу та стиснення відеоданих**

NeuroQuant поєднує адаптивний per-frame rate control на основі R-λ моделі з Real-ESRGAN постпроцесингом для досягнення кращої суб'єктивної якості відео при заданому бітрейті.

## Швидкий запуск (GUI)

```bash
# Windows — подвійний клік на:
run.bat

# Або вручну (потрібен Python 3.9):
py -3.9 gui.py

# Світла тема (за замовчуванням):
py -3.9 gui.py --light

# Темна тема:
py -3.9 gui.py --dark
```

## Особливості

- **Аналіз складності кадрів** — Sobel градієнти (spatial), SAD (temporal), PySceneDetect (scene cuts)
- **Адаптивний rate control** — per-frame QP план на основі складності контенту
- **2-pass ABR кодування** — точний контроль бітрейту через libx265
- **Real-ESRGAN постпроцесинг** — селективне відновлення якості для кадрів з VMAF < порогу
- **Порівняльний бенчмарк** — H.264 / HEVC / NeuroQuant з метриками PSNR, SSIM, VMAF, BD-Rate
- **Генерація звітів** — RD-криві, HTML звіти, CSV таблиці

## Вимоги

### Системні

| Компонент | Мінімум | Рекомендовано |
|-----------|---------|---------------|
| Python | 3.9 | 3.9-3.11 (для torch з CUDA) |
| CPU | 4 ядра | 8+ ядер |
| RAM | 8 GB | 16 GB |
| GPU | — | NVIDIA RTX 3060+ |
| VRAM | 4 GB | 8+ GB |
| FFmpeg | 4.3+ | 5.0+ |

### FFmpeg

Потрібна збірка з підтримкою кодеків:

```bash
# Перевірка
ffmpeg -encoders 2>/dev/null | grep -E "libx264|libx265"
ffmpeg -filters 2>/dev/null | grep libvmaf

# Очікуваний вивід:
# V..... libx264    libx264 H.264 / AVC / MPEG-4 AVC
# V..... libx265    libx265 H.265 / HEVC
```

**Windows:** Завантажити повну збірку з [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)

**Linux:**
```bash
sudo apt install ffmpeg
# або збірка з --enable-libx265 --enable-libvmaf
```

## Встановлення

```bash
# Клонування
git clone https://github.com/your-username/neuroquant.git
cd neuroquant

# Віртуальне середовище
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Залежності
pip install -r requirements.txt
```

## Швидкий старт

### Кодування відео

```bash
# Базове кодування з адаптивним rate control
python -m neuroquant encode input.mp4 output.mp4 --bitrate 1M

# З Real-ESRGAN постпроцесингом (потрібен GPU)
python -m neuroquant encode input.mp4 output.mp4 --bitrate 500k --sr --sr-threshold 70
```

### Аналіз складності

```bash
python -m neuroquant analyze input.mp4 --output complexity.json --plot
```

### Бенчмарк

```bash
# Порівняння методів на тестових відео
python -m neuroquant benchmark ./test_videos \
    --methods h264,hevc,nq,nq_sr \
    --bitrates 300k,600k,1200k \
    --output ./results

# Генерація HTML звіту
python -m neuroquant report results/benchmark_report.json \
    --format html,png \
    --output ./report
```

### Інформація про систему

```bash
python -m neuroquant info
```

## Архітектура

```
input.mp4 → ComplexityAnalyzer → RLambdaController → FFmpegEncoder → SRPostProcessor → output.mp4
                  ↓                      ↓                  ↓                ↓
           complexity_map[]        qp_plan[]         encoded.mp4      +Real-ESRGAN
```

### Модулі

| Модуль | Призначення |
|--------|-------------|
| `analyzer.py` | Аналіз складності кадрів (Sobel, SAD, PySceneDetect) |
| `controller.py` | Генерація per-frame QP плану |
| `encoder.py` | FFmpeg обгортка з 2-pass ABR та адаптивним AQ |
| `sr_processor.py` | Real-ESRGAN x2 постпроцесинг |
| `metrics.py` | PSNR, SSIM, VMAF, BD-Rate |
| `benchmark.py` | Порівняльне тестування методів |
| `report.py` | RD-криві, HTML звіти, CSV таблиці |

## Формула складності

```
complexity_i = 0.4 × spatial_i + 0.5 × temporal_i + 0.1 × cut_i
```

- `spatial_i` — нормований градієнт Sobel (деталізація кадру)
- `temporal_i` — SAD між сусідніми кадрами (рух)
- `cut_i` — 1 при зміні сцени, 0 інакше

## Алгоритм кодування

1. **Аналіз складності** — обчислення complexity для кожного кадру
2. **QP планування** — адаптивний QP на основі складності:
   ```
   QP_i = QP_base - round(delta_max × (complexity_i - mean) / std)
   ```
3. **2-pass ABR** — точний контроль бітрейту через libx265
4. **Adaptive Quantization** — AQ strength пропорційний варіації складності
5. **SR постпроцесинг** (опційно) — Real-ESRGAN для кадрів з VMAF < threshold

## Конфігурація

Параметри в `config.yaml`:

```yaml
# Аналіз складності
complexity:
  spatial_weight: 0.4      # вага просторової складності
  temporal_weight: 0.5     # вага тимчасової складності
  cut_weight: 0.1          # вага scene cuts
  scene_threshold: 27.0    # поріг PySceneDetect

# Rate control
rate_control:
  qp_min: 10               # мінімальний QP
  qp_max: 51               # максимальний QP
  delta_max: 8             # макс. відхилення від базового QP
  i_frame_bonus: 4         # бонус якості для I-кадрів
  gop_seconds: 2.0         # тривалість GOP

# SR постпроцесинг
sr:
  vmaf_threshold: 70       # поріг активації SR
  model: RealESRNet_x2plus # модель Real-ESRGAN
  tile_size: 512           # розмір тайлу (для економії VRAM)

# Бенчмарк
benchmark:
  methods: [h264, hevc, nq, nq_sr]
  bitrates: [150000, 300000, 600000, 1200000, 2500000]
```

## Метрики якості

| Метрика | Опис | Діапазон |
|---------|------|----------|
| **PSNR** | Peak Signal-to-Noise Ratio | 20-50 dB (більше = краще) |
| **SSIM** | Structural Similarity | 0-1 (більше = краще) |
| **VMAF** | Netflix Video Multi-Method Assessment | 0-100 (більше = краще) |
| **BD-Rate** | Bjøntegaard Delta Rate | % (менше = краще) |

## Очікувані результати

| Метод | BD-Rate vs H.264 (VMAF) | Примітка |
|-------|-------------------------|----------|
| HEVC (libx265) | -35% to -40% | Базовий HEVC |
| NeuroQuant | -38% to -45% | Адаптивний rate control |
| NeuroQuant+SR | -45% to -55% | При низькому бітрейті |

*Результати залежать від контенту та цільового бітрейту*

## Обмеження

1. **Час обробки** — SR постпроцесинг: ~0.15-0.25 сек/кадр на RTX 3060
2. **Offline режим** — система призначена для offline обробки, не real-time
3. **VRAM** — для 4K відео потрібно 8+ GB VRAM (tile-based inference)

## Тестування

```bash
# Запуск тестів
pytest tests/ -v

# З покриттям
pytest tests/ --cov=neuroquant --cov-report=html
```

## Структура проекту

```
neuroquant/
├── neuroquant/
│   ├── __init__.py
│   ├── analyzer.py      # ComplexityAnalyzer
│   ├── controller.py    # RLambdaController
│   ├── encoder.py       # FFmpegEncoder
│   ├── sr_processor.py  # SRPostProcessor
│   ├── metrics.py       # MetricsCollector, BDRateCalculator
│   ├── benchmark.py     # BenchmarkEngine
│   ├── report.py        # ReportGenerator
│   ├── utils.py         # Допоміжні функції
│   └── types.py         # Dataclasses
├── tests/
│   ├── test_controller.py
│   ├── test_metrics.py
│   └── test_utils.py
├── cli.py               # Click CLI
├── config.yaml          # Конфігурація
├── requirements.txt
└── README.md
```

## CLI команди

```bash
# Довідка
python -m neuroquant --help

# Команди
python -m neuroquant encode   # Кодування відео
python -m neuroquant analyze  # Аналіз складності
python -m neuroquant benchmark # Порівняльне тестування
python -m neuroquant report   # Генерація звітів
python -m neuroquant info     # Інформація про систему
```

## Приклади виводу

```
============================================================
     _   _                      ___                    _
    | \ | | ___ _   _ _ __ ___ / _ \ _   _  __ _ _ __ | |_
    |  \| |/ _ \ | | | '__/ _ \ | | | | | |/ _` | '_ \| __|
    | |\  |  __/ |_| | | | (_) | |_| | |_| | (_| | | | | |_
    |_| \_|\___|\__,_|_|  \___/ \__\_\\__,_|\__,_|_| |_|\__|

    Intelligent Video Compression System
============================================================

[NeuroQuant] Input video: test.mp4
[NeuroQuant] Resolution: 1920x1080 @ 30fps
[NeuroQuant] Duration: 60.0 sec (1800 frames)
[NeuroQuant] Target bitrate: 1M

[Step 1/4] Analyzing frame complexity...
[NeuroQuant] Середня складність: 0.412 | Макс: 0.891 | Мін: 0.043

[Step 2/4] Generating QP plan...
[NeuroQuant] Базовий QP: 28
[NeuroQuant] Діапазон QP: 22–34

[Step 3/4] Encoding video...
[NeuroQuant] Режим: 2-pass ABR @ 1000k, AQ strength: 1.35
  Кодування ████████████████████████████████ 100% | ETA: 0:01:23

[Step 4/4] SR skipped

Encoding complete!
[NeuroQuant] Output file: output.mp4
[NeuroQuant] Size: 7.32 MB
[NeuroQuant] Actual bitrate: 998k
```

## Ліцензія

MIT License

## Автор

Дипломний проект, 2024-2025

## Посилання

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — Super Resolution
- [x265](https://x265.org/) — HEVC Encoder
- [VMAF](https://github.com/Netflix/vmaf) — Video Quality Metric
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — Scene Detection
- [FFmpeg](https://ffmpeg.org/) — Video Processing
