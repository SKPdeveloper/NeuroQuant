# NeuroQuant

**Інтелектуальна система порівняльного аналізу відеокодеків**

NeuroQuant — освітній інструмент для порівняння ефективності відеокодеків. Система кодує відео 5 різними методами паралельно і наочно демонструє різницю в якості та розмірі.

## Головна ідея

Одне натискання — порівняння 5 методів кодування:

| Метод | Кодек | Особливість |
|-------|-------|-------------|
| **H.264** | libx264 | Baseline — найпоширеніший |
| **HEVC** | libx265 | На 30-40% ефективніший |
| **VVC** | libvvenc | Найновіший стандарт H.266 |
| **NeuroQuant** | libx265 + R-λ | Адаптивний QP за складністю |
| **NeuroQuant+SR** | libx265 + R-λ + Real-ESRGAN | Відновлення деталей після стиснення |

## Швидкий запуск

```bash
# Windows — подвійний клік:
run.bat

# Або вручну (Python 3.9 обов'язково для torch):
py -3.9 gui.py
```

## Як працює

1. **Обираєш відео** на вкладці "Кодування"
2. **Обираєш бітрейт** (150k — 4M)
3. **Тиснеш "Кодувати"**
4. Система паралельно кодує 5 методами
5. **Результат**: таблиця з метриками + 4 відео-екрани для візуального порівняння

## GUI

### Вкладка "Кодування"
- Вибір відео та бітрейту
- Запуск порівняльного кодування
- Журнал операцій

### Вкладка "Бенчмарк"
- **Таблиця**: розмір, бітрейт, PSNR, SSIM для кожного методу
- **4 екрани**: синхронне відтворення H.264 / HEVC / VVC / NeuroQuant
- **Висновок**: автоматичне визначення найкращого кодека

### Вкладка "Аналіз"
- Графік складності кадрів (spatial + temporal)
- QP план NeuroQuant
- Статистика та висновки

### Вкладка "Порівняння"
- 2 екрани: оригінал vs стиснене
- Покадрове порівняння

### Вкладка "Теорія"
- Математика R-λ моделі
- Формули QP адаптації

## Вимоги

| Компонент | Мінімум | Рекомендовано |
|-----------|---------|---------------|
| Python | 3.9 | 3.9-3.11 |
| FFmpeg | 5.0+ з libx265, libvvenc | gyan.dev full build |
| GPU | — | NVIDIA RTX (для SR) |
| RAM | 8 GB | 16 GB |

### FFmpeg

Потрібна збірка з кодеками:
- libx264 (H.264)
- libx265 (HEVC)
- libvvenc (VVC)
- libvmaf (метрики)

**Windows:** [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) — full build

## Встановлення

```bash
git clone https://github.com/SKPdeveloper/NeuroQuant.git
cd NeuroQuant
pip install -r requirements.txt
```

## R-λ модель (NeuroQuant)

### Формула складності кадру
```
complexity = 0.4 × spatial + 0.5 × temporal + 0.1 × scene_cut
```

- `spatial` — градієнт Sobel (деталізація)
- `temporal` — SAD між кадрами (рух)
- `scene_cut` — 1 при зміні сцени

### Адаптивний QP
```
QP_i = QP_base - round(delta_max × (complexity_i - mean) / std)
```

Складний кадр → нижчий QP → більше біт → краща якість
Простий кадр → вищий QP → менше біт → економія

## Real-ESRGAN (SR)

Постпроцесинг для відновлення деталей:
1. Кадри з низьким VMAF (< 70) проходять через нейромережу
2. Upscale x2 → downscale до оригінального розміру
3. Відновлюються текстури, втрачені при агресивному стисненні

## Метрики

| Метрика | Опис | Краще |
|---------|------|-------|
| **PSNR** | Peak Signal-to-Noise Ratio | Більше |
| **SSIM** | Structural Similarity | Більше |
| **Розмір** | Розмір файлу | Менше |

## CLI (опційно)

```bash
# Кодування
py -3.9 -m neuroquant encode input.mp4 output.mp4 --bitrate 1M

# Аналіз складності
py -3.9 -m neuroquant analyze input.mp4 --plot

# Бенчмарк
py -3.9 -m neuroquant benchmark ./videos -m h264,hevc,vvc,nq -b 600k
```

## Структура

```
NeuroQuant/
├── gui.py              # PyQt6 GUI
├── run.bat             # Лаунчер Windows
├── neuroquant/
│   ├── analyzer.py     # Аналіз складності
│   ├── controller.py   # R-λ QP планування
│   ├── encoder.py      # FFmpeg кодування
│   ├── sr_processor.py # Real-ESRGAN
│   ├── metrics.py      # PSNR, SSIM, VMAF
│   └── benchmark.py    # Порівняльне тестування
├── models/             # Ваги Real-ESRGAN
└── test_videos/        # Тестові відео
```

## Очікувані результати

При бітрейті 600k для 1080p відео:

| Метод | PSNR | Розмір |
|-------|------|--------|
| H.264 | ~32 dB | 100% (baseline) |
| HEVC | ~34 dB | ~70% |
| VVC | ~35 dB | ~60% |
| NeuroQuant | ~35 dB | ~65% |
| NeuroQuant+SR | ~36 dB | ~65% |

*Результати залежать від контенту*

## Ліцензія

MIT License

## Автор

Дипломний проект, 2026

## Посилання

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [x265](https://x265.org/)
- [VVenC](https://github.com/fraunhoferhhi/vvenc)
- [FFmpeg](https://ffmpeg.org/)
