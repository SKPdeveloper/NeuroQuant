"""
NeuroQuant - Інтелектуальна система аналізу та стиснення відеоданих

Поєднує per-frame rate control на основі R-λ теорії
з Real-ESRGAN постпроцесингом для оптимального
співвідношення якість/бітрейт.
"""

__version__ = "0.1.0"
__author__ = "NeuroQuant Team"


def __getattr__(name):
    """Ледачі імпорти для уникнення циклічних залежностей."""
    if name == "ComplexityAnalyzer":
        from .analyzer import ComplexityAnalyzer
        return ComplexityAnalyzer
    elif name == "RLambdaController":
        from .controller import RLambdaController
        return RLambdaController
    elif name == "FFmpegEncoder":
        from .encoder import FFmpegEncoder
        return FFmpegEncoder
    elif name == "SRPostProcessor":
        from .sr_processor import SRPostProcessor
        return SRPostProcessor
    elif name == "MetricsCollector":
        from .metrics import MetricsCollector
        return MetricsCollector
    elif name == "BenchmarkEngine":
        from .benchmark import BenchmarkEngine
        return BenchmarkEngine
    elif name == "ReportGenerator":
        from .report import ReportGenerator
        return ReportGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ComplexityAnalyzer",
    "RLambdaController",
    "FFmpegEncoder",
    "SRPostProcessor",
    "MetricsCollector",
    "BenchmarkEngine",
    "ReportGenerator",
]
