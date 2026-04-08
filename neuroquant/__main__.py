"""
Точка входу для запуску як модуля: python -m neuroquant
"""

import sys
from pathlib import Path

# Додаємо батьківську директорію до шляху для доступу до cli.py
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from cli import main
except ImportError:
    # Fallback для випадку коли структура інша
    from neuroquant.cli import main

if __name__ == "__main__":
    main()
