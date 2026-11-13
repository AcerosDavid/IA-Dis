"""
Punto de entrada de la aplicacion de escritorio.

Uso:
    python app.py
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from src.ui.app import main


if __name__ == "__main__":
    main()
