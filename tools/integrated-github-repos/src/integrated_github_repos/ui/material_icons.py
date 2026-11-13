from __future__ import annotations

import qtawesome as qta
from PySide6.QtGui import QIcon


def material_icon(name: str, color: str = "#27313a") -> QIcon:
    return qta.icon(f"mdi6.{name}", color=color)

