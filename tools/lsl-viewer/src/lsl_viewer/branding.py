from __future__ import annotations

import tkinter as tk
from pathlib import Path

BRAND_COLOR = "#231815"
MARK_PATH = Path(__file__).resolve().parent / "assets" / "active-intelligence-mark@2x.png"


def stacked_brand(parent: tk.Misc, *, background: str) -> tk.Frame:
    frame = tk.Frame(parent, background=background, borderwidth=0)
    mark = tk.PhotoImage(file=str(MARK_PATH)).subsample(4, 4)
    mark_label = tk.Label(frame, image=mark, background=background, borderwidth=0)
    mark_label.image = mark
    mark_label.pack()
    wordmark = tk.Frame(frame, background=background, borderwidth=0)
    wordmark.pack(pady=(2, 0))
    tk.Label(
        wordmark,
        text="主动科技",
        background=background,
        foreground=BRAND_COLOR,
        font=("TkDefaultFont", 10, "bold"),
    ).pack(side="left")
    tk.Label(
        wordmark,
        text="|",
        background=background,
        foreground=BRAND_COLOR,
        font=("TkDefaultFont", 15),
        padx=4,
    ).pack(side="left")
    tk.Label(
        wordmark,
        text="Active Intelligence\nTechnologies",
        background=background,
        foreground=BRAND_COLOR,
        font=("TkDefaultFont", 6, "bold"),
        justify="left",
    ).pack(side="left")
    return frame
