"""Scrollable full-resolution image canvas with fit, anchored zoom and pan (from macaque GUI)."""
import math
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from .theme import CANVAS_BG, MUTED


class ZoomCanvas(ttk.Frame):
    def __init__(self, parent, on_zoom=None, placeholder='选择记录和 section 后在此查看图像'):
        super().__init__(parent)
        self.image = None
        self.photo = None
        self.scale = 1.0
        self.fit_mode = 'whole'
        self.on_zoom = on_zoom
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, background=CANVAS_BG, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        vertical = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        horizontal = ttk.Scrollbar(self, orient='horizontal', command=self.canvas.xview)
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        self.canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.canvas.bind('<Configure>', self.resized)
        self.canvas.bind('<MouseWheel>', self.wheel)
        self.canvas.bind('<Button-4>', lambda e: self.zoom(1.2, e.x, e.y))
        self.canvas.bind('<Button-5>', lambda e: self.zoom(1 / 1.2, e.x, e.y))
        self.canvas.bind('<ButtonPress-1>', lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind('<B1-Motion>', lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
        self.canvas.create_text(40, 50, anchor='nw', text=placeholder, fill=MUTED, font=('Microsoft YaHei UI', 12))

    def load(self, path):
        with Image.open(path) as image:
            self.image = image.convert('RGB')
        self.after_idle(lambda: self.fit(whole=True))

    def fit(self, whole=False):
        """Fit to the canvas width (scroll vertically); whole=True fits the entire page."""
        if self.image is None:
            return
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.scale = width / self.image.width
        if whole:
            self.scale = min(self.scale, height / self.image.height)
        self.fit_mode = 'whole' if whole else 'width'
        self.render()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def actual(self):
        if self.image is not None:
            self.fit_mode = False
            self.scale = 1.0
            self.render()

    def resized(self, event=None):
        if self.fit_mode:
            self.fit(whole=self.fit_mode == 'whole')

    def wheel(self, event):
        if event.state & 0x0004:  # Ctrl + wheel zooms, plain wheel scrolls
            self.zoom(1.2 if event.delta > 0 else 1 / 1.2, event.x, event.y)
        else:
            self.canvas.yview_scroll(int(-event.delta / 120), 'units')
        return 'break'

    def zoom(self, factor, x=None, y=None):
        if self.image is None:
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        x, y = (width / 2 if x is None else x), (height / 2 if y is None else y)
        image_x, image_y = self.canvas.canvasx(x) / self.scale, self.canvas.canvasy(y) / self.scale
        maximum = min(4., math.sqrt(24_000_000 / (self.image.width * self.image.height)))
        self.scale = max(.02, min(maximum, self.scale * factor))
        self.fit_mode = False
        self.render()
        region_width = max(width, self.image.width * self.scale)
        region_height = max(height, self.image.height * self.scale)
        self.canvas.xview_moveto(max(0, (image_x * self.scale - x) / region_width))
        self.canvas.yview_moveto(max(0, (image_y * self.scale - y) / region_height))

    def render(self):
        width, height = max(1, round(self.image.width * self.scale)), max(1, round(self.image.height * self.scale))
        rendered = self.image.resize((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(rendered, master=self.canvas)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, max(width, self.canvas.winfo_width()),
                                            max(height, self.canvas.winfo_height())))
        if self.on_zoom:
            self.on_zoom(f'{self.scale * 100:.0f}%')
