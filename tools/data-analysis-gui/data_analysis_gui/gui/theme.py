"""Colours, fonts and ttk styles shared with the macaque analysis GUI."""
import sys
from tkinter import ttk

FONT = 'Microsoft YaHei UI'
ACCENT = '#2464a5'
ACCENT_DARK = '#174c82'
HEADER = '#152c47'
HEADER_CROSS = '#3b2a52'       # cross-day mode uses its own header colour
BACKGROUND = '#f1f4f8'
CANVAS_BG = '#e8edf3'
MUTED = '#53657c'
OK = '#1f7a4d'
WARN = '#a8641b'
ERROR = '#b3261e'
CARD = '#ffffff'


def enable_dpi_awareness():
    """Per-monitor DPI awareness on Windows; without it Tk is bitmap-scaled and blurry."""
    if sys.platform != 'win32':
        return
    import ctypes
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            pass


def px(widget, value):
    """Scale a 96-dpi pixel size to the current display (Treeview widths, fixed frames)."""
    try:
        factor = float(widget.tk.call('tk', 'scaling')) / (96 / 72)
    except Exception:
        factor = 1.0
    return int(round(value * max(factor, 1.0)))


def apply(root):
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', font=(FONT, 10), background=BACKGROUND)
    style.configure('TFrame', background=BACKGROUND)
    style.configure('Card.TFrame', background=CARD)
    style.configure('TLabel', background=BACKGROUND)
    style.configure('Card.TLabel', background=CARD)
    style.configure('Muted.TLabel', foreground=MUTED, background=BACKGROUND)
    style.configure('CardMuted.TLabel', foreground=MUTED, background=CARD)
    style.configure('Title.TLabel', font=(FONT, 12, 'bold'), background=BACKGROUND)
    style.configure('CardTitle.TLabel', font=(FONT, 11, 'bold'), background=CARD)
    style.configure('Status.TLabel', font=(FONT, 10, 'bold'), background=BACKGROUND)
    style.configure('TLabelframe', background=BACKGROUND)
    style.configure('TLabelframe.Label', font=(FONT, 10, 'bold'), foreground=HEADER, background=BACKGROUND)
    style.configure('TCheckbutton', background=BACKGROUND)
    style.configure('TRadiobutton', background=BACKGROUND)
    style.configure('TButton', padding=(10, 5))
    style.configure('Small.TButton', padding=(6, 2))
    style.configure('Accent.TButton', background=ACCENT, foreground='white')
    style.map('Accent.TButton', background=[('active', ACCENT_DARK), ('disabled', '#9fb3c8')])
    style.configure('Treeview', rowheight=px(root, 20), font=(FONT, 10))
    style.configure('Treeview.Heading', font=(FONT, 10, 'bold'))
    style.configure('TNotebook', background=BACKGROUND)
    style.configure('TNotebook.Tab', padding=(12, 5), font=(FONT, 10))
    style.configure('Task.Horizontal.TProgressbar', background=ACCENT, troughcolor='#e3eaf3',
                    lightcolor=ACCENT, darkcolor=ACCENT, bordercolor='#b8c6d8', thickness=12)
    _tick_checkbuttons(root, style)
    root.configure(background=BACKGROUND)
    return style


def _check_image(size, checked, disabled):
    """Rounded box; checked = accent fill with a white tick (drawn 4x, downsampled)."""
    from PIL import Image, ImageDraw
    scale = 4
    n = size * scale
    image = Image.new('RGBA', (n, n), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    border = '#9fb3c8' if disabled else ('#1d5490' if checked else '#7b8ba1')
    fill = ('#9fb3c8' if disabled else ACCENT) if checked else ('#eef1f5' if disabled else 'white')
    inset = scale
    draw.rounded_rectangle((inset, inset, n - inset - 1, n - inset - 1), radius=3 * scale, fill=fill,
                           outline=border, width=max(scale, int(1.2 * scale)))
    if checked:
        points = [(0.24 * n, 0.52 * n), (0.42 * n, 0.70 * n), (0.77 * n, 0.32 * n)]
        draw.line(points, fill='white', width=max(2 * scale, int(0.13 * n)), joint='curve')
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    padded = Image.new('RGBA', (size + max(4, size // 3), size), (0, 0, 0, 0))   # gap before the label
    padded.paste(image, (0, 0))
    return padded


def _tick_checkbuttons(root, style):
    """Replace clam's '×' indicator with a tick-mark box for every ttk.Checkbutton."""
    from PIL import ImageTk
    size = px(root, 15)
    images = {key: ImageTk.PhotoImage(_check_image(size, *key), master=root)
              for key in ((False, False), (True, False), (False, True), (True, True))}
    root._tick_images = images        # keep references alive
    style.element_create('Tick.indicator', 'image', images[(False, False)],
                         ('disabled', 'selected', images[(True, True)]),
                         ('disabled', images[(False, True)]),
                         ('selected', images[(True, False)]), border=0, sticky='')
    style.layout('TCheckbutton', [('Checkbutton.padding', {'sticky': 'nswe', 'children': [
        ('Tick.indicator', {'side': 'left', 'sticky': ''}),
        ('Checkbutton.focus', {'side': 'left', 'sticky': 'w', 'children': [
            ('Checkbutton.label', {'sticky': 'nswe'})]})]})])
    style.configure('TCheckbutton', padding=(2, 2))
