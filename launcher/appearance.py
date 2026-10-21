"""Small Tk surfaces and line icons, without extra UI dependencies."""

import base64
import math
import struct
import tkinter as tk
import zlib


BG = "#f6f7f9"
SIDEBAR = "#eef1f5"
TOOLBAR = "#fafbfc"
INK = "#202b3b"
MUTED = "#7a8596"
BLUE = "#007aff"
BORDER = "#dfe3ea"
SELECTED = "#dfeafe"


def rounded_rectangle(canvas, x1, y1, x2, y2, radius=10, **options):
    return canvas.create_polygon(
        x1 + radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
        x2, y2 - radius, x2, y2, x2 - radius, y2, x1 + radius, y2,
        x1, y2, x1, y2 - radius, x1, y1 + radius, x1, y1,
        smooth=True, splinesteps=24, **options,
    )


def _photo(master, size, pixel):
    """Rasterize small theme elements with subpixel edge coverage."""
    samples = 3
    rows = bytearray()
    for y in range(size):
        rows.append(0)
        for x in range(size):
            values = [pixel(x + (sx + .5) / samples, y + (sy + .5) / samples)
                      for sy in range(samples) for sx in range(samples)]
            alpha = sum(value[3] for value in values)
            rows.extend(round(sum(value[i] * value[3] for value in values) / alpha)
                        if alpha else 0 for i in range(3))
            rows.append(round(alpha / samples ** 2))

    def chunk(kind, data):
        return (struct.pack("!I", len(data)) + kind + data
                + struct.pack("!I", zlib.crc32(kind + data)))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack("!2I5B", size, size, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(rows))) + chunk(b"IEND", b""))
    return tk.PhotoImage(master=master, data=base64.b64encode(png))


def surface_image(master, color, *, size=22, radius=7):
    rgb = tuple(bytes.fromhex(color.lstrip("#")))

    def pixel(x, y):
        dx = max(radius - x, x - (size - radius), 0)
        dy = max(radius - y, y - (size - radius), 0)
        return (*rgb, 255 if dx * dx + dy * dy <= radius * radius else 0)

    return _photo(master, size, pixel)


def symbol_image(master, name, color, size=18):
    """A consistent small line-icon vocabulary for toolbar and categories."""
    rgb = tuple(bytes.fromhex(color.lstrip("#")))
    lines, circles = [], []
    if name == "grid":
        for x in (3, 11):
            for y in (3, 11):
                lines.extend([((x, y), (x + 4, y)), ((x + 4, y), (x + 4, y + 4)),
                              ((x + 4, y + 4), (x, y + 4)), ((x, y + 4), (x, y))])
    elif name == "terminal":
        lines = [((2, 3), (16, 3)), ((16, 3), (16, 15)), ((16, 15), (2, 15)),
                 ((2, 15), (2, 3)), ((5, 7), (7, 9)), ((7, 9), (5, 11)),
                 ((10, 11), (13, 11))]
    elif name == "signal":
        points = [(1, 10), (4, 10), (6, 4), (9, 14), (12, 7), (14, 10), (17, 10)]
        lines = list(zip(points, points[1:]))
    elif name == "search":
        circles = [(7.5, 7.5, 5.2)]
        lines = [((11.5, 11.5), (16, 16))]
    elif name == "refresh":
        for i in range(22):
            a, b = math.radians(-45 + i * 12), math.radians(-45 + (i + 1) * 12)
            lines.append(((9 + 5.8 * math.cos(a), 9 + 5.8 * math.sin(a)),
                          (9 + 5.8 * math.cos(b), 9 + 5.8 * math.sin(b))))
        lines.extend([((13, 2), (13, 6)), ((13, 6), (9, 6))])
    else:
        lines = [((2, 5), (7, 5)), ((7, 5), (9, 7)), ((9, 7), (16, 7)),
                 ((16, 7), (16, 15)), ((16, 15), (2, 15)), ((2, 15), (2, 5))]

    def distance(x, y, start, end):
        ax, ay = start
        bx, by = end
        length = (bx - ax) ** 2 + (by - ay) ** 2
        t = max(0, min(1, ((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / length))
        return math.hypot(x - ax - t * (bx - ax), y - ay - t * (by - ay))

    def pixel(x, y):
        x, y = x * 18 / size, y * 18 / size
        hit = any(distance(x, y, start, end) <= .8 for start, end in lines)
        hit |= any(abs(math.hypot(x - cx, y - cy) - r) <= .8 for cx, cy, r in circles)
        return (*rgb, 255 if hit else 0)

    return _photo(master, size, pixel)
