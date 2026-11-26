"""Unmodified chip-to-shank reference image accompanying channel stability."""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from ... import config as C

TITLE = 'BGA Chip Layout Mapping'


def append_chip_layout(result, path=None, *, layout=None):
    if layout is not None and layout.get('bga_position') is not None:
        from matplotlib.lines import Line2D
        from ...report_bga import SHANK_COLORS

        xy = layout['bga_position']
        shanks = layout['position'][:, 1]
        fig = Figure(figsize=(16, 13), layout='constrained')
        ax = fig.subplots()
        ax.scatter(xy[:, 1], xy[:, 0], c=[SHANK_COLORS[s] for s in shanks], s=108)
        for ch, (row, col) in enumerate(xy):
            ax.text(col, row, str(ch), ha='center', va='center', fontsize=5)
        ax.set(xlabel='BGA column', ylabel='BGA row', aspect='equal', title=TITLE)
        ax.legend(handles=[Line2D([], [], marker='o', linestyle='', color=color, label=f'Shank {i}')
                           for i, color in enumerate(SHANK_COLORS, 1)], ncol=8, loc='upper center',
                  bbox_to_anchor=(.5, -.06))
        source = Path(layout['source'])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        result.add_figure(TITLE, fig, method=f'Channel ID, BGA position and Shank from {source}; SHA256: {digest}.',
                          data={'mapping': [dict(channel_0based=ch, shank=int(shanks[ch])+1,
                                                 bga_row=int(row), bga_col=int(col))
                                            for ch, (row, col) in enumerate(xy)]})
        result.summary.update(chip_layout_source=str(source), chip_layout_sha256=digest)
        return
    path = Path(path) if path is not None else C.BGA_SHANK_PROJECTION
    if not path.is_file():
        result.notes.append(f'芯片 layout 参考图缺失：{path}')
        return
    with Image.open(path) as source:
        pixels = np.array(source.convert('RGBA'))
    height, width = pixels.shape[:2]
    fig = Figure(figsize=(width / 300, height / 300), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(pixels, interpolation='none')
    ax.set_axis_off()
    fig._preserve_schematic_fonts = True
    fig._reference_image_path = str(path.resolve())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result.add_figure(TITLE, fig, method=(
        f'Original BGA-to-shank reference image, preserved without restyling. '
        f'Source: {path}; SHA256: {digest}. Colors identify shanks and labels '
        'are channel IDs from the supplied image. This is a mapping reference, '
        'not a measurement of channel activity or stability.'),
        data={'reference_source': [{'path': str(path), 'sha256': digest,
                                    'width_px': width, 'height_px': height}]})
    result.summary['chip_layout_source'] = str(path)
    result.summary['chip_layout_sha256'] = digest
