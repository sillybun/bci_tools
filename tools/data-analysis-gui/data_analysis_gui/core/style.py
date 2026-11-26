"""Figure style shared by all sections (English figure text keeps fonts portable)."""
from __future__ import annotations

import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
import warnings

import matplotlib as mpl
import numpy as np

# Eight directions on a perceptually even hue circle (0° = +x, counter-clockwise).
DIRECTION_COLORS = ('#d1495b', '#edae49', '#a3a948', '#4f9d69', '#00798c', '#30638e', '#6a4c93', '#b5446e')
GREY, INK, INK2 = '#b0b7bd', '#1b1b1b', '#55534e'
MISSING = '#d6d4cf'
DEFAULT_FONT_PERCENT = 100  # Legacy request field; exports use fixed point sizes.
TITLE_PT, LABEL_PT, TICK_PT = 18, 14, 11
FONT_POLICY = 'titles18-labels14-ticks11-v1'
_MISSING_GLYPH = r'Glyph .* missing from (?:the )?(?:current )?font'
_CJK = re.compile(r'[㐀-鿿豈-﫿]')
_RUN_ID = re.compile(r'\brun[- ]?\d+', re.IGNORECASE)
MAX_CODE_COLORS = ('#111111', '#FFD91A', '#FF9400', '#EF2727')   # max code 0..3
MODEL_COLORS = {'gt': '#2b2b2b', 'ridge': '#2176ae', 'kf': '#e07b39', 'online': '#009e73'}
# Polar subplot placement for eight directions in a 3x3 grid (centre empty).
DIRECTION_GRID = {0: (1, 2), 1: (0, 2), 2: (0, 1), 3: (0, 0), 4: (1, 0), 5: (2, 0), 6: (2, 1), 7: (2, 2)}


def apply():
    from matplotlib.font_manager import fontManager

    installed = {font.name for font in fontManager.ttflist}
    families = ['Arial', 'Microsoft YaHei', 'Noto Sans CJK SC', 'PingFang SC',
                'SimHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
    mpl.rcParams.update({
        'font.size': 11, 'axes.titlesize': TITLE_PT, 'axes.labelsize': LABEL_PT,
        'xtick.labelsize': TICK_PT, 'ytick.labelsize': TICK_PT,
        'legend.fontsize': TICK_PT, 'figure.titlesize': TITLE_PT,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'font.family': [name for name in families if name in installed],
    })


def finish_figure(figure, font_percent=DEFAULT_FONT_PERCENT):
    """Idempotent role-based typography, including colorbar axes and legends.

    font_percent is retained for old requests; it never changes the fixed sizes.
    In-cell numbers and data annotations keep their explicit sizes.
    """
    from matplotlib.text import Text

    text_artists = figure.findobj(Text)
    for text in text_artists:
        label = text.get_text()
        if _RUN_ID.search(label):
            text.set_text(' '.join(line.strip() for line in label.splitlines()))
            text.set_wrap(False)

    if getattr(figure, '_preserve_schematic_fonts', False):
        if not getattr(figure, '_schematic_legacy_finished', False):
            for text in text_artists:
                text.set_fontfamily(mpl.rcParams['font.family'])
                text.set_fontsize((text.get_fontsize()+1)*1.2)
            figure._schematic_legacy_finished = True
        return

    for text in text_artists:
        text.set_fontfamily(mpl.rcParams['font.family'])
    for ax in figure.axes:
        for annotation in ax.texts:
            if annotation.get_fontsize() > TICK_PT:
                annotation.set_fontsize(TICK_PT)
        for title in (ax.title, ax._left_title, ax._right_title):
            title.set_fontsize(TITLE_PT)
        for axis in (ax.xaxis, ax.yaxis):
            axis.label.set_fontsize(LABEL_PT)
            axis.offsetText.set_fontsize(TICK_PT)
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                tick.label1.set_fontsize(TICK_PT)
                tick.label2.set_fontsize(TICK_PT)
        legend = ax.get_legend()
        if legend:
            for text in legend.findobj(Text):
                text.set_fontsize(TICK_PT)
    for legend in figure.legends:
        for text in legend.findobj(Text):
            text.set_fontsize(TICK_PT)
    for text in figure.texts:
        text.set_fontsize(TITLE_PT if text is getattr(figure, '_suptitle', None) or text.get_position()[1] >= .9 else TICK_PT)
    for name in ('_supxlabel', '_supylabel'):
        text = getattr(figure, name, None)
        if text is not None:
            text.set_fontsize(LABEL_PT)
    _fit_figure_text(figure)
    figure._font_policy = FONT_POLICY


def _fit_figure_text(figure):
    """Wrap long figure captions and reserve room above manually positioned axes."""
    import textwrap
    from matplotlib.textpath import TextToPath
    measure = TextToPath()
    width = figure.get_figwidth()*72*.94
    title = getattr(figure, '_suptitle', None)
    for text in figure.texts:
        if text is not title and text.get_position()[1] > .15:
            continue
        original = text.get_text()
        if _RUN_ID.search(original):
            extent = measure.get_text_width_height_descent(original, text.get_fontproperties(), False)[0]
            if extent > width:
                figure.set_figwidth(max(figure.get_figwidth(), extent / (72 * .94)))
                width = figure.get_figwidth()*72*.94
            continue
        if '$' in original:
            continue
        lines = []
        for line in original.split('\n'):
            extent = measure.get_text_width_height_descent(line, text.get_fontproperties(), False)[0]
            if extent <= width or not line:
                lines.append(line)
                continue
            chars = max(10, int(len(line)*width/extent)-2)
            lines.extend(textwrap.wrap(line, width=chars, break_long_words=True, break_on_hyphens=False))
        text.set_text('\n'.join(lines))
    if title is not None and figure.axes and figure.get_layout_engine() is None:
        height = figure.get_figheight()*72
        lines = len(title.get_text().splitlines())
        top = (title.get_position()[1]*height-lines*TITLE_PT*1.2-8)/height
        positions = [ax.get_position() for ax in figure.axes]
        low, high = min(p.y0 for p in positions), max(p.y1 for p in positions)
        if low+.15 < top < high-.001:
            ratio = (top-low)/(high-low)
            for ax, p in zip(figure.axes, positions):
                ax.set_position([p.x0, low+(p.y0-low)*ratio, p.width, p.height*ratio])


def _install_export_style():
    """Cover direct scientific-script exports as well as runner/GUI exports."""
    from matplotlib.figure import Figure
    if getattr(Figure.savefig, '_fixed_font_policy', False):
        return
    original = Figure.savefig

    def savefig(figure, fname, *args, **kwargs):
        from matplotlib.text import Text

        path = Path(fname) if isinstance(fname, (str, os.PathLike)) else None
        format_name = (kwargs.get('format') or (path.suffix[1:] if path else '')
                       or mpl.rcParams['savefig.format']).lower()
        portable_svg = format_name == 'svg' and any(
            _CJK.search(text.get_text()) for text in figure.findobj(Text))
        settings = {'svg.fonttype': 'path'} if portable_svg else {}

        def render(target):
            with mpl.rc_context(settings), warnings.catch_warnings():
                warnings.filterwarnings('error', _MISSING_GLYPH, category=UserWarning)
                try:
                    finish_figure(figure)
                    return original(figure, target, *args, **kwargs)
                except UserWarning as exc:
                    if re.match(_MISSING_GLYPH, str(exc)):
                        raise RuntimeError(f'图表缺字，导出失败 ({fname})。请安装支持该字符的字体：{exc}') from exc
                    raise

        # Keep the previous file intact when a renderer writes before reporting a missing glyph.
        if path is None or path.suffix.lower() not in ('.png', '.pdf', '.svg'):
            return render(fname)
        with NamedTemporaryFile(dir=path.parent, prefix='.figure-', suffix=path.suffix, delete=False) as file:
            staged = Path(file.name)
        try:
            result = render(staged)
            os.replace(staged, path)
            return result
        finally:
            staged.unlink(missing_ok=True)

    savefig._fixed_font_policy = True
    Figure.savefig = savefig


def direction_color(index: int) -> str:
    return DIRECTION_COLORS[int(index) % 8]


ELLIPSE_METHOD = (' Direction ellipses show a 1-SD sample-covariance contour in the plotted plane: '
                  'centre = group mean, semi-axis lengths = square roots of the sample covariance '
                  'eigenvalues (ddof=1), orientation = covariance eigenvectors. Only finite paired '
                  'coordinates are used; groups with fewer than 3 trials or rank-deficient covariance '
                  'are omitted. These describe trial spread, not uncertainty of the mean or a 68% '
                  'joint confidence region (a bivariate Gaussian has about 39% probability inside). ')


def direction_ellipses(ax, scores, directions, *, fill=True):
    """Draw one sample-covariance 1-SD ellipse per direction in a 2-D plane."""
    from matplotlib.patches import Ellipse

    scores, directions = np.asarray(scores, float), np.asarray(directions)
    finite = np.isfinite(scores).all(1)
    patches = []
    for d in range(8):
        group = scores[finite & (directions == d)]
        if len(group) < 3:
            continue
        covariance = np.cov(group, rowvar=False, ddof=1)
        values, vectors = np.linalg.eigh(covariance)
        if values[0] <= max(values[-1] * 1e-12, 0):
            continue
        major = vectors[:, -1]
        angle = np.degrees(np.arctan2(major[1], major[0]))
        colour = direction_color(d)
        patch = Ellipse(group.mean(0), width=2 * np.sqrt(values[-1]),
                        height=2 * np.sqrt(values[0]), angle=angle,
                        facecolor=mpl.colors.to_rgba(colour, 0.08) if fill else 'none', edgecolor=colour,
                        linewidth=1, zorder=0.8, label='_nolegend_', gid=f'{d * 45}° 1 SD')
        ax.add_patch(patch)
        patches.append(patch)
    return patches


def eight_direction_axes(fig, sharey=True):
    """Axes dict {direction: ax} arranged around a centre cell, plus the centre axis."""
    grid = fig.add_gridspec(3, 3)
    axes, first = {}, None
    for d, (r, c) in DIRECTION_GRID.items():
        ax = fig.add_subplot(grid[r, c], sharey=first if sharey else None, sharex=first)
        first = first or ax
        axes[d] = ax
    centre = fig.add_subplot(grid[1, 1])
    return axes, centre


def signed_log(values, scale=1.0):
    return np.sign(values) * np.log10(1 + np.abs(values) / scale)


def text_panel(ax, lines, fontsize=7.5):
    ax.axis('off')
    ax.text(0, 1, '\n'.join(lines), va='top', ha='left', fontsize=fontsize, family='monospace',
            transform=ax.transAxes)


apply()
_install_export_style()
