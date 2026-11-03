"""Figure style shared by all sections (English figure text keeps fonts portable)."""
from __future__ import annotations

import matplotlib as mpl
import numpy as np

# Eight directions on a perceptually even hue circle (0° = +x, counter-clockwise).
DIRECTION_COLORS = ('#d1495b', '#edae49', '#a3a948', '#4f9d69', '#00798c', '#30638e', '#6a4c93', '#b5446e')
GREY, INK, INK2 = '#b0b7bd', '#1b1b1b', '#55534e'
MISSING = '#d6d4cf'
DEFAULT_FONT_PERCENT = 120
MAX_CODE_COLORS = ('#111111', '#FFD91A', '#FF9400', '#EF2727')   # max code 0..3
MODEL_COLORS = {'gt': '#2b2b2b', 'ridge': '#2176ae', 'kf': '#e07b39', 'online': '#009e73'}
# Polar subplot placement for eight directions in a 3x3 grid (centre empty).
DIRECTION_GRID = {0: (1, 2), 1: (0, 2), 2: (0, 1), 3: (0, 0), 4: (1, 0), 5: (2, 0), 6: (2, 1), 7: (2, 2)}


def apply():
    mpl.rcParams.update({
        'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'font.family': ['Arial', 'Microsoft YaHei', 'DejaVu Sans'],
    })


def finish_figure(figure, font_percent=DEFAULT_FONT_PERCENT):
    """Apply the shared typeface and scale all figure text consistently."""
    from matplotlib.text import Text

    for text in figure.findobj(Text):
        text.set_fontfamily(mpl.rcParams['font.family'])
        text.set_fontsize((text.get_fontsize() + 1) * font_percent / 100)


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
