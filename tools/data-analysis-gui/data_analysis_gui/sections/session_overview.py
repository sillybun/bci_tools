"""Shared recording-selection diagram for cross-session analysis."""
from itertools import groupby

from matplotlib.lines import Line2D

from ..core.crossday import (OTHER_SESSION_STYLE, SESSION_STYLES, CONDITION_STYLES, recording_style,
                             recording_labels, recording_ticks, session_label_color)
from .base import new_figure

TITLE = 'Session Overview'
METHOD = ('Each symbol represents one selected recording. Sessions are equally spaced in date order. '
          'Dashed vertical lines separate dates. Squares denote Center Out 8 and circles denote Grid. '
          'Gray and blue indicate offline recordings; orange and green indicate online recordings. '
          'Purple denotes CO8 online restrained recordings. Initial assist 100 or 200 is shown '
          'beneath each run; assist may decay during the run.')
PAGES = ((None, TITLE), *((kind, style[0]) for kind, style in SESSION_STYLES.items()))


def session_overview(recordings, kind=None):
    if kind is not None:
        recordings = [r for r in recordings if r.data_type == kind]
    recordings = sorted(recordings, key=lambda r: (r.date, r.key))
    labels = recording_labels(recordings)
    rows = [dict(session=label, date=r.date, recording=r.key, session_type=r.data_type,
                 condition=getattr(r, 'condition', ''))
            for r, label in zip(recordings, labels)]
    styles = dict(SESSION_STYLES) if kind is None else {kind: SESSION_STYLES[kind]}
    for key, style in CONDITION_STYLES.items():
        if any(recording_style(r) == style for r in recordings):
            styles[key] = style
    if any(r.data_type not in styles for r in recordings):
        styles['other'] = OTHER_SESSION_STYLE
    # A separate row reserves space for the legend even after font scaling.
    columns = (len(styles) + 1) // 2
    fig = new_figure(max(18, 4 + .49 * len(rows)), 8.5)
    fig._preserve_schematic_fonts = True
    legend_ax, ax = fig.subplots(2, 1, gridspec_kw={'height_ratios': [2.2, 6]})
    legend_ax.axis('off')
    handles = [Line2D([], [], marker=marker, color=color, markersize=26, linestyle='none', label=name)
               for name, color, marker in styles.values()]
    legend = legend_ax.legend(handles=handles, loc='center', ncol=columns, frameon=False, fontsize=28,
                       columnspacing=1.5, handletextpad=.55)
    for text, (kind, (_, color, _)) in zip(legend.get_texts(), styles.items()):
        text.set_color('black' if kind == 'CO8' else color)
    start = 0
    show_year = len({r.date[:4] for r in recordings if r.date}) > 1
    for date, group in groupby(recordings, key=lambda r: r.date):
        end = start + sum(1 for _ in group)
        ax.text((start + end - 1) / 2, .48, (date if show_year else date[5:]) or 'Unknown date',
                fontsize=14, ha='center', va='center')
        if start:
            ax.axvline(start - .5, ymin=0, ymax=.65, color='#a0a0a0', linewidth=1.1, linestyle='--')
        start = end
    for index, r in enumerate(recordings):
        _, color, marker = recording_style(r)
        ax.plot(index, .09, marker=marker, color=color, markersize=10, linestyle='none')
    if not rows:
        message = 'Select recordings to build the session overview' if kind is None else 'No selected sessions of this type'
        ax.text(.5, .38, message,
                ha='center', transform=ax.transAxes, fontsize=24)
    recording_ticks(ax, range(len(rows)), labels, fontsize=11)
    ax.set(xlim=(-.65, max(.65, len(rows) - .35)), ylim=(0, 1), yticks=[])
    ax.tick_params(axis='x', length=0)
    ax.spines[:].set_visible(False)
    return fig, rows
