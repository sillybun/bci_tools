"""Run-level success and successful-trial acquisition time."""
import numpy as np

from ...core import behavior
from ...core.crossday import recording_ticks
from ..base import new_figure

TITLE = 'Online CO8 Run Success and Time to Target'
METHOD = (behavior.METHOD + ' Run statistics pool the selected non-overlapping trial sources within each '
          'run. A complete run CSV takes precedence over overlapping XDF fragments. Success rate is '
          'computed from pooled counts, and time quantiles from pooled successful-trial times; block '
          'percentages or medians are not averaged. Error bars show IQR, not a confidence interval.')


def run_figure(rows):
    fig = new_figure(max(10, len(rows) * .85), 6.5)
    left = fig.subplots()
    right = left.twinx()
    x = np.arange(len(rows))
    success = np.array([r['success_pct'] for r in rows])
    median = np.array([r['rt_median_s'] for r in rows])
    q1 = np.array([r['rt_q1_s'] for r in rows])
    q3 = np.array([r['rt_q3_s'] for r in rows])
    blue, orange = '#286e9e', '#a65b24'
    line, = left.plot(x - .08, success, 'o-', color=blue, label='Success rate')
    times = right.errorbar(x + .08, median, yerr=[median - q1, q3 - median],
                          fmt='s-', color=orange, capsize=4, label='Time to target · median and IQR')
    for i, row in enumerate(rows):
        if np.isfinite(success[i]):
            left.annotate(f'{row["successes"]}/{row["attempts"]}', (i - .08, success[i]),
                          (0, 8), textcoords='offset points', ha='center', color=blue, fontsize=8)
    if not np.isfinite(median).any():
        right.text(.5, .5, 'Time to target: N/A', transform=right.transAxes, ha='center', color=orange)
    left.set(ylim=(0, 112), xlim=(-.5, len(rows) - .5), ylabel='Success rate (%)')
    left.set_yticks(np.arange(0, 101, 20))
    right.set(ylabel='Time to target (s)', ylim=(0, None))
    right.margins(y=.2)
    left.tick_params(axis='y', colors=blue)
    right.tick_params(axis='y', colors=orange)
    left.yaxis.label.set_color(blue)
    right.yaxis.label.set_color(orange)
    left.grid(axis='y', alpha=.23)
    recording_ticks(left, x, [r['session'] for r in rows], fontsize=8)
    for i in range(1, len(rows)):
        if rows[i]['date'] != rows[i - 1]['date']:
            left.axvline(i - .5, color='.75', linestyle='--', linewidth=.8)
    left.legend(handles=[line, times], loc='upper center', bbox_to_anchor=(.5, 1.13),
                ncol=2, frameon=False)
    fig.suptitle('Online CO8 Behavior by Run')
    return fig
