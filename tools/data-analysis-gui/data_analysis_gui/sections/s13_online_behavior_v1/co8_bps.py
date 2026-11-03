"""Three explicitly labelled throughput estimates for online CO8."""
import math

import numpy as np

from ...core.crossday import recording_ticks
from ..base import new_figure


TITLE = 'Online CO8 BPS by Method'
GRID_TITLE = 'Online Grid BPS'
METHOD = (
    'CO8, N=8. T is total active attempt time, including failures/timeouts and excluding ITI. '
    'Neuralink-style BPS = (correct selections − explicit wrong selections) × log₂(7) / T. '
    'Timeouts consume time without counting as wrong selections. Generic MISS without a verified selection '
    'outcome makes this metric unavailable. '
    'Wolpaw estimate = [log₂(8) + P log₂(P) + (1−P) log₂((1−P)/7)] / (T / attempts), '
    'P = successes / known attempts. Timeouts count as unsuccessful; this is an approximation because '
    'Wolpaw assumes wrong choices are uniformly distributed across the other targets. '
    'Fitts = sum of nominal difficulty bits for successful trials / sum of their outbound movement times. '
    'Timing uses behavior CSV when available; XDF fallback uses start to acquisition on success and '
    'start to MISS on failure (never trial end, which can include feedback). Fitts requires its CSV inputs. Missing inputs are shown as N/A.'
)
GRID_METHOD = (
    'Grid BPS = C × log₂(N) / T. C = correct selections; N = verified grid cell count. '
    'T = total active attempt seconds, including failed attempts and timeouts, excluding ITI, '
    'block breaks and post-outcome feedback. Missing required inputs are shown as N/A.'
)


def add_xdf_attempt_times(session, trials, source):
    if source != 'XDF event markers':
        return trials
    durations = {row['trial_index'] + 1: row['acquire_time'] if row['outcome'] == 'success' else row.get('miss_time', math.nan)
                 for row in session.trials if row['complete']}
    return [dict(row, duration_s=durations[row['trial']]) for row in trials]


def selection_metrics(trials, targets):
    known = {'acquired', 'timeout', 'miss', 'incorrect', 'wrong_target'}
    durations = [row['duration_s'] for row in trials]
    n = len(trials)
    successes = sum(row['outcome'] == 'acquired' for row in trials)
    wrong = sum(row['outcome'] in ('incorrect', 'wrong_target') for row in trials)
    ambiguous = sum(row['outcome'] == 'miss' for row in trials)
    timeouts = sum(row['outcome'] == 'timeout' for row in trials)
    seconds = (math.fsum(durations) if n and all(row['outcome'] in known for row in trials)
               and all(math.isfinite(t) and t > 0 for t in durations) else math.nan)
    neuralink = wolpaw = math.nan
    if math.isfinite(seconds) and targets is not None:
        p = successes / n
        bits = math.log2(targets) + sum(
            q * math.log2(q / divisor) for q, divisor in ((p, 1), (1 - p, targets - 1)) if q)
        wolpaw = bits * n / seconds
        if not ambiguous:
            neuralink = (successes - wrong) * math.log2(targets - 1) / seconds
    return dict(neuralink_bps=neuralink, wolpaw_bps=wolpaw,
                active_attempt_seconds=seconds, explicit_wrong_selections=wrong,
                timeouts=timeouts, ambiguous_misses=ambiguous)


def co8_metrics(trials, fitts):
    return dict(selection_metrics(trials, 8), fitts_bps=fitts)


def grid_metrics(trials):
    bits = {row['bits'] for row in trials if math.isfinite(row['bits'])}
    targets = (round(2 ** next(iter(bits))) if len(bits) == 1
               and all(math.isfinite(row['bits']) for row in trials) else None)
    targets = targets if targets is not None and targets >= 2 else None
    return dict(selection_metrics(trials, targets), grid_cells=targets)


def bps_figure(rows, title, *, session_labels=False, grid=False):
    fig = new_figure(max(11, min(22, 5 + .5 * len(rows))) if session_labels else 11, 5.5 if grid else 8)
    axes = fig.subplots(1 if grid else 3, 1, sharex=True, squeeze=False).ravel()
    x = np.arange(len(rows))
    panels = [('bps', 'BPS = C × log₂(N) / T')] if grid else [
        ('neuralink_bps', 'A  Neuralink-style net selection rate'),
        ('wolpaw_bps', 'B  Wolpaw estimate'), ('fitts_bps', 'C  Fitts throughput')]
    for ax, (key, label), color in zip(
            axes, panels,
            ('#286e9e', '#c25e00', '#15803d')):
        values = np.array([row[key] for row in rows], dtype=float)
        ax.plot(x, values, 'o-', color=color, markersize=3)
        ax.set(title=label, ylabel='bits/s')
        ax.grid(axis='y', alpha=.23)
        missing = np.flatnonzero(~np.isfinite(values))
        if len(missing) == len(rows):
            ax.text(.5, .5, 'N/A · required inputs unavailable', transform=ax.transAxes, ha='center')
        else:
            for i in missing:
                ax.text(i, .05, 'N/A', transform=ax.get_xaxis_transform(), ha='center', fontsize=7)
        if session_labels:
            for i in range(1, len(rows)):
                if rows[i]['date'] != rows[i - 1]['date']:
                    ax.axvline(i - .5, color='#c8c8c8', linestyle='--', linewidth=.9, zorder=0)
    axes[-1].set_xlim(-.5, len(rows) - .5)
    if session_labels:
        recording_ticks(axes[-1], x, [row['session'] for row in rows], fontsize=8)
    else:
        ticks = sorted(set(range(0, len(rows), max(1, len(rows) // 12))) | {len(rows) - 1})
        axes[-1].set_xticks(ticks, [str(rows[i]['block'] or 'Unknown') for i in ticks])
        axes[-1].set_xlabel('Task block')
    fig.suptitle(title)
    footer = (
        'A  (correct − explicit wrong selections) × log₂(7) / T. Timeout adds time; ambiguous MISS gives N/A.\n'
        'B  [log₂(8) + P log₂(P) + (1−P) log₂((1−P)/7)] / (T / attempts); P = success rate, including timeouts as failures.\n'
        'C  Sum of successful-trial Fitts difficulty / sum of successful-trial outbound movement time.\n'
        'T = active attempt time in seconds, excluding inter-trial intervals. Wolpaw is approximate for timeout outcomes.')
    if grid:
        footer = 'C = correct selections · N = grid cells · T = active attempt seconds, including timeouts; ITI and feedback excluded'
    fig.supxlabel(footer, fontsize=8, multialignment='left')
    return fig
