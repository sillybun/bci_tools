"""Sixty-second active-time throughput and per-run distributions of those windows."""
from collections import defaultdict
import math
from itertools import groupby
import json
from pathlib import Path

import numpy as np

from ...core import behavior
from ...core.catalog import RUN_PREFIX, run_folder
from ...core.run_config import naming_config
from ...core.crossday import recording_ticks
from ..base import new_figure
from . import co8_bps

WINDOW_SECONDS = 60.0
STEP_SECONDS = 1.0
METHOD = (
    'Trailing 60 seconds of accumulated active attempt time, step 1 second. '
    'ITI, block breaks and feedback excluded; dwell, failed attempts and unfinished attempt time included. '
    'Long runs start at 60s with a strict 1s grid. Only an entire verified run shorter than 60s uses one window at its endpoint: ratio=T/60; BPS=bits/(60*ratio). '
    'Windows never cross sessions/fragments or invalid timing/trial-identity gaps. '
    'Only outcomes completed in (window start, window end] count; future outcomes are never used. '
    'Grid BPS=correct selections*log2(N)/T. CO8 Neuralink/Wolpaw use the same window and T. '
    'Fitts retains successful difficulty/outbound movement time over fully contained trials only. '
    'Run median/IQR and peak use these windows, including ratio-normalized partial windows. '
    'Overlapping windows are descriptive; IQR is not a confidence interval. '
    'TTT pools successful trials once, independently of overlapping BPS windows. '
    'Grid 6 bits/s reference=log2(N)/6 seconds at 100% success, a mean-time budget.'
)


def methods(mode):
    if mode == 'grid':
        return [('bps', 'BPS = C × log₂(N) / T')]
    return [('neuralink_bps', 'Neuralink-style'), ('wolpaw_bps', 'Wolpaw'), ('fitts_bps', 'Fitts')]


def run_key(recording):
    folder = run_folder(recording.path)
    return str(folder.resolve()) if folder else '|'.join((recording.date, recording.task_mode,
                                                         str(recording.online), recording.run or recording.key))


def windows(trials, mode, *, session='', date='', run=''):
    """Trailing active-time windows on a strict 1s grid; short runs use bits/T.

    Outcomes contribute only when completed in (left, right]. In-progress attempt
    time contributes to T, never its future outcome. Partial coverage uses bits/T.
    """
    ordered = sorted(trials, key=lambda r: r.get('trial', 0))
    known = {'acquired', 'timeout', 'miss', 'incorrect', 'wrong_target'}
    segments, segment = [], []
    active_elapsed, previous = 0.0, None
    for position, original in enumerate(ordered, 1):
        index = original.get('trial', position)
        duration = original.get('duration_s', math.nan)
        valid = original['outcome'] in known and math.isfinite(duration) and duration > 0
        if not valid or (previous is not None and index != previous + 1):
            if segment:segments.append(segment)
            segment, previous = [], None
        if not valid:continue
        left = active_elapsed
        active_elapsed += duration
        segment.append((dict(original), left, active_elapsed, index))
        previous = index
    if segment:segments.append(segment)
    rows = []
    short_run = len(segments) == 1 and len(segments[0]) == len(ordered) and active_elapsed < WINDOW_SECONDS-1e-9
    for segment_id, segment in enumerate(segments):
        origin, stop = segment[0][1], segment[-1][2]
        length = stop-origin
        if length < WINDOW_SECONDS-1e-9:
            if not short_run:continue
            ends = [length]
        else:
            ends = np.arange(WINDOW_SECONDS, length+1e-9, STEP_SECONDS)
        starts = np.array([r[1] for r in segment]);stops = np.array([r[2] for r in segment])
        for relative_end in ends:
            right = origin+relative_end;cut = max(origin, right-WINDOW_SECONDS);seconds = right-cut
            a = int(np.searchsorted(stops, cut+1e-9, side='right'))
            b = int(np.searchsorted(starts, right-1e-9, side='left'))
            overlapping = segment[a:b]
            completed = [dict(r, duration_s=end-max(left,cut)) for r,left,end,index in overlapping if end <= right+1e-9]
            fully_contained = [r for r,left,end,index in overlapping if left >= cut-1e-9 and end <= right+1e-9]
            values = behavior.metrics(completed, mode)
            if mode == 'centerout':
                fitts = behavior.metrics(fully_contained, mode)['bps']
                selection = co8_bps.co8_metrics(completed, fitts)
            else:
                selection = co8_bps.grid_metrics(completed)
            completed_seconds = selection['active_attempt_seconds']
            for key in ('neuralink_bps','wolpaw_bps'):
                selection[key] = selection[key]*completed_seconds/seconds if completed else 0.
            selection['active_attempt_seconds'] = seconds
            values.update(selection)
            if mode == 'grid':
                bits = {r['bits'] for r,_,_,_ in overlapping}
                verified = len(bits)==1 and all(math.isfinite(x) for x in bits)
                bit = next(iter(bits)) if verified else math.nan
                values['grid_cells'] = round(2**bit) if verified else None
                values['bps'] = sum(r['outcome']=='acquired' for r in completed)*bit/seconds
            else:values['bps'] = values['fitts_bps']
            rows.append(dict(session=session, date=date, run_key=run or session,
                trial_start=overlapping[0][3], trial_end=overlapping[-1][3], window_trials=len(completed),
                window_attempts=len(overlapping),window_seconds=WINDOW_SECONDS,step_seconds=STEP_SECONDS,
                effective_window_seconds=seconds,window_ratio=seconds/WINDOW_SECONDS,
                partial_window=seconds<WINDOW_SECONDS-1e-9,
                terminal_fractional_step=not math.isclose(relative_end,round(relative_end),abs_tol=1e-9),
                window_start_active_s=cut,window_end_active_s=right,segment_id=segment_id,
                partial_first_trial=overlapping[0][1]<cut-1e-9,
                partial_last_trial=overlapping[-1][2]>right+1e-9,
                fitts_fully_contained_trials=len(fully_contained),valid_active_time=True,invalid_reason='',**values))
    return rows


def daily_summary(rows, mode, *, dates=()):
    groups = defaultdict(list)
    for row in rows:
        groups[row['date']].append(row)
    for date in dates:
        groups.setdefault(date, [])
    result = []
    for date, group in sorted(groups.items()):
        row = dict(date=date, windows=len(group), sessions=len({r['session'] for r in group}))
        for key, _ in methods(mode):
            valid = [r for r in group if math.isfinite(r[key])]
            q1, median, q3 = np.quantile([r[key] for r in valid], [.25, .5, .75]) if valid else [math.nan] * 3
            peak = max(valid, key=lambda r: r[key]) if valid else None
            row.update({f'{key}_n': len(valid), f'{key}_q1': float(q1), f'{key}_median': float(median),
                        f'{key}_q3': float(q3), f'{key}_peak': peak[key] if peak else math.nan,
                        f'{key}_peak_session': peak['session'] if peak else '',
                        f'{key}_peak_trial_start': peak['trial_start'] if peak else None,
                        f'{key}_peak_trial_end': peak['trial_end'] if peak else None,
                        f'{key}_peak_start_active_s': peak['window_start_active_s'] if peak else None,
                        f'{key}_peak_end_active_s': peak['window_end_active_s'] if peak else None})
        result.append(row)
    return result


def ttt_for_bps(grid_cells, target_bps=6.0):
    """Ideal mean successful-attempt time at 100% success; not a median bound."""
    try:
        n = float(grid_cells)
    except (TypeError, ValueError):
        return math.nan
    return math.log2(n) / target_bps if math.isfinite(n) and n > 1 and target_bps > 0 else math.nan


def run_summary(rows, mode, *, runs, trials_by_run=None):
    groups = defaultdict(list)
    for row in rows:
        groups[row['run_key']].append(row)
    summaries = []
    for run in sorted({r['run_key']: r for r in runs}.values(), key=lambda r: (r['date'], r['run_key'])):
        summary, = daily_summary(groups[run['run_key']], mode, dates=[run['date']])
        summary.update(run_key=run['run_key'], run=run['session'])
        if trials_by_run is not None:
            behavior_row = behavior.metrics(trials_by_run.get(run['run_key'], []), mode)
            summary.update({key: behavior_row[key] for key in ('rt_n', 'rt_median_s', 'rt_q1_s', 'rt_q3_s')})
        if mode == 'grid':
            cells = {row.get('grid_cells') for row in groups[run['run_key']] if row.get('grid_cells') is not None}
            grid_cells = next(iter(cells)) if len(cells) == 1 else None
            if trials_by_run is not None:
                grid_cells = co8_bps.grid_metrics(trials_by_run.get(run['run_key'], []))['grid_cells'] or grid_cells
            summary.update(grid_cells=grid_cells, ttt_for_6bps_s=ttt_for_bps(grid_cells))
        summaries.append(summary)
    return summaries


def rolling_figure(rows, mode, title):
    fig = new_figure(11, 5.5 if mode == 'grid' else 9)
    axes = fig.subplots(len(methods(mode)), 1, sharex=True, squeeze=False).ravel()
    for ax, (key, label) in zip(axes, methods(mode)):
        x = [r['window_end_active_s'] / 60 for r in rows]
        for segment_id in dict.fromkeys(r['segment_id'] for r in rows):
            group = [r for r in rows if r['segment_id'] == segment_id]
            ax.plot([r['window_end_active_s'] / 60 for r in group], [r[key] for r in group],
                    '.-', color='#286e9e', markersize=3)
        valid = [r for r in rows if math.isfinite(r[key])]
        if valid:
            peak = max(valid, key=lambda r: r[key])
            ax.scatter(peak['window_end_active_s'] / 60, peak[key], color='#bd5021', marker='*', s=150, zorder=5)
            ax.annotate(f"Peak {peak[key]:.3g} bits/s · trials {peak['trial_start']}–{peak['trial_end']}",
                        (peak['window_end_active_s'] / 60, peak[key]), (0, 9), textcoords='offset points',
                        ha='right' if peak['window_end_active_s'] / 60 > np.median(x) else 'left', fontsize=8)
            ax.margins(y=.25)
        else:
            ax.text(.5, .5, 'N/A · no verified active time or required inputs unavailable',
                    transform=ax.transAxes, ha='center', fontsize=8)
        ax.set(title=label, ylabel='bits/s')
        ax.grid(axis='y', alpha=.23)
    axes[-1].set_xlabel('Accumulated active attempt time (min; 60 s trailing window)')
    fig.suptitle(title)
    fig.supxlabel('60s window · 1s step · partial windows: bits / actual seconds · star = session peak', fontsize=8)
    return fig


def grid_run_config_path(run_key):
    if not run_key:
        return None
    folder = Path(run_key)
    if folder.is_file():
        folder = run_folder(folder) or folder.parent
    return next((p for p in (folder/'config/task_config.json', folder/'task_config.json') if p.is_file()), None)


def _grid_config_cells(row):
    folder = Path(row.get('run_key') or '')
    match = RUN_PREFIX.match(folder.name)
    run = str(int(match['number'])) + (match['part'] or '') if match else row['run']
    path = grid_run_config_path(row.get('run_key') or '')
    try:
        config = json.loads(path.read_text(encoding='utf-8-sig')) if path else {}
    except (OSError, ValueError):
        config = {}
    fields = naming_config(config)

    def multiplier(value, base):
        return f'×{value/base:.3g}' if isinstance(value, (int, float)) and math.isfinite(value) else '—'

    def switch(value):
        return 'On' if value is True else 'Off' if value is False else '—'

    next_target = fields['preview_next_target']
    nearby = fields['nearby_assist']
    if str(row['date']).replace('-', '') == '20261111':
        # Confirmed settings for the November 11 archive before these flags were saved.
        next_target = False if next_target is None else next_target
        nearby = False if nearby is None else nearby
    radius = fields['nearby_radius']
    assist = fields['bci_target_assist_speed']
    return (
        run, multiplier(fields['joystick_gain'], .012), multiplier(fields['bci_velocity_scale'], 1.),
        switch(next_target),
        f'{2*radius+1}×{2*radius+1}' if nearby is True and radius in (1, 2, 3) else switch(nearby),
        f'{assist:g}' if fields['bci_target_assist_enabled'] is True and isinstance(assist, (int, float))
        else switch(fields['bci_target_assist_enabled']),
    )


def _grid_date_label(date):
    value = str(date).replace('-', '')
    return f'{value[4:6]}.{value[6:8]}' if len(value) == 8 and value.isdigit() else str(date)


def _grid_run_table(ax, ttt_ax, table, rows, show_assist):
    fields = ['Date', 'Run #', 'Cursor Speed', 'BC Speed', 'Next Target', 'Nearby Target']
    if show_assist:
        fields.append('Assist Speed')
    table.set_yticks(range(len(fields)), fields)
    table.set_ylim(len(fields)-.5, -.5)
    table.tick_params(axis='both', length=0)
    table.spines[:].set_visible(False)
    table.tick_params(axis='x', labelbottom=False)
    for y in range(len(fields)+1):
        table.axhline(y-.5, color='#dbe3eb', lw=.7)
    phase_starts = []
    for date, group in groupby(enumerate(rows), key=lambda item: item[1]['date']):
        indices = [i for i, _ in group]
        left, right = indices[0]-.5, indices[-1]+.5
        center = (left+right)/2
        shade = '#eaf1f7' if len(phase_starts) % 2 == 0 else '#f4f7fa'
        table.axvspan(left, right, color=shade, zorder=0, ec='none')
        table.text(center, 0, _grid_date_label(date), ha='center', va='center', fontsize=10, fontweight='bold')
        phase_starts.append(left)
    for i, row in enumerate(rows):
        for y, value in enumerate(_grid_config_cells(row)[:len(fields)-1], 1):
            table.text(i, y, value, ha='center', va='center', fontsize=9,
                       fontweight='bold' if y == 1 else 'normal')
    for boundary in phase_starts[1:]:
        for item in (ax, ttt_ax, table):
            item.axvline(boundary, color='#a6b6c4', linestyle='--', lw=.85, zorder=1)


def grid_run_figure(rows, *, online=True, show_assist=None):
    """BPS and TTT panels with a shared date and run-configuration axis."""
    title = 'Online Grid Run Rolling BPS Summary'
    fig = new_figure(max(12, 1.6 + .48*len(rows)), 8.4)
    bps_ax, ttt_ax, table_ax = fig.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [3.4, 3.4, 1.55]})
    axes = (bps_ax, ttt_ax)
    x = np.arange(len(rows))
    for ax, is_time in zip(axes, (False, True)):
        color = '#c25e00' if is_time else '#286e9e'
        median = np.array([r.get('rt_median_s', math.nan) if is_time else r['bps_median'] for r in rows], float)
        q1 = np.array([r.get('rt_q1_s', math.nan) if is_time else r['bps_q1'] for r in rows], float)
        q3 = np.array([r.get('rt_q3_s', math.nan) if is_time else r['bps_q3'] for r in rows], float)
        ax.errorbar(x, median, yerr=[median-q1, q3-median], fmt='s-' if is_time else 'o-',
                    capsize=3, color=color, markersize=4,
                    label='Time to target · median and IQR' if is_time else 'BPS · median and IQR')
        for i, value in enumerate(median):
            if math.isfinite(value):
                ax.annotate(f'{value:.3g}', (i, value), (5, -8), textcoords='offset points',
                            ha='left', va='top', color=color, fontsize=7,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.3))
        if not is_time:
            peaks = np.array([r['bps_peak'] for r in rows], float)
            ax.plot(x, peaks, '*-', color='#bd5021', markersize=7, label='BPS · peak')
            for i, value in enumerate(peaks):
                if math.isfinite(value):
                    ax.annotate(f'{value:.3g}', (i, value), (0, 6), textcoords='offset points',
                                ha='center', color='black', fontsize=7)
        if is_time:
            target = np.array([ttt_for_bps(r.get('grid_cells')) for r in rows], float)
            ax.plot(x, target, '^--', color='#15803d', markersize=4, linewidth=1,
                    label='6 bits/s reference · 100% success')
            for i, value in enumerate(target):
                if math.isfinite(value):
                    ax.annotate(f'{value:.2f}', (i, value), (0, -8), textcoords='offset points',
                                ha='center', va='top', color='#15803d', fontsize=6.5)
        if not np.isfinite(median).any():
            ax.text(.5, .5, 'N/A · required inputs unavailable', transform=ax.transAxes, ha='center', color=color)
        ax.set_ylabel('Time to target (s)' if is_time else 'BPS (bits/s)')
        ax.margins(y=.3)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.23)
        ax.legend(loc='upper right', frameon=False, fontsize=6.5, ncol=2)
    bps_ax.set_xlim(-.5, len(rows)-.5)
    for ax in axes:
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
    _grid_run_table(bps_ax, ttt_ax, table_ax, rows, online if show_assist is None else show_assist)
    fig.suptitle(f'{"Online" if online else "Offline"} Grid · BPS and Time to Target by Run')
    return title, fig


def run_figure(rows, mode, *, peak=False, online=True):
    if mode == 'grid':
        return grid_run_figure(rows, online=online)
    task = behavior.TASK_LABELS[mode]
    combined = mode == 'grid'
    title = f'Online {task} Run Rolling BPS ' + ('Summary' if combined else 'Peak' if peak else 'Median and IQR')
    fig = new_figure(max(10, len(rows) * .8), 6.5 if combined else 9)
    axes = fig.subplots(len(methods(mode)), 1, sharex=True, squeeze=False).ravel()
    x = np.arange(len(rows))
    for ax, (key, label) in zip(axes, methods(mode)):
        values = np.array([r[f'{key}_peak' if peak and not combined else f'{key}_median'] for r in rows])
        if peak and not combined:
            ax.plot(x, values, '*-', color='#bd5021', markersize=9)
            for i, value in enumerate(values):
                if math.isfinite(value):
                    ax.annotate(f'{value:.3g}', (i, value), (0, 7), textcoords='offset points', ha='center')
        else:
            q1 = np.array([r[f'{key}_q1'] for r in rows])
            q3 = np.array([r[f'{key}_q3'] for r in rows])
            ax.errorbar(x, values, yerr=[values - q1, q3 - values], fmt='o-', capsize=4,
                        color='#286e9e', label='Median and IQR')
            ax.legend(frameon=False)
        if combined:
            for i, value in enumerate(values):
                if math.isfinite(value):
                    ax.annotate(f'{value:.3g}', (i, value), (7, -10),
                                textcoords='offset points', ha='left', va='top',
                                color='#286e9e', fontsize=8,
                                bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.4))
            peaks = np.array([r[f'{key}_peak'] for r in rows])
            ax.plot(x, peaks, '*--', color='#bd5021', markersize=9, label='Peak')
            for i, value in enumerate(peaks):
                if math.isfinite(value):
                    ax.annotate(f'{value:.3g}', (i, value), (0, 8), textcoords='offset points', ha='center', fontsize=8)
            ax.legend(frameon=False)
        if not np.isfinite(values).any():
            ax.text(.5, .5, 'N/A · required inputs unavailable', transform=ax.transAxes, ha='center')
        ax.set(title=label, ylabel='bits/s')
        ax.grid(axis='y', alpha=.23)
        ax.margins(y=.25)
    recording_ticks(axes[-1], x, [r['run'] for r in rows], fontsize=8)
    axes[-1].set_xlabel('Run')
    fig.suptitle(title)
    fig.supxlabel('60s window · 1s step · partial windows normalized by actual coverage · ITI and feedback excluded', fontsize=8)
    if combined:
        fig.supxlabel('BPS = C × log₂(N) / T · C = correct selections · T = 60 active attempt seconds, including timeouts; ITI excluded', fontsize=8)
    return title, fig
