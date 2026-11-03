"""S01 v1 — raw data visualisation.

Eight-direction raster and trial-averaged PSTH, a recording-time raster, and a
trial selection overview with task success rates.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
from scipy.stats import theilslopes

from ...core import analysis as A
from ...core import style
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure


def anchors(cohort, align):
    if align == 'target_acquired':
        return cohort.starts + cohort.array('acquire_time')
    return cohort.starts


def pick_electrode(ctx: AnalysisContext, requested: int) -> int:
    """Stored column for an electrode id; 0 = most direction-modulated channel."""
    ids = ctx.session.electrode_ids
    if requested:
        hit = np.flatnonzero(ids == requested)
        if not len(hit):
            raise ValueError(f'电极 {requested} 不在本记录的 512 个传输通道中')
        return int(hit[0])
    stats = ctx.cached(('anova', id(ctx.cohort)),
                       lambda: A.oneway_anova(ctx.cohort.window_rates, ctx.cohort.directions))
    return int(np.nanargmax(stats['F']))


class RawVisualization(Section):
    id = 's01'
    title = 'Raw data visualisation'
    description = '八方向 raster、方向响应随记录时间变化、八方向 trial average、trial 筛选概况、任务成功率'
    params = (
        Param('electrode', '电极 ID（可输入多个）', 'text', '0',
              help='输入电极 ID，以逗号或空格分隔（如 12, 35, 90）。留空或输入 0 自动选方向调制最强的一个电极。'),
        Param('all_active_electrodes', '计算所有活跃电极', 'bool', False,
              help='勾选后按记录列顺序绘制整份记录中至少有一帧计数非零的电极，优先于手动输入的 ID。'),
        Param('activity_stability_atlas', '生成全通道稳定性图册', 'bool', False,
              help='每页绘制 64 个电极并导出逐方向稳定性指标。预览时可跳过，勾选后重新加载 / 计算可补齐图册。'),
        Param('align', '对齐事件', 'choice', 'trial_start', choices=('trial_start', 'target_acquired')),
        Param('tmin', '起始时间 / s', 'float', -0.5),
        Param('tmax', '结束时间 / s', 'float', 2.0),
        Param('bin', 'PSTH bin / s', 'float', 0.05, minimum=0.02, maximum=0.5),
        Param('max_trials', '每方向 raster 最多 trial 数', 'int', 60, minimum=5),
        Param('recording_raster_direction', '整场 raster 方向', 'choice', '0°',
              choices=tuple(f'{d * 45}°' for d in range(8)),
              help='只影响整场时间 raster。使用该方向全部可对齐的保留 trial，不受每方向最多 trial 数限制。'),
    )

    def run(self, ctx: AnalysisContext) -> SectionResult:
        p, cohort, session = ctx.params, ctx.cohort, ctx.session
        if p['tmax'] <= p['tmin']:
            raise ValueError('结束时间必须大于起始时间')
        result = SectionResult()
        columns = []
        if not cohort.n:
            result.notes.append('当前筛选条件下没有保留的 trial，只显示筛选概况和任务成功率。')
        elif p['all_active_electrodes']:
            columns = np.flatnonzero(np.any(session.counts, axis=0)).tolist()
            selection = 'selected because it has a nonzero count in at least one frame of the recording. '
            if not columns:
                result.notes.append('本记录没有活跃电极，只显示筛选概况和任务成功率。')
        else:
            entries = re.split(r'[,，\s]+', str(p['electrode']).strip() or '0')
            if not all(entry.isascii() and entry.isdecimal() for entry in entries):
                raise ValueError('请输入整数电极 ID，以逗号或空格分隔')
            requested = list(dict.fromkeys(map(int, entries)))
            automatic = requested == [0]
            if 0 in requested and not automatic:
                raise ValueError('0 表示自动选择，请单独输入')
            columns = [pick_electrode(ctx, electrode) for electrode in requested]
            selection = ('chosen automatically as the channel with the largest one-way ANOVA F over the eight '
                         'directions of the analysis-window mean rate. ' if automatic else 'chosen by the user. ')
        if columns:
            electrodes = [int(session.electrode_ids[column]) for column in columns]
            pages = [f' {i}' if len(columns) > 1 else '' for i in range(1, len(columns) + 1)]
            anchor = anchors(cohort, p['align'])
            raster_rates = {}
            for page, column, electrode in zip(pages, columns, electrodes):
                ctx.check_cancel()
                ctx.progress(f'raster · E{electrode}')
                fig, events, means = self.raster(cohort, column, electrode, anchor, p)
                raster_rates[str(electrode)] = means.tolist()
                method = (
                f'Included trials of the topbar cohort, grouped by target direction (first {p["max_trials"]} per '
                f'direction in chronological order). Electrode {electrode} = stored column {column + 1}; '
                + selection
                + f'Each tick is one 20 ms frame with a non-zero 2-bit count (marker size proportional to count, 3 = >=3), '
                f'at its LSL receive time minus the {p["align"]} event time, window {p["tmin"]}..{p["tmax"]} s. '
                f'Shaded band = topbar analysis window {cohort.criteria.window_start}-{cohort.criteria.window_end} s '
                'after T_TRIAL_START. Centre polar plot: mean rate (count / 0.020 s) in that window per direction.')
                result.add_figure(f'Eight-direction raster{page}', fig, method=method, data=dict(events=events))
            direction = int(p['recording_raster_direction'].removesuffix('°')) // 45
            selected = np.flatnonzero(cohort.directions == direction)
            rows = selected[np.isfinite(anchor[selected])]
            for page, column, electrode in zip(pages, columns, electrodes):
                ctx.check_cancel()
                ctx.progress(f'recording-time raster · E{electrode}')
                fig, events, trials = self.recording_raster(cohort, column, electrode, direction, rows, anchor, p)
                result.add_figure(f'Direction raster over recording time{page}', fig, method=(
                    f'Electrode {electrode}; direction {direction * 45}°. All {len(rows)} of {len(selected)} '
                    'included trials in that direction with a finite alignment timestamp; no trial cap. '
                    'Shared y-axis: trial start in minutes relative to the first received neural frame. '
                    f'Left: each nonzero received frame at time relative to {p["align"]}, in the half-open '
                    f'window [{p["tmin"]}, {p["tmax"]}) s; marker size scales with the stored count (3 means ≥3). '
                    'Faint horizontal lines mark trial rows. Right: stored counts summed over the same window '
                    'divided by the number of received frames times the integration duration per frame. '
                    'Crosses mark trials with no received frames in the window.'), data=dict(events=events, trials=trials))
            binned, edges = session.binned_rates(anchor, p['tmin'], p['tmax'], p['bin'])
            population = binned.mean(axis=2)
            for page, column, electrode in zip(pages, columns, electrodes):
                ctx.check_cancel()
                ctx.progress(f'PSTH · E{electrode}')
                fig, psth = self.average(cohort, column, electrode, p, binned, edges, population)
                method = (
                f'For every included trial, frames in consecutive {p["bin"] * 1000:.0f} ms bins from {p["tmin"]} to '
                f'{p["tmax"]} s around {p["align"]} are averaged and converted to Hz (count / 0.020 s); bins without '
                'any received frame are NaN and skipped. Per direction: mean across trials +/- SEM '
                '(SD / sqrt(number of trials with a valid bin)). Outer panels: electrode '
                f'{electrode}; centre: mean over all 512 stored columns, per direction.')
                result.add_figure(f'Eight-direction trial average{page}', fig, method=method, data=dict(psth=psth))
            result.summary.update(electrode=electrodes[0], stored_column=columns[0] + 1,
                                  electrodes=electrodes, align=p['align'], raster_window_mean_hz=raster_rates)
        result.add_figure('Filtered trial overview', self.overview(cohort), method=(
            'Trials are reconstructed from the event stream (T_TRIAL_START ... T_TARGET_ACQUIRED|MISS ... '
            'T_TRIAL_END) and checked against cursor trial/block identity while active_trial == 1. Every row lists '
            'all reasons that excluded it under the topbar criteria: '
            + ', '.join(f'{k}={v}' for k, v in cohort.criteria.to_dict().items())
            + '. Top-left: number of trials per reason (a trial can have several). Top-right: trials with a '
            'direction label (grey) and included trials (coloured) per direction. Bottom: trial duration '
            '(T_TRIAL_END - T_TRIAL_START) against start time; red crosses are excluded trials; pink bands mark '
            'neural receive gaps > 0.5 s. Top-left bars: red = the single primary reason of each excluded trial '
            '(first failing criterion in the fixed priority order); orange = trials of the direction cohort that leave '
            'the velocity cohort because of velocity-bin quality (velocity/KF analyses only); blue = flags, which do '
            'not exclude. Top-right: labelled / direction-cohort / velocity-cohort trials per direction.'),
            data=dict(trials=cohort.exclusion_table()))
        result.tables['trial_selection'] = cohort.exclusion_table()
        result.summary.update(cohort.summary())
        fig, rates = self.task_success(cohort)
        result.add_figure('Task success rate', fig, method=(
            'At each trial start, plot the success rate through that trial in a trailing window of up to '
            '40 completed trials with an unambiguous outcome. Time is measured from the first neural frame. '
            'Uses recording events before Topbar filters. Overall includes trials without a direction label. '
            'Directions with no attempts in the window have a gap in the line.'),
            data=dict(task_success=rates))
        result.tables['task_success'] = rates
        if cohort.n and p['activity_stability_atlas']:
            self.activity_stability(ctx, result, anchors(cohort, p['align']))
        return result

    def activity_stability(self, ctx, result, anchor):
        cohort, session, p = ctx.cohort, ctx.session, ctx.params
        rows = np.flatnonzero(np.isfinite(anchor))
        rows = rows[np.argsort(cohort.starts[rows], kind='stable')]
        times = (cohort.starts[rows] - session.neural_times[0]) / 60
        directions = cohort.directions[rows]
        edges = np.searchsorted(session.neural_times, [anchor[rows] + p['tmin'], anchor[rows] + p['tmax']])
        rates = np.full((len(rows), len(session.electrode_ids)), np.nan)
        for i, (lo, hi) in enumerate(zip(*edges)):
            if hi > lo:
                rates[i] = session.counts[lo:hi].mean(axis=0) / session.frame_seconds

        by_direction = [np.flatnonzero(directions == d) for d in range(8)]
        stability_rows = []
        for page, start in enumerate(range(0, len(session.electrode_ids), 64), 1):
            ctx.check_cancel()
            ctx.progress(f'neural activity stability · page {page}')
            fig = new_figure(16, 12)
            axes = fig.subplots(8, 8, sharex=True).ravel()
            page_rows = []
            stop = min(start + 64, len(session.electrode_ids))
            for column, ax in zip(range(start, stop), axes):
                electrode = int(session.electrode_ids[column])
                for direction, indices in enumerate(by_direction):
                    x, y = times[indices], rates[indices, column]
                    ax.plot(x, y, color=style.direction_color(direction), lw=0.75, alpha=0.8,
                            marker='.', ms=2, label=f'{direction * 45}°' if column == start else None,
                            gid=f'electrode-{electrode}-direction-{direction * 45}')
                    valid = np.isfinite(y)
                    x, y = x[valid], y[valid]
                    median = float(np.median(y)) if len(y) else None
                    mad = float(np.median(np.abs(y - median))) if len(y) >= 2 else None
                    slope = float(theilslopes(y, x).slope) if len(np.unique(x)) >= 3 else None
                    page_rows.append(dict(electrode=electrode, direction_deg=direction * 45, n_valid_trials=len(y),
                                          median_hz=median, mad_hz=mad, slope_hz_per_min=slope))
                ax.set_title(f'E{electrode}', fontsize=8, pad=3)
                ax.set_ylim(bottom=0)
                ax.tick_params(labelsize=6)
            for ax in axes[stop - start:]:
                ax.set_visible(False)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside upper center', ncol=8, frameon=False, fontsize=8,
                       title=f'{session.label} · neural activity stability · stored columns {start + 1}–{stop}',
                       title_fontsize=11)
            fig.supxlabel('Trial start since first neural frame (min)', fontsize=10)
            fig.supylabel('Mean rate (Hz)', fontsize=10)
            result.add_figure(f'Neural activity stability {page}', fig, method=(
                f'All recorded electrodes in stored column order, 64 per 8×8 page. Each panel shows one '
                f'electrode with its own rate scale and a shared time scale. Each line connects successive '
                f'included trials of the same target direction. Each point is the mean count per received frame '
                f'divided by the frame duration, over [{p["tmin"]}, {p["tmax"]}) s from {p["align"]}. '
                'Trials without received frames leave gaps. Time is trial start relative to the first neural frame. '
                'The per-electrode, per-direction table includes the median, MAD and Theil–Sen slope.'),
                data=dict(stability=page_rows, trial_rates=dict(
                    trial_id=cohort.trial_ids[rows], elapsed_min=times, direction_deg=directions * 45,
                    **{f'E{session.electrode_ids[col]}_hz': rates[:, col] for col in range(start, stop)})))
            stability_rows.extend(page_rows)
        result.tables['neural_activity_stability'] = stability_rows

    def task_success(self, cohort, *, figure=None):
        attempts = sorted(
            (r for r in cohort.table if r['complete'] and r['outcome'] in ('success', 'miss')),
            key=lambda r: r['start'])
        window_size = 40
        t0 = cohort.session.neural_times[0] if len(cohort.session.neural_times) else np.nan
        times = [(r['start'] - t0) / 60 for r in attempts]
        groups = [('Overall', None, style.INK)]
        groups.extend((f'{d * 45}°', d, style.direction_color(d)) for d in range(8))
        rows = []
        fig = new_figure(11, 5) if figure is None else figure
        ax = fig.subplots()
        for label, direction, color in groups:
            rates = []
            for i, elapsed in enumerate(times):
                window = attempts[max(0, i + 1 - window_size):i + 1]
                trials = [r for r in window if direction is None or r['direction'] == direction]
                successes = sum(r['outcome'] == 'success' for r in trials)
                rate = successes / len(trials) if trials else None
                rows.append(dict(elapsed_min=elapsed, trial_id=attempts[i]['trial_id'], direction=label,
                                 successes=successes, attempts=len(trials), success_rate=rate))
                rates.append(100 * rate if rate is not None else np.nan)
            ax.step(times, rates, where='post', color=color, label=label, lw=2 if direction is None else 1)
        if not times or not np.isfinite(t0):
            ax.text(0.5, 0.5, 'No trials with known outcomes and a neural time reference',
                    transform=ax.transAxes, ha='center')
        ax.legend(ncol=9, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1))
        ax.set_ylim(-3, 103)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel('Success rate / %')
        ax.set_xlabel('Trial start since first neural frame / min')
        fig.suptitle(f'{cohort.session.label} · task success rate · trailing {window_size} trials', fontsize=10)
        return fig, rows

    # ------------------------------------------------------------------ figures
    def recording_raster(self, cohort, column, electrode, direction, rows, anchor, p):
        session = cohort.session
        nt = session.neural_times
        origin = nt[0]
        elapsed = (cohort.starts[rows] - origin) / 60
        left, right = np.searchsorted(nt, [anchor[rows] + p['tmin'], anchor[rows] + p['tmax']])
        counts = session.counts[:, column]
        events, trials, event_times, event_minutes, event_counts = [], [], [], [], []
        for row, minute, lo, hi in zip(rows, elapsed, left, right):
            received = counts[lo:hi]
            hit = np.flatnonzero(received)
            relative = nt[lo + hit] - anchor[row]
            total = int(received.sum())
            trial = dict(electrode=electrode, direction_deg=direction * 45,
                         trial_id=int(cohort.trial_ids[row]), block=int(cohort.blocks[row]),
                         elapsed_min=float(minute), trial_start_lsl_s=float(cohort.starts[row]),
                         alignment_lsl_s=float(anchor[row]))
            trials.append(dict(trial, received_frames=len(received), stored_count=total,
                               mean_hz=total / (len(received) * session.frame_seconds) if len(received) else np.nan))
            events.extend(dict(trial, time_s=float(t), count=int(received[j])) for t, j in zip(relative, hit))
            event_times.extend(relative)
            event_minutes.extend([minute] * len(hit))
            event_counts.extend(received[hit])

        fig = new_figure(11, 7)
        raster, rates = fig.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [3, 1]})
        color = style.direction_color(direction)
        raster.hlines(elapsed, p['tmin'], p['tmax'], color='#e0e0e0', linewidth=0.45, zorder=0)
        raster.scatter(event_times, event_minutes, s=9 * np.asarray(event_counts), c=color,
                       marker='|', linewidths=0.8, rasterized=True)
        raster.axvline(0, color=style.GREY, linewidth=0.8, linestyle='--')
        raster.set(xlim=(p['tmin'], p['tmax']), xlabel=f'Time from {p["align"]} (s)',
                   ylabel='Trial start since first neural frame (min)', title='A  Event raster by trial start')
        duration = max((nt[-1] - origin) / 60, elapsed.max(initial=0), 1 / 60)
        raster.set_ylim(duration, elapsed.min(initial=0))
        hz = np.asarray([trial['mean_hz'] for trial in trials])
        valid = np.isfinite(hz)
        rates.scatter(hz[valid], elapsed[valid], s=12, color=color)
        if (~valid).any():
            rates.scatter(np.full((~valid).sum(), 0.04), elapsed[~valid], marker='x',
                          color=style.GREY, transform=rates.get_yaxis_transform(), label='No frames')
            rates.legend(loc='best', fontsize=8)
        rates.set(xlabel='Observed window mean rate (Hz)', title='B  Per-trial mean rate')
        rates.set_xlim(left=0, right=max(hz[valid].max(initial=0) * 1.06, 1))
        if not len(rows):
            raster.text(0.5, 0.5, 'No alignable trials for this direction', transform=raster.transAxes,
                        ha='center', va='center')
        fig.suptitle(f'{session.label} · E{electrode} · {direction * 45}° · {len(rows)} trials', fontsize=11)
        return fig, events, trials

    def raster(self, cohort, column, electrode, anchor, p):
        session = cohort.session
        nt, counts = session.neural_times, session.counts[:, column]
        fig = new_figure(11, 8.5)
        axes, centre = style.eight_direction_axes(fig)
        means = np.zeros(8)
        events = []
        for d, ax in axes.items():
            rows = np.flatnonzero(cohort.directions == d)[:p['max_trials']]
            for k, i in enumerate(rows):
                lo, hi = np.searchsorted(nt, [anchor[i] + p['tmin'], anchor[i] + p['tmax']])
                hit = np.flatnonzero(counts[lo:hi] > 0)
                for j in hit:
                    events.append(dict(direction_deg=d * 45, trial_id=int(cohort.trial_ids[i]), raster_row=k,
                                       time_s=float(nt[lo + j] - anchor[i]), count=int(counts[lo + j])))
                if len(hit):
                    ax.scatter(nt[lo:hi][hit] - anchor[i], np.full(len(hit), k), s=1.5 * counts[lo:hi][hit],
                               c=style.direction_color(d), marker='|', linewidths=0.6)
            ax.axvline(0, color=style.GREY, lw=0.6)
            ax.axvspan(cohort.criteria.window_start, cohort.criteria.window_end, color='#f1efe8', zorder=-1)
            ax.set_title(f'{d * 45}°  (n={len(rows)})', fontsize=8)
            ax.set_xlim(p['tmin'], p['tmax'])
            ax.set_ylim(-1, max(len(rows), 1))
            ax.set_ylabel('trial')
            ax.set_xlabel(f't from {p["align"]} / s')
        rates = cohort.window_rates[:, column]
        for d in range(8):
            sel = cohort.directions == d
            means[d] = np.nanmean(rates[sel]) if sel.any() else np.nan
        self._polar(fig, centre, means, f'E{electrode}\nwindow mean Hz')
        fig.suptitle(f'{session.label} · electrode {electrode} (column {column + 1}) · marker size ∝ count',
                     fontsize=9)
        return fig, events, means

    def average(self, cohort, column, electrode, p, binned, edges, population):
        session = cohort.session
        centres = (edges[:-1] + edges[1:]) / 2
        fig = new_figure(11, 8.5)
        axes, centre = style.eight_direction_axes(fig)
        psth = []
        for d, ax in axes.items():
            sel = cohort.directions == d
            if not sel.any():
                continue
            y = binned[sel, :, column]
            m, se = np.nanmean(y, 0), np.nanstd(y, 0) / np.sqrt(np.maximum(np.sum(np.isfinite(y), 0), 1))
            pop = np.nanmean(population[sel], 0)
            for t, a, b, c in zip(centres, m, se, pop):
                psth.append(dict(direction_deg=d * 45, n_trials=int(sel.sum()), time_s=float(t),
                                 electrode_mean_hz=float(a), electrode_sem_hz=float(b), population_mean_hz=float(c)))
            ax.fill_between(centres, m - se, m + se, color=style.direction_color(d), alpha=0.25, lw=0)
            ax.plot(centres, m, color=style.direction_color(d), lw=1.2)
            ax.axvline(0, color=style.GREY, lw=0.6)
            ax.set_title(f'{d * 45}° (n={sel.sum()})', fontsize=8)
            ax.set_xlabel('t / s')
            ax.set_ylabel(f'E{electrode} Hz')
        for d in range(8):
            sel = cohort.directions == d
            if sel.any():
                centre.plot(centres, np.nanmean(population[sel], 0), color=style.direction_color(d), lw=1,
                            label=f'{d * 45}°')
        centre.axvline(0, color=style.GREY, lw=0.6)
        centre.set_title('Population mean (all 512)', fontsize=8)
        centre.set_xlabel('t / s')
        centre.set_ylabel('Hz')
        centre.legend(ncol=2, fontsize=5.5, frameon=False)
        fig.suptitle(f'{session.label} · electrode {electrode} (column {column + 1}) · '
                     f'trial-averaged rate ± SEM · bin {p["bin"] * 1000:.0f} ms', fontsize=9)
        return fig, psth

    def overview(self, cohort):
        table = cohort.table
        fig = new_figure(12, 7.5)
        grid = fig.add_gridspec(2, 3)
        primary = Counter(r['primary_reason'] for r in table if r['primary_reason'])
        flags = Counter(f for r in table for f in r['flags'])
        velocity = Counter(x for reasons in (cohort.velocity_audit['reasons'] if cohort.n else [])
                           for x in set(reasons))
        ax = fig.add_subplot(grid[0, :2])
        items = ([(k, v, '#c76d4b', 'excluded (primary reason)') for k, v in primary.most_common()]
                 + [(k, v, '#e0a458', 'velocity cohort only') for k, v in velocity.most_common()]
                 + [(k, v, '#6a8caf', 'flag (kept)') for k, v in flags.most_common()])
        if items:
            seen = set()
            for i, (k, v, colour, kind) in enumerate(items):
                ax.barh(i, v, color=colour, label=kind if kind not in seen else None)
                seen.add(kind)
                ax.text(v, i, f' {v}', va='center', fontsize=7)
            ax.set_yticks(range(len(items)))
            ax.set_yticklabels([k for k, *_ in items], fontsize=7)
            ax.invert_yaxis()
            ax.legend(frameon=False, fontsize=6.5, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        else:
            ax.text(0.5, 0.5, 'No trial excluded or flagged', ha='center', va='center', transform=ax.transAxes)
        status = cohort.session_status()
        ax.set_title(f'Primary reason per trial · velocity-bin quality · flags  |  session '
                     f'{"qualified" if status["qualified"] else "NOT qualified: " + "; ".join(status["issues"])}'
                     f' · {status["session_type"].split("（")[0]}', fontsize=8)
        ax.set_xlabel('trials')

        ax = fig.add_subplot(grid[0, 2])
        total = np.bincount([r['direction'] for r in table if r['direction'] >= 0], minlength=8)
        kept = np.bincount(cohort.directions, minlength=8) if cohort.n else np.zeros(8)
        vel = np.bincount(cohort.directions[cohort.velocity_include], minlength=8) if cohort.n else np.zeros(8)
        x = np.arange(8)
        ax.bar(x, total, color=style.GREY, label='labelled')
        ax.bar(x, kept, color=[style.direction_color(d) for d in x], alpha=0.45, label='direction cohort')
        ax.bar(x, vel, color=[style.direction_color(d) for d in x], label='velocity cohort')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{d * 45}' for d in x])
        ax.set_xlabel('direction / °')
        ax.set_title('Trials per direction')
        ax.legend(frameon=False, fontsize=6.5, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.16))

        ax = fig.add_subplot(grid[1, :])
        t0 = min((r['start'] for r in table if np.isfinite(r['start'])), default=0)
        for r in table:
            if not np.isfinite(r['start']) or not np.isfinite(r['duration']):
                continue
            color = style.direction_color(r['direction']) if r['included'] else '#c76d4b'
            ax.scatter(r['start'] - t0, r['duration'], s=8 if r['included'] else 14, c=color,
                       marker='o' if r['included'] else 'x', linewidths=0.8)
        if cohort.criteria.overtime_enabled and cohort.criteria.overtime_mode == 'fixed' \
                and cohort.criteria.overtime_basis == 'start_to_end':
            ax.axhline(cohort.criteria.overtime_seconds, color='#c76d4b', ls='--', lw=0.8, label='overtime threshold')
        ax.axhline(cohort.criteria.window_end, color=style.GREY, ls=':', lw=0.8, label='analysis window end')
        gaps = np.diff(cohort.session.neural_times)
        for i in np.flatnonzero(gaps > 0.5):
            ax.axvspan(cohort.session.neural_times[i] - t0, cohort.session.neural_times[i + 1] - t0,
                       color='#fde0dd', zorder=-1)
        ax.set_xlabel('trial start / s from first trial (pink = neural gap > 0.5 s)')
        ax.set_ylabel('trial duration / s')
        ax.set_title(f'{cohort.n} included (dots, coloured by direction) · '
                     f'{len(table) - cohort.n} excluded (red ×)')
        ax.legend(frameon=False, loc='upper right')
        fig.suptitle(f'{cohort.session.label} · trial selection', fontsize=9)
        return fig

    @staticmethod
    def _polar(fig, centre, values, title):
        spec = centre.get_subplotspec()
        centre.remove()
        ax = fig.add_subplot(spec, projection='polar')
        theta = np.radians(np.r_[np.arange(0, 360, 45), 0])
        ax.plot(theta, np.r_[values, values[0]], color=style.INK, lw=1.2)
        ax.scatter(theta[:-1], values, c=list(style.DIRECTION_COLORS), s=18, zorder=3)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=6)
        return ax


SECTION = RawVisualization()
