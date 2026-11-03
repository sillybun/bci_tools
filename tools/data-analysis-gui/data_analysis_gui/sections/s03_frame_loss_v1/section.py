"""S03 v1 — every frame-timing / frame-loss figure of neural_behavior data_qc_v1.

Includes transport overview, relative delivery delay, loss by time and trial, and per-gap zooms.

Inputs: the spike_counts_metadata rows (window_index, start_timestamp [us, host-derived],
end_timestamp, raw_total_count, device_timestamp_100us [FPGA clock, unit actually 1 ms])
and the raw LSL receive stamps (no clock sync, no dejitter). All received frames are used;
no trial or channel filter except in the last figure.
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator

from .clock_histogram import OVERVIEW_FPGA_LOSS, OVERVIEW_NORMAL, OVERVIEW_TEXT_SIZE

from ...core import style
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure
from ..shared_frame_loss import (BASE_PERIOD_S, FPGA_UNIT_S, fpga_analysis, meta_column, metadata_frame_indices,
                                 overlap, transmission_totals)

FRAME_S = BASE_PERIOD_S
NORMAL_MAX = 20             # FPGA step of 19 or 20 ms is one frame
NOMINAL_MS = 20.0           # nominal frame period for the counter fallback
ZOOM_N = 50
GAP_HALF_MS = 200.0
DELAY_BIN_MS = 2.0
DELAY_CENTRAL_MS = 150.0
SERIES, ACCENT, INK2, GOLD, GREEN = '#2a78d6', '#c0392b', '#52514e', '#eda100', '#1baf7a'
RED, BLUE = ACCENT, SERIES
BOX = dict(boxstyle='round,pad=0.45', fc='#fbf6d9', ec='#c9c3a0', lw=0.8)


def _thin(n, keep, target=4000):
    """Row indices for drawing a long full-record trace: every n/target-th frame plus the frames around `keep`."""
    step = max(1, n // target)
    rows = np.r_[np.arange(0, n, step), n - 1, keep, np.asarray(keep) + 1]
    return np.unique(rows[(rows >= 0) & (rows < n)])


def _grid(ax):
    ax.grid(True, color='#d8d7d2', lw=0.6)
    ax.set_axisbelow(True)


def draw_clock_mapping(ax, dev, t, widx, gaps, slope, icpt, resid_ms, zoom_start, missing_by_step, *, title, compact=False):
    lost = int(missing_by_step.sum())
    show = _thin(len(dev), gaps)
    ends = np.array([dev[0], dev[-1]])
    ax.plot(dev[show], t[show], color=OVERVIEW_NORMAL if compact else SERIES, lw=1.0, zorder=3)
    ax.plot(ends, slope * ends + icpt, color='#404854' if compact else ACCENT, lw=1.0, ls='--', zorder=4,
            label='Clock fit' if compact else f'fit: {slope:.8f} host s per device s')
    ax.plot(ends, ends, color=INK2, lw=1.0, ls=':', zorder=2, label='Equal clock rates' if compact else 'identity (equal rates)')
    if len(gaps):
        ax.plot(dev[gaps + 1], t[gaps + 1], ls='none', marker='v', ms=5, mfc=OVERVIEW_FPGA_LOSS if compact else ACCENT, mec='white', mew=.5,
                zorder=5, label='Inferred loss events' if compact else f'acquisition gap ({len(gaps)})')
    ax.set_xlabel('FPGA elapsed / s' if compact else 't_fpga / s (acquisition clock, re. first frame)')
    ax.set_ylabel('Host elapsed / s' if compact else 'host LSL timestamp / s (re. first frame)')
    _grid(ax)
    ax.set_title(title, loc='center' if compact else 'left')
    steps = ', '.join(f'{g:.0f}' for g in np.unique(np.diff(widx)))
    box = (f'frames: {len(widx):,}  ({widx[0]:,.0f} → {widx[-1]:,.0f}, window_index steps = {{{steps}}})\n'
           f'device span {dev[-1]:.1f} s / host span {t[-1]:.1f} s\n'
           f'clock rate ratio: {slope:.8f}  ({(slope - 1) * 1e6:+.1f} ppm)\n'
           f'residual to fit: sd {resid_ms.std():.1f} ms, range {resid_ms.min():.0f} to {resid_ms.max():.0f} ms'
           f'  (delivery jitter)\nacquisition gaps: {len(gaps)}, {lost} frames lost '
           f'({lost / (len(widx) + lost):.3%}, {lost * FRAME_S:.2f} s)')
    if compact:
        box = (f'Clock rate difference: {(slope - 1) * 1e6:+.1f} ppm\n'
               f'Residual SD: {resid_ms.std():.1f} ms\n'
               f'Inferred loss events: {len(gaps):,}')
    ax.text(.985, .03, box, transform=ax.transAxes, ha='right', va='bottom', fontsize=OVERVIEW_TEXT_SIZE if compact else 7.8, bbox=BOX)
    a, b = zoom_start, zoom_start + ZOOM_N
    axi = ax.inset_axes([.065, .52, .42, .42])
    xz, yz = (dev[a:b] - dev[a]) * 1e3, (t[a:b] - t[a]) * 1e3
    axi.step(xz, yz, where='post', color=OVERVIEW_NORMAL if compact else SERIES, lw=1.1, zorder=3)
    axi.plot(xz, yz, ls='none', marker='o', ms=2.8, mfc=OVERVIEW_NORMAL if compact else SERIES, mec='white', mew=.5, zorder=4)
    axi.plot(xz, slope * xz, color='#404854' if compact else ACCENT, lw=1.0, ls='--', zorder=2)
    inside = gaps[(gaps >= a) & (gaps + 1 < min(b, len(dev)))]
    if len(inside):
        gap_widths = (dev[inside + 1] - dev[inside]) * 1e3
        missing = int(missing_by_step[inside].sum())
        frame_text = 'frame' if missing == 1 else 'frames'
        gap_text = (f'Gap: {gap_widths[0]:.0f} ms' if len(inside) == 1 else
                    f'{len(inside)} gaps: {gap_widths.min():.0f}–{gap_widths.max():.0f} ms')
        for i, j in enumerate(inside):
            axi.axvspan(xz[j - a], xz[j + 1 - a], color=OVERVIEW_FPGA_LOSS, alpha=.12,
                        label=f'{gap_text}, {missing} {frame_text} lost' if i == 0 else None)
        endpoints = np.unique(np.r_[inside, inside + 1]) - a
        axi.plot(xz[endpoints], yz[endpoints], ls='none', marker='v', ms=4,
                 mfc=OVERVIEW_FPGA_LOSS, mec='white', mew=.5, zorder=5)
    handles, labels = ax.get_legend_handles_labels()
    gap_handles, gap_labels = axi.get_legend_handles_labels()
    if compact:
        ax.set_title(title, pad=30 if len(handles) + len(gap_handles) > 2 else 20)
        ax.legend(handles + gap_handles, labels + gap_labels, loc='lower center',
                  bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False, borderaxespad=0,
                  fontsize=OVERVIEW_TEXT_SIZE)
    else:
        ax.legend(handles + gap_handles, labels + gap_labels, loc='upper left', bbox_to_anchor=(.54, 1.0),
                  fontsize=7.5, frameon=True, framealpha=.95, edgecolor='#c9c3a0')
    axi.set_title(f'Zoom: {len(xz)} frames at index {widx[a]:,.0f}',
                  fontsize=7, color=INK2, loc='left', pad=3)
    axi.set_xlabel('t_fpga (ms)', fontsize=6.5, labelpad=1)
    axi.set_ylabel('Host / ms' if compact else 'host (ms)', fontsize=6.5, labelpad=1)
    if compact:
        axi.yaxis.set_label_position('right')
    axi.xaxis.set_major_locator(MaxNLocator(nbins=3 if compact else 4))
    if compact:
        axi.yaxis.set_major_locator(MaxNLocator(nbins=3))
    axi.tick_params(labelsize=6)
    axi.grid(True, color='#e6e5e0', lw=.5)


class FrameLoss(Section):
    id = 's03'
    cache_code_dependencies = ('sections/shared_frame_loss.py',)
    title = '丢包率分析'
    description = ('首先展示传输状态总览、相对投递延迟、丢失随时间与 trial 的影响；'
                   '随后补充间隔/时钟分布、时钟拟合及逐缺口放大。无 FPGA 时钟时明确标注未知。')
    params = (
        Param('max_gap_figures', '逐缺口放大图最多张数（0 = 全部）', 'int', 12, minimum=0, maximum=500),
        Param('delay_window_s', '投递延迟时间窗 / s', 'float', 60.0, minimum=5, maximum=600),
        Param('gap_threshold', '接收缺口阈值 / s', 'float', 0.15, minimum=0.02, maximum=10),
        Param('minute_bin', '时间分箱 / min', 'float', 1.0, minimum=0.1, maximum=30),
    )

    # ------------------------------------------------------------------ run
    def run(self, ctx: AnalysisContext, *, overview_target=None) -> SectionResult:
        from .overview import transmission_overview, delivery_overview

        session, cohort, p = ctx.session, ctx.cohort, ctx.params
        result = SectionResult()
        m, lsl = session.metadata, session.neural_times
        if len(lsl) < 3:
            raise ValueError('神经帧太少，无法分析丢包')
        wcol = meta_column(session, 'window_index', 0)
        widx = m[:, wcol] if wcol is not None else np.arange(len(m), dtype=float)
        frame_indices = metadata_frame_indices(session)
        fpga = (ctx.cached(('s03_fpga', id(session)), lambda: fpga_analysis(session, frame_indices))
                if frame_indices is not None else None)
        summary = dict(recording=session.label, n_frames=int(len(lsl)), fpga_clock=fpga is not None,
                       window_index_steps=[float(v) for v in np.unique(np.diff(widx))])
        if len(m) != len(lsl):
            summary.update(metadata_rows=len(m), metadata_matched_frames=0 if frame_indices is None else len(frame_indices))
            result.notes.append(f'神经计数流有 {len(lsl):,} 帧，元数据流有 {len(m):,} 行。接收统计保留全部神经帧。'
                                + ('FPGA 分析仅使用按原始时间戳匹配的元数据，跨越缺元数据帧的步长不计为缺帧。'
                                   if frame_indices is not None else '无法可靠匹配元数据，跳过 FPGA 分析。'))
        if session.stream_audit:
            merged = [a for a in session.stream_audit if a.get('name') == 'spike_counts_metadata']
            if merged:
                summary['metadata_duplicate_streams_merged'] = merged
                result.notes.append('spike_counts_metadata 有重复的流；按 window_index/时间戳合并后共 '
                                    f'{len(lsl):,} 帧（neural_behavior 只取最长的一条，帧数可能相差几帧）。')

        ctx.progress('inter-frame intervals')
        self.interval_stats(session, summary, frame_indices)

        clock_data = None
        delay_data = None
        delay_windows = []
        if fpga is not None:
            ctx.progress('FPGA clock')
            clock = m[:, meta_column(session, 'device_timestamp_100us', 4)].astype(np.int64)
            dev = (clock - clock[0]) * FPGA_UNIT_S
            t = lsl[frame_indices] - lsl[frame_indices[0]]
            if fpga['n_nonpositive_steps']:
                result.notes.append(f'FPGA 时钟有 {fpga["n_nonpositive_steps"]} 个非正步长（回绕/重置），'
                                    '这些步长不计入丢帧；拟合和延迟仍按原始时钟计算。')
            self.step_stats(fpga, summary)
            slope, icpt = np.polyfit(dev, t, 1)
            metadata_steps = fpga['k'][frame_indices[:-1]]
            gaps = np.flatnonzero(metadata_steps > 1)
            clock_data = self.clock_stats(dev, t, widx, gaps, slope, icpt, summary)
            limit = 0 if overview_target is not None else (p['max_gap_figures'] or len(gaps))
            ctx.progress(f'{min(limit, len(gaps))} gap figures')
            gap_rows = []
            for i, j in enumerate(gaps, start=1):
                row = self.gap_stats(i, dev, t, widx, metadata_steps, int(j), slope)
                gap_rows.append(row)
                if i <= limit:
                    ctx.check_cancel()
                    self.gap_figure(result, session, row, dev, t, widx, int(j), slope)
            if overview_target is None and len(gaps) > limit:
                result.notes.append(f'共有 {len(gaps)} 个采集缺口，只画了前 {limit} 个放大图'
                                    '（参数「逐缺口放大图最多张数」，0 = 全部）；全部缺口见表 acquisition_gaps。')
            result.tables['acquisition_gaps'] = [{k: v for k, v in r.items() if not k.startswith('_')}
                                                 for r in gap_rows]
            summary['gap_figures_shown'] = int(min(limit, len(gaps)))
            ctx.progress('relative delivery delay')
            delay, delay_data = self.delay_distribution(dev, t, widx, slope, summary)
            delay_windows = self.delay_windows(dev, delay, p['delay_window_s'], summary)
        elif frame_indices is not None and len(frame_indices) >= 2:
            result.notes.append(f'本记录的 spike_counts_metadata 只有 {m.shape[1]} 个通道'
                                f'（{", ".join(session.metadata_labels)}），没有 device_timestamp_100us（FPGA 采集时钟）：'
                                '无法检测采集端丢帧，也无法计算 FPGA 步长、时钟比、逐缺口放大和相对投递延迟；'
                                'window_index 连续不能证明没有丢帧。')
            self.counter_stats(lsl[frame_indices], widx, summary)

        ctx.progress('loss over time and per trial')
        self.loss_over_time(result, session, cohort, fpga, p, summary)
        totals = ctx.cached(('s03_transmission_totals', id(session), p['gap_threshold']),
                            lambda: transmission_totals(lsl, p['gap_threshold'], fpga))
        transmission_overview(result, session, fpga, widx, p, clock_data,
                              frame_period_ms=FRAME_S * 1000, totals=totals, figure=overview_target)
        if overview_target is not None:
            result.summary.update(summary)
            return result
        delivery_overview(result, session, summary, fpga, delay_data, delay_windows,
                          delay_reference_ms=DELAY_CENTRAL_MS)
        primary = ['Transmission overview', 'Delivery delay overview', 'Loss over time and per trial']
        order = [i for title in primary for i, (name, _) in enumerate(result.figures) if name == title]
        order += [i for i, (name, _) in enumerate(result.figures) if name not in primary]
        result.figures = [(result.figures[i][0] if result.figures[i][0] in primary else
                           'Supplementary ' + result.figures[i][0], result.figures[i][1]) for i in order]
        result.methods = [result.methods[i] for i in order]
        result.figure_data = [result.figure_data[i] for i in order]
        summary['primary_figures'] = [title for title in primary if any(name == title for name, _ in result.figures)]
        result.summary.update(summary)
        return result

    # ------------------------------------------------------- 1 interval histogram
    def interval_stats(self, session, summary, frame_indices):
        m, lsl = session.metadata, session.neural_times
        scol = meta_column(session, 'start_timestamp', 1)
        if scol is not None and frame_indices is not None and len(frame_indices) >= 2:
            t_start = m[:, scol]
            d = np.diff(t_start) / 1000.0                    # us -> ms
            host_offset = float(np.abs(t_start - np.round((lsl[frame_indices] - 0.020) * 1e6)).max())
        else:
            d = np.diff(lsl) * 1000
            host_offset = float('nan')
        counts_int = np.bincount(np.floor(np.maximum(d, 0)).astype(int))
        dominant_bin = int(np.argmax(counts_int))
        le20, gt20, exact0 = int((d <= 20).sum()), int((d > 20).sum()), int((d == 0).sum())
        sel = d[(d > 8) & (d < 24)]
        base = float(np.median(sel)) if len(sel) else FRAME_S * 1000

        summary.update(interval_ms=dict(median=float(np.median(d)), mean=float(d.mean()), sd=float(d.std()),
                                        min=float(d.min()), max=float(d.max()),
                                        p1=float(np.percentile(d, 1)), p99=float(np.percentile(d, 99))),
                       interval_n_exact_zero=exact0, interval_n_le_20ms=le20, interval_n_gt_20ms=gt20,
                       interval_dominant_bin_ms=dominant_bin, link_interval_ms=base,
                       start_timestamp_vs_lsl_max_abs_us=host_offset)


    def step_stats(self, fpga, summary):
        d = fpga['delta'][fpga['valid']]
        n = fpga['n_frames'] - 1 - fpga['n_steps_crossing_missing_metadata']
        normal = int((d <= NORMAL_MAX).sum())
        summary.update(fpga_steps_le_20ms=normal, fpga_steps_gt_20ms=n - normal,
                       fpga_frames_lost=fpga['frames_lost'], fpga_drop_rate=fpga['drop_rate'],
                       fpga_lost_seconds=fpga['lost_seconds'], fpga_max_lost_at_one_step=fpga['max_lost_at_one_step'],
                       fpga_nonpositive_steps=fpga['n_nonpositive_steps'],
                       fpga_max_residual_vs_19p2ms_ms=fpga['max_residual_ms'],
                       fpga_unit_ratio_at_1ms=fpga['unit_ratio_at_1ms'])


    def clock_stats(self, dev, t, widx, gaps, slope, icpt, summary):
        resid_ms = (t - (slope * dev + icpt)) * 1e3
        zoom_start = (int(max(min(gaps[0] - ZOOM_N // 2, len(dev) - ZOOM_N - 1), 0)) if len(gaps)
                      else min(50_000, max(len(dev) - ZOOM_N - 1, 0)))
        summary.update(clock_rate_ratio=float(slope), clock_rate_ppm=float((slope - 1) * 1e6),
                       clock_fit_intercept_s=float(icpt), device_span_s=float(dev[-1]), host_span_s=float(t[-1]),
                       clock_residual_ms=dict(sd=float(resid_ms.std()), min=float(resid_ms.min()),
                                              max=float(resid_ms.max()), p1=float(np.percentile(resid_ms, 1)),
                                              p99=float(np.percentile(resid_ms, 99))),
                       acquisition_gaps=int(len(gaps)),
                       acquisition_gap_window_index=[float(widx[g + 1]) for g in gaps])
        return {
            'per_frame': dict(window_index=widx, t_fpga_s=dev, host_s=t, fit_s=slope * dev + icpt,
                              residual_ms=resid_ms),
            'fit': [dict(slope_host_s_per_device_s=float(slope), intercept_s=float(icpt),
                         ppm=float((slope - 1) * 1e6), zoom_start_row=zoom_start)],
        }

    # ------------------------------------------------------- 4 per-gap zoom
    @staticmethod
    def gap_stats(index, dev, t, widx, k, j, slope):
        half = GAP_HALF_MS / 1e3
        lo, hi = np.searchsorted(dev, [dev[j] - half, dev[j] + half + 1e-12])
        sl = slice(lo, min(hi + 1, len(dev)))
        x = (dev[sl] - dev[j]) * 1e3
        y = (t[sl] - t[j]) * 1e3
        resid = y - slope * x
        return dict(gap=index, window_index_before=float(widx[j]), window_index_after=float(widx[j + 1]),
                    delta_fpga_ms=float((dev[j + 1] - dev[j]) * 1e3), frames_lost=int(k[j] - 1),
                    frame_periods=int(k[j]), device_time_s=float(dev[j]), host_time_s=float(t[j]),
                    frames_in_window=int(len(x)), host_jitter_sd_ms=float(resid.std()),
                    host_jitter_min_ms=float(resid.min()), host_jitter_max_ms=float(resid.max()),
                    _slice=sl, _x=x, _y=y, _resid=resid)

    def gap_figure(self, result, session, row, dev, t, widx, j, slope):
        x, y, resid, sl = row['_x'], row['_y'], row['_resid'], row['_slice']
        delta_ms, lost = row['delta_fpga_ms'], row['frames_lost']
        fig = new_figure(7.4, 4.8)
        ax = fig.add_subplot(111)
        ax.axvspan(0, delta_ms, color=ACCENT, alpha=.10, zorder=0,
                   label=f'gap: {delta_ms:.0f} ms, {lost} frame{"s" if lost != 1 else ""} lost')
        ax.plot(x, slope * x, color=ACCENT, lw=1.0, ls='--', zorder=2, label='fitted host/device rate')
        ax.step(x, y, where='post', color=SERIES, lw=1.2, zorder=3)
        ax.plot(x, y, ls='none', marker='o', ms=3.4, mfc=SERIES, mec='white', mew=.6, zorder=4)
        ax.plot([0, delta_ms], [0, (t[j + 1] - t[j]) * 1e3], ls='none', marker='v', ms=6, mfc=ACCENT, mec='white',
                mew=.6, zorder=5)
        ax.set_xlabel('t_fpga − t_fpga(last frame before gap) / ms')
        ax.set_ylabel('host LSL timestamp / ms (same reference)')
        _grid(ax)
        ax.set_title(f'Acquisition gap {row["gap"]}  |  window_index {widx[j]:,.0f} → {widx[j + 1]:,.0f} · '
                     f'{session.label}', loc='left', fontsize=8.5)
        ax.legend(loc='upper left', fontsize=7, frameon=True, framealpha=.95, edgecolor='#c9c3a0')
        steps = ', '.join(f'{v:.0f}' for v in np.unique(np.diff(widx[sl])))
        box = (f'frames shown: {len(x)}  ({x[0]:.0f} to {x[-1]:.0f} ms)\nwindow_index steps here: {{{steps}}}\n'
               f'host jitter about the rate line: sd {resid.std():.1f} ms, range {resid.min():.0f} to '
               f'{resid.max():.0f} ms')
        ax.text(.985, .03, box, transform=ax.transAxes, ha='right', va='bottom', fontsize=7, bbox=BOX)
        result.add_figure(
            f'Acquisition gap {row["gap"]}', fig,
            method=(f'Gap {row["gap"]}: the step between window_index {widx[j]:,.0f} and {widx[j + 1]:,.0f} with '
                    f'k = rint(Δt_fpga / 19.2 ms) = {row["frame_periods"]} (> 1), so {lost} frame(s) lost. Frames '
                    f'within ±{GAP_HALF_MS:.0f} ms of acquisition time around the last frame before the gap. x = '
                    '(t_fpga − t_fpga of that frame) ms, so the hole occupies (0, Δ]; y = (host LSL − host LSL of that '
                    'frame) ms. Dashed line: y = b·x with b the whole-record OLS clock-rate ratio. Jitter = y − b·x '
                    '(population sd, min, max). Red band = the gap; triangles = the two frames bracketing it.'),
            data={'frames': dict(window_index=widx[sl], x_ms=x, y_ms=y, fitted_ms=slope * x, jitter_ms=resid),
                  'gap': [{k: v for k, v in row.items() if not k.startswith('_')}]})

    # ------------------------------------------------------- 5 delay distribution
    def delay_distribution(self, dev, t, widx, slope, summary):
        r = t - slope * dev
        anchor = int(np.argmin(r))
        a = float(r[anchor])
        d = (r - a) * 1e3
        n = len(d)
        names = ['min', 'p1', 'p5', 'p25', 'median', 'p75', 'p90', 'p95', 'p99', 'p99_9', 'max']
        stats = dict(zip(names, map(float, np.percentile(d, [0, 1, 5, 25, 50, 75, 90, 95, 99, 99.9, 100]))))
        stats.update(mean=float(d.mean()), sd_population=float(d.std()))
        full_max = max(100., math.ceil(d.max() / 100) * 100.)
        # Keep the central 2 ms resolution and bound the tail size for long stalls.
        tail_width = DELAY_BIN_MS * max(1, math.ceil((full_max - DELAY_CENTRAL_MS) / (500 * DELAY_BIN_MS)))
        if tail_width == DELAY_BIN_MS:
            edges = np.arange(0, full_max + DELAY_BIN_MS / 2, DELAY_BIN_MS)
        else:
            edges = np.r_[np.arange(0, DELAY_CENTRAL_MS, DELAY_BIN_MS),
                          np.arange(DELAY_CENTRAL_MS, full_max, tail_width), full_max]
        counts, _ = np.histogram(d, bins=edges)
        pct = counts / n * 100
        summary.update(delay_ms=stats, delay_zero_frame_window_index=float(widx[anchor]),
                       delay_fraction_exceeding_ms={str(v): float(np.mean(d > v)) for v in (20, 50, 100, 150, 200, 300, 500)})
        return d, {
            'histogram_2ms': dict(bin_left_ms=edges[:-1], bin_right_ms=edges[1:], count=counts,
                                  percent_of_frames=pct),
            'tail_bin_width_ms': tail_width,
            'quantiles_ms': [dict(name=k, value=v) for k, v in stats.items()],
            'frames_above_reference': int(np.count_nonzero(d > DELAY_CENTRAL_MS)),
        }

    # ------------------------------------------------------- 6 delay over time
    def delay_windows(self, dev, d, seconds, summary):
        group = np.floor(dev / seconds).astype(np.int64)
        rows = []
        for k in range(int(group[-1]) + 1):
            chosen = group == k
            lo, nominal_hi = k * seconds, (k + 1) * seconds
            observed_hi = min(nominal_hi, float(dev[-1]))
            count = int(chosen.sum())
            q = np.quantile(d[chosen], [.5, .95], method='linear') if count else [np.nan, np.nan]
            rows.append(dict(window=k, start_s=float(lo), nominal_end_s=float(nominal_hi), observed_end_s=observed_hi,
                             midpoint_s=(lo + observed_hi) / 2, observed_duration_s=observed_hi - lo,
                             is_partial=bool(observed_hi < nominal_hi), n_frames=count,
                             median_ms=float(q[0]), p95_ms=float(q[1])))
        summary.update(delay_window_s=seconds, delay_windows=len(rows),
                       delay_window_median_ms_range=[float(np.nanmin([r['median_ms'] for r in rows])), float(np.nanmax([r['median_ms'] for r in rows]))],
                       delay_window_p95_ms_range=[float(np.nanmin([r['p95_ms'] for r in rows])), float(np.nanmax([r['p95_ms'] for r in rows]))])
        return rows

    # ------------------------------------------------------- 7 counter fallback
    def counter_stats(self, lsl, widx, summary):
        t = lsl - lsl[0]
        x = widx - widx[0]
        slope, icpt = np.polyfit(x, t, 1)
        slope_ms = slope * 1e3
        resid_ms = (t - (slope * x + icpt)) * 1e3
        summary.update(counter_fit_ms_per_frame=float(slope_ms), counter_fit_rate_hz=float(1e3 / slope_ms),
                       counter_fit_vs_nominal=float(slope_ms / NOMINAL_MS - 1),
                       counter_accumulated_offset_s=float((slope_ms - NOMINAL_MS) * len(widx) / 1e3),
                       counter_residual_ms=dict(sd=float(resid_ms.std()), min=float(resid_ms.min()),
                                                max=float(resid_ms.max())))


    # ------------------------------------------------------- 8 loss over time / per trial
    def loss_over_time(self, result, session, cohort, fpga, p, summary):
        nt = session.neural_times
        thr = p['gap_threshold']
        intervals = np.diff(nt)
        gap_idx = np.flatnonzero(intervals > thr)
        lefts = cohort.starts + cohort.criteria.window_start
        rights = cohort.starts + cohort.criteria.window_end
        gap_hit = overlap(nt[gap_idx], nt[gap_idx + 1], lefts, rights)
        if fpga is not None:
            drop_idx = np.flatnonzero(fpga['lost'] > 0)
            drop_hit = overlap(nt[drop_idx], nt[drop_idx + 1], lefts, rights)
            lost_in_window = drop_hit @ fpga['lost'][drop_idx] if len(lefts) else np.zeros(0)
        else:
            drop_idx = np.zeros(0, int)
            drop_hit = np.zeros((len(lefts), 0), bool)
            lost_in_window = np.zeros(len(lefts))
        all_starts = np.array([r['start'] for r in cohort.table if np.isfinite(r['start'])])
        all_l, all_r = all_starts + cohort.criteria.window_start, all_starts + cohort.criteria.window_end
        all_gap = overlap(nt[gap_idx], nt[gap_idx + 1], all_l, all_r).any(1)
        all_drop = (overlap(nt[drop_idx], nt[drop_idx + 1], all_l, all_r).any(1) if fpga is not None
                    else np.zeros(len(all_starts), bool))
        exposure = dict(inc_gap=gap_hit.any(1), inc_drop=drop_hit.any(1), all_gap=all_gap, all_drop=all_drop,
                        lost_in_window=lost_in_window)
        fig, per_bin = self.time_figure(session, cohort, intervals, gap_idx, thr, fpga, exposure, p['minute_bin'])
        t0 = nt[0]
        trials = [dict(trial_index=r['trial_index'], trial_id=r['trial_id'], block=r['block'],
                       start_s=round(r['start'] - t0, 3), window_touches_gap=bool(exposure['inc_gap'][i]),
                       window_touches_fpga_drop=bool(exposure['inc_drop'][i]),
                       frames_lost_in_window=int(exposure['lost_in_window'][i])) for i, r in enumerate(cohort.rows)]
        bars = [dict(group='included trials', event='receive gap', trials=int(exposure['inc_gap'].sum()), total=cohort.n),
                dict(group='all trial rows', event='receive gap', trials=int(all_gap.sum()), total=len(all_gap))]
        if fpga is not None:
            bars += [dict(group='included trials', event='FPGA drop', trials=int(exposure['inc_drop'].sum()),
                          total=cohort.n),
                     dict(group='all trial rows', event='FPGA drop', trials=int(all_drop.sum()), total=len(all_drop))]
        for b in bars:
            b['percent'] = round(100 * b['trials'] / b['total'], 4) if b['total'] else 0
        crit = cohort.criteria
        result.tables['receive_gaps'] = [dict(frame=int(i), time_s=round(float(nt[i] - t0), 4),
                                              duration_s=round(float(intervals[i]), 4)) for i in gap_idx]
        result.tables['included_trials'] = trials
        result.tables['per_bin'] = per_bin
        summary.update(gap_threshold_s=thr, n_receive_gaps=int(len(gap_idx)),
                       receive_gap_total_s=float(intervals[gap_idx].sum()), included_trials=cohort.n,
                       included_trials_window_touching_gap=int(exposure['inc_gap'].sum()),
                       trial_rows_window_touching_gap=int(all_gap.sum()), trial_rows_with_start=int(len(all_starts)))
        if fpga is not None:
            summary.update(included_trials_window_touching_drop=int(exposure['inc_drop'].sum()),
                           trial_rows_window_touching_drop=int(all_drop.sum()))
        if crit.neural_check and thr >= crit.max_gap_seconds:
            result.notes.append(f'顶栏筛选已剔除分析窗口内缺口 > {crit.max_gap_seconds} s 的 trial，'
                                '因此保留 trial 中触及接收缺口的比例接近 0；"trial rows" 统计包括所有有起点的 trial。')
        result.add_figure(
            'Loss over time and per trial', fig,
            method=(f'Time bins of {p["minute_bin"]:g} min from the first frame. Receive loss per bin = sum of receive '
                    f'intervals (LSL) > {thr:g} s whose first frame falls in the bin, as % of the bin duration. FPGA '
                    'loss per bin = frames lost (k = rint(Δt_fpga / 19.2 ms), lost = max(k − 1, 0)) at steps starting '
                    'in the bin / (frames received in bin + lost). Per trial: analysis window [T_TRIAL_START + '
                    f'{crit.window_start:g} s, + {crit.window_end:g} s); a window touches a loss event if the interval '
                    'between the two frames bracketing the gap/drop overlaps the window at all (looser than the topbar '
                    'in-window gap criterion). "included trials" = trials kept by the topbar filter; "all trial rows" '
                    '= every reconstructed trial with a start time. Both groups use analysis windows only. Bottom '
                    'right: counts of included trials by positive inferred missing-frame count, on a linear scale.'),
            data={'per_bin': per_bin, 'trial_exposure': bars, 'included_trials': trials})


    def time_figure(self, session, cohort, intervals, gap_idx, thr, fpga, exposure, minute_bin):
        nt = session.neural_times
        t0 = nt[0]
        width = minute_bin * 60
        edges = np.arange(0, nt[-1] - t0 + width, width)
        if len(edges) < 2:
            edges = np.array([0, nt[-1] - t0 + 1e-9])
        rel = nt - t0
        received = np.histogram(rel, edges)[0]
        gap_time = np.histogram(rel[gap_idx], edges, weights=intervals[gap_idx])[0]
        durations = np.diff(edges)
        durations[-1] = max(min(durations[-1], rel[-1] - edges[-2]), 1e-9)
        rows = []
        fig = new_figure(12, 7.5)
        grid = fig.add_gridspec(2, 3)
        ax = fig.add_subplot(grid[0, :])
        centres = (edges[:-1] + edges[1:]) / 2 / 60
        ax.plot(centres, 100 * gap_time / durations, color=BLUE, marker='o', ms=3, lw=1,
                label=f'receive: time in gaps > {thr:g} s (% of bin)')
        if fpga is not None:
            lost = np.histogram(rel[:-1], edges, weights=fpga['lost'])[0]
            rate = np.divide(lost, received + lost, out=np.zeros(len(lost)), where=(received + lost) > 0)
            ax.plot(centres, 100 * rate, color=RED, marker='o', ms=3, lw=1,
                    label='FPGA: lost / (received + lost) (%)')
        else:
            lost = np.zeros(len(received))
        for i in range(len(received)):
            rows.append(dict(bin_start_min=round(edges[i] / 60, 3), frames_received=int(received[i]),
                             receive_gap_seconds=round(float(gap_time[i]), 4),
                             fpga_frames_lost=int(lost[i]) if fpga is not None else '',
                             fpga_drop_rate=(round(float(lost[i] / max(received[i] + lost[i], 1)), 6)
                                             if fpga is not None else '')))
        ax.set_xlabel('time from first frame / min')
        ax.set_ylabel('loss / %')
        ax.set_ylim(bottom=0)
        ax.set_title(f'Loss over time ({minute_bin:g}-min bins)' +
                     ('' if fpga is not None else ' · no FPGA clock: receive side only'))
        ax.legend(frameon=False, loc='upper right')

        ax = fig.add_subplot(grid[1, :2])
        labels, values, totals, colors = [], [], [], []
        labels.append(f'included trials\nwindow touches receive gap'); values.append(exposure['inc_gap'].sum())
        totals.append(cohort.n); colors.append(BLUE)
        labels.append(f'all trial rows\nwindow touches receive gap'); values.append(exposure['all_gap'].sum())
        totals.append(len(exposure['all_gap'])); colors.append(BLUE)
        if fpga is not None:
            labels.append('included trials\nwindow touches FPGA drop'); values.append(exposure['inc_drop'].sum())
            totals.append(cohort.n); colors.append(RED)
            labels.append('all trial rows\nwindow touches FPGA drop'); values.append(exposure['all_drop'].sum())
            totals.append(len(exposure['all_drop'])); colors.append(RED)
        frac = [100 * v / t if t else 0 for v, t in zip(values, totals)]
        y = np.arange(len(labels))
        ax.barh(y, frac, color=colors, height=0.6)
        for yi, f, v, t in zip(y, frac, values, totals):
            ax.text(f, yi, f' {f:.1f}%  ({int(v)}/{t})', va='center', fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, max(max(frac, default=0) * 1.35, 5))
        ax.set_xlabel(f'% of trials · analysis window [{cohort.criteria.window_start:g}, '
                      f'{cohort.criteria.window_end:g}] s from T_TRIAL_START')
        ax.set_title('Loss events in analysis windows only (both trial groups)')

        ax = fig.add_subplot(grid[1, 2])
        if fpga is not None and cohort.n:
            values = exposure['lost_in_window'].astype(int)
            missing, counts = np.unique(values[values > 0], return_counts=True)
            if missing.size:
                bars = ax.bar(missing, counts, color=RED)
                ax.bar_label(bars, padding=3, fontsize=7)
                ax.set_ylim(0, counts.max() * 1.25)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_xlabel('inferred missing frames per window')
                ax.set_ylabel('included trials')
                ax.set_title('Included trials with inferred frame loss')
            else:
                style.text_panel(ax, ['No included analysis windows', 'overlap an inferred frame loss.'])
        else:
            style.text_panel(ax, ['No FPGA clock in this record:', 'acquisition-side drops cannot',
                                  'be counted per trial.'] if fpga is None else ['No included trials.'])
        fig.suptitle(f'{session.label} · frame loss over time and per trial', fontsize=9)
        return fig, rows



SECTION = FrameLoss()
