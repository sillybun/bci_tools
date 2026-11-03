"""Primary transport-status views; detailed clock and gap diagnostics follow."""
from __future__ import annotations

import numpy as np
from matplotlib.ticker import MultipleLocator

from .clock_histogram import (
    OVERVIEW_FPGA_LOSS, OVERVIEW_NORMAL, OVERVIEW_RECEIVE_GAP, OVERVIEW_TEXT_SIZE,
    STATS_BOX, draw_clock_histogram,
)

from ...core import style
from ..base import new_figure


def _plot_receive_intervals(ax, elapsed, intervals_ms, gaps, show, threshold):
    show = show[intervals_ms[show] <= threshold * 1000]
    ax.scatter(elapsed[show] / 60, intervals_ms[show], s=3, color=OVERVIEW_NORMAL, alpha=.45)
    if len(gaps):
        ax.scatter(elapsed[gaps] / 60, intervals_ms[gaps], s=12, color=OVERVIEW_RECEIVE_GAP, label='Long receive gaps')
    ax.axhline(threshold * 1000, color=OVERVIEW_RECEIVE_GAP, ls='--', lw=.8, label=f'Threshold: {threshold * 1000:g} ms')
    ax.set_yscale('symlog', linthresh=1)
    if intervals_ms.min() >= 0:
        ax.set_ylim(bottom=0)


def _plot_loss_bins(ax, rows, span, minute_bin, fpga):
    starts = np.arange(len(rows)) * minute_bin
    durations = np.minimum(minute_bin * 60, np.maximum(span - starts * 60, 1e-9))
    centres = starts + durations / 120
    burden = np.array([r['receive_gap_seconds'] for r in rows]) / durations * 100
    ax.plot(centres, burden, 'o-', color=OVERVIEW_RECEIVE_GAP, ms=3, label='Long receive gaps (% of time)')
    if fpga is not None:
        ax.plot(centres, [100 * r['fpga_drop_rate'] for r in rows], 's-', color=OVERVIEW_FPGA_LOSS, ms=3,
                label='Inferred frame loss (% of frames)')


def transmission_overview(result, session, fpga, widx, params, clock_data, *, frame_period_ms, totals, figure=None):
    times = session.neural_times
    intervals_ms = np.diff(times) * 1000
    elapsed = times - times[0]
    threshold = params['gap_threshold']
    gaps = np.flatnonzero(intervals_ms > threshold * 1000)
    show = np.unique(np.r_[np.arange(0, len(intervals_ms), max(1, len(intervals_ms) // 4000)),
                              gaps, np.argmax(intervals_ms)])
    sequence_steps = np.diff(widx)
    fig = new_figure(13, 9) if figure is None else figure
    layout = fig.add_gridspec(3, 2)
    receive = fig.add_subplot(layout[0, 0])
    events = fig.add_subplot(layout[1, 0], sharex=receive)
    loss = fig.add_subplot(layout[2, 0], sharex=receive)
    ax = fig.add_subplot(layout[0, 1])
    # Keep off-view counts in the export without drawing them in the 0–200 ms view.
    view_max = 200
    beyond = int(np.count_nonzero(intervals_ms > view_max))
    edges = np.r_[np.arange(view_max), np.nextafter(float(view_max), np.inf)]
    if intervals_ms.min() < 0:
        edges = np.r_[intervals_ms.min(), edges]
    if beyond:
        edges = np.r_[edges, np.nextafter(intervals_ms.max(), np.inf)]
    hist_counts, edges = np.histogram(intervals_ms, bins=edges)
    visible = (edges[:-1] >= 0) & (edges[:-1] < view_max)
    gap_counts = np.histogram(intervals_ms[gaps], bins=edges)[0]
    normal_counts = hist_counts - gap_counts
    bin_left, bin_width = edges[:-1][visible], np.diff(edges)[visible]
    ax.bar(bin_left, normal_counts[visible], width=bin_width, align='edge',
           color=OVERVIEW_NORMAL, lw=0)
    ax.bar(bin_left, gap_counts[visible], bottom=normal_counts[visible], width=bin_width,
           align='edge', color=OVERVIEW_RECEIVE_GAP, lw=0)
    ax.set_xlim(0, view_max)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.set_yscale('log')
    if hist_counts[visible].max() == 0:
        ax.set_ylim(.8, 10)
    else:
        ax.set_ylim(bottom=.7)
    ax.axvline(threshold * 1000, color=OVERVIEW_RECEIVE_GAP, ls='--', lw=.8)
    ax.set(xlabel='Receive interval / ms', ylabel='Receive interval count (log)',
           title='D  Receive interval distribution')
    ax.text(.98, .97, f'Median: {np.median(intervals_ms):.2f} ms · P95: {np.percentile(intervals_ms,95):.2f} ms\n'
            f'Maximum: {intervals_ms.max():.1f} ms\n'
            f'Long receive gaps (>{threshold * 1000:g} ms): {len(gaps):,}\n'
            f'Zero: {int((intervals_ms == 0).sum()):,} · Negative: {int((intervals_ms < 0).sum()):,}\n'
            f'Above {view_max} ms (outside plot): {beyond:,}',
            transform=ax.transAxes, ha='right', va='top', fontsize=OVERVIEW_TEXT_SIZE, bbox=STATS_BOX)
    ax = fig.add_subplot(layout[1, 1])
    if fpga is None:
        style.text_panel(ax, ['E  No independent acquisition clock',
                              'Receive jitter does not measure acquisition loss.',
                              'Missing-frame count and rate are unknown.'])
    else:
        delta = fpga['delta'][fpga['valid']]
        if len(delta):
            draw_clock_histogram(ax, fpga, compact=True)
            ax.set_title('E  FPGA clock step distribution')
        else:
            style.text_panel(ax, ['E  No positive FPGA clock steps', 'Clock-based loss estimate is not interpretable.'])
    _plot_receive_intervals(receive, elapsed, intervals_ms, gaps, show, threshold)
    receive.set(title='A  Receive intervals over time', ylabel='Receive interval / ms')
    receive.legend(loc='lower right', fontsize=OVERVIEW_TEXT_SIZE)

    event_rows = []
    clock_steps = {}
    if fpga is not None:
        missing = fpga['lost']
        valid = fpga['valid']
        normal = valid & (missing == 0)
        event_indices = np.flatnonzero(missing > 0)
        event_rows = [dict(start_host_s=float(elapsed[i]), inferred_missing_frames=int(missing[i]))
                      for i in event_indices]
        if valid.any():
            step_ms = fpga['delta']
            events.scatter(elapsed[:-1][normal] / 60, step_ms[normal], color=OVERVIEW_NORMAL,
                           s=3, alpha=.35, rasterized=True, label='Clock steps without inferred loss')
            if event_indices.size:
                events.scatter(elapsed[event_indices] / 60, step_ms[event_indices], color=OVERVIEW_FPGA_LOSS,
                               s=16, label='Inferred loss events')
            cutoff = 1.5 * frame_period_ms
            events.axhline(cutoff, color=OVERVIEW_FPGA_LOSS, ls='--', lw=.8, label=f'Threshold: {cutoff:g} ms')
            events.set_yscale('log')
            events.legend(loc='upper right', facecolor='white', framealpha=1, fontsize=OVERVIEW_TEXT_SIZE)
            clock_steps = dict(start_host_s=elapsed[:-1][valid], step_ms=step_ms[valid],
                               inferred_missing_frames=missing[valid])
        else:
            events.text(.5, .5, 'No valid positive FPGA clock steps',
                        transform=events.transAxes, ha='center', va='center')
            events.set_yticks([])
    else:
        events.text(.5, .5, 'Missing-frame events unknown (no FPGA clock)',
                    transform=events.transAxes, ha='center', va='center')
        events.set_yticks([])
    events.set(title='B  FPGA clock steps over time', ylabel='FPGA clock step / ms')

    rows = result.tables['per_bin']
    minute_bin = params['minute_bin']
    _plot_loss_bins(loss, rows, elapsed[-1], minute_bin, fpga)
    loss.set(xlabel='Host elapsed time from first received frame / min', ylabel='Percentage / %')
    loss.set_title(f'C  Long receive gaps and inferred frame loss ({minute_bin:g}-min bins)', pad=20)
    loss.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2, frameon=False,
                borderaxespad=0, fontsize=OVERVIEW_TEXT_SIZE)
    gap_fraction = totals['long_receive_gap_time_fraction']
    fpga_rate = totals['fpga_frame_lost_rate']
    gap_label = f'{gap_fraction:.4%}' if gap_fraction is not None else 'Unknown'
    fpga_label = f'{fpga_rate:.4%}' if fpga_rate is not None else 'Unknown'
    loss.text(.02, .97, f'Session totals\nFPGA frame loss rate: {fpga_label}\n'
              f'Long receive gap time fraction: {gap_label}', transform=loss.transAxes,
              ha='left', va='top', fontsize=OVERVIEW_TEXT_SIZE, bbox=STATS_BOX)
    final_start = (len(rows) - 1) * minute_bin
    if elapsed[-1] / 60 < final_start + minute_bin:
        loss.axvspan(final_start, elapsed[-1] / 60, color=style.GREY, alpha=.15, zorder=0)
    for ax in (receive, events, loss):
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    for ax in (receive, events):
        ax.tick_params(labelbottom=False)
    mapping = fig.add_subplot(layout[2, 1])
    mapping_title = 'F  Host time vs FPGA time'
    if fpga is None:
        style.text_panel(mapping, [mapping_title, 'Unavailable without an FPGA clock'])
    else:
        from .section import draw_clock_mapping

        frames, fit = clock_data['per_frame'], clock_data['fit'][0]
        metadata_loss = fpga['lost'][fpga['frame_indices'][:-1]]
        draw_clock_mapping(mapping, frames['t_fpga_s'], frames['host_s'], frames['window_index'],
                           np.flatnonzero(metadata_loss > 0), fit['slope_host_s_per_device_s'],
                           fit['intercept_s'], frames['residual_ms'], fit['zoom_start_row'],
                           metadata_loss, title=mapping_title, compact=True)
    for ax in fig.axes:
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_size(9)
        ax.yaxis.label.set_size(9)
        ax.tick_params(axis='both', which='both', labelsize=8)
    clock_status = ('FPGA loss UNKNOWN (no independent clock)' if fpga is None else
                    f'Inferred missing frames: {fpga["frames_lost"]:,} ({fpga["drop_rate"]:.3%}) · '
                    f'Nonpositive FPGA steps: {fpga["n_nonpositive_steps"]:,}')
    fig.suptitle(f'{session.label} · Transmission overview\n'
                 f'{len(times):,} received frames · {elapsed[-1]/60:.2f} min · '
                 f'Nonconsecutive frame indices: {int((sequence_steps != 1).sum()):,}\n{clock_status}', fontsize=9)
    result.add_figure(
        'Transmission overview', fig,
        method=('All received frames. A, B and C share the raw host LSL time axis. A shows receive intervals '
                'at the first frame, retaining all long gaps in green. B plots all positive FPGA clock steps '
                'at the first frame, highlighting inferred loss in orange with a threshold of 1.5 frame periods. '
                'C assigns events to their starting host bin. Green is long-gap duration / observed bin duration; '
                'orange is inferred missing / expected frames. Shading marks the partial final bin. '
                'The session totals use unrounded values: long-gap duration / first-to-last receive span, '
                'and inferred missing frames / (FPGA-matched received frames + inferred missing frames). '
                'D is the raw LSL receive-interval histogram, with a linear 0–200 ms view and 1 ms bins. '
                'Off-view counts are retained in the export. '
                'E counts FPGA clock steps, with steps above 100 ms in an inset. Nonpositive FPGA steps '
                'are excluded from the estimate. F shows the host/FPGA clock fit and a 50-frame zoom, '
                'with shaded gaps and the inferred missing frames in the zoom. '
                'Missing FPGA clock means unknown loss.'),
        data={'receive_intervals': dict(start_host_s=elapsed[:-1], interval_ms=intervals_ms),
              'receive_interval_histogram': dict(bin_left_ms=edges[:-1], bin_right_ms=edges[1:], count=hist_counts),
              **(clock_data or {}), 'per_bin': rows, 'fpga_loss_events': event_rows, 'fpga_clock_steps': clock_steps,
              'session_totals': [totals],
              'clock_steps': [dict(delta_ms=int(d), count=int(n)) for d, n in
                              zip(*np.unique(fpga['delta'][fpga['valid']], return_counts=True))] if fpga is not None else []})


def delivery_overview(result, session, summary, fpga, delay_data, windows, *, delay_reference_ms):
    if fpga is None:
        return
    fig = new_figure(12, 8)
    grid = fig.add_gridspec(2, 2)
    main, full = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    trend = fig.add_subplot(grid[1, :])
    histogram = delay_data['histogram_2ms']
    if fpga['n_nonpositive_steps']:
        for ax, title in zip((main, full, trend), ('A  Main distribution', 'B  Full distribution', 'C  Delay over time')):
            style.text_panel(ax, [title, 'Whole-record delay unavailable across FPGA clock resets or wraps.'])
        fig.suptitle(f'{session.label} · Relative delivery delay', fontsize=11)
    else:
        left_edges = histogram['bin_left_ms']
        widths = histogram['bin_right_ms'] - left_edges
        central = histogram['bin_right_ms'] <= delay_reference_ms
        main.bar(left_edges[central], histogram['percent_of_frames'][central], width=widths[central],
                 align='edge', color=OVERVIEW_NORMAL)
        main.set(xlim=(0, delay_reference_ms), ylabel='Frames per 2 ms bin / %',
                 title=f'A  Main distribution (0–{delay_reference_ms:g} ms)')
        full.bar(left_edges, histogram['count'], width=widths, align='edge', color=OVERVIEW_NORMAL)
        full.set_yscale('log')
        full.set_ylim(bottom=.7)
        tail_width = delay_data['tail_bin_width_ms']
        full.set(ylabel='Frame count per bin (log)', title='B  Full distribution including the tail')
        if tail_width > 2:
            full.set_title(f'B  Full distribution (2 ms central / up to {tail_width:g} ms tail bins)')
        q = summary['delay_ms']
        for ax in (main, full):
            ax.axvline(q['median'], color=OVERVIEW_NORMAL, ls='--', lw=1, label='Median')
            ax.axvline(q['p95'], color='#c25e00', ls=':', lw=1, label='P95')
            ax.set_xlabel('Relative delivery delay / ms')
        main.set_xlim(0, delay_reference_ms)
        full.axvline(delay_reference_ms, color='#52514e', ls='--', lw=1,
                     label=f'{delay_reference_ms:g} ms delay reference')
        full.legend(loc='upper right', fontsize=8)
        n_above = delay_data['frames_above_reference']
        n_frames = int(histogram['count'].sum())
        full.text(.98, .71, f'Above {delay_reference_ms:g} ms: {n_above:,} frames ({n_above / n_frames:.3%})',
                  transform=full.transAxes, ha='right', va='top', fontsize=8, bbox=STATS_BOX)
        x = [r['midpoint_s'] / 60 for r in windows]
        trend.plot(x, [r['median_ms'] for r in windows], 'o-', color=OVERVIEW_NORMAL, ms=3, label='Median')
        trend.plot(x, [r['p95_ms'] for r in windows], 's--', color='#c25e00', ms=3, label='P95')
        for row in windows:
            if row['is_partial']:
                trend.axvspan(row['start_s'] / 60, row['observed_end_s'] / 60, color=style.GREY, alpha=.15,
                              label=f'Final partial window ({row["observed_duration_s"]:.1f} s)')
        trend.set(xlabel='FPGA elapsed time / min', ylabel='Relative delivery delay / ms',
                  title=f'C  Delay over time ({summary["delay_window_s"]:g} s FPGA windows)', ylim=(0, None))
        trend.legend(loc='upper right', fontsize=8)
        fig.suptitle(f'{session.label} · Relative delivery delay\n'
                     f'{n_frames:,} frames · Median {q["median"]:.1f} ms · P95 {q["p95"]:.1f} ms · '
                     f'P99 {q["p99"]:.1f} ms · Maximum {q["max"]:.1f} ms\n'
                     'Extra delay relative to the fastest frame', fontsize=10)
    result.add_figure(
        'Delivery delay overview', fig,
        method=('All received frames. The relative delay is (host elapsed − b × FPGA elapsed − minimum residual) '
                '× 1000 ms, with b fitted across the full record. A shows percentages of all frames. B shows full '
                'counts on a log scale and marks the 150 ms display reference. The central bins are 2 ms wide. '
                'The tail uses wider bins when needed; see the bin edges in the data table. C shows median and P95 in FPGA-time '
                'windows. All panels use the same fitted clock ratio and zero reference. Shading marks a partial '
                'final window. Absolute latency is unknown. Clock resets or wraps prevent whole-record interpretation.'),
        data={('histogram' if delay_data['tail_bin_width_ms'] > 2 else 'histogram_2ms'): histogram,
              'quantiles_ms': delay_data['quantiles_ms'], 'windows': windows,
              'interpretable': [dict(whole_record_clock_continuous=not bool(fpga['n_nonpositive_steps']))]})
