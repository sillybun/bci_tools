"""Comparable recording-level readouts from the single-day sections."""
from __future__ import annotations

import numpy as np

from ..core.crossday import recording_labels, recording_ticks
from ..core.style import DEFAULT_FONT_PERCENT, finish_figure
from ..decoders import label as decoder_label
from ..sections.base import new_figure


# A series is (display label, path into the single-day summary, optional scale).
PANELS = {
    's01': [('Trials', [('Complete', ('complete_trials',)), ('Included', ('included_trials',))])],
    's02': [('Electrodes by maximum code', [(str(i), ('channels_by_max_code', str(i))) for i in range(4)]),
            ('Frames by code (%)', [(str(i), ('overall_code_fraction', str(i)), 100) for i in range(4)])],
    's03': [('Receive interval (ms)', [('Median', ('interval_ms', 'median')), ('P99', ('interval_ms', 'p99'))]),
            ('FPGA frame loss (%)', [('Inferred loss', ('fpga_drop_rate',), 100)])],
    's04': [('Explained variance (%)', [(f'PC{i + 1}', ('pc_explained_pct', i)) for i in range(3)]),
            ('TDR angle error (°)', [('CV median absolute error', ('tdr_cv_median_abs_error_deg',))])],
    's05': [('ANOVA electrode counts', [('Active', ('active_channels',)), ('Significant', ('significant_channels',)),
                                         ('Gain-controlled significant', ('significant_gain_controlled',))])],
    's06': [('Responsive electrodes', [('Up', ('responsive_up',)), ('Down', ('responsive_down',))]),
            ('Response latency (ms)', [('Median', ('median_latency_s',), 1000)])],
    's07': [('Tuning overlap (electrodes)', [('Direction', ('tuning_direction_channels',)),
                                            ('Temporal', ('tuning_time_channels',)),
                                            ('Both', ('tuning_both_channels',))])],
    's08': [('Transmitted electrodes by tier', [(tier.title(), ('transmitted_by_tier', tier))
                                               for tier in ('good', 'uncertain', 'bad', 'unlisted')])],
    's09': [('Direction decoding accuracy (%)', [('Balanced accuracy', ('balanced_accuracy',), 100),
                                                 ('Permutation P95', ('null_p95',), 100),
                                                 ('Chance', ('chance',), 100)]),
            ('Angle error (°)', [('Mean absolute error', ('mean_abs_error_deg',))])],
    's12': [('Trials', [('Complete', ('complete_trials',)), ('Successful', ('success_trials',)),
                       ('Direction cohort', ('direction_trials',)), ('Velocity cohort', ('velocity_trials',))]),
            ('Neural outages (s)', [('Total duration', ('outage_total_s',))]),
            ('Mean saturation (%)', [('Code ≥3', ('overview_saturation_mean',), 100)]),
            ('PC1 + PC2 variance (%)', [('Explained variance', ('overview_pca_pc12_variance',), 100)]),
            ('Tuning overlap (electrodes)', [('Direction', ('overview_direction_tuned',)),
                                            ('Temporal', ('overview_time_tuned',)),
                                            ('Both', ('overview_both_tuned',))])],
}


def _panels(section_id, summaries):
    if section_id == 's01':
        electrodes = [s['raster_window_mean_hz'] for s in summaries if s and s.get('raster_window_mean_hz')]
        common = set(electrodes[0]).intersection(*electrodes[1:]) if electrodes else set()
        if not common:
            return PANELS[section_id] + [('Window mean (Hz)', [])]
        electrode = next(key for key in electrodes[0] if key in common)
        directions = [(f'{i * 45}°', ('raster_window_mean_hz', electrode, i)) for i in range(8)]
        return PANELS[section_id] + [(f'E{electrode} window mean (Hz)', directions)]
    if section_id in PANELS:
        return PANELS[section_id]
    names = dict.fromkeys(name for summary in summaries if summary for name in summary.get('decoders', []))
    if section_id == 's10':
        return [('Within-recording velocity R²', [(decoder_label(name), (f'{name}_r2_mean',)) for name in names])]
    models = [('Online (closed-loop)', ('online_vs_gt_r2_xy',))]
    models.extend((decoder_label(name), (f'{name}_vs_gt_r2_xy',)) for name in names)
    return [('Velocity R² vs cursor', models)]


def _value(summary, path, scale=1):
    try:
        for key in path:
            summary = summary[key]
        return float(summary) * scale
    except (KeyError, IndexError, TypeError, ValueError):
        return np.nan


def comparison_figure(section_id, recordings, summaries, *, font_percent=DEFAULT_FONT_PERCENT):
    panels = _panels(section_id, summaries)
    columns = 2 if len(panels) > 2 else 1
    rows = (len(panels) + columns - 1) // columns
    fig = new_figure(11, 3 + 1.7 * rows)
    fig.suptitle(f'{section_id.upper()} · Session Metrics')
    axes = fig.subplots(rows, columns, sharex='col', squeeze=False).ravel()
    xs = np.arange(len(recordings))
    for ax, (ylabel, series) in zip(axes, panels):
        for name, path, *scale in series:
            ys = [_value(summary, path, *scale) for summary in summaries]
            if np.isfinite(ys).any():
                ax.plot(xs, ys, 'o-', ms=4, lw=1.2, label=name)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.25)
        if ax.lines:
            ax.legend(ncol=min(4, len(ax.lines)), fontsize=8)
        else:
            message = 'No shared raster electrode' if section_id == 's01' and not series else 'No available values'
            ax.text(0.5, 0.5, message, transform=ax.transAxes, ha='center')
    labels = recording_labels(recordings)
    for ax in axes[len(panels):]:
        ax.remove()
    for ax in axes[len(panels) - columns:len(panels)]:
        recording_ticks(ax, xs, labels, fontsize=8)
        ax.tick_params(labelbottom=True)
        ax.set_xlim(-0.4, len(xs) - 0.6)
    finish_figure(fig, font_percent)
    return fig
