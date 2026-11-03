"""Electrode-aligned overlap of the actual S05 direction and S06 temporal tests."""
from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
from matplotlib import colormaps

from ...core import analysis as A, style
from ...core.channels import read_channel_config, column_tiers
from ..base import new_figure
from ..s05_anova_v1.section import channel_anova
from ..s06_psth_v1.section import temporal_statistics

PARTITIONS = ('inactive', 'active_not_significant', 'direction_only', 'time_only', 'both')
LABELS = ('Inactive', 'Active, not significant', 'Direction only', 'Time only', 'Both')
COLORS = ('#cccccc', '#2471a3', '#c0392b', '#d77b35', '#806ba5')
TIERS = ('good', 'uncertain', 'bad', 'unlisted')


def classify(direction_active, time_active, direction_sig, time_sig):
    """Active is the union of both source definitions, making both tests subsets."""
    active = np.asarray(direction_active, bool) | np.asarray(time_active, bool)
    direction = np.asarray(direction_sig, bool)
    temporal = np.asarray(time_sig, bool)
    if not (active.shape == direction.shape == temporal.shape):
        raise ValueError('S05/S06 channel masks do not align')
    if np.any(direction & ~np.asarray(direction_active, bool)) or np.any(temporal & ~np.asarray(time_active, bool)):
        raise ValueError('Significant channels must be active in their own source test')
    categories = np.full(len(active), 'active_not_significant', dtype=object)
    categories[~active] = 'inactive'
    categories[direction & ~temporal] = 'direction_only'
    categories[temporal & ~direction] = 'time_only'
    categories[direction & temporal] = 'both'
    return dict(active=active, direction=direction, temporal=temporal,
                both=direction & temporal, categories=categories)


def collect(ctx):
    sources = ctx.params['_source_sections']
    p5, p6 = sources['s05']['params'], sources['s06']['params']
    direction_ctx = replace(ctx, params=p5)
    if ctx.cohort.n < 16:
        raise ValueError('S05 方向检验需要至少 16 个保留 trial')
    stats, x, dirs = channel_anova(direction_ctx)
    if len(x) < 16 or len(np.unique(dirs)) < 2:
        raise ValueError('S05 方向检验的有限 trial 或方向数不足')
    direction_active = x.std(0) > 1e-12
    direction_q = A.bh_fdr(stats['p'], np.flatnonzero(direction_active))
    direction_sig = direction_active & (direction_q < p5['alpha'])
    time = temporal_statistics(replace(ctx, params=p6))
    groups = classify(direction_active, time['active'], direction_sig, time['sig'])
    rows = [dict(stored_column=j + 1, electrode=int(e), active=bool(groups['active'][j]),
                 direction_test_active=bool(direction_active[j]), time_test_active=bool(time['active'][j]),
                 direction_q=float(direction_q[j]), time_q=float(time['q'][j]),
                 direction_tuned=bool(direction_sig[j]), time_tuned=bool(time['sig'][j]),
                 both_tuned=bool(groups['both'][j]), category=groups['categories'][j])
            for j, e in enumerate(ctx.session.electrode_ids)]
    return dict(**groups, rows=rows, direction_trials=len(x), time_trials=len(time['anchor']),
                time_baseline_trials=int(time['valid'].sum()), time_notes=list(time['notes']), sources=sources)


def overlap_rows(groups):
    total, active = len(groups['active']), int(groups['active'].sum())
    counts = [('transmitted', total), ('active', active), ('direction_tuned', int(groups['direction'].sum())),
              ('time_tuned', int(groups['temporal'].sum())), ('both_tuned', int(groups['both'].sum()))]
    return [dict(category=name, channels=count, total_transmitted=total, total_active=active,
                 pct_transmitted=100 * count / total if total else None,
                 pct_active=100 * count / active if active and name != 'transmitted' else None)
            for name, count in counts]


def draw_overlap(ctx, groups, rows):
    fig = new_figure(13, 4.8)
    axes = fig.subplots(1, 3)
    names = ['All\ntransmitted', 'Active', 'Direction-\ntuned', 'Time-\ntuned', 'Both']
    colours = ['#cbd4db', '#2471a3', '#c0392b', '#d77b35', '#806ba5']
    for ax, selected, labels, field, ylabel in (
            (axes[0], rows, names, 'channels', 'Channels'),
            (axes[1], rows[2:], names[2:], 'pct_transmitted', '% of all transmitted channels'),
            (axes[2], rows[2:], names[2:], 'pct_active', '% of active channels (S05 ∪ S06)')):
        values = [r[field] if r[field] is not None else 0 for r in selected]
        bars = ax.bar(np.arange(len(selected)), values, color=colours if len(selected) == 5 else colours[2:])
        ax.set_xticks(np.arange(len(selected)), labels)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values + [1]) * 1.23)
        for bar, row in zip(bars, selected):
            value = row[field]
            label = 'N/A' if value is None else (str(int(value)) if field == 'channels' else f'{value:.1f}%')
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha='center', va='bottom', fontsize=8)
    for ax, title in zip(axes, ('A  Channel counts', 'B  Fraction of all channels', 'C  Fraction of active channels')):
        ax.set_title(title)
    partitions = {name: int((groups['categories'] == name).sum()) for name in PARTITIONS}
    p5, p6 = groups['sources']['s05']['params'], groups['sources']['s06']['params']
    fig.suptitle(f'{ctx.session.label} · direction / time modulation overlap\n'
                 f'S05 q<{p5["alpha"]:g}; S06 q<{p6["alpha"]:g}, {p6["align"]}, '
                 f'baseline [{p6["baseline_start"]:g}, {p6["baseline_end"]:g}) s, '
                 f'bin {p6["bin"] * 1000:g} ms', fontsize=9)
    fig.supxlabel(f'Direction and time bars both include Both; do not add these bars. '
                  f'Direction only {partitions["direction_only"]}; time only {partitions["time_only"]}; '
                  f'both {partitions["both"]}; active not significant {partitions["active_not_significant"]}.\n'
                  f'S05: {groups["direction_trials"]} trials; S06: {groups["time_baseline_trials"]} '
                  'baseline-eligible trials. Not significant does not establish absence of modulation.', fontsize=7)
    return fig


def tier_rows(groups, tiers):
    rows = []
    for tier in TIERS:
        selected = tiers == tier
        total = int(selected.sum())
        for name in PARTITIONS:
            count = int(((groups['categories'] == name) & selected).sum())
            rows.append(dict(tier=tier, category=name, channels=count, tier_channels=total,
                             percent=100 * count / total if total else None))
    return rows


def draw_composition(left, rows, tiers):
    x = np.arange(len(tiers))
    totals = np.array([sum(row['channels'] for row in rows if row['tier'] == tier) for tier in tiers])
    bottom = np.zeros(len(tiers))
    # Draw responsive groups at the bottom, inactive last (matching the reference convention).
    for name, label, colour in zip(reversed(PARTITIONS), reversed(LABELS), reversed(COLORS)):
        counts = np.array([next(r['channels'] for r in rows if r['tier'] == tier and r['category'] == name)
                           for tier in tiers])
        left.bar(x, counts, bottom=bottom, color=colour, edgecolor='white', lw=.5, label=label)
        for i, count in enumerate(counts):
            if count:
                # Small slices are labelled in a callout, not squeezed into their segment.
                if count < max(float(totals.max()) * .035, 4):
                    left.annotate(str(count), (x[i] + .4, bottom[i] + count / 2), xytext=(8, 0),
                                  textcoords='offset points', fontsize=6, va='center')
                else:
                    left.text(x[i], bottom[i] + count / 2, str(count), ha='center', va='center',
                              color=style.INK if name == 'inactive' else 'white', fontsize=8)
        bottom += counts
    left.set_xticks(x, [tier.title() for tier in tiers])
    left.set(ylabel='Transmitted channels', ylim=(0, max(float(totals.max()), 1) * 1.35),
             title='A  Mutually exclusive channel composition')
    for i, count in enumerate(totals):
        left.text(i, count + max(float(totals.max()) * .015, .5), str(count), ha='center', fontweight='bold')
    left.legend(frameon=False, fontsize=7, ncol=2, loc='upper right')
    return totals


def draw_tiers(ctx, groups, tiers, rows):
    fig = new_figure(12, 5.8)
    left, right = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1.1, 1]})
    totals = draw_composition(left, rows, TIERS)
    x = np.arange(len(TIERS))
    masks = (~groups['active'], groups['active'], groups['direction'], groups['temporal'], groups['both'])
    values = np.full((5, len(TIERS)), np.nan)
    counts = np.zeros((5, len(TIERS)), int)
    for j, tier in enumerate(TIERS):
        for i, mask in enumerate(masks):
            counts[i, j] = int((mask & (tiers == tier)).sum())
            if totals[j]:
                values[i, j] = 100 * counts[i, j] / totals[j]
    cmap = colormaps['Blues'].with_extremes(bad='#eeeeee')
    image = right.imshow(np.ma.masked_invalid(values), vmin=0, vmax=100, cmap=cmap, aspect='auto')
    right.set_xticks(x, [tier.title() for tier in TIERS])
    right.set_yticks(np.arange(5), ['Inactive', 'Active', 'Direction-tuned', 'Time-tuned', 'Both-tuned'])
    for (i, j), value in np.ndenumerate(values):
        text = 'N/A\n(n=0)' if not np.isfinite(value) else f'{counts[i, j]}\n{value:.1f}%'
        right.text(j, i, text, ha='center', va='center', fontsize=7,
                   color='white' if np.isfinite(value) and value > 60 else style.INK)
    right.set_title('B  Incidence within each config tier (rows overlap)')
    fig.colorbar(image, ax=right, label='% of transmitted channels in tier')
    fig.suptitle(f'{ctx.session.label} · config tiers × modulation', fontsize=10)
    fig.supxlabel('Left: five disjoint groups sum to each tier total. Right: Active includes tuned groups; '
                  'Direction and Time include Both.\nOnly transmitted electrodes are counted; '
                  'Unlisted is separate from Bad; config indices converted from 0-based to 1-based.', fontsize=7)
    incidence = [dict(tier=tier, group=label, channels=int(counts[i, j]), tier_channels=int(totals[j]),
                      percent=float(values[i, j]) if np.isfinite(values[i, j]) else None)
                 for i, label in enumerate(('inactive', 'active', 'direction_tuned', 'time_tuned', 'both_tuned'))
                 for j, tier in enumerate(TIERS)]
    return fig, incidence


def append_overlap(ctx, result):
    title = 'Direction and time tuning overlap'
    try:
        groups = collect(ctx)
    except ValueError as exc:
        fig = new_figure(9, 3)
        style.text_panel(fig.subplots(), ['Direction / time overlap unavailable', str(exc),
                                         'Uncomputed tests are not classified as inactive or untuned.'])
        result.add_figure(title, fig, method='Source S05/S06 test unavailable; reason displayed; no zero counts imputed.')
        result.summary['tuning_overlap_status'] = str(exc)
        result.notes.append('S05/S06 重叠汇总未计算：' + str(exc))
        return
    rows = overlap_rows(groups)
    sources = groups['sources']
    settings = json.dumps(sources, sort_keys=True, ensure_ascii=False)
    method = ('Direction-tuned = ordinary S05 one-way direction ANOVA, BH within its nonconstant-window-rate '
              'active family, q<source S05 alpha (the additional gain-controlled result is not substituted). '
              'Time-tuned = EXACT shared S06 unsmoothed paired-bin-vs-baseline t-test, min-bin Bonferroni then '
              'BH over its positive-mean binned-rate family, q<source S06 alpha; temporal modulation is not '
              'direction tuning. Both = physical-electrode intersection. Active denominator = UNION of S05 '
              'and S06 active definitions; this is distinct from S07 baseline-change active. Each source keeps '
              'its own valid-trial set, windows, alignment and multiple-testing family; they are reported, not '
              'silently forced to the same trial subset. Bars for Direction/Time INCLUDE Both and must not be '
              'summed. Zero denominators produce N/A. Not significant is not evidence of no modulation. '
              f'Source settings: {settings}; S05 valid trials={groups["direction_trials"]}; '
              f'S06 aligned trials={groups["time_trials"]}, baseline-eligible={groups["time_baseline_trials"]}. ')
    result.add_figure(title, draw_overlap(ctx, groups, rows), method=method,
                      data={'counts_and_fractions': rows, 'channels': groups['rows']})
    result.tables['tuning_overlap_counts'] = rows
    result.tables['tuning_overlap_channels'] = groups['rows']
    result.tables['tuning_source_settings'] = [dict(section=sid, version=info['version'],
                                                   parameter=name, value=value)
                                              for sid, info in sources.items() for name, value in info['params'].items()]
    result.notes.extend('S06 重叠汇总：' + note for note in groups['time_notes'])
    result.notes.append('重叠图 Active 为 S05/S06 活跃通道并集；S07 原基线变化面板使用自身活跃定义，'
                        '两者分母可能不同。Direction/Time 都包含 Both，不能相加。')
    result.summary.update(tuning_overlap_status='computed', tuning_transmitted_channels=len(groups['active']),
                          tuning_active_channels=int(groups['active'].sum()),
                          tuning_direction_channels=int(groups['direction'].sum()),
                          tuning_time_channels=int(groups['temporal'].sum()),
                          tuning_both_channels=int(groups['both'].sum()), tuning_source_settings=sources)
    if not ctx.channel_config_path:
        result.summary['tuning_config_status'] = 'no_configs_ini'
        result.notes.append('当天无通道配置 INI / LOG，跳过配置层级的调谐组成图。')
        return groups
    try:
        config = read_channel_config(ctx.channel_config_path)
        membership = [set(config.get(tier, [])) for tier in TIERS[:3]]
        if any(membership[i] & membership[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError('good / uncertain / bad 列表存在重叠电极，不能唯一归类')
        tiers = column_tiers(ctx.session.electrode_ids, config)
    except (OSError, ValueError) as exc:
        result.summary['tuning_config_status'] = str(exc)
        result.notes.append('配置层级汇总未计算：' + str(exc))
        return groups
    config_rows = tier_rows(groups, tiers)
    fig, incidence = draw_tiers(ctx, groups, tiers, config_rows)
    result.add_figure('Tuning composition by config tier', fig,
                      method=method + ' Config groups use the recording day file: ' + str(ctx.channel_config_path) +
                      '. Only transmitted electrode IDs are assigned; good/uncertain/bad are converted from '
                      'zero-based config indices. Unlisted is not merged into Bad. Left stacks are disjoint '
                      'Inactive / Active-not-significant / Direction-only / Time-only / Both; right matrix '
                      'reports overlapping incidences with each tier total as denominator.',
                      data={'disjoint_composition': config_rows, 'overlapping_incidence': incidence,
                            'channels': [dict(row, config_tier=str(tier)) for row, tier in zip(groups['rows'], tiers)]})
    result.tables['tuning_config_composition'] = config_rows
    result.tables['tuning_config_incidence'] = incidence
    result.summary['tuning_config_status'] = 'computed'
    return groups
