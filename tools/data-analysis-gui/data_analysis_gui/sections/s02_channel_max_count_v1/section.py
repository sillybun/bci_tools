"""S02 v1 — electrode map of the maximum encoded count, code distribution, saturation.

Counts are 2-bit codes: 0, 1, 2 and 3 where 3 means ">= 3", so a per-channel rate
is censored at 3 codes per frame. Every transmitted column is placed on the array
(physical workbook layout when available, plus the 128×8 electrode-id grid);
untransmitted electrodes stay grey (NaN), never zero.

Scope: the whole recording (raw frames, no trial selection, as in
data_qc_v1/channel_maximum_heatmaps.py) or only the frames inside the included
trials [T_TRIAL_START, T_TRIAL_END] of the topbar filter.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from ... import config as C
from ...core import style
from ...core.channels import TIERS, column_tiers, draw_channel_map, id_grid, layout_grid, read_channel_config
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure

SCOPES = ('whole_recording', 'included_trials')
CODES = np.arange(C.MAX_CODE + 1)


def frame_mask(cohort, scope):
    """Boolean per frame: frames inside included trials, or all frames."""
    nt = cohort.session.neural_times
    if scope == 'whole_recording':
        return np.ones(len(nt), bool)
    mask = np.zeros(len(nt), bool)
    if cohort.n:
        lo = np.searchsorted(nt, cohort.starts, side='left')
        hi = np.searchsorted(nt, cohort.ends, side='right')
        delta = np.zeros(len(nt) + 1, int)
        np.add.at(delta, lo, 1)
        np.add.at(delta, hi, -1)
        mask = np.cumsum(delta[:-1]) > 0
    return mask


def code_stats(session, mask):
    """Per-column code histogram (frames × code), max code, saturation and mean Hz."""
    counts = session.counts if mask.all() else session.counts[mask]
    n = len(counts)
    hist = np.zeros((counts.shape[1], len(CODES)), np.int64)
    if n:
        clipped = np.minimum(counts, C.MAX_CODE)
        for k in CODES:
            hist[:, k] = np.count_nonzero(clipped == k, axis=0)
        maximum = counts.max(0).astype(int)
        mean_hz = counts.mean(0, dtype=np.float64) / session.frame_seconds
    else:
        maximum = np.full(counts.shape[1], -1)
        mean_hz = np.full(counts.shape[1], np.nan)
    frac = hist / max(n, 1)
    return dict(n_frames=n, hist=hist, frac=frac, max=maximum, sat=frac[:, C.MAX_CODE], mean_hz=mean_hz)


# Zones describe how strongly a channel's counts are clipped by the 2-bit ceiling (share of frames at code 3),
# not whether the channel is abnormal: a high-rate channel that fires like a Poisson process is clipped but healthy.
# Abnormal channels are the silent ones (no / almost no events) and the bursty flag (far above Poisson).
ZONES = ('rarely saturated', 'sometimes saturated', 'often saturated', 'silent')
ZONE_COLORS = {'rarely saturated': '#2e9d5b', 'sometimes saturated': '#e6a700', 'often saturated': '#d1342f',
               'silent': '#8c8c8c'}
BURSTY_COLOR = '#ff00e6'        # vivid magenta: distinct from the zone colours and black max-0
# Display categories of the zone figure: silent is split into "max code 0" (no event at all in scope,
# i.e. dead / disconnected) and the remaining silent channels (some events, rate < rate_min).
DEAD = 'silent · max 0'
DISPLAY = ('rarely saturated', 'sometimes saturated', 'often saturated', 'silent', DEAD)
DISPLAY_COLORS = {**ZONE_COLORS, DEAD: style.MAX_CODE_COLORS[0]}


def poisson_reference(frame_seconds, n=400):
    """Poisson spiking seen through the 2-bit code: (censored mean Hz, % frames at code 3) per rate λ.

    For X ~ Poisson(λ) per frame the stored code is min(X, 3), so the observed mean is
    E[min(X, 3)] = 3 - 3·p0 - 2·p1 - p2 and the saturation is P(X >= 3) = 1 - p0 - p1 - p2.
    Both are computed in the observed (censored) coordinates, so they line up with the scatter.
    """
    lam = np.logspace(-4, np.log10(60), n)
    p0 = np.exp(-lam)
    p1, p2 = lam * p0, lam ** 2 / 2 * p0
    observed = (C.MAX_CODE - 3 * p0 - 2 * p1 - p2) / frame_seconds
    return observed, 100 * (1 - p0 - p1 - p2)


def saturation_zones(stats, frame_seconds, sat_ok, sat_bad, rate_min, burst_ratio):
    """Saturation zone per channel + Poisson-expected saturation + bursty flag."""
    pct, hz = 100 * stats['sat'], stats['mean_hz']
    zone = np.where(pct > sat_bad, 'often saturated',
                    np.where(pct > sat_ok, 'sometimes saturated', 'rarely saturated')).astype(object)
    zone[~(hz >= rate_min) | (stats['max'] == 0)] = 'silent'      # NaN (no frames); max 0 even if rate_min = 0
    ref_x, ref_y = poisson_reference(frame_seconds)
    expected = np.interp(np.nan_to_num(hz), ref_x, ref_y)
    floor = 100 / max(stats['n_frames'], 1)               # one frame at code 3
    bursty = (zone != 'silent') & (pct > sat_ok) & (pct > burst_ratio * np.maximum(expected, floor))
    return zone, expected, bursty


def config_tiers(ctx, ids):
    """(tier per column from the day's configs.ini or None, note). Optional: S02 runs without it."""
    if not ctx.channel_config_path:
        return None, ''
    try:
        return column_tiers(ids, read_channel_config(ctx.channel_config_path)), ''
    except (OSError, ValueError) as exc:
        return None, f'通道配置 INI / LOG 无法读取，saturation zone 未按 good/uncertain/bad 分组：{exc}'


def zone_groups(display, tiers):
    """Bars of the zone breakdown: (name, column mask). All transmitted, then one per configs.ini tier."""
    groups = [('all transmitted', np.ones(len(display), bool))]
    if tiers is not None:
        groups += [(t, tiers == t) for t in TIERS]
        if np.any(tiers == 'unlisted'):
            groups.append(('unlisted', tiers == 'unlisted'))
    return groups


def zone_breakdown(display, bursty, tiers):
    """One row per bar: channels per display category, its share of the bar, bursty count."""
    rows = []
    for name, m in zone_groups(display, tiers):
        n = int(m.sum())
        row = dict(group=name, channels=n, bursty=int(np.sum(m & bursty)))
        for z in DISPLAY:
            k = int(np.sum(m & (display == z)))
            row[z] = k
            row[f'{z} pct'] = round(100 * k / n, 3) if n else 0.0
        silent = row['silent'] + row[DEAD]
        row.update(silent_total=silent, max0_pct_of_silent=round(100 * row[DEAD] / silent, 3) if silent else 0.0)
        rows.append(row)
    return rows


ZERO_COLOR = '#ffffff'          # default colour of 0 on a log colour scale (below the floor)


def map_norm(values, scale, floor, cmap, zero=ZERO_COLOR, zero_name='white'):
    """Colour norm + cmap for an electrode map. Returns (norm, cmap, extend, label suffix).

    linear: 0..99.5th percentile, larger values drawn at the top colour.
    log:    floor..max on a log scale, floor = max(resolution floor, 1st percentile of the positive values);
            values below the floor (including 0) use the `zero` colour, shown as the colourbar's lower extension.
    """
    from matplotlib import colormaps
    from matplotlib.colors import LogNorm, Normalize
    cmap = colormaps[cmap] if isinstance(cmap, str) else cmap
    v = values[np.isfinite(values)]
    if scale == 'log':
        positive = v[v > 0]
        lo = max(floor, float(np.percentile(positive, 1))) if len(positive) else floor
        hi = max(float(positive.max()) if len(positive) else lo * 10, lo * 10)
        return (LogNorm(lo, hi), cmap.with_extremes(under=zero), 'min',
                f'log scale; {zero_name} = 0 or < {lo:.3g}')
    hi = max(float(np.percentile(v, 99.5)) if len(v) else 1.0, 1e-3)
    return Normalize(0, hi), cmap, 'neither', 'clipped at p99.5'


def max_code_cmap():
    cmap = ListedColormap(style.MAX_CODE_COLORS)
    return cmap, BoundaryNorm(np.arange(-0.5, len(CODES) + 0.5), cmap.N)


class ChannelMaxCount(Section):
    id = 's02'
    title = '通道稳定性分析'
    description = ('每个传输通道的最大编码、编码分布、饱和度（编码 3 占比）和平均发放率，按截断程度分区，并在阵列上定位。')
    params = (
        Param('scope', '统计范围', 'choice', 'whole_recording', choices=SCOPES,
              help='whole_recording = 全部帧；included_trials = 仅筛选保留 trial 的 [start, end] 内的帧'),
        Param('top_n', '饱和度 Top-N 表行数', 'int', 15, minimum=1, maximum=100),
        Param('zoom_n', '编码组成放大图通道数', 'int', 60, minimum=10, maximum=200,
              help='Maximum code & code distribution 图中「code 2 / 3 放大」面板显示 code 3 占比最高的前 N 个通道'),
        Param('map_scale', '电极映射图色标', 'choice', 'log', choices=('log', 'linear'),
              help='Saturation 与 Mean rate 电极映射图的颜色刻度：log 让低值区也有层次（0 显示为白色）；linear = 0..p99.5'),
        Param('sat_ok', '轻度截断饱和度上限 (%)', 'float', 1.0, minimum=0.0, maximum=100.0,
              help='饱和度 ≤ 该值且有信号 → rarely saturated（轻度截断）；也是 bursty 判定的最低饱和度'),
        Param('sat_bad', '重度截断饱和度下限 (%)', 'float', 5.0, minimum=0.0, maximum=100.0,
              help='饱和度 > 该值 → often saturated（重度截断）；介于两者之间 → sometimes saturated。mean rate 图中也按此加框'),
        Param('rate_min', '无信号阈值 (Hz)', 'float', 0.5, minimum=0.0, maximum=150.0,
              help='mean rate 低于该值 → 无信号 silent（不论饱和度）'),
        Param('burst_ratio', '爆发判定倍数', 'float', 3.0, minimum=1.0, maximum=100.0,
              help='饱和度 > sat_ok 且 > Poisson 预期饱和度的该倍数 → 标记为 bursty'),
    )

    def run(self, ctx: AnalysisContext, *, overview_target=None) -> SectionResult:
        session, cohort, p = ctx.session, ctx.cohort, ctx.params
        result = SectionResult()
        ctx.progress('code histogram')
        stats = ctx.cached(('s02_code_stats', id(session), p['scope'], id(cohort) if p['scope'] != SCOPES[0] else 0),
                           lambda: code_stats(session, frame_mask(cohort, p['scope'])))
        if stats['n_frames'] == 0:
            result.notes.append('所选范围内没有神经帧（included_trials 为空？），图中所有通道为空。')
        if session.counts.max() > C.MAX_CODE:
            result.notes.append(f'计数最大值 {int(session.counts.max())} 超过 2-bit 上限 {C.MAX_CODE}；'
                                f'≥{C.MAX_CODE} 的帧都计入编码 {C.MAX_CODE}。')
        ids = session.electrode_ids
        scope_text = ('whole recording' if p['scope'] == 'whole_recording'
                      else f'frames inside {cohort.n} included trials')
        sat = stats['sat']
        if p['sat_bad'] < p['sat_ok']:
            result.notes.append(f'sat_bad ({p["sat_bad"]:g}%) < sat_ok ({p["sat_ok"]:g}%)，已按 sat_bad = sat_ok 处理。')
        sat_bad = max(p['sat_bad'], p['sat_ok'])
        zone, expected, bursty = saturation_zones(stats, session.frame_seconds, p['sat_ok'], sat_bad, p['rate_min'],
                                             p['burst_ratio'])
        n_by_max = {int(k): int(np.sum(stats['max'] == k)) for k in CODES}
        order = np.argsort(-sat, kind='stable')
        rows = []
        for j in range(len(ids)):
            e = int(ids[j])
            pos = {}
            if ctx.layout is not None:
                r, c = ctx.layout['position'][e - 1]
                pos = dict(layout_row=int(r) + 1, layout_column=chr(65 + int(c)))
            rows.append(dict(column=j + 1, electrode_id=e, idgrid_column=(e - 1) // 128 + 1,
                             idgrid_row=(e - 1) % 128 + 1, **pos, max_code=int(stats['max'][j]),
                             **{f'n_code{k}': int(stats['hist'][j, k]) for k in CODES},
                             **{f'frac_code{k}': round(float(stats['frac'][j, k]), 6) for k in CODES},
                             saturation_frac=round(float(sat[j]), 6),
                             mean_hz=round(float(stats['mean_hz'][j]), 4),
                             zone=str(zone[j]), bursty=bool(bursty[j]),
                             poisson_expected_saturation_pct=round(float(expected[j]), 5)))
        scope_method = (f'Scope: {"all frames of the recording (no trial selection)" if p["scope"] == "whole_recording" else "only frames whose receive time lies inside [T_TRIAL_START, T_TRIAL_END] of a trial kept by the topbar filter"}; '
                        f'n_frames = {stats["n_frames"]}. Counts are 2-bit codes 0..3 per 20 ms frame, 3 meaning >=3 '
                        '(values above 3, if any, are counted as code 3). ')
        map_method = ('Electrode placement: physical workbook layout (position 1 at bottom, 128 at top; columns left to right) when the layout '
                      'xlsx is available, and the default layout = 128x8 electrode-id grid (column-major, electrode e at column '
                      '(e-1)//128+1, row (e-1)%128+1). Grey = electrode not among the 512 transmitted columns. ')
        ctx.progress('maps')

        def scale_method(quantity, resolution, zero='white'):
            if p['map_scale'] == 'log':
                return (f'Map colour: {quantity} on a log scale from max(resolution of {resolution} in scope, 1st '
                        'percentile of the non-zero values) to the maximum; values below that floor, including 0, '
                        f'are {zero}. ')
            return f'Map colour: {quantity} on a linear scale 0..99.5th percentile, larger values at the top colour. '
        frac = stats['frac']
        by_total = np.lexsort((frac[:, 3], frac[:, 2], 1 - frac[:, 0]))[::-1]     # non-zero share, then code 2, 3
        sorted_rows = [dict(rank=i + 1, column=rows[j]['column'], electrode_id=rows[j]['electrode_id'],
                            max_code=rows[j]['max_code'], **{f'frac_code{k}': rows[j][f'frac_code{k}'] for k in CODES})
                       for i, j in enumerate(by_total)]
        by_sat = np.lexsort((frac[:, 2], frac[:, 3]))[::-1][:min(p['zoom_n'], len(ids))]
        zoom_rows = [dict(rank=i + 1, column=rows[j]['column'], electrode_id=rows[j]['electrode_id'],
                          frac_code2=rows[j]['frac_code2'], frac_code3=rows[j]['frac_code3'])
                     for i, j in enumerate(by_sat)]
        result.add_figure(
            'Maximum code and code distribution', self.code_overview(ctx, stats, scope_text, by_total, by_sat),
            method=scope_method + map_method + 'Maps: colour = per-electrode maximum code over the frames in scope '
                   '(0 black, 1 yellow, 2 orange, 3 red); legend counts = transmitted electrodes per maximum. '
                   'Top left: number of channels by maximum code. Top right: for each channel with at least one event, '
                   'the fraction of frames at code 3, 2 and 1 stacked in that order from the baseline (remainder = '
                   'code 0), channels sorted by the non-zero fraction (1 - fraction at code 0), descending; channels '
                   'without any event (max code 0) are omitted and counted in the panel. Middle (zoom): the '
                   f'{len(by_sat)} channels with the highest fraction at code 3 (ties by code 2), fraction at code 3 '
                   'and code 2 only, stacked, x labels = electrode ids. Bottom: histograms across the transmitted '
                   'channels of the per-channel fraction of frames at code k (k = 0..3), 40 equal bins from 0 to the '
                   'maximum, log y-axis.',
            data={'channels': [{k: r[k] for k in r if not k.startswith(('n_code', 'frac_code', 'saturation', 'mean',
                                                                         'zone', 'bursty', 'poisson'))}
                               for r in rows],
                  'channels_by_max_code': [dict(max_code=int(k), channels=n_by_max[int(k)]) for k in CODES],
                  'code_fractions_sorted': sorted_rows, 'zoom_code2_code3': zoom_rows})
        result.add_figure(
            'Saturation per electrode', self.saturation(ctx, stats, scope_text, p['top_n'], (p['sat_ok'], sat_bad), p['map_scale']),
            method=scope_method + map_method + 'Saturation = fraction of frames in scope at code 3 (the 2-bit '
                   'ceiling), per electrode, in %. ' + scale_method('saturation (%)', '1 frame', 'light blue') + 'Top right: '
                   'saturation sorted descending, log y, channels with 0 omitted; dashed/dotted lines at sat_ok and sat_bad with '
                   'the number of channels above each. Middle: saturation vs mean rate, mean rate = mean count / '
                   f'{session.frame_seconds * 1000:.0f} ms configured window (censored by the ceiling), ceiling line '
                   f'at {C.MAX_CODE}/{session.frame_seconds:g} s; colour = max code. Bottom: most saturated electrodes.',
            data={'channels': [dict(column=r['column'], electrode_id=r['electrode_id'], max_code=r['max_code'],
                                    saturation_pct=round(100 * r['saturation_frac'], 5), mean_hz=r['mean_hz'])
                               for r in rows],
                  'top_saturated': [rows[j] for j in order[:p['top_n']]]})
        zone_counts = {z: int(np.sum(zone == z)) for z in ZONES}
        dead = stats['max'] == 0
        n_dead = int(dead.sum())
        dead_of_silent = 100 * n_dead / zone_counts['silent'] if zone_counts['silent'] else 0.0
        tiers, tier_note = config_tiers(ctx, ids)
        if tier_note:
            result.notes.append(tier_note)
        display = np.where(dead, DEAD, zone).astype(object)
        breakdown = zone_breakdown(display, bursty, tiers)
        ini = ctx.channel_config_path.replace(chr(92), '/').split('/')[-1] if tiers is not None else ''
        tier_method = ((f'Bottom bars: tiers from {ini} (good / uncertain / bad lists, 0-based ini indices converted to '
                        '1-based electrode ids; unlisted = transmitted electrode in no list, bar drawn only if any). ')
                       if tiers is not None else
                       'Bottom bar: no INI or Neuralinsight log for this recording day, so only the all-transmitted bar is drawn. ')
        result.add_figure(
            'Channel saturation zones', self.zone_figure(ctx, stats, scope_text, zone, bursty, p['sat_ok'], sat_bad,
                                                    p['rate_min'], p['burst_ratio'], tiers, figure=overview_target),
            method=scope_method + map_method + 'Each transmitted channel is placed by its mean rate (x, Hz = mean code / '
                   f'{session.frame_seconds * 1000:.0f} ms, censored) and saturation (y, % of frames at code 3). Zones = how '
                   'strongly the counts are clipped by the 2-bit ceiling (not a judgement of channel health): '
                   f'for fr ≥ {p["rate_min"]:g} Hz, sat ≤ {p["sat_ok"]:g}%, {p["sat_ok"]:g}% < sat ≤ {sat_bad:g}%, '
                   f'or sat > {sat_bad:g}%. Silent, fr = 0 Hz (black) has maximum code 0 in scope, '
                   'i.e. not a single event (dead / disconnected electrode or empty column); they are always silent and '
                   'are drawn as black x at (0, 0) in the scatter. The remaining silent channels (grey) had some '
                   'events. Both axes symlog (linear below 1 Hz / 0.01%, so zeros are drawn). '
                   'Black curve = Poisson reference: for spike count X ~ Poisson(λ) per frame the stored code is '
                   'min(X, 3), giving observed mean (3 - 3p0 - 2p1 - p2)/frame and saturation 1 - p0 - p1 - p2, '
                   f'drawn in the same censored coordinates; λ = expected events per frame = true rate x frame length. Bursty (magenta ring) = saturation > {p["sat_ok"]:g}% and > '
                   f'{p["burst_ratio"]:g} x the Poisson saturation expected at the channel\'s observed rate '
                   '(expectation floored at one frame), i.e. code-3 frames far more often than a regular spiker of '
                   'the same rate would produce; this, together with silent, is the abnormal-channel signal. Maps: zone per '
                   'electrode, magenta box = bursty. Bottom: stacked bars of '
                   'channels per zone in the shared legend order (bottom to top); labels = channels and % of '
                   'that bar; dashed outline = silent (both silent parts); '
                   'bold number = bar total; bursty count under each bar. ' + tier_method,
            data={'zone_counts': [dict(zone=z, channels=zone_counts[z]) for z in ZONES]
                                 + [dict(zone='silent · max code 0', channels=n_dead,
                                         pct_of_transmitted=round(100 * n_dead / max(len(ids), 1), 3),
                                         pct_of_silent=round(dead_of_silent, 3)),
                                    dict(zone='bursty (flag)', channels=int(bursty.sum()))],
                  'zones_by_tier': breakdown,
                  'channels': [dict(column=r['column'], electrode_id=r['electrode_id'], mean_hz=r['mean_hz'],
                                    saturation_pct=round(100 * r['saturation_frac'], 5), zone=r['zone'],
                                    max_code_0=r['max_code'] == 0, bursty=r['bursty'],
                                    **({'tier': str(tiers[r['column'] - 1])} if tiers is not None else {}),
                                    poisson_expected_saturation_pct=r['poisson_expected_saturation_pct'])
                               for r in rows]})
        result.add_figure(
            'Mean rate per electrode', self.rate_map(ctx, stats, scope_text, sat_bad, p['map_scale']),
            method=scope_method + map_method + f'Colour = per-electrode mean rate in Hz = mean code over the frames in '
                   f'scope / {session.frame_seconds * 1000:.0f} ms (codes 3 count as 3, so this is a lower bound of the '
                   f'true rate, at most {C.MAX_CODE / session.frame_seconds:.0f} Hz). ' + scale_method('mean rate (Hz)', '1 event') +
                   f'Boxed electrodes: saturation > {sat_bad:g}% (often saturated zone). Right: histogram '
                   'of the per-channel mean rate across transmitted channels, 40 equal bins, log y; dashed line = ceiling.',
            data={'channels': [dict(column=r['column'], electrode_id=r['electrode_id'], mean_hz=r['mean_hz'],
                                    saturation_pct=round(100 * r['saturation_frac'], 5),
                                    boxed=bool(100 * r['saturation_frac'] > sat_bad)) for r in rows]})

        result.summary.update(
            recording=session.label, scope=p['scope'], n_frames=stats['n_frames'],
            transmitted_electrodes=int(len(ids)), layout='workbook' if ctx.layout else 'electrode-id grid',
            channels_by_max_code=n_by_max,
            overall_code_fraction={int(k): float(stats['hist'][:, k].sum() / max(stats['hist'].sum(), 1))
                                   for k in CODES},
            saturation_mean=float(sat.mean()), saturation_max=float(sat.max()),
            channels_saturation_any=int((sat > 0).sum()),
            channels_saturation_gt_1pct=int((sat > 0.01).sum()),
            channels_saturation_gt_5pct=int((sat > 0.05).sum()),
            top_saturated_electrodes=[int(ids[j]) for j in order[:min(10, len(order))]],
            zone_thresholds=dict(sat_ok_pct=p['sat_ok'], sat_bad_pct=sat_bad, rate_min_hz=p['rate_min'],
                                 burst_ratio=p['burst_ratio']),
            channels_by_zone=zone_counts, channels_bursty=int(bursty.sum()),
            channels_max_code_0=n_dead, max_code_0_pct_of_silent=round(dead_of_silent, 3),
            max_code_0_electrodes=[int(ids[j]) for j in np.flatnonzero(dead)],
            channel_config=ctx.channel_config_path if tiers is not None else '',
            zones_by_tier={r['group']: {k: r[k] for k in ('channels', *DISPLAY, 'bursty')} for r in breakdown},
            often_saturated_electrodes=[int(ids[j]) for j in np.flatnonzero(zone == 'often saturated')],
            bursty_electrodes=[int(ids[j]) for j in np.flatnonzero(bursty)])
        result.tables['channel_codes'] = rows
        result.tables['top_saturated'] = [rows[j] for j in order[:p['top_n']]]
        result.tables['zones_by_tier'] = breakdown
        first = ('Channel saturation zones',)
        order_fig = sorted(range(len(result.figures)),
                           key=lambda i: (first.index(result.figures[i][0]) if result.figures[i][0] in first else 2, i))
        result.figures = [result.figures[i] for i in order_fig]
        result.methods = [result.methods[i] for i in order_fig]
        result.figure_data = [result.figure_data[i] for i in order_fig]
        return result

    # ------------------------------------------------------------------ figures
    # Electrode maps: every S02 map figure uses the same figure height, footer height and cell width, so a
    # square cell has the same size on all maps of all S02 figures. The legend or
    # colour bar sits in a footer under the maps.
    MAP_HEIGHT_IN = 10.0
    MAP_FOOTER_IN = 1.5             # fits a one-column legend of 7 entries + title
    MAP_MIN_BLOCK_IN = 2.6
    LEGEND_TWO_COL_IN = 4.4         # footer legends use two columns only if the map block is this wide

    @staticmethod
    def _map_geometry(ctx):
        """Layouts drawn, map slot widths, map-row height and total block width in inches."""
        cls = ChannelMaxCount
        layouts = ([ctx.layout] if ctx.layout is not None else []) + [None]
        shapes = [(lay['rows'], lay['columns']) if lay is not None else (C.N_ELECTRODES // 8, 8) for lay in layouts]
        map_h = cls.MAP_HEIGHT_IN - cls.MAP_FOOTER_IN - 0.9     # suptitle, map titles and x tick labels
        cell_h = map_h / max(r for r, _ in shapes)
        # + y tick labels / y label, plus slack so the map height (identical in every figure) is always the
        # binding constraint under constrained layout — the cell size then never depends on the right panels
        widths = [cell_h * c + 1.1 for _, c in shapes]
        return layouts, widths, map_h, max(sum(widths), cls.MAP_MIN_BLOCK_IN)

    @staticmethod
    def _map_block(fig, slot, geometry, footer_in):
        """Map axes (one per layout) above one footer axis spanning them, inside `slot`."""
        layouts, widths, map_h, block_w = geometry
        pad = (block_w - sum(widths)) / 2
        cols = ([pad] if pad > 0 else []) + widths + ([pad] if pad > 0 else [])
        sub = slot.subgridspec(2, len(cols), width_ratios=cols, height_ratios=[map_h, footer_in])
        first = 1 if pad > 0 else 0
        axes = [fig.add_subplot(sub[0, first + i]) for i in range(len(layouts))]
        footer = fig.add_subplot(sub[1, :])
        footer.axis('off')
        return axes, footer

    @staticmethod
    def _draw_maps(axes, geometry, ids, values, cmap, norm, boxes=None, box_color='black', box_lw=0.8):
        from matplotlib.patches import Rectangle
        image = None
        for ax, layout in zip(axes, geometry[0]):
            name = 'Physical layout' if layout is not None else 'Electrode ID layout'
            image = draw_channel_map(ax, ids, values, layout=layout, cmap=cmap, norm=norm, title=name,
                                     colorbar=False, missing=style.MISSING)
            if boxes is not None and boxes.any():
                mask = np.where(boxes, 1.0, np.nan)
                cells = layout_grid(layout, ids, mask) if layout is not None else id_grid(ids, mask)
                for r, c in zip(*np.nonzero(cells == 1)):
                    ax.add_patch(Rectangle((c - 0.5, r + 0.5), 1, 1, fill=False, ec=box_color, lw=box_lw, zorder=3))
        return image

    @staticmethod
    def _footer_legend(footer, geometry, handles, title):
        """Legend under the maps. Kept out of constrained layout: a legend wider than the map block would
        otherwise make the layout engine shrink the maps (and so change the cell size)."""
        ncol = 2 if geometry[3] >= ChannelMaxCount.LEGEND_TWO_COL_IN else 1
        legend = footer.legend(handles=handles, loc='upper center', frameon=False, fontsize=7, ncol=ncol,
                               title=title, title_fontsize=7.5, handlelength=1.4, columnspacing=1.0)
        legend.set_in_layout(False)
        return legend

    @staticmethod
    def _footer_colorbar(fig, footer, image, label, extend, note=''):
        cax = footer.inset_axes([0.06, 0.70, 0.88, 0.13])
        cb = fig.colorbar(image, cax=cax, orientation='horizontal', extend=extend)
        cb.ax.tick_params(labelsize=6)
        cb.set_label(label, fontsize=7)
        if note:
            footer.text(0.5, 0.0, note, transform=footer.transAxes, ha='center', va='bottom', fontsize=6.5,
                        color=style.INK2)

    def code_overview(self, ctx, stats, scope_text, by_total, by_sat):
        """Maximum-code maps (left) + code distribution panels (right)."""
        session = ctx.session
        ids, frac = session.electrode_ids, stats['frac']
        height, footer_in = self.MAP_HEIGHT_IN, self.MAP_FOOTER_IN
        geometry = self._map_geometry(ctx)
        right_w = 11.0
        fig = new_figure(geometry[3] + right_w, height)
        outer = fig.add_gridspec(1, 2, width_ratios=[geometry[3], right_w])
        axes, footer = self._map_block(fig, outer[0, 0], geometry, footer_in)

        # maps
        cmap, norm = max_code_cmap()
        values = np.where(stats['max'] >= 0, np.minimum(stats['max'], C.MAX_CODE), np.nan).astype(float)
        self._draw_maps(axes, geometry, ids, values, cmap, norm)
        counts = [int(np.sum(stats['max'] == k)) for k in CODES]
        handles = [Patch(color=style.MAX_CODE_COLORS[k], label=f'max code {k}' + (' (≥3)' if k == C.MAX_CODE else '')
                         + f': {counts[k]}') for k in CODES]
        handles.append(Patch(color=style.MISSING, label=f'not transmitted: {C.N_ELECTRODES - len(ids)}'))
        self._footer_legend(footer, geometry, handles, 'Per-electrode maximum')
        footer.text(0.5, 0.0, f'{len(ids)} / {C.N_ELECTRODES} electrodes transmitted', transform=footer.transAxes,
                    ha='center', va='bottom', fontsize=6.5, color=style.INK2).set_in_layout(False)

        grid = outer[0, 1].subgridspec(3, 4, height_ratios=[1.15, 1.0, 0.8])
        # channels by maximum code
        ax = fig.add_subplot(grid[0, 0])
        ax.bar(CODES, counts, color=style.MAX_CODE_COLORS, edgecolor=style.INK, lw=0.4)
        for k, c in zip(CODES, counts):
            ax.text(k, c, f'{c}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(CODES)
        ax.set_xlabel('maximum code')
        ax.set_ylabel('channels')
        ax.set_ylim(0, max(max(counts), 1) * 1.15)
        ax.set_title(f'Channels by maximum code (n={len(ids)})', fontsize=8)

        # composition of every channel with events, rarest code on the baseline
        ax = fig.add_subplot(grid[0, 1:])
        active = by_total[(1 - frac[by_total, 0]) > 0]
        x = np.arange(len(active))
        bottom = np.zeros(len(active))
        for k in (3, 2, 1):
            ax.bar(x, frac[active, k], bottom=bottom, width=1.0, color=style.MAX_CODE_COLORS[k], lw=0,
                   label=f'code {k}' + (' (≥3)' if k == C.MAX_CODE else ''))
            bottom += frac[active, k]
        ax.set_xlim(-0.5, max(len(active), 1) - 0.5)
        ax.set_ylim(0, max(float(bottom.max(initial=0)), 1e-3) * 1.05)
        ax.set_xlabel('channel rank (sorted by non-zero fraction = 1 − fraction at code 0)')
        ax.set_ylabel('fraction of frames')
        ax.set_title(f'Non-zero code composition · {len(active)} channels with events '
                     f'({len(ids) - len(active)} with max code 0 omitted)', fontsize=8)
        ax.legend(frameon=False, loc='upper right', fontsize=7)

        # zoom: code 2 and 3 of the most saturated channels
        ax = fig.add_subplot(grid[1, :])
        x = np.arange(len(by_sat))
        ax.bar(x, frac[by_sat, 3], width=0.8, color=style.MAX_CODE_COLORS[3], lw=0, label='code 3 (≥3)')
        ax.bar(x, frac[by_sat, 2], width=0.8, bottom=frac[by_sat, 3], color=style.MAX_CODE_COLORS[2], lw=0,
               label='code 2')
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(ids[j])) for j in by_sat], rotation=90, fontsize=5.5)
        ax.set_xlim(-0.6, max(len(by_sat), 1) - 0.4)
        ax.set_xlabel('electrode id (sorted by fraction at code 3)')
        ax.set_ylabel('fraction of frames')
        ax.set_title(f'Zoom · code 2 and 3 of the {len(by_sat)} most saturated channels', fontsize=8)
        ax.legend(frameon=False, loc='upper right', fontsize=7)

        # per-code histograms across channels
        for k in CODES:
            ax = fig.add_subplot(grid[2, k])
            values = frac[:, k]
            hi = max(float(values.max(initial=0)), 1e-4)
            ax.hist(values, bins=np.linspace(0, hi * 1.0001, 41), color=style.MAX_CODE_COLORS[k],
                    edgecolor=style.INK if k == 0 else 'none', lw=0.3)
            ax.set_yscale('log')
            ax.set_xlabel(f'fraction of frames at code {k}', fontsize=7)
            ax.set_ylabel('channels' if k == 0 else '')
            ax.set_title(f'code {k}: median {np.median(values):.3g}, max {values.max(initial=0):.3g}', fontsize=7.5)
        fig.suptitle(f'{session.label}\nmaximum code and code distribution · {stats["n_frames"]:,} frames '
                     f'({scope_text})', fontsize=8.5)
        return fig

    def saturation(self, ctx, stats, scope_text, top_n, thresholds=(1.0, 5.0), scale='log'):
        session = ctx.session
        ids, sat, mean_hz = session.electrode_ids, stats['sat'], stats['mean_hz']
        height, footer_in = self.MAP_HEIGHT_IN, self.MAP_FOOTER_IN
        geometry = self._map_geometry(ctx)
        fig = new_figure(geometry[3] + 9.0, height)
        outer = fig.add_gridspec(1, 2, width_ratios=[geometry[3], 9.0])
        axes, footer = self._map_block(fig, outer[0, 0], geometry, footer_in)
        pct = 100 * sat
        norm, cmap, extend, suffix = map_norm(pct, scale, 100 / max(stats['n_frames'], 1), 'YlOrRd',
                                              zero='#9ecae1', zero_name='blue')
        image = self._draw_maps(axes, geometry, ids, pct, cmap, norm)
        self._footer_colorbar(fig, footer, image, f'saturation (%)\n{suffix}', extend)
        grid = outer[0, 1].subgridspec(3, 1)
        ax = fig.add_subplot(grid[0])
        ranked = np.sort(pct)[::-1]
        positive = ranked[ranked > 0]
        if len(positive):
            ax.plot(np.arange(1, len(positive) + 1), positive, color='#c0392b', lw=1.2)
        ax.set_xlim(0, len(ranked) + 1)
        ax.text(0.99, 0.04, f'{int((ranked <= 0).sum())} channels never at code 3 (not drawn on log axis)',
                transform=ax.transAxes, ha='right', fontsize=7, color=style.INK2)
        for thr, ls in zip(thresholds, ('--', ':')):
            n_above = int((sat > thr / 100).sum())
            ax.axhline(thr, color=style.INK2, ls=ls, lw=0.8, label=f'> {thr:g}%: {n_above} channels')
        ax.set_yscale('log')
        ax.set_xlabel('channel rank')
        ax.set_ylabel('saturation (%)')
        ax.set_title(f'Sorted saturation · {int((sat > 0).sum())} channels ever reach code 3')
        ax.legend(frameon=False, loc='upper right')

        ax = fig.add_subplot(grid[1])
        ax.scatter(mean_hz, pct, s=6, c=np.minimum(stats['max'], C.MAX_CODE),
                   cmap=max_code_cmap()[0], norm=max_code_cmap()[1], lw=0)
        ceiling = C.MAX_CODE / session.frame_seconds
        ax.axvline(ceiling, color=style.GREY, ls='--', lw=0.8)
        ax.text(ceiling, ax.get_ylim()[1], f' ceiling {ceiling:.0f} Hz', fontsize=6.5, va='top', color=style.INK2)
        ax.set_xlabel(f'mean rate / Hz (count / {session.frame_seconds * 1000:.0f} ms, censored)')
        ax.set_ylabel('saturation (%)')
        ax.set_title('Saturation vs mean rate (colour = max code)')

        ax = fig.add_subplot(grid[2])
        order = np.argsort(-sat, kind='stable')[:min(top_n, 12)]
        lines = [f'{"rank":>4} {"electrode":>9} {"column":>6} {"sat %":>7} {"mean Hz":>8}']
        for r, j in enumerate(order, 1):
            lines.append(f'{r:>4} {int(ids[j]):>9} {j + 1:>6} {pct[j]:>7.3f} {mean_hz[j]:>8.1f}')
        if top_n > 12:
            lines.append(f'(full top {top_n} in table "top_saturated")')
        style.text_panel(ax, lines, fontsize=7)
        ax.set_title('Most saturated electrodes', loc='left', fontsize=8)
        fig.suptitle(f'{session.label}\ncount saturation (code 3 = ≥3 per frame) · mean {pct.mean():.3f}% · '
                     f'{scope_text}', fontsize=8.5)
        return fig

    def zone_figure(self, ctx, stats, scope_text, zone, bursty, sat_ok, sat_bad, rate_min, burst_ratio, tiers=None, *, figure=None):
        session = ctx.session
        ids = session.electrode_ids
        shown = np.where(stats['max'] == 0, DEAD, zone).astype(object)   # display category: silent split in two
        height, footer_in = self.MAP_HEIGHT_IN, self.MAP_FOOTER_IN
        geometry = self._map_geometry(ctx)
        fig = new_figure(geometry[3] + 9.5, height) if figure is None else figure
        outer = fig.add_gridspec(1, 2, width_ratios=[geometry[3], 9.5])
        axes, _ = self._map_block(fig, outer[0, 0], geometry, footer_in)
        grid = outer[0, 1].subgridspec(2, 1, height_ratios=[1.75, 1.25])

        # zone maps
        cmap = ListedColormap([DISPLAY_COLORS[z] for z in DISPLAY])
        norm = BoundaryNorm(np.arange(-0.5, len(DISPLAY) + 0.5), cmap.N)
        index = np.array([DISPLAY.index(z) for z in shown], float)
        self._draw_maps(axes, geometry, ids, index, cmap, norm, boxes=bursty, box_color=BURSTY_COLOR, box_lw=1.4)
        ax = fig.add_subplot(grid[0])
        self._zone_scatter(ax, session, stats, shown, bursty, sat_ok, sat_bad, rate_min, burst_ratio)
        labels = (f'sat ≤ {sat_ok:g}%, fr ≥ {rate_min:g} Hz',
                  f'{sat_ok:g}% < sat ≤ {sat_bad:g}%, fr ≥ {rate_min:g} Hz',
                  f'sat > {sat_bad:g}%, fr ≥ {rate_min:g} Hz',
                  f'Silent, 0 < fr < {rate_min:g} Hz', 'Silent, fr = 0 Hz')
        handles = [Patch(color=DISPLAY_COLORS[z], label=f'{label} ({int(np.sum(shown == z))})')
                   for z, label in zip(DISPLAY, labels)]
        handles += [
            Patch(fc='none', ec=BURSTY_COLOR, lw=1.4, label=f'Bursty ({int(bursty.sum())})'),
            Patch(color=style.MISSING, label=f'Not transmitted ({C.N_ELECTRODES - len(ids)})'),
            Patch(fc='none', ec=style.INK, ls=(0, (3, 2)), label='Combined silent count'),
        ]
        handles.extend(ax.get_legend_handles_labels()[0])
        ax.legend(handles=handles, title='Session totals and references', frameon=True, framealpha=0.9,
                  loc='upper left', fontsize=7, title_fontsize=7)

        # zone counts: stacked bars, all transmitted channels and (if configs.ini) one bar per tier
        ax = fig.add_subplot(grid[1])
        self._zone_bars(ax, zone_breakdown(shown, bursty, tiers))
        fig.suptitle(f'{session.label}\nchannel saturation zones · {stats["n_frames"]:,} frames ({scope_text})',
                     fontsize=8.5)
        return fig

    @staticmethod
    def _zone_scatter(ax, session, stats, shown, bursty, sat_ok, sat_bad, rate_min, burst_ratio):
        """Saturation vs censored mean rate (symlog axes) with zone bands and the Poisson reference."""
        hz, pct = stats['mean_hz'], 100 * stats['sat']
        ok = np.isfinite(hz)
        ceiling = C.MAX_CODE / session.frame_seconds
        x_hi, y_hi = ceiling * 1.15, 100.0
        ax.axhspan(0, sat_ok, color=ZONE_COLORS['rarely saturated'], alpha=0.10, lw=0)
        ax.axhspan(sat_ok, sat_bad, color=ZONE_COLORS['sometimes saturated'], alpha=0.14, lw=0)
        ax.axhspan(sat_bad, y_hi, color=ZONE_COLORS['often saturated'], alpha=0.10, lw=0)
        ax.axvspan(0, rate_min, color=ZONE_COLORS['silent'], alpha=0.30, lw=0)
        for y in (sat_ok, sat_bad):
            ax.axhline(y, color=style.INK2, ls='--', lw=0.7)
        ax.axvline(rate_min, color=style.INK2, ls=':', lw=0.7)
        ax.axvline(ceiling, color=style.GREY, ls='--', lw=0.8)
        ax.text(ceiling, y_hi, f'ceiling {ceiling:.0f} Hz ', fontsize=6.5, va='top', ha='right', color=style.INK2)
        ref_x, ref_y = poisson_reference(session.frame_seconds)
        ax.plot(ref_x, ref_y, color=style.INK, lw=1.0, label='Poisson reference (expected sat. of random spiking)')
        ax.plot(ref_x, np.minimum(burst_ratio * ref_y, y_hi), color=style.INK, lw=0.7, ls=':',
                label=f'{burst_ratio:g}× Poisson (bursty above, if > {sat_ok:g}%)')
        for z in ZONES:
            m = ok & (shown == z)
            ax.scatter(hz[m], pct[m], s=9, color=DISPLAY_COLORS[z], lw=0, zorder=3)
        dead = ok & (shown == DEAD)
        ax.scatter(hz[dead], pct[dead], s=30, marker='x', color=DISPLAY_COLORS[DEAD], lw=0.9, zorder=4)
        m = ok & bursty
        ax.scatter(hz[m], pct[m], s=40, facecolors='none', edgecolors=BURSTY_COLOR, lw=1.3, zorder=4)
        # symlog, not log: many channels have saturation 0 and silent channels rate 0, which a log axis cannot
        # draw; symlog is linear below 1 Hz / 0.01 % and logarithmic above, so zeros stay on the plot.
        ax.set_xscale('symlog', linthresh=1.0, linscale=0.4)
        ax.set_yscale('symlog', linthresh=0.01, linscale=0.4)
        ax.set_xticks([0, 0.5, 1, 2, 5, 10, 20, 50, 100, 150])
        ax.set_xticklabels(['0', '0.5', '1', '2', '5', '10', '20', '50', '100', '150'])
        ax.set_yticks([0, 0.01, 0.1, 1, 5, 10, 100])
        ax.set_yticklabels(['0', '0.01', '0.1', '1', '5', '10', '100'])
        ax.set_xlim(0, x_hi)
        ax.set_ylim(0, y_hi)
        ax.set_xlabel(f'mean rate / Hz (count / {session.frame_seconds * 1000:.0f} ms, censored; symlog)')
        ax.set_ylabel('saturation (%, symlog)')
        ax.set_title('Saturation and Firing Rate')

    @staticmethod
    def _zone_bars(ax, breakdown, width=0.6):
        """Stacked zone bars; small segments labelled outside with a leader line, silent outlined."""
        from matplotlib.patches import Rectangle
        top = max(max(r['channels'] for r in breakdown), 1)
        inside_min, gap = 0.075 * top, 0.06 * top
        dark_text = {'sometimes saturated', 'silent'}
        for i, r in enumerate(breakdown):
            n, bottom, outside = r['channels'], 0, []
            for z in DISPLAY:
                k = r[z]
                if k:
                    ax.bar(i, k, width, bottom=bottom, color=DISPLAY_COLORS[z], edgecolor='white', lw=0.6)
                    label = f'{k} ({100 * k / n:.0f}%)'
                    if k >= inside_min:
                        ax.text(i, bottom + k / 2, label, ha='center', va='center', fontsize=7,
                                color=style.INK if z in dark_text else 'white')
                    else:
                        outside.append([bottom + k / 2, label, z])
                bottom += k
            silent = r['silent_total']
            if silent:
                ax.add_patch(Rectangle((i - width / 2, n - silent), width, silent, fill=False, ec=style.INK,
                                       lw=0.9, ls=(0, (3, 2)), zorder=3))
            last = -np.inf
            for item in outside:                                  # bottom to top; push up to avoid overlap
                item.append(max(item[0], last + gap))
                last = item[3]
            for y, label, z, y_text in outside:
                x0, x1 = i + width / 2 + 0.02, i + width / 2 + 0.12
                color = style.INK2 if z in ('silent', DEAD, 'rarely saturated') else DISPLAY_COLORS[z]
                ax.plot([x0, x1], [y, y_text], color=color, lw=0.7, clip_on=False)
                ax.text(x1 + 0.02, y_text, label, va='center', fontsize=7, color=color)
            ax.text(i, n + 0.015 * top, f'{n}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        ax.set_xticks(range(len(breakdown)))
        ax.set_xticklabels([f"{r['group']}\nbursty {r['bursty']}" for r in breakdown], fontsize=7.5)
        if len(breakdown) > 1:
            ax.axvline(0.5, color=style.GREY, lw=0.6, ls=':')
        pad = (max(len(breakdown), 4) - len(breakdown)) / 2      # same bar width with or without configs.ini
        ax.set_xlim(-0.55 - pad, len(breakdown) - 0.25 + pad)
        ax.set_ylim(0, top * 1.12)
        ax.set_ylabel('channels')
        ax.set_title('Channels per saturation zone' + (' · by config tier' if len(breakdown) > 1 else
                                                  ' (no channel config for this day)'), fontsize=8.5)

    def rate_map(self, ctx, stats, scope_text, sat_box, scale='log'):
        session = ctx.session
        ids, hz = session.electrode_ids, stats['mean_hz']
        boxed = 100 * stats['sat'] > sat_box
        ceiling = C.MAX_CODE / session.frame_seconds
        height, footer_in = self.MAP_HEIGHT_IN, self.MAP_FOOTER_IN
        geometry = self._map_geometry(ctx)
        fig = new_figure(geometry[3] + 7.0, height)
        outer = fig.add_gridspec(1, 2, width_ratios=[geometry[3], 7.0])
        axes, footer = self._map_block(fig, outer[0, 0], geometry, footer_in)
        grid = outer[0, 1].subgridspec(2, 1)
        finite = hz[np.isfinite(hz)]
        one_event = 1 / max(stats['n_frames'] * session.frame_seconds, 1e-9)     # Hz of a single event
        norm, cmap, extend, suffix = map_norm(hz, scale, one_event, 'viridis')
        image = self._draw_maps(axes, geometry, ids, hz, cmap, norm, boxes=boxed, box_color='#e8112d')
        self._footer_colorbar(fig, footer, image, f'mean rate (Hz)\n{suffix}', extend,
                              note=f'red box = saturation > {sat_box:g}%')

        ax = fig.add_subplot(grid[0])
        if len(finite):
            ax.hist(finite, bins=np.linspace(0, max(float(finite.max()), 1e-3) * 1.0001, 41), color='#3b7ea1',
                    lw=0)
            ax.set_yscale('log')
        ax.axvline(ceiling, color=style.GREY, ls='--', lw=0.8)
        ax.text(ceiling, 1, f' ceiling {ceiling:.0f} Hz', transform=ax.get_xaxis_transform(), fontsize=6.5,
                va='top', ha='right', color=style.INK2)
        ax.set_xlabel(f'mean rate / Hz (count / {session.frame_seconds * 1000:.0f} ms, censored)')
        ax.set_ylabel('channels')
        ax.set_title(f'Mean rate across {len(finite)} channels')

        ax = fig.add_subplot(grid[1])
        lines = ['mean rate (Hz) over transmitted channels', '']
        if len(finite):
            q = np.percentile(finite, [0, 25, 50, 75, 100])
            lines += [f'{"min":>8} {q[0]:8.2f}', f'{"p25":>8} {q[1]:8.2f}', f'{"median":>8} {q[2]:8.2f}',
                      f'{"p75":>8} {q[3]:8.2f}', f'{"max":>8} {q[4]:8.2f}', f'{"mean":>8} {finite.mean():8.2f}', '',
                      f'silent (0 Hz):      {int((finite == 0).sum())}',
                      f'boxed (sat > {sat_box:g}%): {int(boxed.sum())}', '',
                      'red box = saturated: rate is a lower',
                      'bound, the true rate is higher']
        style.text_panel(ax, lines, fontsize=7.5)
        fig.suptitle(f'{session.label}\nmean rate per electrode · {stats["n_frames"]:,} frames ({scope_text})',
                     fontsize=8.5)
        return fig


SECTION = ChannelMaxCount()
