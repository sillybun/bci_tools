"""X01 v1 — cross-day channel max count + saturation stability.

Follows macaque_data_analysis `sdk/channel_stability.py`: electrodes are aligned by
their one-based id (XDF desc/channel_ids); untransmitted electrodes are missing
(grey), never zero. Statistics use every received frame of each recording (not
only the filtered trials), because max code and saturation describe the channel,
not the task. Identity is the electrode/contact, not a verified neuron.

Per electrode and day:
  max code     maximum 2-bit code (0..3; 3 means >= 3 events per frame)
  saturation   fraction of frames at code 3
  mean rate    count / frame window (Hz)
State per session: silent · low · active · saturated
(saturation >= threshold). An electrode is *stable* if it was transmitted on at
least two recordings and kept the same state on all of them.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import re

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path as MarkerPath

from ... import config as C
from ...core.analysis import bh_fdr, oneway_anova
from ...core import crossday as X
from ...core import behavior
from ...core import style
from ...core.catalog import day_config_path, day_folder
from ...core.channels import draw_channel_map, read_channel_config
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure
from ..s02_channel_max_count_v1.section import code_stats


def tuning_counts(values, directions):
    """Separate observed across-trial variability from an estimable direction F test."""
    values = np.asarray(values, float)
    directions = np.asarray(directions)
    finite = np.isfinite(values).all(1)
    values, directions = values[finite], directions[finite]
    n = len(values)
    groups = len(np.unique(directions))
    active = values.std(0) > 1e-12 if n >= 2 else None
    result = dict(trials=n, active=int(active.sum()) if active is not None else None,
                  direction_tuned=None, represented_directions=groups,
                  anova_df_between=groups - 1, anova_df_within=n - groups,
                  tuning_unavailable_reason='')
    if groups < 2 or n <= groups:
        result['tuning_unavailable_reason'] = 'Fewer than two directions or no within-direction residual degrees of freedom'
    else:
        stats = oneway_anova(values, directions)
        q = bh_fdr(stats['p'], np.flatnonzero(active))
        result['direction_tuned'] = int((active & (q < .05)).sum())
    return result


def config_candidates(recordings):
    """Find current and earlier dated configs for the selected recording roots."""
    latest = {}
    for recording in recordings:
        folder = day_folder(recording.path)
        if folder and recording.date:
            latest[folder.parent] = max(latest.get(folder.parent, ''), recording.date.replace('-', ''))
    candidates = {}
    for root, end in latest.items():
        dated = []
        for folder in root.iterdir():
            if not folder.is_dir() or not re.fullmatch(r'\d{8}|\d{4}-\d{2}-\d{2}', folder.name):
                continue
            date = folder.name.replace('-', '')
            if date <= end and (path := day_config_path(folder / '_')):
                dated.append((date, path))
        candidates[root] = sorted(dated, reverse=True)
    return candidates


def config_changes(sessions):
    """Infer config from the actual good → uncertain → bad column order."""
    candidates = config_candidates(s.recording for s in sessions)
    configs = {}
    for dated in candidates.values():
        for _, path in dated:
            try:
                cfg = read_channel_config(path)
                groups = tuple(tuple(sorted(set(cfg[k]))) for k in ('good', 'uncertain', 'bad'))
                if sum(map(len, groups)) == len(set().union(*groups)):
                    configs[path] = groups
            except (OSError, ValueError):
                continue
    rows = [{} for _ in sessions]
    previous = {}
    marked_unverified = set()
    for i in sorted(range(len(sessions)), key=lambda i: (sessions[i].recording.date, sessions[i].recording.key)):
        session = sessions[i]
        recording = session.recording
        folder = day_folder(recording.path)
        root = folder.parent if folder else None
        date = recording.date.replace('-', '')
        ids = list(session.electrode_ids)
        matches = []
        for cfg_date, path in candidates.get(root, ()):
            if cfg_date > date or path not in configs:
                continue
            good, uncertain, bad = configs[path]
            end = len(good) + len(uncertain)
            if (ids[:len(good)] == list(good) and set(ids[len(good):end]) == set(uncertain)
                    and set(ids[end:]).issubset(bad)):
                matches.append(path)
        identities = {configs[path] for path in matches}
        path = matches[0] if len(identities) == 1 else ''
        changed = False
        if path:
            identity = configs[path]
            changed = previous.get(root) != identity
            previous[root] = identity
        directory_config = recording.channel_config
        unverified_change = bool(not path and directory_config and (root, date) not in marked_unverified)
        if unverified_change:
            marked_unverified.add((root, date))
        rows[i] = dict(recording=recording.key, date=recording.date, directory_config=directory_config,
                       matched_config=path, config_changed=changed,
                       unverified_change=unverified_change,
                       match_status='matched' if path else 'ambiguous' if matches else 'unverified')
    return rows

STATES = ('untransmitted', 'silent', 'low', 'active', 'saturated')
STATE_COLORS = ('#d6d4cf', '#111111', '#9ecae1', '#2176ae', '#ef2727')


class ChannelStability(Section):
    id = 'x01'
    mode = 'cross'
    title = '电极活跃度与稳定性'
    description = ('先比较各记录的可检验和方向调谐通道数，再按电极 ID 对齐 max code、饱和比例、'
                   '平均发放率矩阵及传输集合重叠（Jaccard），'
                   '每日分布，跨天稳定/不稳定电极，共同电极的平均发放率相关，以及每日通道布局图')
    cache_code_dependencies = ('sections/s02_channel_max_count_v1', 'sections/s05_anova_v1')
    params = (
        Param('sat_threshold', '饱和判定：code=3 帧比例 ≥', 'float', 0.05, minimum=0.0, maximum=1.0),
        Param('silent_threshold', '静默阈值 / Hz', 'float', 0.5, minimum=0.0),
        Param('rate_threshold', '活跃判定：平均发放率 ≥ / Hz', 'float', 1.0, minimum=0.0),
        Param('map_quantity', '布局图显示量', 'choice', 'max_code', choices=('max_code', 'saturation', 'mean_rate')),
        Param('map_layout', '布局图排列', 'choice', 'default', choices=('default', 'physical'),
              help='default：8 列 × 128 行电极 ID 网格；physical：使用已加载的真实布局工作簿。'),
    )

    def resolve_params(self, raw=None):
        values = super().resolve_params(raw)
        if values['silent_threshold'] > values['rate_threshold']:
            raise ValueError('静默阈值不能高于活跃阈值')
        return values

    def run(self, ctx: AnalysisContext) -> SectionResult:
        p = ctx.params
        sessions = ctx.sessions
        n = len(sessions)
        labels = X.recording_labels(s.recording for s in sessions)
        result = SectionResult()
        mats = {k: np.full((n, C.N_ELECTRODES), np.nan) for k in ('max', 'sat', 'rate', 'nonzero')}
        for i, s in enumerate(sessions):
            ctx.check_cancel()
            ctx.progress(f'channel statistics {labels[i]}')
            stats = ctx.cached(('s02_code_stats', id(s), 'whole_recording', 0),
                               lambda: code_stats(s, np.ones(len(s.neural_times), dtype=bool)))
            st = dict(max=stats['max'], sat=stats['sat'], rate=stats['mean_hz'], nonzero=1 - stats['frac'][:, 0])
            for k in mats:
                mats[k][i] = X.full_vector(s.electrode_ids, st[k])
        observed = np.isfinite(mats['max'])
        silent_mask = observed & ((mats['rate'] < p['silent_threshold']) | (mats['max'] == 0))
        state = np.zeros((n, C.N_ELECTRODES), int)                       # 0 untransmitted
        state[observed] = 3
        state[observed & (mats['rate'] < p['rate_threshold'])] = 2
        state[observed & (mats['sat'] >= p['sat_threshold']) & (mats['max'] > 0)] = 4
        state[silent_mask] = 1

        # ---------------------------------------------------------- overlap
        jac = np.eye(n)
        shared = np.zeros((n, n), int)
        rate_r = np.eye(n)
        active_jac = np.eye(n)
        pair_rows = []
        for i, j in combinations(range(n), 2):
            both = observed[i] & observed[j]
            union = observed[i] | observed[j]
            shared[i, j] = shared[j, i] = both.sum()
            jac[i, j] = jac[j, i] = both.sum() / union.sum() if union.any() else np.nan
            ai, aj = both & (mats['max'][i] > 0), both & (mats['max'][j] > 0)
            au = (ai | aj).sum()
            active_jac[i, j] = active_jac[j, i] = (ai & aj).sum() / au if au else np.nan
            a, b = np.log1p(mats['rate'][i, both]), np.log1p(mats['rate'][j, both])
            r = float(np.corrcoef(a, b)[0, 1]) if both.sum() > 2 and a.std() > 0 and b.std() > 0 else np.nan
            rate_r[i, j] = rate_r[j, i] = r
            same_state = float(np.mean(state[i, both] == state[j, both])) if both.any() else np.nan
            pair_rows.append(dict(recording_a=sessions[i].label, recording_b=sessions[j].label,
                                  days_apart=abs(X.day_number(sessions[j]) - X.day_number(sessions[i])),
                                  transmitted_a=int(observed[i].sum()), transmitted_b=int(observed[j].sum()),
                                  shared_transmitted=int(both.sum()), jaccard_transmitted=round(jac[i, j], 4),
                                  jaccard_nonzero_on_shared=round(float(active_jac[i, j]), 4),
                                  log_rate_pearson_r_on_shared=round(r, 4) if np.isfinite(r) else '',
                                  same_state_fraction_on_shared=round(same_state, 4) if np.isfinite(same_state) else ''))
        for i in range(n):
            shared[i, i] = observed[i].sum()

        # ---------------------------------------------------------- stability per electrode
        n_obs = observed.sum(0)
        multi = n_obs >= 2
        st_obs = np.where(observed, state, -1)
        same = np.array([len(set(col[col >= 0])) == 1 for col in st_obs.T])
        stable = multi & same
        unstable = multi & ~same
        all_common = observed.all(0)
        electrode_rows = []
        for e in range(C.N_ELECTRODES):
            row = dict(electrode_id=e + 1, transmitted_recordings=int(n_obs[e]),
                       category='stable' if stable[e] else 'unstable' if unstable[e] else
                       ('single_recording' if n_obs[e] == 1 else 'never_transmitted'))
            if stable[e]:
                row['category'] = f'stable_{STATES[st_obs[st_obs[:, e] >= 0, e][0]]}'
            for i, lab in enumerate(labels):
                row[f'{lab}_max'] = int(mats['max'][i, e]) if observed[i, e] else ''
                row[f'{lab}_sat'] = round(float(mats['sat'][i, e]), 5) if observed[i, e] else ''
                row[f'{lab}_rate_hz'] = round(float(mats['rate'][i, e]), 3) if observed[i, e] else ''
                row[f'{lab}_state'] = STATES[state[i, e]]
            electrode_rows.append(row)
        day_rows = []
        for i, s in enumerate(sessions):
            o = observed[i]
            nonsilent = o & ~silent_mask[i]
            day_rows.append(dict(recording=s.label, date=s.recording.date, transmitted=int(o.sum()),
                                 id_min=int(s.electrode_ids.min()), id_max=int(s.electrode_ids.max()),
                                 **{f'max_code_{c}': int((mats['max'][i, o] == c).sum()) for c in range(4)},
                                 **{f'state_{STATES[k]}': int((state[i] == k).sum()) for k in range(1, 5)},
                                 median_sat_nonsilent=round(float(np.median(mats['sat'][i, nonsilent])), 5)
                                 if nonsilent.any() else '',
                                 median_rate_hz=round(float(np.median(mats['rate'][i, o])), 3)))

        common_text = f'{int(all_common.sum())} electrodes transmitted in all {n} recordings'
        ctx.progress('figures')
        activity_rows = []
        threshold = p['rate_threshold']
        for label, rates, transmitted, silent_channels in zip(labels, mats['rate'], observed, silent_mask):
            silent = int(silent_channels.sum())
            low = int((transmitted & ~silent_channels & (rates < threshold)).sum())
            rates = rates[transmitted]
            active = len(rates) - silent - low
            q1, median, q3 = np.percentile(rates, [25, 50, 75])
            activity_rows.append(dict(session=label, transmitted=len(rates), silent=silent, low=low, active=active,
                                      active_fraction=active / len(rates), q1_hz=float(q1),
                                      median_hz=float(median), q3_hz=float(q3)))
        tuning_rows = []
        success_rows = behavior.run_success_summaries(sessions)
        for session, cohort, row in zip(sessions, ctx.cohorts, activity_rows):
            ctx.check_cancel()
            counts = tuning_counts(cohort.window_rates, cohort.directions)
            tuning_rows.append(dict(session=row['session'], date=session.recording.date,
                                    transmitted=row['transmitted'], **counts))
        for row, success in zip(tuning_rows, success_rows):
            row.update(success)
        result.add_figure('Electrode Activity and Direction Tuning', self.activity(tuning_rows), method=(
            'One bar per selected session. Active channels have nonzero across-trial SD of '
            'the mean firing rate in the configured analysis window, after the topbar trial filter and removal '
            'of nonfinite trials. Direction-tuned channels are the subset with q < 0.05 by one-way ANOVA '
            'across direction groups and Benjamini–Hochberg correction over the active channels of that session. '
            'No fixed minimum trial-count threshold is imposed. Across-trial variability requires two finite trials. '
            'Direction ANOVA requires at least two directions and positive within-direction residual degrees of freedom; '
            'otherwise only the tuned count is unavailable. '
            'The dashed horizontal line shows the transmitted channel count when equal across sessions. '
            'Right axis: overall mean success percentage (successes / known attempts) within each run, with Q1–Q3 of task-block success percentages; '
            'successes / known attempts including failures and timeouts, independent of neural trial filters. '
            'A matching complete behavior CSV supersedes early XDF fragments of that run; otherwise '
            'only recorded complete XDF trials are counted. Missing block outcomes are unavailable.'),
            data={'direction_tuning_by_session': tuning_rows})
        result.tables['direction_tuning_by_session'] = tuning_rows
        result.tables['channel_activity'] = activity_rows
        base = (f'Inputs: every received spike_counts frame of each recording (not only filtered trials); '
                f'{n} recordings: {", ".join(labels)}. Stored columns are mapped to one-based electrode ids '
                f'(XDF desc/channel_ids) and placed on a 1..1024 id axis; electrodes a recording did not transmit '
                f'are missing (NaN, grey), never zero. Per electrode and recording: max code = maximum 2-bit count '
                f'over all frames (0..3, 3 means >= 3 events); saturation fraction = fraction of frames with code 3; '
                f'mean rate = mean count / frame window ({sessions[0].frame_seconds:g} s) in Hz. '
                f'State: silent = max 0 or rate < {p["silent_threshold"]:g} Hz; '
                f'otherwise saturated if saturation fraction >= {p["sat_threshold"]:g}; low = mean rate '
                f'< {p["rate_threshold"]:g} Hz; otherwise active.')
        long_rows = [dict(recording=labels[i], electrode_id=e + 1, transmitted=bool(observed[i, e]),
                          max_code=int(mats['max'][i, e]) if observed[i, e] else '',
                          saturation_fraction=float(mats['sat'][i, e]) if observed[i, e] else '',
                          mean_rate_hz=float(mats['rate'][i, e]) if observed[i, e] else '',
                          state=STATES[state[i, e]])
                     for i in range(n) for e in range(C.N_ELECTRODES)]
        for title, fig, field, explanation in self.matrices(mats, state, labels, p):
            result.add_figure(title, fig, method=base + ' Rows = recordings, columns = electrode ID 1–1024. '
                              + explanation, data={'electrode_values': [
                                  {key: row[key] for key in ('recording', 'electrode_id', 'transmitted', field)}
                                  for row in long_rows]})
        result.add_figure(
            'Electrode Overlap and Rate Correlation',
            self.overlap(jac, active_jac, shared, rate_r, labels, observed, common_text),
            method=base + ' For every pair of recordings i, j: Jaccard of transmitted id sets = |Ti ∩ Tj| / |Ti ∪ Tj|; '
                          'shared = |Ti ∩ Tj| (diagonal = electrodes transmitted); Pearson r of log(1 + mean rate) '
                          'over the shared electrodes; Jaccard of nonzero electrodes (max code > 0) computed within '
                          'the shared set. Strip: which ids each recording transmitted.',
            data={'pairwise_matrices': X.matrix_rows(
                      {'jaccard_transmitted': jac, 'shared_transmitted': shared.astype(float),
                       'log1p_rate_pearson_r': rate_r, 'jaccard_nonzero_on_shared': active_jac}, labels),
                  'transmitted': [dict(recording=labels[i], electrode_id=e + 1, transmitted=bool(observed[i, e]))
                                  for i in range(n) for e in range(C.N_ELECTRODES)]})
        result.add_figure(
            'Electrode Distributions by Session', self.distributions(mats, state, observed, labels, p),
            method=base + ' Panels: (1) transmitted electrodes per recording stacked by max code; (2) stacked by '
                          'state; (3) violin of log10 saturation fraction over non-silent transmitted electrodes '
                          '(floor 1e-6), dashed line = saturation threshold; (4) violin of log10(1 + mean Hz) over '
                          'transmitted electrodes, dashed line = rate threshold. Violin bars mark min/median/max.',
            data={'per_recording_counts': day_rows,
                  'violin_values': [dict(recording=labels[i], electrode_id=e + 1,
                                         log10_saturation=float(np.log10(max(mats['sat'][i, e], 1e-6)))
                                         if not silent_mask[i, e] else '',
                                         log10_1p_rate=float(np.log10(1 + mats['rate'][i, e])))
                                    for i in range(n) for e in np.flatnonzero(observed[i])]})
        trans = self._transitions(state, observed)
        result.add_figure(
            'Electrode Stability', self.stability(state, observed, stable, unstable, n_obs, labels, p),
            method=base + ' An electrode transmitted in >= 2 recordings is stable if its state is identical in every '
                          'recording that transmitted it, otherwise unstable. Panels: (1) histogram of the number of '
                          'recordings transmitting each of the 1024 ids; (2) stable electrodes by state and unstable '
                          'count; (3) state transition counts between consecutive recordings (selection order), '
                          'pooled over pairs, on ids transmitted by both; (4) per-recording state of unstable '
                          'electrodes ordered by id (first 300 by coverage when more).',
            data={'coverage': [dict(n_recordings=k, electrodes=int(v))
                               for k, v in enumerate(np.bincount(n_obs, minlength=n + 1))],
                  'categories': [dict(category=f'stable_{STATES[k]}',
                                      electrodes=int((stable & (st_obs.max(0) == k)).sum())) for k in range(1, 5)]
                  + [dict(category='unstable', electrodes=int(unstable.sum()))],
                  'transitions': X.matrix_rows({'count': trans}, list(STATES[1:]), 'state_earlier', 'state_later'),
                  'unstable_states': [dict(electrode_id=e + 1, **{labels[i]: STATES[state[i, e]] for i in range(n)})
                                      for e in np.flatnonzero(unstable)]})
        pairs = self._pairs(n)
        result.add_figure(
            'Firing Rate Agreement', self.rate_scatter(mats, observed, labels),
            method=base + ' Each panel: mean rate (Hz, symlog axes, linear below 1 Hz) of electrodes transmitted by '
                          'both recordings, consecutive pairs plus first vs last; red = max code 3 on either '
                          'recording; grey line = identity; r = Pearson r of log(1 + rate).',
            data={'pair_rates': [dict(recording_x=labels[i], recording_y=labels[j], electrode_id=e + 1,
                                      rate_x_hz=float(mats['rate'][i, e]), rate_y_hz=float(mats['rate'][j, e]),
                                      max3_either=bool(mats['max'][i, e] == 3 or mats['max'][j, e] == 3))
                                 for i, j in pairs for e in np.flatnonzero(observed[i] & observed[j])]})
        map_layout = ctx.layout if p['map_layout'] == 'physical' else None
        if p['map_layout'] == 'physical' and map_layout is None:
            result.notes.append('未加载可用的真实布局工作簿，布局图使用默认电极 ID 网格。')
        where = 'physical workbook layout (selected ChannelMap xlsx)' if map_layout is not None else \
            '8-column × 128-row electrode-id grid (ids 1–128 in column 1)'
        map_rows = []
        mv = self._map_values(mats, p['map_quantity'])
        for i in range(n):
            for e in range(C.N_ELECTRODES):
                row = dict(recording=labels[i], electrode_id=e + 1,
                           value=float(mv[i, e]) if observed[i, e] else '')
                if map_layout is not None:
                    r, c = map_layout['position'][e]
                    row.update(layout_row=int(r) + 1, layout_column=chr(65 + int(c)))
                else:
                    row.update(grid_row=e % 128 + 1, grid_column=e // 128 + 1)
                map_rows.append(row)
        configs = config_changes(sessions)
        result.add_figure(
            'Electrode Maps by Session', self.maps(map_layout, mats, labels, p, configs),
            method=base + f' One map per recording on the {where}; quantity = {p["map_quantity"]} '
                          '(max_code: discrete 0..3; saturation: log scale floored at 1e-5; mean_rate: '
                          'log(1 + Hz), colour range 0..99th percentile). Grey = not transmitted. '
                          'A solid green arrow marks the initial matched configuration or a verified change '
                          'in channel tiers from the previous matched session. A hollow green arrow marks '
                          'the first unmatched session on a date with '
                          'a directory config. Configs are matched to the recorded electrode column order.',
            data={'map_values': map_rows, 'config_changes': configs})
        # Keep the spatial view next to the activity overview, before the flattened matrices.
        for entries in (result.figures, result.methods, result.figure_data):
            entries.insert(1, entries.pop())
        result.tables['per_recording'] = day_rows
        result.tables['pairwise'] = pair_rows
        result.tables['electrodes'] = electrode_rows
        result.summary.update(
            recordings=labels, transmitted_per_recording=[int(v) for v in observed.sum(1)],
            active_fraction_per_recording=[row['active_fraction'] for row in activity_rows],
            transmitted_all_recordings=int(all_common.sum()), transmitted_any_recording=int(observed.any(0).sum()),
            stable_electrodes=int(stable.sum()), unstable_electrodes=int(unstable.sum()),
            stable_by_state={STATES[k]: int((stable & (st_obs.max(0) == k)).sum()) for k in range(1, 5)},
            mean_pairwise_jaccard_transmitted=round(float(jac[np.triu_indices(n, 1)].mean()), 3),
            mean_pairwise_log_rate_r=round(float(np.nanmean(rate_r[np.triu_indices(n, 1)])), 3),
            sat_threshold=p['sat_threshold'], rate_threshold_hz=p['rate_threshold'],
            silent_threshold_hz=p['silent_threshold'])
        result.notes.append('方向调谐图使用筛选后的 trial，其余统计使用每份记录的全部接收帧。未传输电极记为缺失。'
                            '电极 ID 稳定只表示同一触点的信号统计相似，不证明是同一神经元。')
        return result

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _transitions(state, observed):
        trans = np.zeros((4, 4))
        for i in range(len(state) - 1):
            both = observed[i] & observed[i + 1]
            for a in range(1, 5):
                for b in range(1, 5):
                    trans[a - 1, b - 1] += np.sum(both & (state[i] == a) & (state[i + 1] == b))
        return trans

    @staticmethod
    def _pairs(n):
        pairs = [(i, i + 1) for i in range(n - 1)]
        if n > 2:
            pairs.append((0, n - 1))
        return pairs

    @staticmethod
    def _map_values(mats, q):
        if q == 'max_code':
            return mats['max']
        if q == 'saturation':
            return np.where(np.isfinite(mats['sat']), np.maximum(mats['sat'], 1e-5), np.nan)
        return np.log1p(mats['rate'])

    # ------------------------------------------------------------------ figures
    @staticmethod
    def _id_axis(ax):
        ticks = [0, 127, 255, 383, 511, 639, 767, 895, 1023]
        ax.set_xticks(ticks)
        ax.set_xticklabels([t + 1 for t in ticks])
        ax.set_xlabel('electrode id')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#c7ccd1')
            spine.set_linewidth(.65)

    def matrices(self, mats, state, labels, p):
        floor = 1e-5
        sat = np.ma.masked_invalid(np.maximum(mats['sat'], floor))
        panels = (
            ('Maximum Code', 'max_code', mats['max'],
             ListedColormap(style.MAX_CODE_COLORS).with_extremes(bad=style.MISSING),
             BoundaryNorm(np.arange(-.5, 4), 4), 'Max code', [0, 1, 2, 3],
             'Colour shows the maximum observed code.'),
            ('Electrode Saturation', 'saturation_fraction', sat, style_cmap('magma'),
             LogNorm(vmin=floor, vmax=max(np.nanmax(sat.filled(np.nan)), floor * 10)),
             'Fraction of frames at code 3', None,
             'Log colour scale; saturation fractions below 1e-5 are displayed at that floor.'),
            ('Electrode Firing Rate', 'mean_rate_hz', np.log1p(mats['rate']), style_cmap('viridis'),
             None, 'log(1 + Hz)', None, 'Colour shows log(1 + mean rate in Hz).'),
            ('Electrode State', 'state', state, ListedColormap(STATE_COLORS),
             BoundaryNorm(np.arange(-.5, 5), 5), None, None,
             'Colour shows the state assigned using the configured saturation and rate thresholds.'),
        )
        for title, field, values, cmap, norm, color_label, ticks, explanation in panels:
            fig = new_figure(13, 2.1 + .22 * len(labels))
            ax = fig.subplots()
            image = ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap=cmap, norm=norm,
                              interpolation='nearest')
            X.recording_yticks(ax, labels)
            self._id_axis(ax)
            if color_label:
                fig.colorbar(image, ax=ax, ticks=ticks, fraction=.03, pad=.01).set_label(color_label, fontsize=7)
                ax.set_title(f'{title} · grey = not transmitted')
            else:
                ax.set_title(f'{title} · low < {p["rate_threshold"]:g} Hz · '
                             f'saturated ≥ {p["sat_threshold"]:.1%}', pad=22)
                ax.legend(handles=[Patch(color=c, label=s) for c, s in zip(STATE_COLORS, STATES)],
                          ncol=5, frameon=False, loc='lower center', bbox_to_anchor=(.5, 1.01),
                          borderaxespad=0, fontsize=7)
            yield title, fig, field, explanation

    @staticmethod
    def activity(rows):
        fig = new_figure(max(10, min(22, 4 + .6 * len(rows))), 5.8)
        ax = fig.subplots()
        x = np.arange(len(rows))
        active = np.array([row['active'] for row in rows], float)
        tuned = np.array([row['direction_tuned'] for row in rows], float)
        transmitted = {row['transmitted'] for row in rows}
        height = max(transmitted)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.22)
        for i in range(1, len(rows)):
            if rows[i]['date'] != rows[i - 1]['date']:
                ax.axvline(i - .5, color=style.GREY, lw=.7, ls='--')
        ax.bar(x, active, width=.7, color='#d4dde4', label='Active channels (nonzero trial SD)')
        ax.bar(x, tuned, width=.7, color='#2176ae', label='Direction-tuned (BH q < 0.05)')
        if len(transmitted) == 1:
            ax.axhline(height, color=style.INK2, lw=.7, ls='--', label=f'{height} transmitted')
        for i, (total, count) in enumerate(zip(active, tuned)):
            if not np.isfinite(total):
                ax.annotate('N/A', (i, 0), (0, 5), textcoords='offset points', ha='center', fontsize=7)
                continue
            if total == 0:
                ax.annotate('0 / 0', (i, 0), (0, 4), textcoords='offset points', ha='center', fontsize=8)
                continue
            ax.annotate(f'{total:.0f}', (i, total), (0, 4), textcoords='offset points', ha='center', fontsize=8)
            if not np.isfinite(count):
                ax.annotate('N/A', (i, 0), (0, 5), textcoords='offset points',
                            ha='center', color='#2176ae', fontsize=7)
                continue
            if count >= .06 * height:
                ax.annotate(f'{count:.0f}', (i, count), (0, -4), textcoords='offset points',
                            ha='center', va='top', color='white', fontsize=8)
            else:
                ax.annotate(f'{count:.0f}', (i, count), (0, 4), textcoords='offset points',
                            ha='center', va='bottom', color='#2176ae', fontsize=7)
        ax.set(ylabel='Channels', xlabel='Session', ylim=(0, C.N_COLUMNS))
        ax.set_yticks([0, 100, 200, 300, 400, C.N_COLUMNS])
        success_ax = ax.twinx()
        color = '#bd5021'
        mean = np.array([row.get('success_pct_mean', np.nan) for row in rows], float)
        q1 = np.array([row.get('success_pct_q1', np.nan) for row in rows], float)
        q3 = np.array([row.get('success_pct_q3', np.nan) for row in rows], float)
        finite = np.isfinite(mean)
        if finite.any():
            # The mean can fall outside the IQR; draw the quartiles independently.
            quartiles = finite & np.isfinite(q1) & np.isfinite(q3)
            success_ax.vlines(x[quartiles], q1[quartiles], q3[quartiles], color=color, linewidth=1)
            success_ax.hlines(q1[quartiles], x[quartiles] - .05, x[quartiles] + .05, color=color, linewidth=1)
            success_ax.hlines(q3[quartiles], x[quartiles] - .05, x[quartiles] + .05, color=color, linewidth=1)
            success_ax.plot(x[finite], mean[finite], 'D', color=color, markersize=4,
                            label='Run success rate: average (block IQR)', zorder=5, clip_on=False)
        success_ax.set(ylabel='Block success rate (%)', ylim=(0, 100))
        success_ax.set_yticks([0, 20, 40, 60, 80, 100])
        success_ax.tick_params(axis='y', colors=color)
        success_ax.yaxis.label.set_color(color)
        success_ax.spines['right'].set_visible(True)
        success_ax.spines['right'].set_color(color)
        handles, names = ax.get_legend_handles_labels()
        extra_handles, extra_names = success_ax.get_legend_handles_labels()
        ax.legend(handles + extra_handles, names + extra_names, loc='lower center',
                  bbox_to_anchor=(.5, 1.01), ncol=4, frameon=False)
        X.recording_ticks(ax, x, [row['session'] for row in rows], fontsize=8)
        fig.suptitle('Electrode Activity and Direction Tuning')
        return fig

    def overlap(self, jac, active_jac, shared, rate_r, labels, observed, common_text):
        fig = new_figure(13, 9)
        grid = fig.add_gridspec(2, 3, height_ratios=[1, 0.8])
        X.heatmap(fig.add_subplot(grid[0, 0]), jac, labels, cmap='Blues', vmin=0, vmax=1,
                  title='Jaccard of transmitted electrode sets', xlabel='', ylabel='')
        X.heatmap(fig.add_subplot(grid[0, 1]), shared, labels, fmt='{:.0f}', cmap='Blues', vmin=0,
                  vmax=C.N_COLUMNS, title='Shared transmitted electrodes (diag = transmitted)', xlabel='', ylabel='')
        X.heatmap(fig.add_subplot(grid[0, 2]), rate_r, labels, cmap='viridis', vmin=0, vmax=1,
                  title='Pearson r of log(1+rate) on shared electrodes', xlabel='', ylabel='')
        ax = fig.add_subplot(grid[1, :2])
        n = len(labels)
        ax.imshow(observed.astype(float), aspect='auto', cmap=ListedColormap([style.MISSING, '#2176ae']),
                  interpolation='nearest', vmin=0, vmax=1)
        X.recording_yticks(ax, labels)
        self._id_axis(ax)
        ax.set_title(f'Transmitted electrode ids (blue) · {common_text}')
        X.heatmap(fig.add_subplot(grid[1, 2]), active_jac, labels, cmap='Blues', vmin=0, vmax=1,
                  title='Jaccard of nonzero electrodes on shared set', xlabel='', ylabel='')
        fig.suptitle('Electrode Overlap and Rate Correlation', fontsize=9)
        return fig

    def distributions(self, mats, state, observed, labels, p):
        n = len(labels)
        fig = new_figure(13, 7.5)
        axes = fig.subplots(2, 2)
        x = np.arange(n)
        ax = axes[0, 0]
        bottom = np.zeros(n)
        for c in range(4):
            v = np.array([(mats['max'][i, observed[i]] == c).sum() for i in range(n)])
            ax.bar(x, v, bottom=bottom, color=style.MAX_CODE_COLORS[c], label=f'max {c}', width=0.7)
            bottom += v
        ax.set_title('Transmitted electrodes by max code')
        ax.set_ylabel('electrodes')
        ax.legend(frameon=False, ncol=4, fontsize=6.5, loc='lower center', bbox_to_anchor=(0.5, 1.07))
        ax = axes[0, 1]
        bottom = np.zeros(n)
        for k in range(1, 5):
            v = (state == k).sum(1)
            ax.bar(x, v, bottom=bottom, color=STATE_COLORS[k], label=STATES[k], width=0.7)
            bottom += v
        ax.set_title('Transmitted electrodes by state')
        ax.legend(frameon=False, ncol=4, fontsize=6.5, loc='lower center', bbox_to_anchor=(0.5, 1.07))
        ax = axes[1, 0]
        data = [np.log10(np.maximum(mats['sat'][i, observed[i] & (state[i] != 1)], 1e-6)) for i in range(n)]
        self._violin(ax, data, '#ef2727')
        ax.axhline(np.log10(max(p['sat_threshold'], 1e-6)), color=style.INK2, ls='--', lw=0.8,
                   label='saturation threshold')
        ax.set_ylabel('log10 saturation fraction')
        ax.set_title('Saturation fraction of non-silent electrodes (floor 1e-6)')
        ax.legend(frameon=False)
        ax = axes[1, 1]
        data = [np.log10(1 + mats['rate'][i, observed[i]]) for i in range(n)]
        self._violin(ax, data, '#2176ae')
        ax.axhline(np.log10(1 + p['rate_threshold']), color=style.INK2, ls='--', lw=0.8, label='rate threshold')
        ax.set_ylabel('log10(1 + mean Hz)')
        ax.set_title('Mean rate of transmitted electrodes')
        ax.legend(frameon=False, loc='upper right')
        for ax in axes.flat:
            X.recording_ticks(ax, x, labels)
        fig.suptitle('Electrode Distributions by Session', fontsize=9)
        return fig

    @staticmethod
    def _violin(ax, data, color):
        pos = [i for i, d in enumerate(data) if len(d) > 1]
        if pos:
            parts = ax.violinplot([data[i] for i in pos], positions=pos, widths=0.75, showmedians=True)
            for body in parts['bodies']:
                body.set_facecolor(color)
                body.set_alpha(0.35)
            for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                if key in parts:
                    parts[key].set_color(color)
                    parts[key].set_linewidth(0.8)

    def stability(self, state, observed, stable, unstable, n_obs, labels, p):
        n = len(labels)
        fig = new_figure(13, 8)
        grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.2])
        ax = fig.add_subplot(grid[0, 0])
        counts = np.bincount(n_obs, minlength=n + 1)
        ax.bar(np.arange(n + 1), counts, color='#2176ae')
        for k, v in enumerate(counts):
            ax.text(k, v, f'{v}', ha='center', va='bottom', fontsize=6.5)
        ax.set_xlabel('number of recordings transmitting the electrode')
        ax.set_ylabel('electrodes (of 1024)')
        ax.set_title('Transmission coverage')
        ax = fig.add_subplot(grid[0, 1])
        st_obs = np.where(observed, state, -1)
        top = st_obs.max(0)
        names, values, colors = [], [], []
        for k in range(1, 5):
            names.append(f'stable {STATES[k]}')
            values.append(int((stable & (top == k)).sum()))
            colors.append(STATE_COLORS[k])
        names.append('unstable')
        values.append(int(unstable.sum()))
        colors.append('#edae49')
        ax.barh(range(len(names)), values, color=colors)
        for k, v in enumerate(values):
            ax.text(v, k, f' {v}', va='center', fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel('electrodes transmitted in ≥ 2 recordings')
        ax.set_title('Same state in every transmitted recording?')
        # transitions between consecutive recordings
        ax = fig.add_subplot(grid[0, 2])
        trans = self._transitions(state, observed)
        X.heatmap(ax, trans, list(STATES[1:]), fmt='{:.0f}', cmap='Blues', title='State transitions, consecutive recordings (pooled)',
                  xlabel='state in later recording', ylabel='state in earlier recording', recording_columns=False)
        # unstable electrodes trajectory
        ax = fig.add_subplot(grid[1, :])
        ids = np.flatnonzero(unstable)
        if len(ids):
            order = ids[np.lexsort((ids, -n_obs[ids]))][:300]
            show = state[:, np.sort(order)]
            ax.imshow(show, aspect='auto', cmap=ListedColormap(STATE_COLORS), norm=BoundaryNorm(np.arange(-0.5, 5), 5),
                      interpolation='nearest')
            ticks = np.linspace(0, len(order) - 1, min(12, len(order))).astype(int)
            ax.set_xticks(ticks)
            ax.set_xticklabels(np.sort(order)[ticks] + 1, fontsize=6)
            ax.set_xlabel('electrode id (unstable electrodes only, ordered by id' +
                          (', first 300 by coverage)' if len(ids) > 300 else ')'))
            ax.legend(handles=[Patch(color=c, label=s) for c, s in zip(STATE_COLORS, STATES)], ncol=5,
                      frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22), fontsize=7)
        else:
            ax.text(0.5, 0.5, 'No unstable electrode', ha='center', va='center', transform=ax.transAxes)
        X.recording_yticks(ax, labels)
        ax.set_title(f'{len(ids)} unstable electrodes: state per recording')
        fig.suptitle(f'Electrode stability · saturation ≥ {p["sat_threshold"]:g} · active ≥ {p["rate_threshold"]:g} Hz',
                     fontsize=9)
        return fig

    def rate_scatter(self, mats, observed, labels):
        n = len(labels)
        pairs = self._pairs(n)
        cols = len(pairs) if len(pairs) <= 5 else 4
        rows = int(np.ceil(len(pairs) / cols))
        fig = new_figure(3.3 * cols + 0.5, 3.2 * rows + 0.6)
        axes = np.atleast_1d(fig.subplots(rows, cols, squeeze=False)).ravel()
        for ax in axes[len(pairs):]:
            ax.axis('off')
        for ax, (i, j) in zip(axes, pairs):
            both = observed[i] & observed[j]
            a, b = mats['rate'][i, both], mats['rate'][j, both]
            sat = (mats['max'][i, both] == 3) | (mats['max'][j, both] == 3)
            ax.scatter(a[~sat], b[~sat], s=5, color='#2176ae', alpha=0.5, lw=0, label='max < 3 on both')
            ax.scatter(a[sat], b[sat], s=6, color='#ef2727', alpha=0.6, lw=0, label='max = 3 on either')
            hi = max(np.nanmax(np.r_[a, b, 1]), 1) * 1.2
            ax.plot([0, hi], [0, hi], color=style.GREY, lw=0.8)
            ax.set_xscale('symlog', linthresh=1)
            ax.set_yscale('symlog', linthresh=1)
            r = np.corrcoef(np.log1p(a), np.log1p(b))[0, 1] if both.sum() > 2 else np.nan
            ax.set_title(f'{labels[i]} vs {labels[j]}\n{both.sum()} shared · r(log) = {r:.2f}', fontsize=7.5)
            ax.set_xlabel(f'{labels[i]} Hz', color=X.session_label_color(labels[i]))
            ax.set_ylabel(f'{labels[j]} Hz', color=X.session_label_color(labels[j]))
        axes[0].legend(frameon=False, fontsize=6)
        fig.suptitle('Firing Rate Agreement on Shared Electrodes',
                     fontsize=9)
        return fig

    def maps(self, layout, mats, labels, p, configs=()):
        n = len(labels)
        map_height = .815 * (210 / 25.4)
        rows, cols = (128, 8) if layout is None else (layout['rows'], layout['columns'])
        map_width = map_height * cols / rows
        grid_width = n * (map_width + .18)
        width = max(297 / 25.4, grid_width + 4.4)
        bottom = 2.15
        height = map_height + bottom + .55
        fig = new_figure(width, height)
        fig.set_layout_engine(None)
        left = (width - grid_width - 1.4) / (2 * width)
        right = left + grid_width / width
        grid = fig.add_gridspec(1, n, left=left, right=right, top=(bottom + map_height) / height,
                                bottom=bottom / height, wspace=.1)
        q = p['map_quantity']
        values = self._map_values(mats, q)
        if q == 'max_code':
            cmap, norm = ListedColormap(style.MAX_CODE_COLORS), BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
            label = 'Maximum code'
        elif q == 'saturation':
            cmap, norm = 'magma', LogNorm(vmin=1e-5, vmax=max(np.nanmax(mats['sat']), 1e-4))
            label = 'Saturation fraction'
        else:
            from matplotlib.colors import Normalize
            cmap, norm = 'viridis', Normalize(0, np.nanpercentile(values, 99))
            label = 'Log(1 + rate in Hz)'
        image = None
        ids = np.arange(1, C.N_ELECTRODES + 1)
        for k in range(n):
            ax = fig.add_subplot(grid[0, k])
            image = draw_channel_map(ax, ids, values[k], layout=layout, cmap=cmap, norm=norm,
                                     colorbar=False, show_labels=False)
            ax.text(.5, -.015, X.recording_label_lines(labels[k]), transform=ax.transAxes, fontsize=7.5,
                    rotation=45, rotation_mode='anchor', ha='right', va='top', fontstyle='normal',
                    color=X.session_label_color(labels[k]))
            config = configs[k] if configs else {}
            if config.get('config_changed') or config.get('unverified_change'):
                ax.annotate('', xy=(.5, 1.004), xytext=(.5, 1.03), xycoords='axes fraction',
                            annotation_clip=False, arrowprops=dict(arrowstyle='simple,tail_width=.48',
                                                                  facecolor='white' if config.get('unverified_change') else '#57a832',
                                                                  edgecolor='#285b23',
                                                                  mutation_scale=18, shrinkA=0, shrinkB=0))
        arrow = MarkerPath([(-.24, .5), (.24, .5), (.24, -.02), (.5, -.02),
                            (0, -.5), (-.5, -.02), (-.24, -.02), (-.24, .5)])
        handles = []
        if q == 'max_code':
            handles.extend(Patch(facecolor=color, label=f'Code {code}')
                           for code, color in reversed(list(enumerate(style.MAX_CODE_COLORS))))
        else:
            color_axis = fig.add_axes([right + .16 / width, .48, .14 / width, .28])
            cb = fig.colorbar(image, cax=color_axis)
            cb.set_label(label, fontsize=7)
        handles.append(Patch(facecolor=style.MISSING, label='Not transmitted'))
        if any(row['config_changed'] for row in configs):
            handles.append(Line2D([], [], ls='none', marker=arrow, ms=12, mfc='#57a832',
                                  mec='#285b23', label='New Config'))
        if any(row.get('unverified_change') for row in configs):
            handles.append(Line2D([], [], ls='none', marker=arrow, ms=12, mfc='white',
                                  mec='#285b23', label='New Config (unverified)'))
        fig.legend(handles=handles, loc='center left',
                   bbox_to_anchor=(right + .12 / width, .55 if q == 'max_code' else .36),
                   frameon=False, fontsize=8)
        fig.suptitle(f'{label} by Session', y=.995, fontsize=11)
        return fig


def style_cmap(name):
    from matplotlib import colormaps
    cm = colormaps[name].copy()
    cm = cm.with_extremes(bad=style.MISSING)
    return cm


SECTION = ChannelStability()
