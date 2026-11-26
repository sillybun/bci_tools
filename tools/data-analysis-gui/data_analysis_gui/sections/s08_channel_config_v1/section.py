"""S08 v1 — impact of the NeuralInsight configs.ini channel selection (compensation).

The CLI programs the 512 transmitted channels as: every `good` electrode (ascending),
then every `uncertain` one, then a random draw from `bad` to pad the frame to 512.
ini indices are 0-based; `read_channel_config` converts them to the 1-based XDF ids.
Tiers are derived from the stable good/uncertain/bad labels intersected with the
record's own electrode ids (not from `last_programmed_channels`, which is a
snapshot of the last programming event; see independent_recheck_v1/channel_quality.py).

Recordings from 10-26/10-27 predate channel selection and transmit electrodes 1..512
unselected; for them the overlap with any ini is only reported, not interpreted.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from ... import config as C
from ...core import analysis as A
from ...core import style
from ...core.channels import column_tiers, draw_channel_map, read_channel_config
from ...core.features import velocity_bins
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure

TIER_ORDER = ('good', 'uncertain', 'bad', 'unlisted')
TIER_COLORS = {'good': '#2e8b57', 'uncertain': '#e0a526', 'bad': '#c0392b', 'unlisted': '#5b6670'}
SUBSETS = (('good', ('good',)), ('good+uncertain', ('good', 'uncertain')),
           ('all 512', TIER_ORDER), ('bad only', ('bad',)))


def programming_order(tiers, ids):
    """Does the column order follow good(ascending) → uncertain → bad?"""
    rank = np.array([TIER_ORDER.index(t) for t in tiers])
    monotone_tiers = bool(np.all(np.diff(rank[rank < 3]) >= 0))
    good = ids[tiers == 'good']
    good_cols = np.flatnonzero(tiers == 'good')
    good_first = bool(len(good_cols) and good_cols.max() == len(good_cols) - 1)
    return dict(tiers_in_cli_order=monotone_tiers, good_block_first=good_first,
                good_ascending=bool(np.all(np.diff(good) > 0)) if len(good) > 1 else True)


def marker_positions(layout, electrodes):
    """(x, y) image coordinates for electrodes on the map drawn by draw_channel_map."""
    e = np.asarray(electrodes, int) - 1
    if layout is not None:
        pos = layout['position'][e]
        return pos[:, 1], pos[:, 0] + 1
    return e // 128, e % 128 + 1


class ChannelConfig(Section):
    id = 's08'
    title = 'Compensation 通道筛选影响（当日 INI / Neuralinsight 日志）'
    description = ('按 INI 或 Neuralinsight 日志的 good / uncertain / bad 标签给 512 个传输通道分层：阵列位置、列顺序、'
                   '各层平均发放率 / 饱和度 / 方向显著比例，以及不同通道子集的方向解码与速度解码')
    requires = ('channel_config',)
    params = (
        Param('alpha', 'ANOVA FDR 阈值 q', 'float', 0.05, minimum=1e-4, maximum=0.5),
        Param('ridge_lambda', 'Ridge λ', 'float', 1.0, minimum=1e-4, maximum=1e4),
        Param('decode', '运行子集解码', 'bool', True),
    )

    def run(self, ctx: AnalysisContext) -> SectionResult:
        session, cohort, p = ctx.session, ctx.cohort, ctx.params
        result = SectionResult()
        try:
            cfg = read_channel_config(ctx.channel_config_path)
        except OSError as exc:
            raise ValueError(f'无法读取通道配置：{exc}') from exc
        ids = session.electrode_ids
        tiers = column_tiers(ids, cfg)
        counts = {t: int(np.sum(tiers == t)) for t in TIER_ORDER}
        order = programming_order(tiers, ids)
        listed = {t: len(cfg.get(t, [])) for t in ('good', 'uncertain', 'bad')}
        crosstalk = sorted(set(int(e) for e in ids) & set(cfg.get('crosstalk_channels', [])))
        unselected = bool(np.array_equal(np.sort(ids), np.arange(1, C.N_COLUMNS + 1)))
        if unselected:
            result.notes.append('本记录传输的是电极 1..512（未经通道筛选，10-26/10-27 记录如此）；'
                                'tier 只是通道配置标签与这 512 个电极的交集，不代表当天的编程结果。')
        missing_good = sorted(set(cfg['good']) - set(int(e) for e in ids))
        if missing_good:
            result.notes.append(f'通道配置中有 {len(missing_good)} 个 good 电极未被本记录传输'
                                '（ini 可能不是这份记录当日的配置）。')

        ctx.progress('channel metrics')
        mean_hz = ctx.cached(('s08_mean_hz', id(session)),
                             lambda: session.counts.mean(0, dtype=np.float64) / session.frame_seconds)
        sat = ctx.cached(('s08_sat', id(session)),
                         lambda: np.count_nonzero(session.counts >= C.MAX_CODE, axis=0) / len(session.counts))
        if cohort.n >= 16:
            anova = ctx.cached(('anova', id(cohort)), lambda: A.oneway_anova(cohort.window_rates, cohort.directions))
            q = A.bh_fdr(anova['p'])
            significant = q < p['alpha']
        else:
            result.notes.append('保留 trial 太少（<16），跳过 ANOVA 与解码。')
            significant = np.zeros(len(ids), bool)

        tier_fig = self.tier_figure(ctx, tiers, counts, listed, order, crosstalk, unselected)
        quality_fig = self.quality_figure(session, tiers, mean_hz, sat, significant, p['alpha'], cohort.n >= 16)
        decoding, decoding_fig = [], None
        if p['decode'] and cohort.n >= 16:
            decoding = self.decode(ctx, tiers, p['ridge_lambda'])
            decoding_fig = self.decoding_figure(session, cohort, decoding)
        elif not p['decode']:
            result.notes.append('已关闭子集解码。')

        tier_rows = []
        for t in TIER_ORDER:
            m = tiers == t
            tier_rows.append(dict(tier=t, listed_in_ini=listed.get(t, ''), transmitted=int(m.sum()),
                                  mean_hz=round(float(mean_hz[m].mean()), 3) if m.any() else '',
                                  silent_lt_0p5hz=int((mean_hz[m] < 0.5).sum()),
                                  saturation_mean=round(float(sat[m].mean()), 6) if m.any() else '',
                                  anova_significant=int(significant[m].sum()),
                                  anova_significant_frac=round(float(significant[m].mean()), 4) if m.any() else ''))
        result.tables['tiers'] = tier_rows
        result.tables['columns'] = [dict(column=j + 1, electrode_id=int(ids[j]), tier=tiers[j],
                                         crosstalk=int(ids[j]) in crosstalk, mean_hz=round(float(mean_hz[j]), 4),
                                         saturation_frac=round(float(sat[j]), 6),
                                         anova_significant=bool(significant[j])) for j in range(len(ids))]
        p_values = anova['p'] if cohort.n >= 16 else np.full(len(ids), np.nan)
        q_values = q if cohort.n >= 16 else np.full(len(ids), np.nan)
        for row, pv, qv in zip(result.tables['columns'], p_values, q_values):
            row.update(anova_p=float(pv), anova_q=float(qv))
        positions = []
        for j, e in enumerate(int(x) for x in ids):
            pos = dict(column=j + 1, electrode_id=e, tier=tiers[j], crosstalk=e in crosstalk,
                       idgrid_column=(e - 1) // 128 + 1, idgrid_row=(e - 1) % 128 + 1)
            if ctx.layout is not None:
                r, c = ctx.layout['position'][e - 1]
                pos.update(layout_row=int(r) + 1, layout_column=chr(65 + int(c)))
            positions.append(pos)
        ini = ctx.channel_config_path.replace(chr(92), '/').split('/')[-1]
        result.add_figure(
            'Tiers on the array and in column order', tier_fig,
            method=(f'Input: {ini}. INI or the last saved Neuralinsight results_applied classification; '
                    'good / uncertain / bad / crosstalk lists are zero-based in the source and '
                    'converted to 1-based electrode ids. Each of the 512 stored columns gets the tier of its electrode '
                    'id (unlisted if in no list); tiers come from the labels intersected with the recording own '
                    'electrode ids, not from last_programmed_channels. Maps: physical workbook layout and 128x8 '
                    'electrode-id grid (electrode e at column (e-1)//128+1, row (e-1)%128+1), grey = not transmitted, '
                    'x = transmitted crosstalk electrode. Bottom strip: tier per stored column in acquisition order. '
                    'Programming-order check: the CLI programs good (ascending), then uncertain, then a random draw '
                    'of bad; "good block first" = good columns occupy columns 1..n_good; "tiers in CLI order" = tier '
                    'rank never decreases along the columns.'),
            data={'columns': positions, 'tiers': tier_rows})
        result.add_figure(
            'Per-tier channel quality', quality_fig,
            method=('Mean rate = mean count over all frames of the recording / configured window '
                    f'({session.frame_seconds * 1000:.0f} ms). Saturation = fraction of all frames at code >= 3 (2-bit '
                    'ceiling). Both shown per channel on a log axis with zeros drawn at 1e-3; black bar = tier median. '
                    'Right: one-way ANOVA across the 8 directions per channel on included-trial mean rates in the '
                    f'topbar analysis window [{cohort.criteria.window_start:g}, {cohort.criteria.window_end:g}] s from '
                    f'T_TRIAL_START; Benjamini-Hochberg over all 512 channels; significant = q < {p["alpha"]:g}; bar = '
                    '% of the tier channels significant (skipped with < 16 included trials).'),
            data={'channels': result.tables['columns'], 'tiers': tier_rows})
        if decoding_fig is not None:
            result.add_figure(
                'Decoding by channel subset', decoding_fig,
                method=('Channel subsets by tier: good; good+uncertain; all 512; bad only (subsets with < 2 channels '
                        'skipped). Direction: features = included-trial mean rates in the topbar analysis window; '
                        'one-hot ridge classifier on standardized features (penalty n_train * lambda, lambda = '
                        f'{p["ridge_lambda"]:g}), 5 contiguous chronological block folds, training drops blocks within '
                        '+-1 of the test range; metric = balanced accuracy over held-out trials (chance 1/8). Velocity: '
                        'bins of 100 ms from T_TRIAL_START + 0.2 s (10 bins per trial), velocity = cursor displacement '
                        '/ 0.1 s from interpolated active cursor positions; features = mean rate in the 200 ms window '
                        'ending at the bin right edge (>= 4 frames, gap <= 0.15 s); ridge regression with the same '
                        'block folds; metric = R^2 on held-out bins, averaged over x and y.'),
                data={'decoding': [{k: v for k, v in d.items() if not k.startswith('_')} for d in decoding]})
        if decoding:
            result.tables['decoding'] = [{k: v for k, v in d.items() if not k.startswith('_')} for d in decoding]
        result.summary.update(
            recording=session.label, config=ctx.channel_config_path, listed_in_ini=listed,
            transmitted_by_tier=counts, unselected_1_to_512=unselected,
            ini_good_not_transmitted=len(missing_good), crosstalk_transmitted=crosstalk, **order,
            anova_significant_by_tier={t: int(significant[tiers == t].sum()) for t in TIER_ORDER},
            decoding={d['subset']: dict(n_channels=d['n_channels'], direction_bacc=d['direction_bacc'],
                                        velocity_r2=d['velocity_r2']) for d in decoding})
        return result

    # ------------------------------------------------------------------ analysis
    def decode(self, ctx, tiers, lam):
        cohort = ctx.cohort
        x = np.nan_to_num(cohort.window_rates)
        y, blocks = cohort.directions, cohort.blocks
        ctx.progress('velocity bins')
        bins = ctx.cached(('velocity_bins_default', id(cohort)), lambda: velocity_bins(cohort))
        out = []
        for name, members in SUBSETS:
            ctx.check_cancel()
            cols = np.flatnonzero(np.isin(tiers, members))
            row = dict(subset=name, n_channels=int(len(cols)), direction_bacc=None, velocity_r2=None,
                       velocity_r2_x=None, velocity_r2_y=None, n_trials=cohort.n, n_bins=int(bins.n))
            if len(cols) >= 2:
                ctx.progress(f'decoding · {name}')
                scores, _, _, tested = A.ridge_classifier_cv(x[:, cols], y, blocks, lam=lam)
                if tested.any():
                    row['direction_bacc'] = A.balanced_accuracy(y[tested], scores[tested].argmax(1))
                if bins.n > 50:
                    pred, folds, _ = A.cv_velocity(bins.features[:, cols], bins.velocity, bins.trial, bins.sequence,
                                                   bins.block, models=('ridge',), lam_ridge=lam)
                    ok = folds >= 0
                    if ok.sum() > 10:
                        r2 = A.r2_score(bins.velocity[ok], pred['ridge'][ok])
                        row.update(velocity_r2=float(np.nanmean(r2)), velocity_r2_x=float(r2[0]),
                                   velocity_r2_y=float(r2[1]))
            out.append(row)
        return out

    # ------------------------------------------------------------------ figures
    def tier_figure(self, ctx, tiers, counts, listed, order, crosstalk, unselected):
        session = ctx.session
        ids = session.electrode_ids
        values = np.array([TIER_ORDER.index(t) for t in tiers], float)
        cmap = ListedColormap([TIER_COLORS[t] for t in TIER_ORDER])
        norm = BoundaryNorm(np.arange(-0.5, len(TIER_ORDER) + 0.5), cmap.N)
        layouts = ([ctx.layout] if ctx.layout is not None else []) + [None]
        fig = new_figure(11, 10)
        grid = fig.add_gridspec(2, len(layouts) + 1, height_ratios=[1, 0.09],
                                width_ratios=[1] * len(layouts) + [0.8])
        for i, layout in enumerate(layouts):
            ax = fig.add_subplot(grid[0, i])
            draw_channel_map(ax, ids, values, layout=layout, cmap=cmap, norm=norm, colorbar=False,
                             title='Physical layout' if layout is not None else 'Electrode ID layout')
            if crosstalk:
                xs, ys = marker_positions(layout, crosstalk)
                ax.scatter(xs, ys, marker='x', s=22, c='white', linewidths=1.6, zorder=3)
                ax.scatter(xs, ys, marker='x', s=16, c=style.INK, linewidths=0.8, zorder=4)
        side = fig.add_subplot(grid[0, len(layouts)])
        side.axis('off')
        handles = [Patch(color=TIER_COLORS[t], label=f'{t}: {counts[t]} transmitted'
                         + (f' / {listed[t]} listed' if t in listed else '')) for t in TIER_ORDER]
        handles.append(Patch(color=style.MISSING, label=f'not transmitted: {C.N_ELECTRODES - len(ids)}'))
        if crosstalk:
            from matplotlib.lines import Line2D
            handles.append(Line2D([], [], marker='x', ls='', color=style.INK, label=f'crosstalk: {len(crosstalk)}'))
        side.legend(handles=handles, loc='upper left', frameon=False, title='Channel config tier', fontsize=7.5)
        lines = ['Programming-order check', '(good asc → uncertain → bad)', '',
                 f'good block first   {"yes" if order["good_block_first"] else "no"}',
                 f'good ascending     {"yes" if order["good_ascending"] else "no"}',
                 f'tiers in CLI order {"yes" if order["tiers_in_cli_order"] else "no"}']
        if unselected:
            lines += ['', 'Record transmits electrodes', '1..512 (no selection):', 'overlap only, not a tiering']
        if crosstalk:
            lines += ['', 'crosstalk electrodes:'] + [', '.join(map(str, crosstalk[i:i + 6]))
                                                     for i in range(0, min(len(crosstalk), 30), 6)]
        side.text(0, 0.6, '\n'.join(lines), va='top', fontsize=7, family='monospace', transform=side.transAxes)

        ax = fig.add_subplot(grid[1, :])
        ax.imshow(values[None, :], aspect='auto', cmap=cmap, norm=norm, interpolation='nearest',
                  extent=[0.5, len(ids) + 0.5, 0, 1])
        ax.set_yticks([])
        ax.set_xlabel('stored column (acquisition order)')
        ax.set_title('Tier per stored column', fontsize=8)
        ax.set_xlim(0.5, len(ids) + 0.5)
        fig.suptitle(f'{session.label} · channel tiers from {ctx.channel_config_path.replace(chr(92), "/").split("/")[-1]}',
                     fontsize=9)
        return fig

    def quality_figure(self, session, tiers, mean_hz, sat, significant, alpha, have_anova):
        fig = new_figure(12, 4.8)
        grid = fig.add_gridspec(1, 3)
        present = [t for t in TIER_ORDER if np.any(tiers == t)]
        rng = np.random.default_rng(0)
        for col, (values, label, log) in enumerate(((mean_hz, 'mean rate / Hz (whole recording)', True),
                                                    (100 * sat, '% frames at code 3', True))):
            ax = fig.add_subplot(grid[0, col])
            for i, t in enumerate(present):
                v = values[tiers == t]
                shown = np.maximum(v, 1e-3) if log else v
                ax.scatter(i + rng.uniform(-0.25, 0.25, len(v)), shown, s=5, color=TIER_COLORS[t], alpha=0.6, lw=0)
                ax.plot([i - 0.32, i + 0.32], [np.median(shown)] * 2, color=style.INK, lw=1.4)
            if log:
                ax.set_yscale('log')
            ax.set_xticks(range(len(present)))
            ax.set_xticklabels([f'{t}\nn={int((tiers == t).sum())}' for t in present])
            ax.set_ylabel(label + ' · log, 0 → 1e-3', fontsize=7)
            ax.set_title(('Mean firing rate' if col == 0 else 'Count saturation') + ' per tier (bar = median)')
        ax = fig.add_subplot(grid[0, 2])
        frac = [100 * significant[tiers == t].mean() for t in present]
        ax.bar(range(len(present)), frac, color=[TIER_COLORS[t] for t in present])
        for i, t in enumerate(present):
            ax.text(i, frac[i], f'{int(significant[tiers == t].sum())}/{int((tiers == t).sum())}', ha='center',
                    va='bottom', fontsize=7)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present)
        ax.set_ylim(0, max(max(frac, default=0) * 1.2, 5))
        ax.set_ylabel('% channels direction-modulated')
        ax.set_title(f'One-way ANOVA on window rate, BH q < {alpha:g}' if have_anova else 'ANOVA skipped (too few trials)')
        fig.suptitle(f'{session.label} · channel quality by config tier', fontsize=9)
        return fig

    def decoding_figure(self, session, cohort, decoding):
        fig = new_figure(11, 4.6)
        grid = fig.add_gridspec(1, 2)
        names = [d['subset'] for d in decoding]
        x = np.arange(len(names))
        colors = ['#2e8b57', '#7fa650', style.INK2, '#c0392b']
        for col, (key, label, chance) in enumerate((('direction_bacc', 'direction balanced accuracy', 1 / 8),
                                                    ('velocity_r2', 'velocity R² (ridge, mean of x/y)', 0.0))):
            ax = fig.add_subplot(grid[0, col])
            vals = np.array([np.nan if d[key] is None else d[key] for d in decoding], float)
            ax.bar(x, np.nan_to_num(vals), color=colors[:len(names)])
            for i, (v, d) in enumerate(zip(vals, decoding)):
                text = 'n/a' if not np.isfinite(v) else f'{v:.3f}'
                ax.text(i, max(np.nan_to_num(v), 0), f'{text}\n{d["n_channels"]} ch', ha='center', va='bottom',
                        fontsize=7)
            ax.axhline(chance, color=style.GREY, ls='--', lw=0.8, label='chance' if col == 0 else 'R² = 0')
            ax.set_xticks(x)
            ax.set_xticklabels(names)
            finite = vals[np.isfinite(vals)]
            lo = min(0.0, float(finite.min()) * 1.2) if len(finite) else 0
            hi = max(float(finite.max()) * 1.25 if len(finite) else 1, chance * 1.5, 0.1)
            ax.set_ylim(lo, hi)
            ax.set_ylabel(label)
            ax.legend(frameon=False, loc='upper right')
            if col == 0:
                ax.set_title(f'Direction: one-hot ridge, 5 block folds (±1 block purge) · {cohort.n} trials')
            else:
                ax.set_title(f'Velocity: ridge on 100 ms bins, 200 ms trailing window · {decoding[0]["n_bins"]} bins')
        fig.suptitle(f'{session.label} · decoding with channel subsets defined by config', fontsize=9)
        return fig


SECTION = ChannelConfig()
