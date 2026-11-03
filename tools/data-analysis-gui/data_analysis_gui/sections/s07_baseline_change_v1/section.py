"""S07 v1 — eight-direction channel response change relative to the pre-trial baseline.

Per trial and channel: Δ = response[T_TRIAL_START + window_start, + window_end]
− baseline[T_TRIAL_START + baseline_start, + baseline_end] (topbar window,
section baseline; default −0.2..0 s). Only trials with neural frames in both
windows are kept (trial-paired). Directions are then averaged with equal trial
weight, as in macaque_data_analysis `sdk/psth/direction_delta.py`.

The heatmaps and maps are descriptive (no per-channel z-score; colour is ΔHz,
not significance). The significance panel tests each channel × direction with a
paired test across that direction's trials and controls BH-FDR over active
channels within each direction.
"""
from __future__ import annotations

import warnings

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

from ...core import analysis as A
from ...core import style
from ...core.channels import draw_channel_map
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure

UP, DOWN = '#c0392b', '#2471a3'
LOG_TICKS = np.array([0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500], float)


def signed_log_ticks(colorbar, max_abs):
    """Colour in signed log, label in original ΔHz."""
    ticks = LOG_TICKS[LOG_TICKS <= max_abs]
    if len(ticks) < 2:
        ticks = np.array([0, max_abs])
    ticks = np.r_[-ticks[:0:-1], ticks]
    colorbar.set_ticks(style.signed_log(ticks))
    colorbar.set_ticklabels([f'{v:g}' for v in ticks])


def paired_p(delta, test):
    """Two-sided paired test of Δ against 0 per column; columns without variation get p = 1."""
    p = np.ones(delta.shape[1])
    if len(delta) < 3:
        return p
    use = np.flatnonzero(np.any(delta != 0, 0) & (np.std(delta, 0) > 1e-12))
    if not len(use):
        return p
    if test == 'wilcoxon':
        # normal approximation: 'auto' falls back to a permutation null with ties at small n (minutes)
        res = stats.wilcoxon(delta[:, use], axis=0, zero_method='wilcox', method='approx')
    else:
        res = stats.ttest_1samp(delta[:, use], 0.0, axis=0)
    p[use] = np.where(np.isfinite(res.pvalue), res.pvalue, 1.0)
    return p


class BaselineChange(Section):
    id = 's07'
    title = '方向/时间调谐汇总与基线响应变化'
    description = ('先汇总 S05/S06 当前设置的 Direction/Time/Both 调谐比例及配置层级组成，'
                   '再展示原有逐 trial 配对基线变化、八方向布局与显著升降。单独计算也会运行来源统计。')
    source_sections = ('s05', 's06')
    params = (
        Param('baseline_start', '基线起点 / s（相对 T_TRIAL_START）', 'float', -0.2),
        Param('baseline_end', '基线终点 / s', 'float', 0.0),
        Param('test', '配对检验', 'choice', 'wilcoxon', choices=('wilcoxon', 't-test')),
        Param('alpha', '显著性水平 α（BH q）', 'float', 0.05, minimum=1e-6, maximum=0.5),
        Param('active_only', '热图只显示活跃通道', 'bool', False),
    )

    def run(self, ctx: AnalysisContext) -> SectionResult:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return self._run(ctx)

    def _run(self, ctx: AnalysisContext) -> SectionResult:
        p, cohort, session = ctx.params, ctx.cohort, ctx.session
        if p['baseline_end'] <= p['baseline_start']:
            raise ValueError('基线终点必须大于起点')
        if cohort.n < 8:
            raise ValueError(f'保留的 trial 太少（{cohort.n}）')
        result = SectionResult()
        from .overlap import append_overlap
        ctx.progress('S05 / S06 tuning overlap')
        groups = append_overlap(ctx, result)
        if groups is not None:
            from .directional import append_directional
            append_directional(ctx, result, groups)
        crit = cohort.criteria
        ctx.progress('response - baseline')
        response = cohort.window_rates
        base, nb = session.window_mean_rates(cohort.starts + p['baseline_start'], cohort.starts + p['baseline_end'])
        valid = np.isfinite(response).all(1) & (nb > 0)
        if valid.sum() < 8:
            raise ValueError('响应与基线窗口均有神经帧的 trial 太少')
        if (~valid).any():
            result.notes.append(f'{int((~valid).sum())} 个 trial 缺少有效基线或响应窗口，已剔除（只保留配对 trial）。')
        delta = (response - base)[valid]
        dirs = cohort.directions[valid]
        active = (np.r_[response[valid], base[valid]].std(0) > 1e-12)
        counts = np.bincount(dirs, minlength=8)
        means = np.full((8, delta.shape[1]), np.nan)
        for d in range(8):
            if counts[d]:
                means[d] = delta[dirs == d].mean(0)
        means[:, ~active] = np.nan

        ctx.progress(f'{p["test"]} per direction')
        pvals = np.ones((8, delta.shape[1]))
        qvals = np.ones_like(pvals)
        family = np.flatnonzero(active)
        for d in range(8):
            sel = dirs == d
            if sel.sum() >= 3:
                pvals[d, active] = paired_p(delta[sel][:, active], p['test'])
                qvals[d] = A.bh_fdr(pvals[d], family)
        sig = (qvals < p['alpha']) & active[None, :]
        up = sig & (means > 0)
        down = sig & (means < 0)

        win = f'{crit.window_start:g}–{crit.window_end:g} s'
        bwin = f'[{p["baseline_start"]:g}, {p["baseline_end"]:g}) s'
        banner = (f'{session.label} · {int(valid.sum())} paired trials · T_TRIAL_START response {win} '
                  f'minus baseline {bwin} · {int(active.sum())} active channels')
        ids = session.electrode_ids
        ctx.progress('figures')
        common = (
            f'Input: included trials with neural frames in both windows ({int(valid.sum())} of {cohort.n}). Per trial '
            f'and stored channel, response = mean rate (spike count / frame duration, Hz, frame-weighted) in '
            f'T_TRIAL_START + [{crit.window_start:g}, {crit.window_end:g}) s (topbar window) and baseline = mean rate '
            f'in T_TRIAL_START + [{p["baseline_start"]:g}, {p["baseline_end"]:g}) s; delta = response - baseline '
            f'(trial-paired). Mean delta per direction = equal-weight mean over that direction\'s trials. Active '
            f'channels = nonzero SD of the pooled response and baseline rates ({int(active.sum())}); inactive and '
            f'untransmitted channels are grey (NaN). No per-channel z-score. ')
        test_txt = (
            f'Test: per direction and active channel, two-sided paired '
            + ('Wilcoxon signed-rank test (zero differences dropped, normal approximation)' if p['test'] == 'wilcoxon'
               else 't-test') + ' of delta against 0 across the trials of that direction; Benjamini-Hochberg FDR '
            f'over active channels within each direction; significant up / down = q < {p["alpha"]:g} with mean delta '
            f'> 0 / < 0. ')
        order = np.argsort(ids)
        delta_rows = []
        for j in order:
            row = dict(electrode=int(ids[j]), stored_column=int(j) + 1, active=bool(active[j]))
            for d in range(8):
                row[f'delta_{d * 45}deg_hz'] = float(means[d, j])
                row[f'signed_log_{d * 45}deg'] = float(style.signed_log(means[d, j]))
            delta_rows.append(row)
        finite = np.abs(means[np.isfinite(means)])
        clip = float(np.percentile(finite, 99)) if finite.size else 0.0
        result.add_figure(
            'Channel x direction delta', self.matrix(means, ids, active, p, banner),
            method=common + (
                'Left: mean delta in Hz, rows = stored channels sorted by electrode id'
                + (' (active only)' if p['active_only'] else '') + f', columns = 8 directions, diverging colour '
                f'centred at 0 and clipped at +-{clip:.3g} Hz (99th percentile of |mean delta|). Right: same values '
                'coloured by sign(delta) x log10(1 + |delta| / 1 Hz), limit = signed log of the max |delta|, colourbar '
                'ticks labelled in raw Hz. Descriptive only; colour is not significance.'),
            data=dict(delta=delta_rows))
        for scale in ('linear', 'signed_log'):
            result.add_figure(
                f'Eight-direction layout delta ({scale})', self.maps(means, ids, ctx.layout, scale, banner),
                method=common + (
                    'Eight maps (one per direction) of mean delta on the '
                    + ('physical workbook layout (columns A-H, rows 1-128)' if ctx.layout is not None
                       else '128 x 8 electrode-id grid (no layout file)') + '; one shared colour scale across the '
                    'eight maps: ' + (f'linear, clipped at +-{clip:.3g} Hz (99th percentile)' if scale == 'linear'
                                      else 'signed log sign(d) x log10(1 + |d| / 1 Hz), ticks in raw Hz') + '.'),
                data=dict(delta=delta_rows))
        pop_trial = np.nanmean(delta[:, active], 1)
        pop_tbl = dict(direction_deg=np.arange(8) * 45, n_trials=counts,
                       population_mean_delta_hz=[pop_trial[dirs == d].mean() if counts[d] else np.nan
                                                 for d in range(8)],
                       sem_hz=[pop_trial[dirs == d].std(ddof=1) / np.sqrt(counts[d]) if counts[d] > 1 else 0
                               for d in range(8)],
                       channels_up=up.sum(1), channels_down=down.sum(1),
                       pct_up=100 * up.sum(1) / max(int(active.sum()), 1),
                       pct_down=100 * down.sum(1) / max(int(active.sum()), 1),
                       median_channel_delta_hz=[np.median(means[d, active]) for d in range(8)])
        act = np.flatnonzero(active)
        chan_rows = [dict(electrode=int(ids[j]), stored_column=int(j) + 1, direction_deg=d * 45,
                          mean_delta_hz=float(means[d, j]), p=float(pvals[d, j]), q=float(qvals[d, j]),
                          change='up' if up[d, j] else ('down' if down[d, j] else ''))
                     for j in act for d in range(8)]
        result.add_figure(
            'Significant change per direction',
            self.significance(delta, dirs, means, up, down, active, counts, p, banner),
            method=common + test_txt + (
                'Panels: (a) % of active channels significantly above (positive bars) / below (negative bars) '
                'baseline per direction, counts printed; (b) polar plot of the population mean delta per direction: '
                'per trial delta averaged over active channels, then mean +- SEM over the trials of each direction '
                '(radius shifted so the minimum is >= 0 when values are negative; dashed ring = 0 Hz; tick labels in '
                'Hz); (c) box plots over active channels of the per-direction mean delta (box = IQR, line = median, '
                'whiskers = 1.5 IQR, outliers hidden).'),
            data=dict(per_direction=pop_tbl, channel_direction=chan_rows))

        rows = []
        for j in range(len(ids)):
            row = dict(stored_column=j + 1, electrode=int(ids[j]), active=bool(active[j]),
                       baseline_hz=round(float(base[valid, j].mean()), 3),
                       response_hz=round(float(response[valid, j].mean()), 3))
            for d in range(8):
                row[f'delta_{d * 45}deg_hz'] = round(float(means[d, j]), 3) if np.isfinite(means[d, j]) else ''
                row[f'q_{d * 45}deg'] = float(qvals[d, j])
            row['up_directions'] = ' '.join(f'{d * 45}' for d in range(8) if up[d, j])
            row['down_directions'] = ' '.join(f'{d * 45}' for d in range(8) if down[d, j])
            rows.append(row)
        result.tables['baseline_change_channels'] = rows
        pop = np.nanmean(means, 1)
        result.summary.update(
            paired_trials=int(valid.sum()), active_channels=int(active.sum()), test=p['test'],
            trials_per_direction=counts.tolist(),
            channels_up_per_direction=up.sum(1).tolist(), channels_down_per_direction=down.sum(1).tolist(),
            channels_up_any_direction=int(up.any(0).sum()), channels_down_any_direction=int(down.any(0).sum()),
            population_mean_delta_hz_per_direction=[round(float(v), 3) for v in pop],
            max_abs_mean_delta_hz=round(float(np.nanmax(np.abs(means))), 2))
        return result

    # ------------------------------------------------------------------ figures
    def matrix(self, means, ids, active, p, banner):
        order = np.argsort(ids)
        if p['active_only']:
            order = order[active[order]]
        grid = means[:, order].T
        finite = np.abs(grid[np.isfinite(grid)])
        max_abs = max(float(finite.max()) if finite.size else 0, 1e-6)
        lin = max(float(np.percentile(finite, 99)) if finite.size else 0, 1e-6)
        cmap = colormaps['RdBu_r'].with_extremes(bad=style.MISSING)
        fig = new_figure(11, 9)
        axes = fig.subplots(1, 2)
        n = len(order)
        tick_rows = np.unique(np.r_[np.arange(0, n, max(n // 16, 1)), n - 1])
        for ax, scale in zip(axes, ('linear', 'signed_log')):
            if scale == 'linear':
                data, norm = grid, TwoSlopeNorm(0, -lin, lin)
            else:
                limit = float(style.signed_log(max_abs))
                data, norm = style.signed_log(grid), TwoSlopeNorm(0, -limit, limit)
            im = ax.imshow(np.ma.masked_invalid(data), aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
            ax.set_xticks(range(8))
            ax.set_xticklabels([f'{d * 45}' for d in range(8)])
            ax.set_xlabel('direction / °')
            ax.set_yticks(tick_rows)
            ax.set_yticklabels([str(int(ids[order[r]])) for r in tick_rows], fontsize=6)
            ax.set_ylabel(f'electrode id (stored columns{", active only" if p["active_only"] else ""}, n={n})')
            if scale == 'linear':
                cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02, extend='both')
                cb.set_label('response − baseline / Hz (clipped at 99th pct)', fontsize=7)
                ax.set_title('Linear ΔHz, centred at 0', fontsize=8)
            else:
                cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.02)
                signed_log_ticks(cb, max_abs)
                cb.set_label('response − baseline / Hz (signed-log colour)', fontsize=7)
                ax.set_title('sign(Δ)·log10(1 + |Δ| / 1 Hz)', fontsize=8)
            cb.ax.tick_params(labelsize=6)
        fig.suptitle(banner + '\nred = above baseline, blue = below, grey = inactive; descriptive, no z-score',
                     fontsize=8)
        return fig

    def maps(self, means, ids, layout, scale, banner):
        finite = np.abs(means[np.isfinite(means)])
        max_abs = max(float(finite.max()) if finite.size else 0, 1e-6)
        if scale == 'linear':
            lim = max(float(np.percentile(finite, 99)) if finite.size else 0, 1e-6)
            values, norm = means, TwoSlopeNorm(0, -lim, lim)
        else:
            limit = float(style.signed_log(max_abs))
            values, norm = style.signed_log(means), TwoSlopeNorm(0, -limit, limit)
        fig = new_figure(15, 8.5)
        axes = fig.subplots(1, 8, sharey=True)
        image = None
        for d, ax in enumerate(axes):
            image = draw_channel_map(ax, ids, values[d], layout=layout, cmap='RdBu_r', norm=norm, colorbar=False)
            ax.set_title(f'{d * 45}°', fontsize=9, color=style.direction_color(d))
            ax.set_xlabel('')
            if d:
                ax.set_ylabel('')
                ax.tick_params(labelleft=False)
        cb = fig.colorbar(image, ax=list(axes), shrink=0.7, pad=0.01, extend='both' if scale == 'linear' else 'neither')
        if scale == 'linear':
            cb.set_label('mean response − baseline / Hz (shared scale, clipped at 99th pct)', fontsize=7)
        else:
            signed_log_ticks(cb, max_abs)
            cb.set_label('mean response − baseline / Hz (signed-log colour)', fontsize=7)
        cb.ax.tick_params(labelsize=6)
        where = 'workbook layout' if layout is not None else '128×8 electrode-id grid (no layout file)'
        fig.suptitle(banner + f'\n{where} · grey = untransmitted or inactive · {scale.replace("_", " ")}',
                     fontsize=8)
        return fig

    def significance(self, delta, dirs, means, up, down, active, counts, p, banner):
        n_act = max(int(active.sum()), 1)
        fig = new_figure(14, 5.2)
        grid = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.9, 1.1])
        ax = fig.add_subplot(grid[0, 0])
        x = np.arange(8)
        fu, fd = 100 * up.sum(1) / n_act, 100 * down.sum(1) / n_act
        ax.bar(x, fu, color=UP, label='above baseline')
        ax.bar(x, -fd, color=DOWN, label='below baseline')
        for i in x:
            ax.text(i, fu[i], f'{up[i].sum()}', ha='center', va='bottom', fontsize=6)
            ax.text(i, -fd[i], f'{down[i].sum()}', ha='center', va='top', fontsize=6)
        ax.axhline(0, color=style.INK, lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{d * 45}°\nn={counts[d]}' for d in x], fontsize=6)
        lim = max(fu.max(), fd.max(), 1) * 1.25
        ax.set_ylim(-lim, lim)
        ax.set_ylabel('% of active channels (down plotted negative)')
        ax.set_title(f'Significant change per direction · {p["test"]}, BH q < {p["alpha"]:g} within direction',
                     fontsize=8)
        ax.legend(frameon=False, fontsize=6, loc='upper right')

        ax = fig.add_subplot(grid[0, 1], projection='polar')
        pop_trial = np.nanmean(delta[:, active], 1)            # per-trial population mean Δ
        mu = np.array([pop_trial[dirs == d].mean() if counts[d] else np.nan for d in range(8)])
        se = np.array([pop_trial[dirs == d].std(ddof=1) / np.sqrt(counts[d]) if counts[d] > 1 else 0
                       for d in range(8)])
        theta = np.radians(np.r_[np.arange(0, 360, 45), 0])
        offset = min(0.0, np.nanmin(mu - se)) * 1.2         # polar radius must stay ≥ 0: shift and label
        ax.plot(theta, np.r_[mu, mu[0]] - offset, color=style.INK, lw=1.2)
        ax.fill_between(theta, np.r_[mu - se, mu[0] - se[0]] - offset, np.r_[mu + se, mu[0] + se[0]] - offset,
                        color=style.GREY, alpha=0.4, lw=0)
        ax.scatter(theta[:-1], mu - offset, c=list(style.DIRECTION_COLORS), s=22, zorder=3)
        if offset < 0:
            ring = np.linspace(0, 2 * np.pi, 100)
            ax.plot(ring, np.full(100, -offset), color='#c76d4b', lw=0.8, ls='--')
        top = float(np.nanmax(mu + se)) - offset
        ax.set_ylim(0, top * 1.05 if top > 0 else 1)
        from matplotlib.ticker import MaxNLocator
        values = [v for v in MaxNLocator(5).tick_values(offset, top + offset) if offset <= v <= top + offset]
        ax.set_yticks([v - offset for v in values])
        ax.set_yticklabels([f'{v:g}' for v in values], fontsize=6)
        ax.set_xticks(theta[:-1])
        ax.set_xticklabels([f'{d}°' for d in range(0, 360, 45)], fontsize=6)
        ax.set_title('Population mean Δ / Hz (± SEM over trials)' + ('\ndashed = 0 Hz' if offset < 0 else ''),
                     fontsize=8)

        ax = fig.add_subplot(grid[0, 2])
        data = [means[d, active] for d in range(8)]
        parts = ax.boxplot(data, positions=x, widths=0.6, showfliers=False, patch_artist=True)
        for d, box in enumerate(parts['boxes']):
            box.set_facecolor(style.direction_color(d))
            box.set_alpha(0.6)
        for med in parts['medians']:
            med.set_color(style.INK)
        ax.axhline(0, color=style.INK, lw=0.6, ls=':')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{d * 45}' for d in x])
        ax.set_xlabel('direction / °')
        ax.set_ylabel('channel mean Δ / Hz')
        ax.set_title('Distribution over active channels (whiskers 1.5 IQR, no outliers)', fontsize=8)
        fig.suptitle(banner, fontsize=8)
        return fig


SECTION = BaselineChange()
