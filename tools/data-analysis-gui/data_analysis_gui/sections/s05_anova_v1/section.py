"""S05 v1 — per-channel one-way ANOVA over the eight directions.

Window rates come from the topbar analysis window. BH-FDR runs within active
channels only (silent channels carry no testable signal). The optional
gain-controlled ANOVA follows neural_behavior `tuning_curves.py`: each channel's
rate is residualized on the trial's leave-one-out total array rate before the
same F test (one extra covariate, df_w − 1).
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
from scipy import stats

from ...core import analysis as A
from ...core import style
from ...core.channels import draw_channel_map, id_grid, layout_grid
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure


def channel_anova(ctx: AnalysisContext):
    """Shared with S01: plain ANOVA on window rates, cached per cohort."""
    cohort = ctx.cohort
    x = cohort.window_rates
    ok = np.isfinite(x).all(1)
    if ok.all():
        stats_ = ctx.cached(('anova', id(cohort)), lambda: A.oneway_anova(x, cohort.directions))
    else:
        stats_ = ctx.cached(('anova_finite', id(cohort)), lambda: A.oneway_anova(x[ok], cohort.directions[ok]))
    return stats_, x[ok], cohort.directions[ok]


def gain_controlled(rate, direction):
    """Direction F on residuals after regressing out the leave-one-out total rate."""
    n, n_ch = rate.shape
    present = [g for g in range(8) if np.any(direction == g)]
    counts = np.array([np.sum(direction == g) for g in range(8)])
    grand = rate.mean(0)
    loo = rate.sum(1, keepdims=True) - rate
    lc = loo - loo.mean(0)
    beta = (lc * (rate - grand)).sum(0) / np.maximum((lc ** 2).sum(0), 1e-12)
    resid = rate - grand - lc * beta
    rmean = np.stack([resid[direction == g].mean(0) if counts[g] else np.zeros(n_ch) for g in range(8)])
    rssb = (counts[:, None] * rmean ** 2).sum(0)
    rssw = np.maximum((resid ** 2).sum(0) - rssb, 0)
    df_b, df_w = len(present) - 1, n - len(present) - 1
    valid = (rssw > 1e-12) & (df_b > 0) & (df_w > 0)
    F = np.zeros(n_ch)
    F[valid] = (rssb[valid] / df_b) / (rssw[valid] / df_w)
    p = np.ones(n_ch)
    p[valid] = stats.f.sf(F[valid], df_b, df_w)
    return F, p, (df_b, df_w)


def outline_channels(ax, layout, electrode_ids, mask, color='black', lw=0.7):
    """Draw a box around masked channels on a `draw_channel_map` axis."""
    values = np.where(mask, 1.0, np.nan)
    grid = layout_grid(layout, electrode_ids, values) if layout is not None else id_grid(electrode_ids, values)
    for r, c in zip(*np.nonzero(grid == 1)):
        ax.add_patch(Rectangle((c - 0.5, r + 0.5), 1, 1, fill=False, ec=color, lw=lw, zorder=3))


class Anova(Section):
    id = 's05'
    title = 'ANOVA'
    description = '逐通道八方向单因素 ANOVA、BH-FDR、可选扣除全局增益、调谐曲线与偏好方向'
    params = (
        Param('alpha', '显著性水平 α（BH q）', 'float', 0.05, minimum=1e-6, maximum=0.5),
        Param('gain_control', '同时做扣除全局增益的 ANOVA', 'bool', False),
        Param('map_value', '布局图数值', 'choice', '-log10 q', choices=('-log10 q', 'eta2', 'F')),
        Param('top_n', '调谐曲线通道数（按 F 排序）', 'int', 16, minimum=1, maximum=64),
    )

    def run(self, ctx: AnalysisContext) -> SectionResult:
        p, cohort, session = ctx.params, ctx.cohort, ctx.session
        result = SectionResult()
        if cohort.n < 16:
            result.tables['anova_channels'] = [dict(electrode=int(e), active=False, tested=False,
                significant=False, q='', p='', F='') for e in session.electrode_ids]
            result.summary.update(recording=session.label, trials=cohort.n, tested=False, alpha=p['alpha'])
            fig = new_figure(8, 4)
            fig.subplots().text(.5,.5,f'Not tested: {cohort.n} eligible trials (minimum 16)',ha='center')
            result.add_figure('ANOVA not tested',fig,method='Insufficient trials; excluded from tuning summary denominators.')
            result.notes.append('Trial 不足，不将未检验通道视为没有 tuning。')
            return result
        ctx.progress('ANOVA')
        st, x, dirs = channel_anova(ctx)
        if len(x) < cohort.n:
            result.notes.append(f'{cohort.n - len(x)} 个 trial 分析窗口无神经帧，已剔除。')
        active = x.std(0) > 1e-12
        family = np.flatnonzero(active)
        q = A.bh_fdr(st['p'], family)
        sig = active & (q < p['alpha'])
        mean = st['mean']
        with np.errstate(invalid='ignore'):
            preferred = np.where(active, np.nanargmax(np.where(np.isfinite(mean), mean, -np.inf), 0), -1)
            depth = np.nanmax(mean, 0) - np.nanmin(mean, 0)
        mean_hz = x.mean(0)
        gain = None
        if p['gain_control']:
            ctx.progress('gain-controlled ANOVA')
            Fg, pg, dfg = gain_controlled(x, dirs)
            qg = A.bh_fdr(pg, family)
            gain = dict(F=Fg, p=pg, q=qg, df=dfg, sig=active & (qg < p['alpha']))
        total = x[:, active].sum(1)
        tot_F, tot_p = stats.f_oneway(*[total[dirs == d] for d in range(8) if np.any(dirs == d)])

        info = dict(st=st, q=q, sig=sig, active=active, preferred=preferred, depth=depth, mean_hz=mean_hz,
                    gain=gain, alpha=p['alpha'], n=len(x), dirs=dirs, x=x)
        win = f'{cohort.criteria.window_start:g}–{cohort.criteria.window_end:g} s'
        banner = (f'{session.label} · {len(x)} trials · {int(active.sum())} active channels · window {win} · '
                  f'df = {st["df"][0]}, {st["df"][1]} · BH within active channels')
        crit = cohort.criteria
        base_txt = (f'Input: per included trial ({len(x)} trials with neural frames), mean rate in Hz (spike count / '
                    f'frame duration, frame-weighted) of every stored channel in T_TRIAL_START + '
                    f'[{crit.window_start:g}, {crit.window_end:g}) s. Active channels = nonzero across-trial SD '
                    f'({int(active.sum())}). Per channel one-way ANOVA of window rate on target direction (8 groups), '
                    f'df = ({st["df"][0]}, {st["df"][1]}); eta2 = SS_between / SS_total; p from the F distribution; '
                    f'q = Benjamini-Hochberg FDR over active channels only (inactive channels q = 1); significant = '
                    f'q < {p["alpha"]:g}. Preferred direction = direction with the highest mean rate; depth = max - min '
                    f'of the 8 direction means. ')
        gain_txt = ('Gain-controlled ANOVA: each channel rate is centred and regressed on the centred leave-one-out '
                    'total rate of the trial (sum of all other channels); the same direction F test is run on the '
                    'residuals with one extra covariate (df_within - 1); BH over active channels. ') if gain else ''
        ids = session.electrode_ids
        act = np.flatnonzero(active)
        chan = dict(stored_column=act + 1, electrode=ids[act], F=st['F'][act], p=st['p'][act], q=q[act],
                    neg_log10_q=-np.log10(np.clip(q[act], 1e-300, 1)), eta2=st['eta2'][act],
                    significant=sig[act], preferred_direction_deg=preferred[act] * 45, depth_hz=depth[act],
                    mean_hz=mean_hz[act])
        if gain:
            chan.update(F_gain=gain['F'][act], p_gain=gain['p'][act], q_gain=gain['q'][act],
                        significant_gain=gain['sig'][act])
        counts_tbl = dict(category=['active', 'significant'] + (['gain_controlled', 'both'] if gain else []),
                          channels=[int(active.sum()), int(sig.sum())] +
                          ([int(gain['sig'].sum()), int((gain['sig'] & sig).sum())] if gain else []))
        pref_tbl = dict(direction_deg=np.arange(8) * 45,
                        significant_channels=np.bincount(preferred[sig], minlength=8)[:8])
        result.add_figure(
            'ANOVA summary', self.summary_figure(info, banner),
            method=base_txt + gain_txt + (
                'Panels: (a) histogram of F over active channels on log-spaced bins, significant channels in red; '
                '(b) histogram of -log10 q (log count axis), dashed = alpha; (c) channel counts (active, significant'
                + (', gain-controlled significant, significant in both' if gain else '') + '); '
                '(d) polar histogram of the preferred direction of significant channels; '
                '(e) eta2 vs mean window rate (symlog x); '
                + ('(f) plain F vs gain-controlled F, blue = significant in both, red = plain only.' if gain else
                   '(f) F vs tuning depth (log y).')),
            data=dict(channels=chan, counts=counts_tbl, preferred_direction=pref_tbl))
        map_label = {'eta2': 'eta2', 'F': 'F'}.get(p['map_value'], '-log10 q')
        result.add_figure(
            'ANOVA channel maps', self.map_figure(info, ctx, p['map_value'], banner),
            method=base_txt + (
                f'Panels on the physical channel layout ({"workbook layout" if ctx.layout is not None else "128 x 8 electrode-id grid"}): '
                f'(a) {map_label} of active channels, significant channels boxed; (b) eta2, significant boxed; '
                '(c) preferred direction of significant channels only. Grey = untransmitted or inactive.'),
            data=dict(channels=chan))
        order = [j for j in np.argsort(-st['F']) if active[j]][:p['top_n']]
        tuning_rows = [dict(rank=r + 1, electrode=int(ids[j]), stored_column=j + 1, direction_deg=d * 45,
                            n_trials=int(st['counts'][d]), mean_hz=float(st['mean'][d, j]),
                            sem_hz=float(st['sem'][d, j]), overall_mean_hz=float(mean_hz[j]),
                            F=float(st['F'][j]), q=float(q[j]))
                       for r, j in enumerate(order) for d in range(8)]
        result.add_figure(
            'Top tuning curves', self.tuning_figure(info, ids, p['top_n'], banner),
            method=base_txt + (
                f'Top {len(order)} active channels by F. Each panel: mean window rate per direction +- SEM '
                '(SD with ddof = 1 / sqrt(n trials)); the 0 deg value is repeated at 360 deg to close the curve; '
                'dotted line = mean over all trials.'),
            data=dict(tuning=tuning_rows))

        ids = session.electrode_ids
        rows = []
        for j in range(len(ids)):
            row = dict(stored_column=j + 1, electrode=int(ids[j]), active=bool(active[j]),
                       F=round(float(st['F'][j]), 4), p=float(st['p'][j]), q=float(q[j]),
                       eta2=round(float(st['eta2'][j]), 5), significant=bool(sig[j]),
                       preferred_direction_deg=int(preferred[j]) * 45 if preferred[j] >= 0 else '',
                       depth_hz=round(float(depth[j]), 3) if np.isfinite(depth[j]) else '',
                       mean_hz=round(float(mean_hz[j]), 3))
            row.update({f'mean_{d * 45}deg_hz': round(float(mean[d, j]), 3) if np.isfinite(mean[d, j]) else ''
                        for d in range(8)})
            if gain:
                row.update(F_gain=round(float(gain['F'][j]), 4), p_gain=float(gain['p'][j]),
                           q_gain=float(gain['q'][j]), significant_gain=bool(gain['sig'][j]))
            rows.append(row)
        result.tables['anova_channels'] = rows
        result.summary.update(
            recording=session.label, tested=True, trials=len(x), active_channels=int(active.sum()), significant_channels=int(sig.sum()),
            alpha=p['alpha'], df=list(st['df']),
            top_electrodes_by_F=[int(ids[j]) for j in np.argsort(-st['F'])[:10]],
            preferred_direction_counts_deg={f'{d * 45}': int(np.sum(preferred[sig] == d)) for d in range(8)},
            total_rate_vs_direction_F=round(float(tot_F), 3), total_rate_vs_direction_p=float(tot_p))
        if gain:
            result.summary.update(significant_gain_controlled=int(gain['sig'].sum()),
                                  significant_both=int((gain['sig'] & sig).sum()))
        if tot_p < 0.05:
            result.notes.append(f'整体发放率随方向显著变化（F={tot_F:.2f}, p={tot_p:.3g}）：'
                                '部分"调谐"可能来自全局增益，建议勾选扣除全局增益的 ANOVA。')
        from ...core.channel_atlas import append_anova_atlas
        append_anova_atlas(result, info, ids, banner)
        return result

    # ------------------------------------------------------------------ figures
    def summary_figure(self, info, banner):
        st, q, sig, active, alpha, gain = info['st'], info['q'], info['sig'], info['active'], info['alpha'], info['gain']
        fig = new_figure(13, 7.5)
        grid = fig.add_gridspec(2, 3)
        F = st['F'][active]
        ax = fig.add_subplot(grid[0, 0])
        bins = np.logspace(np.log10(max(F[F > 0].min(), 1e-2)) if np.any(F > 0) else -2,
                           np.log10(max(F.max(), 1.1)), 40)
        ax.hist(F[~sig[active]], bins=bins, color=style.GREY, label='n.s.')
        ax.hist(F[sig[active]], bins=bins, color='#c76d4b', label=f'q < {alpha:g}')
        ax.set_xscale('log')
        ax.axvline(1, color=style.INK, lw=0.6, ls=':')
        ax.set_xlabel('F (log scale)')
        ax.set_ylabel('channels')
        ax.set_title('F distribution (active channels)')
        ax.legend(frameon=False)

        ax = fig.add_subplot(grid[0, 1])
        lq = -np.log10(np.clip(q[active], 1e-300, 1))
        ax.hist(lq, bins=40, color='#30638e')
        ax.axvline(-np.log10(alpha), color='#c76d4b', ls='--', lw=0.8, label=f'q = {alpha:g}')
        ax.set_yscale('log')
        ax.set_xlabel('−log10 q (BH)')
        ax.set_ylabel('channels (log)')
        ax.set_title('FDR-corrected significance')
        ax.legend(frameon=False)

        ax = fig.add_subplot(grid[0, 2])
        labels = ['active', f'q<{alpha:g}']
        values = [int(active.sum()), int(sig.sum())]
        colors = [style.GREY, '#c76d4b']
        if gain:
            labels += ['gain-ctrl', 'both']
            values += [int(gain['sig'].sum()), int((gain['sig'] & sig).sum())]
            colors += ['#00798c', '#30638e']
        ax.bar(range(len(values)), values, color=colors)
        for i, v in enumerate(values):
            ax.text(i, v, f'{v}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('channels')
        ax.set_title(f'Significant channels: {int(sig.sum())}/{int(active.sum())} '
                     f'({100 * sig.sum() / max(active.sum(), 1):.0f}%)')

        ax = fig.add_subplot(grid[1, 0], projection='polar')
        counts = np.bincount(info['preferred'][sig], minlength=8)
        theta = np.radians(np.arange(0, 360, 45))
        ax.bar(theta, counts, width=np.radians(40), color=list(style.DIRECTION_COLORS), edgecolor='white')
        ax.set_xticks(theta)
        ax.set_xticklabels([f'{d}°' for d in range(0, 360, 45)], fontsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.set_title(f'Preferred direction of significant channels (n={int(sig.sum())})', fontsize=8)

        ax = fig.add_subplot(grid[1, 1])
        a = active
        ax.scatter(info['mean_hz'][a & ~sig], st['eta2'][a & ~sig], s=6, color=style.GREY, label='n.s.')
        ax.scatter(info['mean_hz'][a & sig], st['eta2'][a & sig], s=8, color='#c76d4b', label='significant')
        ax.set_xscale('symlog', linthresh=1)
        ax.set_xlabel('mean window rate / Hz')
        ax.set_ylabel('η² (direction)')
        ax.set_title('Effect size vs firing rate')
        ax.legend(frameon=False)

        ax = fig.add_subplot(grid[1, 2])
        if gain:
            Fg = gain['F'][a]
            ax.scatter(F, Fg, s=6, c=np.where(sig[a] & gain['sig'][a], '#30638e',
                                              np.where(sig[a], '#c76d4b', style.GREY)))
            lim = max(F.max(), Fg.max()) * 1.05
            ax.plot([0, lim], [0, lim], color=style.INK, lw=0.6, ls='--')
            ax.set_xlabel('F (plain)')
            ax.set_ylabel('F (gain-controlled)')
            ax.set_title('Gain control: blue = significant in both, red = plain only', fontsize=8)
        else:
            ax.scatter(info['depth'][a], F, s=6, c=np.where(sig[a], '#c76d4b', style.GREY))
            ax.set_xlabel('tuning depth (max − min direction mean) / Hz')
            ax.set_ylabel('F')
            ax.set_yscale('log')
            ax.set_title('F vs tuning depth')
        fig.suptitle(banner, fontsize=9)
        return fig

    def map_figure(self, info, ctx, map_value, banner):
        st, q, sig, active = info['st'], info['q'], info['sig'], info['active']
        ids = ctx.session.electrode_ids
        fig = new_figure(13, 9)
        axes = fig.subplots(1, 3)
        if map_value == 'eta2':
            values, label = st['eta2'], 'η²'
        elif map_value == 'F':
            values, label = st['F'], 'F'
        else:
            values, label = -np.log10(np.clip(q, 1e-300, 1)), '−log10 q'
        values = np.where(active, values, np.nan)
        draw_channel_map(axes[0], ids, values, layout=ctx.layout, cmap='magma_r', title=f'{label} (boxed: q < {info["alpha"]:g})',
                         cbar_label=label)
        outline_channels(axes[0], ctx.layout, ids, sig, color='#00798c')
        draw_channel_map(axes[1], ids, np.where(active, st['eta2'], np.nan), layout=ctx.layout, cmap='viridis',
                         title='η² (boxed: significant)', cbar_label='η²')
        outline_channels(axes[1], ctx.layout, ids, sig, color='white')
        pref = np.where(sig, info['preferred'], np.nan).astype(float)
        cmap = ListedColormap(list(style.DIRECTION_COLORS))
        image = draw_channel_map(axes[2], ids, pref, layout=ctx.layout, cmap=cmap,
                                 norm=BoundaryNorm(np.arange(-0.5, 8.5), 8), title='Preferred direction (significant only)',
                                 colorbar=False)
        cb = fig.colorbar(image, ax=axes[2], fraction=0.08, pad=0.03, ticks=range(8))
        cb.ax.set_yticklabels([f'{d * 45}°' for d in range(8)], fontsize=6)
        fig.suptitle(banner + ' · grey = untransmitted or inactive', fontsize=9)
        return fig

    def tuning_figure(self, info, ids, top_n, banner):
        st, q, active = info['st'], info['q'], info['active']
        order = [j for j in np.argsort(-st['F']) if active[j]][:top_n]
        n_cols = min(4, max(len(order), 1))
        n_rows = int(np.ceil(max(len(order), 1) / n_cols))
        fig = new_figure(3.0 * n_cols + 1, 2.3 * n_rows + 0.8)
        axes = np.atleast_1d(fig.subplots(n_rows, n_cols, squeeze=False)).ravel()
        x = np.arange(8)
        for ax, j in zip(axes, order):
            m, se = st['mean'][:, j], st['sem'][:, j]
            ax.plot(np.r_[x, 8], np.r_[m, m[0]], color=style.INK, lw=0.8, zorder=1)
            ax.errorbar(x, m, yerr=se, fmt='none', ecolor=style.INK2, elinewidth=0.8, capsize=2)
            ax.scatter(x, m, c=list(style.DIRECTION_COLORS), s=22, zorder=3)
            ax.scatter([8], [m[0]], color=style.DIRECTION_COLORS[0], s=22, zorder=3, alpha=0.4)
            ax.axhline(info['mean_hz'][j], color=style.GREY, lw=0.6, ls=':')
            ax.set_xticks(np.r_[x, 8])
            ax.set_xticklabels([f'{d * 45}' for d in range(9)], fontsize=6)
            ax.set_title(f'E{int(ids[j])} · F={st["F"][j]:.1f} · q={q[j]:.1e}', fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in axes[len(order):]:
            ax.axis('off')
        fig.supxlabel('direction / °  (dotted = mean over trials)', fontsize=8)
        fig.supylabel('window rate / Hz (mean ± SEM)', fontsize=8)
        fig.suptitle(banner + f' · top {len(order)} by F', fontsize=8)
        return fig


SECTION = Anova()
