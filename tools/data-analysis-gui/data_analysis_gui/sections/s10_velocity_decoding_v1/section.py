"""S10 v1 — continuous velocity decoding with selectable decoders.

Bins follow `core.features.velocity_bins` (cursor-displacement velocity, trailing
neural window). Every decoder in `data_analysis_gui.decoders` selected by the `decoders`
param is fitted per chronological block fold (±1 block purge) by
`decoders.cv_decode`; every number here is held-out. Decoders with a fixed
contract bin (SDK: 20 ms) get native 20 ms bins over the same trial windows and
their predictions are averaged onto the evaluation bins. Metrics compare all
selected decoders on the bins every decoder predicted (common bins).
Trajectories are reconstructed by integrating decoded velocity from the cursor
position at the left edge of the first bin of each trial.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LogNorm

from ... import decoders as D
from ...core import analysis as A
from ...core import style
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure

GT_COLOR = style.MODEL_COLORS['gt']


def spec_of(p) -> D.BinSpec:
    return D.BinSpec(offset=float(p['offset']), bin_s=float(p['bin']),
                     n_bins=int(p['n_bins']) if p['until'] == 'window' else None, until=p['until'], sqrt=bool(p['sqrt']))


def metrics(y, pred):
    r2 = A.r2_score(y, pred)
    return dict(r2_x=float(r2[0]), r2_y=float(r2[1]), r2_mean=float(np.mean(r2)),
                r2_xy=float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean(0)) ** 2)),
                rmse_x=float(A.rmse(y, pred)[0]), rmse_y=float(A.rmse(y, pred)[1]),
                r_x=float(A.pearson(y, pred)[0]), r_y=float(A.pearson(y, pred)[1]))


def block_bootstrap(y, pred, blocks, n_boot, seed=11):
    """Percentile CI of mean R² resampling whole blocks with replacement."""
    rng = np.random.default_rng(seed)
    unique = np.unique(blocks)
    index = {b: np.flatnonzero(blocks == b) for b in unique}
    out = np.empty(n_boot)
    for i in range(n_boot):
        rows = np.concatenate([index[b] for b in rng.choice(unique, len(unique))])
        out[i] = np.mean(A.r2_score(y[rows], pred[rows]))
    return np.percentile(out, [2.5, 97.5])


def paired_bootstrap(y, pred, names, blocks, n_boot, seed=12):
    """Same resampled blocks for every decoder: mean-R² CI per decoder and ΔR² vs the best decoder."""
    point = {m: float(np.mean(A.r2_score(y, pred[m]))) for m in names}
    best = max(names, key=point.get)
    unique = np.unique(blocks)
    index = {b: np.flatnonzero(blocks == b) for b in unique}
    rng = np.random.default_rng(seed)
    draws = {m: np.empty(n_boot) for m in names}
    for i in range(n_boot):
        rows = np.concatenate([index[b] for b in rng.choice(unique, len(unique))])
        for m in names:
            draws[m][i] = np.mean(A.r2_score(y[rows], pred[m][rows]))
    rows = []
    for m in names:
        delta = draws[m] - draws[best]
        ci = np.percentile(draws[m], [2.5, 97.5]) if n_boot else (np.nan, np.nan)
        dci = np.percentile(delta, [2.5, 97.5]) if n_boot else (np.nan, np.nan)
        rows.append(dict(decoder=m, label=D.label(m), r2_mean=point[m], ci_low=float(ci[0]), ci_high=float(ci[1]),
                         best=best, delta_vs_best=point[m] - point[best], delta_ci_low=float(dci[0]),
                         delta_ci_high=float(dci[1]),
                         p_delta_ge_0=float(np.mean(delta >= 0)) if n_boot and m != best else np.nan))
    return rows, best


def integrate(vb, values, bin_s):
    """Per trial: positions at bin edges from the first-bin position (trials with gaps skipped)."""
    out = {}
    for t in np.unique(vb.trial):
        rows = np.flatnonzero(vb.trial == t)
        if len(np.unique(vb.sequence[rows])) != 1 or vb.bin[rows][0] != 0 or not np.isfinite(values[rows]).all():
            continue
        start = vb.position[rows[0]]
        out[int(t)] = np.vstack([start, start + np.cumsum(values[rows], 0) * bin_s])
    return out


class VelocityDecoding(Section):
    id = 's10'
    title = '速度解码 Overview + examples'
    description = ('Ridge 与 Vel-KF的 block CV：'
                   'R²/RMSE/r、每折结果、速度曲线示例、积分轨迹与分方向 R²')
    params = (
        Param('bin', 'bin 宽 / s', 'float', 0.1, minimum=0.02, maximum=0.5),
        Param('offset', '首个 bin 相对 T_TRIAL_START / s', 'float', 0.2, minimum=0.0),
        Param('n_bins', '每 trial bin 数（until=window）', 'int', 10, minimum=1, maximum=200),
        Param('until', 'bin 截止', 'choice', 'window', choices=('window', 'end', 'acquire')),
        Param('sqrt', '使用 sqrt(rate) 特征', 'bool', False),
        Param('lam_ridge', 'Ridge λ（惩罚 n·λ）', 'float', 1.0, minimum=1e-8),
        Param('lam_kf', 'KF 最小二乘 λ', 'float', 1e-4, minimum=0.0),
        Param('q_shrinkage', 'KF 观测协方差收缩 α', 'float', 0.5, minimum=0.0, maximum=1.0),
        Param('n_boot', 'block bootstrap 次数（0=不计算 CI）', 'int', 200, minimum=0, maximum=5000),
        Param('n_examples', '示例 trial 数', 'int', 6, minimum=1, maximum=16),
    )

    def run(self, ctx: AnalysisContext) -> SectionResult:
        p, session = ctx.params, ctx.session
        names = ('ridge', 'velocity_kf')
        result = SectionResult()
        spec = spec_of(p)
        ctx.progress('velocity bins')
        vb = D.build_data(ctx.cohort, spec, cache=ctx.cached)
        if vb.n < 50 or len(np.unique(vb.block)) < 4:
            raise ValueError('有效速度 bin 太少（需 ≥50 个 bin、≥4 个 block）')
        ctx.progress('cross-validated decoders: ' + ', '.join(names))
        cv = D.section_cv(ctx, ctx.cohort, names, spec, p, data=vb)
        pred = {m: cv['results'][m]['pred'] for m in names}
        folds = cv['results'][names[0]]['folds']
        y = vb.velocity
        tested = (folds >= 0) & np.all([np.isfinite(pred[m]).all(1) for m in names], axis=0)
        coverage = {m: int(np.isfinite(pred[m]).all(1).sum()) for m in names}
        stats = {}
        for m in names:
            stats[m] = metrics(y[tested], pred[m][tested])
            if p['n_boot']:
                stats[m]['r2_mean_ci'] = block_bootstrap(y[tested], pred[m][tested], vb.block[tested],
                                                         p['n_boot']).tolist()
        per_fold = []
        for f in np.unique(folds[folds >= 0]):
            sel = tested & (folds == f)
            if sel.sum() < 3:
                continue
            row = dict(fold=int(f), n_bins=int(sel.sum()), blocks=f'{vb.block[sel].min()}-{vb.block[sel].max()}')
            for m in names:
                r2 = A.r2_score(y[sel], pred[m][sel])
                row.update({f'{m}_r2_x': float(r2[0]), f'{m}_r2_y': float(r2[1]), f'{m}_r2_mean': float(r2.mean())})
                fi = {r['fold']: r for r in cv['results'][m]['info']}.get(int(f), {})
                for k in ('kf_spectral_radius', 'spectral_radius'):
                    if fi.get(k) is not None:
                        row[f'{m}_spectral_radius'] = fi[k]
            per_fold.append(row)
        per_dir = []
        for d in range(8):
            sel = tested & (vb.direction == d)
            if sel.sum() < 3:
                continue
            row = dict(direction_deg=d * 45, n_bins=int(sel.sum()))
            for m in names:
                e = y[sel] - pred[m][sel]
                row[f'{m}_r2_xy'] = float(1 - np.sum(e ** 2) / np.sum((y[sel] - y[sel].mean(0)) ** 2))
                row[f'{m}_rmse'] = float(np.sqrt(np.mean(np.sum(e ** 2, 1))))
            per_dir.append(row)
        decoder_rows = []
        for m in names:
            res = cv['results'][m]
            nat = cv['native'].get(m)
            decoder_rows.append(dict(
                decoder=m, label=D.label(m), source=D.source_text(m), cv_seconds=res['seconds'],
                fit_seconds=round(sum(r['fit_seconds'] for r in res['info']), 2),
                predict_seconds=round(sum(r['predict_seconds'] for r in res['info']), 2),
                native_bins=int(nat.n) if nat is not None else '', eval_bins_predicted=coverage[m],
                common_bins=int(tested.sum()), **{k: v for k, v in stats[m].items() if k != 'r2_mean_ci'}))
        ctx.check_cancel()
        ctx.progress('figures')
        base = self.base_method(vb, spec, names, p, tested) + ' ' + D.cohort_rule_text(ctx.cohort, spec)
        result.add_figure('Velocity decoding overview',
                          self.overview(session, vb, y, pred, tested, stats, per_fold, names, p),
                          method=base + ' Panels: (top) held-out R² per axis and mean (R² = 1 − SSE/SST per axis on '
                          'the common held-out bins; error bar = 95% percentile CI of mean R² from block bootstrap, '
                          f'{p["n_boot"]} resamples of whole blocks); RMSE per axis = sqrt(mean squared error); Pearson '
                          'r per axis; per-fold R² (x solid, y dashed) on each fold\'s common test bins. (bottom) '
                          'log-density 2-D histograms (60×60 bins, ±1.1×99.5th percentile |v|) of GT vs decoded '
                          'velocity per axis with the identity line.',
                          data=dict(metrics=self.metric_rows(stats, names), per_fold=per_fold,
                                    heldout_bins=self.bin_table(vb, pred, tested, names)))
        result.add_figure('Velocity traces (examples)', self.traces(session, ctx.cohort, vb, pred, names, p),
                          method=base + f' Up to {p["n_examples"]} trials evenly spaced through the session; GT and '
                          'every decoder\'s held-out velocity per bin, plotted at the bin right edge relative to '
                          'T_TRIAL_START. Bins a decoder did not predict are gaps.',
                          data=dict(traces=self.trace_rows(ctx.cohort, vb, pred, names, p)))
        traj_fig, traj_rows = self.trajectories(session, vb, pred, per_dir, names, p)
        result.add_figure('Integrated trajectories by direction', traj_fig,
                          method=base + ' Top: positions integrated per trial as p₀ + Σ v·Δt from the cursor '
                          'position at the left edge of bin 0, only for trials whose bins form one gap-free sequence '
                          'starting at bin 0 and that the decoder predicted fully; thin lines = up to 40 trials per '
                          'direction, thick = mean over trials of the first L points (L = shortest trial). Bottom: '
                          'per-direction R² = 1 − SSE/SST with x and y pooled and SST around the direction\'s own mean '
                          '(hence lower than pooled R²); RMSE = sqrt(mean ‖v − v̂‖²) on common held-out bins.',
                          data=dict(mean_paths=traj_rows, per_direction=per_dir))
        ctx.progress('decoder comparison')
        n_cmp = int(p['n_boot']) or 200
        paired, best = paired_bootstrap(y[tested], {m: pred[m][tested] for m in names}, names, vb.block[tested],
                                        n_cmp)
        cmp_fig, cmp_data = self.comparison(session, ctx.cohort, vb, y, pred, tested, stats, per_fold, per_dir, paired,
                                            best, decoder_rows, names, p)
        result.add_figure('Decoder comparison', cmp_fig,
                          method=base + ' Decoder comparison, all on the same common held-out bins: (a) R² per axis and '
                          'mean with 95% percentile CI of mean R² from block bootstrap; (b) RMSE per axis; (c) total CV '
                          'wall time (fit + predict over 5 folds, log scale); (d) mean R² per fold, lines join the same '
                          f'fold across decoders; (e) ΔR² = mean R²(decoder) − mean R²(best = highest point estimate, '
                          f'{D.label(best)}) with a paired 95% CI: {n_cmp} block-bootstrap resamples, every decoder '
                          'scored on the same resampled blocks in each draw; (f) per-direction R² (x,y pooled, SST '
                          'around the direction mean); (g, h) one example trial (median-length trial of the picked '
                          'examples), GT and every decoder.', data=cmp_data)
        result.tables['decoder_comparison'] = paired
        result.summary.update(recording=session.label, decoders=list(names), n_bins=int(vb.n),
                              n_common_tested_bins=int(tested.sum()), n_trials=int(len(np.unique(vb.trial))),
                              bin_s=p['bin'], offset_s=p['offset'], until=p['until'],
                              feature='sqrt(Hz)' if p['sqrt'] else 'Hz', bin_exclusions=vb.extra.get('exclusions'),
                              **{f'{m}_{k}': v for m in names for k, v in stats[m].items()},
                              **{f'{m}_cv_seconds': cv['results'][m]['seconds'] for m in names})
        result.tables['per_fold'] = per_fold
        result.tables['per_direction'] = per_dir
        result.tables['metrics'] = self.metric_rows(stats, names)
        result.tables['decoders'] = decoder_rows
        result.tables['fold_info'] = [dict(decoder=m, **{k: v for k, v in r.items() if np.isscalar(v) or v is None})
                                      for m in names for r in cv['results'][m]['info']]
        return result

    # ------------------------------------------------------------------ method / data helpers
    @staticmethod
    def base_method(vb, spec, names, p, tested):
        return (f'Inputs: included trials of this recording; evaluation bins of {spec.bin_s * 1000:g} ms from '
                f'T_TRIAL_START+{spec.offset:g} s ({spec.until}'
                + (f', {spec.n_bins} bins/trial' if spec.n_bins else '') + '); GT velocity = (p(right) − p(left))/Δ '
                f'from active-cursor positions linearly interpolated within the trial (gap ≤ '
                f'{vb.extra["cursor_gap"]:g} s); features = mean rate (count/0.020 s) of the 20 ms frames received in '
                f'the trailing {vb.extra["neural_window"] * 1000:g} ms window, '
                + ('sqrt-transformed, ' if spec.sqrt else '') + 'all transmitted electrodes. Folds: 5 chronological '
                'block groups; training drops blocks within ±1 of the test group; each decoder normalises with '
                'training-row statistics only. Metrics use the ' + f'{int(tested.sum())} held-out bins predicted by every '
                'selected decoder. Decoders: ' + D.describe(names, p, spec))

    @staticmethod
    def metric_rows(stats, names):
        return [dict(decoder=m, label=D.label(m), **{k: v for k, v in stats[m].items() if k != 'r2_mean_ci'},
                     **({'ci_low': stats[m]['r2_mean_ci'][0], 'ci_high': stats[m]['r2_mean_ci'][1]}
                        if 'r2_mean_ci' in stats[m] else {})) for m in names]

    @staticmethod
    def bin_table(vb, pred, tested, names):
        rows = np.flatnonzero(tested)
        out = dict(trial=vb.trial[rows], bin=vb.bin[rows], block=vb.block[rows], direction_deg=vb.direction[rows] * 45,
                   gt_vx=vb.velocity[rows, 0], gt_vy=vb.velocity[rows, 1])
        for m in names:
            out[f'{m}_vx'], out[f'{m}_vy'] = pred[m][rows, 0], pred[m][rows, 1]
        return out

    @staticmethod
    def picks(vb, p):
        trials = np.unique(vb.trial)
        n = min(p['n_examples'], len(trials))
        return trials[np.linspace(0, len(trials) - 1, n).round().astype(int)]

    def trace_rows(self, cohort, vb, pred, names, p):
        rows = []
        for t in self.picks(vb, p):
            for i in np.flatnonzero(vb.trial == t):
                row = dict(trial_id=int(cohort.trial_ids[t]), t_right_edge_s=p['offset'] + (vb.bin[i] + 1) * p['bin'],
                           gt_vx=float(vb.velocity[i, 0]), gt_vy=float(vb.velocity[i, 1]))
                for m in names:
                    row[f'{m}_vx'], row[f'{m}_vy'] = float(pred[m][i, 0]), float(pred[m][i, 1])
                rows.append(row)
        return rows

    # ------------------------------------------------------------------ figures
    def overview(self, session, vb, y, pred, tested, stats, per_fold, names, p):
        k = len(names)
        hist_rows = int(np.ceil(2 * k / 4))
        fig = new_figure(13, 4.2 * (1 + hist_rows))
        grid = fig.add_gridspec(1 + hist_rows, 4)
        width = 0.8 / k
        for col, (title, keys, ylabel) in enumerate((
                ('Held-out R²', ('r2_x', 'r2_y', 'r2_mean'), 'R²'),
                ('RMSE', ('rmse_x', 'rmse_y'), 'RMSE / task units·s⁻¹'),
                ('Pearson r', ('r_x', 'r_y'), 'r'))):
            ax = fig.add_subplot(grid[0, col])
            xs = np.arange(len(keys))
            for j, m in enumerate(names):
                off = (j - (k - 1) / 2) * width
                vals = [stats[m][q] for q in keys]
                bars = ax.bar(xs + off, vals, width, color=D.COLORS[m], label=D.label(m))
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}' if abs(v) < 10 else f'{v:.0f}',
                            ha='center', va='bottom', fontsize=5.5)
                if col == 0 and 'r2_mean_ci' in stats[m]:
                    lo, hi = stats[m]['r2_mean_ci']
                    ax.errorbar(2 + off, stats[m]['r2_mean'],
                                yerr=[[stats[m]['r2_mean'] - lo], [hi - stats[m]['r2_mean']]],
                                color=style.INK, capsize=3, lw=0.8)
            ax.set_xticks(xs, [q.split('_')[1] for q in keys])
            ax.axhline(0, color=style.GREY, lw=0.5)
            ax.set_ylabel(ylabel)
            ax.set_title(title + (' (mean: 95% block-bootstrap CI)' if col == 0 and p['n_boot'] else ''), fontsize=8)
            ax.margins(y=0.15)
            if col == 0:
                ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=min(k, 3), fontsize=6)

        ax = fig.add_subplot(grid[0, 3])
        xs = np.arange(len(per_fold))
        for m in names:
            for axis, mk in (('x', 'o'), ('y', 's')):
                ax.plot(xs, [f[f'{m}_r2_{axis}'] for f in per_fold], marker=mk, ms=4, lw=1,
                        color=D.COLORS[m], ls='-' if axis == 'x' else '--', label=f'{D.label(m)} {axis}')
        ax.set_xticks(xs, [f'F{f["fold"] + 1}\nb{f["blocks"]}' for f in per_fold], fontsize=6)
        ax.axhline(0, color=style.GREY, lw=0.5)
        ax.set_ylabel('R²')
        ax.set_title('Per-fold R²')
        ax.legend(frameon=False, fontsize=5, ncol=2)

        lim = np.nanpercentile(np.abs(y[tested]), 99.5) * 1.1
        for j, m in enumerate(names):
            for axis in range(2):
                cell = 2 * j + axis
                ax = fig.add_subplot(grid[1 + cell // 4, cell % 4])
                ax.hist2d(y[tested, axis], pred[m][tested, axis], bins=60, range=[[-lim, lim], [-lim, lim]],
                          cmap='Greys', cmin=1, norm=LogNorm())
                ax.plot([-lim, lim], [-lim, lim], color=D.COLORS[m], lw=0.8)
                ax.set_xlabel(f'GT v{"xy"[axis]}')
                ax.set_ylabel(f'{D.label(m)} v{"xy"[axis]}', fontsize=7)
                ax.set_title(f'{D.label(m)} · v{"xy"[axis]} · R² {stats[m]["r2_" + "xy"[axis]]:.2f} · '
                             f'r {stats[m]["r_" + "xy"[axis]]:.2f}', fontsize=7)
                ax.set_aspect('equal')
        fig.suptitle(f'{session.label} · held-out velocity decoding · {int(tested.sum())} common bins of '
                     f'{p["bin"] * 1000:.0f} ms from +{p["offset"]:g} s ({p["until"]}) · '
                     f'{len(np.unique(vb.trial))} trials', fontsize=10)
        return fig

    def traces(self, session, cohort, vb, pred, names, p):
        picks = self.picks(vb, p)
        n = len(picks)
        cols = min(n, 3)
        rows = int(np.ceil(n / cols))
        fig = new_figure(13, 2.6 * rows + 1.0)
        grid = fig.add_gridspec(rows, cols)
        for j, t in enumerate(picks):
            sub = grid[j // cols, j % cols].subgridspec(2, 1, hspace=0.05)
            sel = np.flatnonzero(vb.trial == t)
            tt = p['offset'] + (vb.bin[sel] + 1) * p['bin']
            for axis in range(2):
                ax = fig.add_subplot(sub[axis])
                ax.plot(tt, vb.velocity[sel, axis], color=GT_COLOR, lw=1.6, label='GT (cursor)')
                for m in names:
                    ax.plot(tt, pred[m][sel, axis], color=D.COLORS[m], lw=1.1, label=D.label(m))
                ax.axhline(0, color=style.GREY, lw=0.4)
                ax.set_ylabel(f'v{"xy"[axis]}')
                if axis == 0:
                    d = int(vb.direction[sel[0]])
                    ax.set_title(f'trial {int(cohort.trial_ids[t])} · block {int(vb.block[sel[0]])} · '
                                 f'target {d * 45}°', fontsize=7.5, color=style.direction_color(d))
                    ax.tick_params(labelbottom=False)
                    if j == 0:
                        ax.legend(frameon=False, fontsize=5.5, ncol=3)
                else:
                    ax.set_xlabel('t from trial start / s (bin right edge)')
        fig.suptitle(f'{session.label} · velocity traces, GT vs held-out decoders (task units/s)', fontsize=10)
        return fig

    def trajectories(self, session, vb, pred, per_dir, names, p):
        keys = ('gt',) + tuple(names)
        fig = new_figure(max(13, 3.2 * len(keys)), 8.4)
        grid = fig.add_gridspec(2, len(keys), height_ratios=[1.3, 1])
        paths = {'gt': integrate(vb, vb.velocity, p['bin'])}
        for m in names:
            paths[m] = integrate(vb, pred[m], p['bin'])
        shared = None
        table = []
        for k, m in enumerate(keys):
            ax = fig.add_subplot(grid[0, k], sharex=shared, sharey=shared)
            shared = shared or ax
            for d in range(8):
                ids = [t for t in paths[m] if int(vb.direction[vb.trial == t][0]) == d]
                if not ids:
                    continue
                lens = min(len(paths[m][t]) for t in ids)
                for t in ids[:40]:
                    ax.plot(*paths[m][t].T, color=style.direction_color(d), lw=0.3, alpha=0.3)
                mean = np.mean([paths[m][t][:lens] for t in ids], 0)
                ax.plot(*mean.T, color=style.direction_color(d), lw=2)
                ax.scatter(*mean[-1], color=style.direction_color(d), s=14, zorder=3)
                table += [dict(model=m, direction_deg=d * 45, step=i, n_trials=len(ids), x=float(xy[0]),
                               y=float(xy[1])) for i, xy in enumerate(mean)]
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(('GT (cursor)' if m == 'gt' else D.label(m)) + ' — integrated', fontsize=8)
            ax.set_xlabel('x / task units')
            ax.set_ylabel('y / task units')
        half = max(1, len(keys) * 2 // 3)
        ax = fig.add_subplot(grid[1, :half])
        xs = np.arange(len(per_dir))
        width = 0.8 / len(names)
        for j, m in enumerate(names):
            vals = [r[f'{m}_r2_xy'] for r in per_dir]
            ax.bar(xs + (j - (len(names) - 1) / 2) * width, vals, width, color=D.COLORS[m], label=D.label(m))
        ax.set_xticks(xs, [f'{r["direction_deg"]}°\nn={r["n_bins"]}' for r in per_dir])
        ax.axhline(0, color=style.GREY, lw=0.5)
        ax.set_ylabel('R² (x,y pooled, within direction)')
        ax.set_title('Per-direction R² — variance is around each direction mean, so lower than the pooled R²',
                     fontsize=8)
        ax.legend(frameon=False, fontsize=6)
        ax = fig.add_subplot(grid[1, half:])
        for j, m in enumerate(names):
            ax.bar(xs + (j - (len(names) - 1) / 2) * width, [r[f'{m}_rmse'] for r in per_dir], width,
                   color=D.COLORS[m])
        ax.set_xticks(xs, [f'{r["direction_deg"]}' for r in per_dir])
        ax.set_xlabel('direction / °')
        ax.set_ylabel('RMSE ‖v‖ / task units·s⁻¹')
        ax.set_title('Per-direction RMSE', fontsize=8)
        fig.suptitle(f'{session.label} · trajectories reconstructed from decoded velocity (thin: up to 40 trials '
                     'per direction; thick: mean)', fontsize=10)
        return fig, table


    def comparison(self, session, cohort, vb, y, pred, tested, stats, per_fold, per_dir, paired, best, decoder_rows,
                   names, p):
        k = len(names)
        fig = new_figure(15, 11.5)
        grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.9])
        labels = [D.label(m) for m in names]
        colors = [D.COLORS[m] for m in names]
        width = 0.8 / k
        # (a) R²
        ax = fig.add_subplot(grid[0, 0])
        for j, m in enumerate(names):
            off = (j - (k - 1) / 2) * width
            ax.bar(np.arange(3) + off, [stats[m][q] for q in ('r2_x', 'r2_y', 'r2_mean')], width, color=colors[j],
                   label=labels[j])
            row = next(r for r in paired if r['decoder'] == m)
            if np.isfinite(row['ci_low']):
                ax.errorbar(2 + off, row['r2_mean'], yerr=[[row['r2_mean'] - row['ci_low']],
                                                           [row['ci_high'] - row['r2_mean']]],
                            color=style.INK, capsize=2, lw=0.8)
        ax.set_xticks(range(3), ['x', 'y', 'mean'])
        ax.set_ylabel('held-out R²')
        ax.axhline(0, color=style.GREY, lw=0.5)
        lo = min(0, min(stats[m]['r2_y'] for m in names) - 0.05)
        ax.set_ylim(max(lo, -1), 1)
        ax.set_title('(a) R² (mean: 95% block-bootstrap CI)', fontsize=8)
        ax.legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=3)
        # (b) RMSE
        ax = fig.add_subplot(grid[0, 1])
        for j, m in enumerate(names):
            ax.bar(np.arange(2) + (j - (k - 1) / 2) * width, [stats[m]['rmse_x'], stats[m]['rmse_y']], width,
                   color=colors[j])
        ax.set_xticks(range(2), ['x', 'y'])
        ax.set_ylabel('RMSE / task units·s⁻¹')
        ax.set_title('(b) RMSE per axis', fontsize=8)
        # (c) runtime
        ax = fig.add_subplot(grid[0, 2])
        secs = [max(r['cv_seconds'], 1e-3) for r in decoder_rows]
        ax.barh(range(k), secs, color=colors)
        for i, v in enumerate(secs):
            ax.text(v, i, f' {v:.2f} s', va='center', fontsize=6.5)
        ax.set_xscale('log')
        ax.set_yticks(range(k), labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel('CV wall time / s (5 folds, log)')
        ax.set_title('(c) runtime', fontsize=8)
        ax.margins(x=0.3)
        # (d) per-fold paired lines
        ax = fig.add_subplot(grid[1, 0])
        for f in per_fold:
            ax.plot(range(k), [f[f'{m}_r2_mean'] for m in names], color=style.GREY, lw=0.8, marker='o', ms=3,
                    zorder=1)
            ax.text(k - 1 + 0.08, f[f'{names[-1]}_r2_mean'], f'F{f["fold"] + 1}', fontsize=6, va='center')
        for j, m in enumerate(names):
            ax.scatter([j] * len(per_fold), [f[f'{m}_r2_mean'] for f in per_fold], color=colors[j], s=22, zorder=2)
        ax.set_xticks(range(k), labels, rotation=20, ha='right', fontsize=6.5)
        ax.set_xlim(-0.4, k - 0.4)
        ax.set_ylabel('mean R² in fold')
        ax.set_title('(d) per fold (lines = same fold)', fontsize=8)
        # (e) ΔR² vs best
        ax = fig.add_subplot(grid[1, 1])
        if k == 1:
            ax.text(0.5, 0.5, 'Only one decoder selected:\nno pairwise comparison', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
        else:
            for i, r in enumerate(paired):
                ax.barh(i, r['delta_vs_best'], color=D.COLORS[r['decoder']], height=0.6)
                if np.isfinite(r['delta_ci_low']) and r['decoder'] != best:
                    ax.errorbar(r['delta_vs_best'], i, xerr=[[r['delta_vs_best'] - r['delta_ci_low']],
                                                            [r['delta_ci_high'] - r['delta_vs_best']]],
                                color=style.INK, capsize=2, lw=0.8)
                ax.text(0, i - 0.38, f' {r["delta_vs_best"]:+.4f}' if r['decoder'] != best else ' best',
                        va='center', fontsize=6.5)
            ax.axvline(0, color=style.INK, lw=0.7)
            ax.set_yticks(range(k), labels, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel(f'ΔR² vs {D.label(best)} (paired block bootstrap 95% CI)')
        ax.set_title('(e) difference from the best decoder', fontsize=8)
        # (f) per-direction heat strip
        ax = fig.add_subplot(grid[1, 2])
        heat = np.array([[r[f'{m}_r2_xy'] for r in per_dir] for m in names])
        from matplotlib import colormaps
        cmap = colormaps['viridis'].with_extremes(under='#3b0f4f')
        vmax = max(0.1, float(np.nanmax(heat)))
        image = ax.imshow(heat, aspect='auto', cmap=cmap, vmin=0, vmax=vmax)
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                ax.text(j, i, f'{heat[i, j]:.2f}', ha='center', va='center', fontsize=5.5,
                        color='white' if heat[i, j] < 0.5 * vmax else 'black')
        ax.set_xticks(range(len(per_dir)), [f'{r["direction_deg"]}°' for r in per_dir], fontsize=6.5)
        ax.set_yticks(range(k), labels, fontsize=6.5)
        fig.colorbar(image, ax=ax, fraction=0.05, pad=0.02, extend='min').ax.tick_params(labelsize=6)
        ax.set_title('(f) per-direction R² (x,y pooled; < 0 dark purple)', fontsize=8)
        # (g, h) one example trial
        picks = self.picks(vb, p)
        t = picks[len(picks) // 2]
        sel = np.flatnonzero(vb.trial == t)
        tt = p['offset'] + (vb.bin[sel] + 1) * p['bin']
        trace = dict(t_right_edge_s=tt, gt_vx=vb.velocity[sel, 0], gt_vy=vb.velocity[sel, 1])
        for axis in range(2):
            ax = fig.add_subplot(grid[2, :2] if axis == 0 else grid[2, 2])
            ax.plot(tt, vb.velocity[sel, axis], color=GT_COLOR, lw=2, label='GT (cursor)')
            for j, m in enumerate(names):
                ax.plot(tt, pred[m][sel, axis], color=colors[j], lw=1.1, marker='.', ms=3, label=labels[j])
                trace[f'{m}_v{"xy"[axis]}'] = pred[m][sel, axis]
            ax.axhline(0, color=style.GREY, lw=0.4)
            ax.set_xlabel('t from trial start / s (bin right edge)')
            ax.set_ylabel(f'v{"xy"[axis]} / units·s⁻¹')
            d = int(vb.direction[sel[0]])
            ax.set_title(f'({"gh"[axis]}) trial {int(cohort.trial_ids[t])} · target {d * 45}° · v{"xy"[axis]}',
                         fontsize=8)
            if axis == 0:
                ax.legend(frameon=False, fontsize=6, ncol=3)
        note = '' if k > 1 else ' · only one decoder selected'
        fig.suptitle(f'{session.label} · decoder comparison on {int(tested.sum())} common held-out bins{note}',
                     fontsize=10)
        data = dict(metrics=self.metric_rows(stats, names), paired_bootstrap=paired, per_fold=per_fold,
                    per_direction=per_dir, runtime=[{k_: r[k_] for k_ in ('decoder', 'cv_seconds', 'fit_seconds',
                                                                          'predict_seconds')} for r in decoder_rows],
                    example_trial=trace)
        return fig, data


SECTION = VelocityDecoding()
