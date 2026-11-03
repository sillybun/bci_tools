"""S09 v1 — eight-direction population decoding.

Ridge regression onto one-hot direction (standardized features, penalty n·λ,
argmax class), five chronological block folds with a ±1-block purge, and a
within-block label-permutation null (exact refits through the CV operator).
Channel-count curves rank channels by the one-way ANOVA F computed on each
training fold only, so the test fold never influences which channels are used.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

from ...core import analysis as A
from ...core import style
from ...core.target_decoding import grid_target_decoding
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure

DEG = [f'{d * 45}' for d in range(8)]


def trial_features(cohort, sqrt_rates: bool):
    x = np.asarray(cohort.window_rates, float)
    ok = np.isfinite(x).all(1)
    if sqrt_rates:
        x = np.sqrt(np.clip(x, 0, None))
    return x, ok


def decode(x, y, blocks, lam, n_perm, seed=20261029, n_folds=5, *, drop_constant=True):
    """Held-out predictions, confusion matrix and permutation null for one feature matrix."""
    active = x.std(0) > 1e-12
    features = x[:, active] if drop_constant else x
    scores, operator, folds, tested = A.ridge_classifier_cv(features, y, blocks, lam, 8, n_folds)
    pred = scores.argmax(1)
    yt, pt = y[tested], pred[tested]
    observed = A.balanced_accuracy(yt, pt)
    counts = np.zeros((8, 8), int)
    np.add.at(counts, (yt, pt), 1)
    fraction = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    null = A.permutation_null_accuracy(operator, y, blocks, tested, n_perm, seed) if n_perm else np.zeros(0)
    circ = np.abs(((pt - yt + 4) % 8) - 4)
    per_fold = []
    for f in np.unique(folds[tested]):
        sel = folds == f
        per_fold.append(dict(fold=int(f), n_test=int(sel.sum()),
                             blocks=f'{blocks[sel].min()}-{blocks[sel].max()}',
                             balanced_accuracy=A.balanced_accuracy(y[sel], pred[sel]),
                             accuracy=float(np.mean(pred[sel] == y[sel]))))
    return dict(scores=scores, pred=pred, folds=folds, tested=tested, observed=observed, counts=counts,
                fraction=fraction, null=null, per_fold=per_fold, n_active=int(active.sum()),
                accuracy=float(np.mean(pt == yt)), within45=float(np.mean(circ <= 1)),
                mean_abs_error_deg=float(np.mean(circ) * 45),
                p=float((1 + np.sum(null >= observed)) / (1 + len(null))) if len(null) else float('nan'))


def cached_decode(ctx, cohort, x, y, blocks, lam, n_perm, *, sqrt_rates=False, seed=20261029, n_folds=5,
                  drop_constant=True):
    key = ('direction_decode', id(cohort), sqrt_rates, lam, n_perm, seed, n_folds, drop_constant)
    return ctx.cached(key, lambda: decode(x, y, blocks, lam, n_perm, seed, n_folds, drop_constant=drop_constant))


def _kernel_ridge_scores(a, b, ya, lam):
    mu, sd, keep = A.standardize_fit(a)
    za = (a[:, keep] - mu[keep]) / sd[keep]
    zb = (b[:, keep] - mu[keep]) / sd[keep]
    gram = za @ za.T + len(za) * lam * np.eye(len(za))
    onehot = np.eye(8)[ya]
    return zb @ za.T @ linalg.solve(gram, onehot - onehot.mean(0), assume_a='pos') + onehot.mean(0)


def channel_curve(x, y, blocks, lam, ks, draws, seed=7):
    """Balanced accuracy vs channel count: train-fold ANOVA ranking and random subsets."""
    active = np.flatnonzero(x.std(0) > 1e-12)
    ranked = {k: np.full(len(y), -1) for k in ks}
    random = {k: np.full((draws, len(y)), -1) for k in ks}
    rng = np.random.default_rng(seed)
    tested = np.zeros(len(y), bool)
    for _, tr, te in A.block_folds(blocks):
        tested[te] = True
        F = A.oneway_anova(x[tr][:, active], y[tr])['F']
        order = active[np.argsort(-F, kind='stable')]
        for k in ks:
            cols = order[:k]
            ranked[k][te] = _kernel_ridge_scores(x[tr][:, cols], x[te][:, cols], y[tr], lam).argmax(1)
            for d in range(draws):
                cols = rng.choice(active, size=min(k, len(active)), replace=False)
                random[k][d, te] = _kernel_ridge_scores(x[tr][:, cols], x[te][:, cols], y[tr], lam).argmax(1)
    out = []
    for k in ks:
        rb = [A.balanced_accuracy(y[tested], random[k][d, tested]) for d in range(draws)]
        out.append(dict(n_channels=int(k), anova_ranked=A.balanced_accuracy(y[tested], ranked[k][tested]),
                        random_mean=float(np.mean(rb)) if rb else float('nan'),
                        random_sd=float(np.std(rb)) if rb else float('nan')))
    return out


class DirectionDecoding(Section):
    id = 's09'
    title = '方向解码 Overview + examples'
    description = '八分类 ridge 解码（block CV + 置换检验）：混淆矩阵、每折准确率、通道数曲线、逐 trial 预测示例'
    params = (
        Param('lam', 'Ridge λ（惩罚 n·λ）', 'float', 1.0, minimum=1e-6),
        Param('sqrt', '使用 sqrt(rate) 特征', 'bool', False),
        Param('n_perm', '置换次数', 'int', 200, minimum=0, maximum=10000),
        Param('channel_curve', '计算通道数曲线', 'bool', True),
        Param('random_draws', '随机通道子集次数', 'int', 5, minimum=0, maximum=50),
        Param('n_examples', '示例 trial 数', 'int', 8, minimum=1, maximum=24),
    )

    def run(self, ctx: AnalysisContext) -> SectionResult:
        p, cohort, session = ctx.params, ctx.cohort, ctx.session
        result = SectionResult()
        x, ok = trial_features(cohort, p['sqrt'])
        if not ok.all():
            result.notes.append(f'{int((~ok).sum())} 个 trial 分析窗口内无神经帧，已从解码中剔除。')
        rows = np.flatnonzero(ok)
        x, y, blocks = x[rows], cohort.directions[rows], cohort.blocks[rows]
        if len(rows) < 16 or len(np.unique(blocks)) < 4 or len(np.unique(y)) < 2:
            raise ValueError('保留 trial 太少（需 ≥16 个 trial、≥4 个 block、≥2 个方向）')
        ctx.progress('ridge CV + permutation null')
        r = cached_decode(ctx, cohort, x, y, blocks, p['lam'], p['n_perm'], sqrt_rates=p['sqrt'])
        curve = []
        if p['channel_curve']:
            ctx.progress('accuracy vs channel count')
            n_active = r['n_active']
            ks = sorted({k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if k < n_active} | {n_active})
            curve = ctx.cached(('direction_channel_curve', id(cohort), p['sqrt'], p['lam'], p['random_draws']),
                               lambda: channel_curve(x, y, blocks, p['lam'], ks, p['random_draws']))
        ctx.check_cancel()
        confusion = [dict(target_deg=int(t * 45), predicted_deg=int(q * 45), n=int(r['counts'][t, q]),
                          fraction=float(r['fraction'][t, q])) for t in range(8) for q in range(8)]
        predictions = [
            dict(order=int(j), cohort_row=int(i), trial_id=int(cohort.trial_ids[i]), block=int(cohort.blocks[i]),
                 fold=int(r['folds'][j]), target_deg=int(y[j] * 45), predicted_deg=int(r['pred'][j] * 45),
                 correct=bool(r['pred'][j] == y[j]),
                 error_steps=int(abs(((r['pred'][j] - y[j] + 4) % 8) - 4)),
                 **{f'score_{d * 45}': float(r['scores'][j, d]) for d in range(8)})
            for j, i in enumerate(rows)]
        feature = 'sqrt(mean Hz)' if p['sqrt'] else 'mean Hz'
        base = (f'Input: {len(y)} included trials (topbar filter) of {session.label}; feature per trial = {feature} '
                f'of each of the 512 stored channels in the analysis window T_TRIAL_START+'
                f'{cohort.criteria.window_start:g} to +{cohort.criteria.window_end:g} s (count / 20 ms frame, averaged '
                f'over frames in the window); trials without frames in the window dropped; constant channels dropped '
                f'({r["n_active"]} active). Classifier: ridge regression onto the 8-column one-hot target direction, '
                f'features z-scored with training-fold mean/SD, penalty n_train*lambda (lambda={p["lam"]:g}), '
                f'intercept = training class frequency, predicted class = argmax of the 8 scores. CV: 5 contiguous '
                f'chronological groups of blocks; for each test group the training set excludes blocks within +-1 of '
                f'the test block range (purge); every trial is predicted once by the model that did not see it. ')
        method_overview = base + (
            'Panel confusion: counts of (target, predicted) over held-out trials, each row divided by its row total, '
            'shown in %. Panel null: balanced accuracy (mean over directions of per-direction recall) of the observed '
            f'held-out predictions vs {len(r["null"])} permutations where direction labels are shuffled within each '
            'block and every fold is refitted exactly (linear CV operator); p = (1 + #null >= observed) / (1 + n_perm). '
            'Panel per-fold: balanced accuracy restricted to the test trials of each fold; pooled line = overall balanced '
            'accuracy; chance = 1/8. Panel channel curve: for k channels, inside each training fold channels are '
            'ranked by one-way ANOVA F (direction) computed on training trials only and the top k are used (no test '
            f'information); random curve = mean +- SD over {p["random_draws"]} random k-subsets of active channels per '
            'fold; balanced accuracy over all held-out trials. Text panel: accuracy = fraction exactly correct; '
            'within +-45 deg = predicted within one neighbouring direction; mean |error| in degrees on the circle.')
        method_examples = base + (
            'Top: held-out predicted direction (dot) vs target (open circle) for every included trial in '
            'chronological order; dot colour = correct / off by 45 deg / larger error; grey vertical lines connect '
            'wrong predictions to their targets; dotted lines = fold boundaries. Middle: rolling fraction correct '
            f'over a centred window of max(8, n/20) trials (zero-padded at the ends, numpy convolve mode=same). '
            f'Bottom: the 8 raw ridge class scores of {min(p["n_examples"], len(y))} trials evenly spaced in '
            'chronological order; black outline = target class.')
        window = max(8, len(y) // 20)
        rolling = np.convolve((r['pred'] == y).astype(float), np.ones(window) / window, mode='same')
        picks = np.linspace(0, len(y) - 1, min(p['n_examples'], len(y))).round().astype(int)
        null_rows = dict(permutation=np.arange(len(r['null'])), balanced_accuracy=r['null'])
        stats_rows = [dict(balanced_accuracy=r['observed'], accuracy=r['accuracy'], within_45deg=r['within45'],
                           mean_abs_error_deg=r['mean_abs_error_deg'], p_value=r['p'], n_perm=len(r['null']),
                           n_trials=len(y), n_active_channels=r['n_active'], chance=0.125)]
        data_overview = dict(confusion=confusion, permutation_null=null_rows, per_fold=r['per_fold'],
                             statistics=stats_rows)
        if curve:
            data_overview['channel_curve'] = curve
        result.add_figure('Direction decoding overview', self.overview(session, cohort, r, curve, p),
                          method=method_overview, data=data_overview)
        result.add_figure('Direction decoding examples', self.examples(session, cohort, rows, y, blocks, r, p),
                          method=method_examples,
                          data=dict(trial_predictions=predictions,
                                    rolling_accuracy=dict(order=np.arange(len(y)), rolling_accuracy=rolling,
                                                          window_trials=np.full(len(y), window)),
                                    example_scores=[predictions[i] for i in picks]))
        if session.task_mode == 'grid':
            ctx.progress('Grid target decoding')
            g = ctx.cached(('grid_target_decoding', id(cohort), p['lam'], p['n_perm']),
                           lambda: grid_target_decoding(cohort, p['lam'], p['n_perm']))
            fig, data = self.grid_figure(session, g)
            result.add_figure('Grid target decoding', fig, method=grid_method(g, p), data=data)
            result.summary.update(grid_n_targets=g['n_targets'], grid_target_r2_xy=g['position']['r2_xy'],
                                  grid_target_median_error=g['position']['median_error'],
                                  grid_displacement_r2_xy=g['displacement_metrics']['r2_xy'])
            if g['identity'] is not None:
                result.summary.update(grid_identity_balanced_accuracy=g['identity']['balanced_accuracy'],
                                      grid_identity_chance=g['identity']['chance'], grid_identity_p=g['identity']['p'])
        result.summary.update(
            recording=session.label, n_trials=int(len(y)), n_blocks=int(len(np.unique(blocks))),
            n_active_channels=r['n_active'], feature='sqrt(Hz)' if p['sqrt'] else 'Hz', ridge_lambda=p['lam'],
            balanced_accuracy=r['observed'], accuracy=r['accuracy'], within_45deg=r['within45'],
            mean_abs_error_deg=r['mean_abs_error_deg'], chance=0.125, n_perm=int(p['n_perm']), p_value=r['p'],
            null_mean=float(r['null'].mean()) if len(r['null']) else float('nan'),
            null_p95=float(np.percentile(r['null'], 95)) if len(r['null']) else float('nan'))
        result.tables['per_fold'] = r['per_fold']
        result.tables['confusion'] = confusion
        result.tables['trial_predictions'] = predictions
        if curve:
            result.tables['channel_curve'] = curve
        result.notes.append('通道排序只用每个训练折内的 ANOVA F，测试折不参与选通道。')
        return result

    # ------------------------------------------------------------------ figures

    def grid_figure(self, session, g):
        fig = new_figure(14, 5.2)
        grid = fig.add_gridspec(1, 4, width_ratios=[1.2, 0.8, 0.8, 1.2])
        targets, true, pred = g['targets'], g['target'], g['position_pred']
        cls = np.unique(np.round(true, 3), axis=0, return_inverse=True)[1].ravel()
        ax = fig.add_subplot(grid[0])
        means = []
        for k, t in enumerate(targets):
            sel = (cls == k) & np.isfinite(pred).all(1)
            if sel.any():
                m = pred[sel].mean(0)
                ax.plot([t[0], m[0]], [t[1], m[1]], color=style.GREY, lw=0.7)
                means.append(dict(target_x=t[0], target_y=t[1], mean_pred_x=m[0], mean_pred_y=m[1],
                                  n_trials=int(sel.sum())))
        ax.scatter(targets[:, 0], targets[:, 1], marker='s', s=30, facecolors='none', edgecolors=style.INK)
        ax.scatter([m['mean_pred_x'] for m in means], [m['mean_pred_y'] for m in means], s=14,
                   color=style.MODEL_COLORS['ridge'])
        ax.set_aspect('equal', adjustable='datalim')
        pm = g['position']
        ax.set_title(f'Target position: R² {pm["r2_xy"]:.2f}, median error {pm["median_error"]:.0f}\n'
                     f'(mean-only baseline {pm["baseline_median_error"]:.0f})', fontsize=8)
        for axis in range(2):
            ax = fig.add_subplot(grid[1 + axis])
            ax.scatter(true[:, axis], pred[:, axis], s=6, alpha=0.5, color=style.MODEL_COLORS['ridge'])
            lim = np.nanmax(np.abs(np.r_[true[:, axis], pred[:, axis]])) * 1.05
            ax.plot([-lim, lim], [-lim, lim], color=style.INK, lw=0.6)
            ax.set_aspect('equal')
            ax.set_xlabel(f'true target {"xy"[axis]}')
            ax.set_ylabel(f'predicted {"xy"[axis]}')
            ax.set_title(f'{"xy"[axis]}: R² {pm["r2_" + "xy"[axis]]:.2f}', fontsize=8)
        ax = fig.add_subplot(grid[3])
        ident = g['identity']
        data = dict(target_means=means, trials=dict(target_x=true[:, 0], target_y=true[:, 1], pred_x=pred[:, 0],
                                                    pred_y=pred[:, 1], fold=g['folds']),
                    statistics=[dict(n_targets=g['n_targets'], **{f'position_{k}': v for k, v in pm.items()},
                                     **{f'displacement_{k}': v for k, v in g['displacement_metrics'].items()})])
        if ident is not None:
            ax.imshow(ident['confusion'], cmap='Blues', vmin=0, vmax=max(0.3, ident['confusion'].max()))
            ax.set_xlabel('predicted target index')
            ax.set_ylabel('true target index (sorted by x, y)')
            ax.set_title(f'Target identity: BA {100 * ident["balanced_accuracy"]:.1f}% vs chance '
                         f'{100 * ident["chance"]:.1f}%\np = {ident["p"]:.3g} (n_perm={len(ident["null"])})', fontsize=8)
            k = len(targets)
            data['identity_confusion'] = [dict(true_index=i, predicted_index=j, true_x=targets[i, 0],
                                               true_y=targets[i, 1], fraction=float(ident['confusion'][i, j]))
                                          for i in range(k) for j in range(k)]
            data['statistics'][0].update(identity_balanced_accuracy=ident['balanced_accuracy'],
                                         identity_chance=ident['chance'], identity_p=ident['p'])
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'identity classification skipped\n(> 64 targets or < 3 trials per target)',
                    ha='center', va='center', transform=ax.transAxes)
        fig.suptitle(f'{session.label} · Grid target decoding (held-out, block CV); displacement R² '
                     f'{g["displacement_metrics"]["r2_xy"]:.2f}', fontsize=10)
        return fig, data
    def overview(self, session, cohort, r, curve, p):
        fig = new_figure(13, 8.4)
        grid = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1])
        ax = fig.add_subplot(grid[0, 0])
        image = ax.imshow(r['fraction'], cmap='Blues', vmin=0, vmax=max(0.5, r['fraction'].max()),
                          interpolation='nearest')
        for t in range(8):
            for q in range(8):
                v = r['fraction'][t, q]
                if r['counts'][t, q]:
                    ax.text(q, t, f'{v * 100:.0f}', ha='center', va='center', fontsize=6.5,
                            color='white' if v > 0.45 else style.INK)
        ax.set_xticks(range(8), DEG)
        ax.set_yticks(range(8), DEG)
        ax.set_xlabel('Predicted direction / °')
        ax.set_ylabel('Target direction / °')
        ax.set_title('Confusion (row-normalised, %)')
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)

        ax = fig.add_subplot(grid[0, 1])
        if len(r['null']):
            ax.hist(100 * r['null'], bins=30, color=style.GREY, label=f'null ({len(r["null"])} perms)')
        ax.axvline(100 * r['observed'], color=style.MODEL_COLORS['ridge'], lw=2, label='observed')
        ax.axvline(12.5, color='#c76d4b', ls='--', lw=1, label='chance 12.5%')
        ax.set_xlabel('Balanced accuracy / %')
        ax.set_ylabel('permutations')
        ax.set_title(f'Within-block permutation null · p = {r["p"]:.3g}' if len(r['null']) else 'No permutation null')
        ax.legend(frameon=False, fontsize=6.5)

        ax = fig.add_subplot(grid[0, 2])
        folds = r['per_fold']
        xs = np.arange(len(folds))
        ax.bar(xs, [100 * f['balanced_accuracy'] for f in folds], color=style.MODEL_COLORS['ridge'], alpha=0.85)
        for i, f in enumerate(folds):
            ax.text(i, 100 * f['balanced_accuracy'], f'{100 * f["balanced_accuracy"]:.0f}', ha='center',
                    va='bottom', fontsize=6.5)
        ax.axhline(100 * r['observed'], color=style.INK, lw=0.8, label='pooled')
        ax.axhline(12.5, color='#c76d4b', ls='--', lw=0.8, label='chance')
        ax.set_xticks(xs, [f'F{f["fold"] + 1}\nb{f["blocks"]}\nn={f["n_test"]}' for f in folds], fontsize=6)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Balanced accuracy / %')
        ax.set_title('Per-fold accuracy (chronological folds)')
        ax.legend(frameon=False, fontsize=6.5, loc='upper right')

        ax = fig.add_subplot(grid[1, :2])
        if curve:
            k = [c['n_channels'] for c in curve]
            ax.plot(k, [100 * c['anova_ranked'] for c in curve], 'o-', color=style.MODEL_COLORS['ridge'],
                    label='top-k by train-fold ANOVA F')
            if p['random_draws']:
                m = np.array([100 * c['random_mean'] for c in curve])
                s = np.array([100 * c['random_sd'] for c in curve])
                ax.fill_between(k, m - s, m + s, color=style.MODEL_COLORS['kf'], alpha=0.2, lw=0)
                ax.plot(k, m, 's-', color=style.MODEL_COLORS['kf'], ms=4,
                        label=f'random k channels (mean ± SD, {p["random_draws"]} draws)')
            ax.axhline(12.5, color='#c76d4b', ls='--', lw=0.8, label='chance')
            ax.set_xscale('log', base=2)
            ax.set_xticks(k, [str(v) for v in k])
            ax.set_ylim(0, 100)
            ax.legend(frameon=False, loc='upper left')
        else:
            ax.text(0.5, 0.5, 'Channel curve disabled', ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('number of channels')
        ax.set_ylabel('Balanced accuracy / %')
        ax.set_title('Accuracy vs channel count (ranking inside each training fold, no test leakage)')

        ax = fig.add_subplot(grid[1, 2])
        lines = [
            f'{session.label}',
            f'trials        {len(r["pred"])}   blocks {len(np.unique(cohort.blocks))}',
            f'channels      {r["n_active"]} active / 512',
            f'feature       {"sqrt(Hz)" if p["sqrt"] else "Hz"}, window '
            f'{cohort.criteria.window_start:g}-{cohort.criteria.window_end:g} s',
            f'ridge λ       {p["lam"]:g} (penalty n·λ)',
            f'CV            5 block folds, ±1 block purge',
            '',
            f'balanced acc  {100 * r["observed"]:.1f} %',
            f'accuracy      {100 * r["accuracy"]:.1f} %',
            f'within ±45°   {100 * r["within45"]:.1f} %',
            f'mean |error|  {r["mean_abs_error_deg"]:.1f}°',
            f'chance        12.5 %',
        ]
        if len(r['null']):
            lines += [f'null mean     {100 * r["null"].mean():.1f} %',
                      f'null 95th     {100 * np.percentile(r["null"], 95):.1f} %',
                      f'p (perm)      {r["p"]:.3g}  (n={len(r["null"])})']
        style.text_panel(ax, lines)
        fig.suptitle(f'{session.label} · eight-direction ridge decoding · held-out predictions', fontsize=10)
        return fig

    def examples(self, session, cohort, rows, y, blocks, r, p):
        fig = new_figure(13, 8.6)
        n_ex = min(p['n_examples'], len(y))
        grid = fig.add_gridspec(3, 1, height_ratios=[1.25, 0.25, 1.2])
        ax = fig.add_subplot(grid[0])
        order = np.arange(len(y))
        correct = r['pred'] == y
        circ = np.abs(((r['pred'] - y + 4) % 8) - 4)
        ax.scatter(order, y * 45, s=22, facecolors='none', edgecolors=style.GREY, lw=0.8, label='target')
        colors = np.where(correct, style.MODEL_COLORS['online'], np.where(circ <= 1, '#edae49', '#d1495b'))
        wrong = np.flatnonzero(~correct)
        ax.vlines(wrong, y[wrong] * 45, r['pred'][wrong] * 45, color=style.GREY, lw=0.4, alpha=0.6)
        ax.scatter(order, r['pred'] * 45, s=7, c=colors, zorder=3)
        for c, lab in ((style.MODEL_COLORS['online'], 'predicted: correct'), ('#edae49', 'predicted: ±45°'),
                       ('#d1495b', 'predicted: wrong')):
            ax.scatter([], [], s=10, c=c, label=lab)
        change = np.flatnonzero(np.diff(r['folds']) != 0)
        for i in change:
            ax.axvline(i + 0.5, color=style.INK2, lw=0.6, ls=':')
        for f in np.unique(r['folds'][r['folds'] >= 0]):
            idx = np.flatnonzero(r['folds'] == f)
            ax.text(idx.mean(), 352, f'fold {f + 1}', ha='center', fontsize=6.5, color=style.INK2)
        ax.set_yticks(np.arange(0, 360, 45))
        ax.set_ylim(-25, 375)
        ax.set_xlim(-1, len(y))
        ax.set_ylabel('direction / °')
        ax.set_xlabel('included trial (chronological)')
        ax.set_title('Per-trial held-out prediction over the session', loc='left')
        ax.legend(frameon=False, ncol=4, fontsize=6.5, loc='lower right', bbox_to_anchor=(1, 1))

        ax = fig.add_subplot(grid[1])
        window = max(8, len(y) // 20)
        rolling = np.convolve(correct.astype(float), np.ones(window) / window, mode='same')
        ax.plot(order, 100 * rolling, color=style.MODEL_COLORS['ridge'], lw=1)
        ax.axhline(12.5, color='#c76d4b', ls='--', lw=0.7)
        ax.set_ylim(0, 100)
        ax.set_xlim(-1, len(y))
        ax.set_ylabel('acc %', fontsize=7)
        ax.set_title(f'Rolling accuracy ({window} trials)', fontsize=7.5)

        sub = grid[2].subgridspec(1, n_ex)
        picks = np.linspace(0, len(y) - 1, n_ex).round().astype(int)
        for j, i in enumerate(picks):
            ax = fig.add_subplot(sub[0, j])
            s = r['scores'][i]
            ax.bar(range(8), s, color=[style.direction_color(d) for d in range(8)], alpha=0.85)
            ax.bar([y[i]], [s[y[i]]], fill=False, edgecolor=style.INK, lw=1.4)
            ax.axhline(0, color=style.GREY, lw=0.5)
            ax.set_xticks(range(8), DEG, fontsize=5.5, rotation=90)
            ax.tick_params(axis='y', labelsize=5.5)
            mark = 'correct' if r['pred'][i] == y[i] else 'wrong'
            ax.set_title(f'trial {int(cohort.trial_ids[rows[i]])} (b{int(blocks[i])})\n'
                         f'target {y[i] * 45}° · pred {r["pred"][i] * 45}° {mark}', fontsize=6.5)
            if j == 0:
                ax.set_ylabel('ridge class score')
        fig.suptitle(f'{session.label} · held-out class scores for {n_ex} evenly spaced trials '
                     '(black outline = target)', fontsize=10)
        return fig


def grid_method(g, p):
    return (f'Grid task: {g["n_trials"]} direction-cohort trials, {g["n_targets"]} unique target positions (rounded '
            'to 1e-3). Feature = mean Hz per channel in the analysis window (constant channels dropped). Target '
            'position (x, y) and reach displacement (target - start cursor position) regressed with standardized '
            f'ridge (training-fold z-score, penalty n*lambda, lambda {p["lam"]:g}) over 5 chronological block folds '
            '(+-1 block purge). Left: each target (open square) joined to the mean held-out prediction of its trials. '
            'Middle: held-out predicted vs true target x and y per trial. R^2 per axis; pooled R^2 = 1 - sum SSE / '
            'sum SST over x and y; median error = median Euclidean distance. Right: target identity, one-hot ridge '
            'over the unique targets (argmax), row-normalised confusion with targets ordered by (x, y); balanced '
            f'accuracy vs chance 1/K; p from {p["n_perm"]} within-block label permutations with exact refits. '
            'Identity is skipped when there are more than 64 targets or fewer than 3 trials for some target.')


SECTION = DirectionDecoding()
