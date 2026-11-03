"""X04 v1 — cross-day eight-direction decoding accuracy.

Decoder: one-hot ridge classifier on standardized topbar-window rates (as
neural_behavior `compare_decoding_records.py` / core `ridge_classifier_cv`).

  * within-day: five chronological block folds with ±1-block purge; balanced
    accuracy; null = within-block label permutations refitted exactly through the
    linear CV operator
  * train day i / test day j: fitted on all trials of day i using only the
    electrodes both recordings transmitted; day j features standardized with day i
    statistics ('train') and/or re-standardized with day j's own statistics
    ('test_own'). Cross-day null: test labels permuted within block, predictions fixed.
"""
from __future__ import annotations

import numpy as np

from ...core import analysis as A
from ...core import crossday as X
from ...core import style
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure
from ..s09_direction_decoding_v1.section import cached_decode


def within_day(ctx, cohort, d, lam, n_folds, n_perm, seed):
    x = d['x']
    # Match S09 whenever its globally dropped columns are exactly constant.
    # Otherwise keep X04's original fold-local feature selection.
    dropped = x.std(0) <= 1e-12
    equivalent = not np.any(np.ptp(x[:, dropped], axis=0) > 0)
    own = cached_decode(ctx, cohort, x, d['y'], d['blocks'], lam, n_perm, seed=seed, n_folds=n_folds,
                        drop_constant=equivalent)
    acc, null = own['observed'], own['null']
    p = float((1 + np.sum(null >= acc - 1e-12)) / (1 + len(null))) if len(null) else np.nan
    return dict(acc=acc, null=null, p=p, conf=own['fraction'], n_tested=int(own['tested'].sum()))


class CrossDayDirectionDecoding(Section):
    id = 'x04'
    mode = 'cross'
    title = '跨天方向解码准确性'
    description = ('每天 block 交叉验证的平衡准确率（含机会水平与置换检验 p），共同电极上的第 i 天训练 / 第 j 天测试准确率矩阵'
                   '（特征用训练日统计量标准化，可选用测试日自身统计量重新标准化），以及准确率随间隔天数的变化')
    cache_code_dependencies = ('sections/s09_direction_decoding_v1',)
    params = (
        Param('lam', 'ridge λ（×n_train）', 'float', 1.0, minimum=1e-6),
        Param('n_folds', '交叉验证 block 折数', 'int', 5, minimum=2, maximum=10),
        Param('n_perm', '置换次数（日内与跨天）', 'int', 50, minimum=0, maximum=1000),
        Param('test_standardize', '测试日特征标准化', 'choice', 'both', choices=('both', 'train', 'test_own')),
        Param('seed', '随机种子', 'int', 20261029),
    )

    def run(self, ctx: AnalysisContext) -> SectionResult:
        p = ctx.params
        sessions, n = ctx.sessions, len(ctx.sessions)
        labels = X.recording_labels(s.recording for s in sessions)
        modes = ('train', 'test_own') if p['test_standardize'] == 'both' else (p['test_standardize'],)
        result = SectionResult()
        data, own = [], []
        for k, s in enumerate(sessions):
            ctx.check_cancel()
            ctx.progress(f'within-day CV {labels[k]}')
            d = X.direction_data(ctx, k)
            if len(d['x']) < 24 or len(np.unique(d['blocks'])) < 3:
                raise ValueError(f'{s.label}: trial 太少（{len(d["x"])}）或 block 不足，无法交叉验证')
            data.append(d)
            own.append(within_day(ctx, ctx.cohorts[k], d, p['lam'], p['n_folds'], p['n_perm'], p['seed']))
        rng = np.random.default_rng(p['seed'])
        acc = {m: np.full((n, n), np.nan) for m in modes}
        pval = {m: np.full((n, n), np.nan) for m in modes}
        n_common = np.full((n, n), np.nan)
        rows = []
        for i in range(n):
            ctx.check_cancel()
            ctx.progress(f'train {labels[i]} → other recordings')
            n_common[i, i] = int((data[i]['x'].std(0) > 1e-12).sum())
            for m in modes:
                acc[m][i, i], pval[m][i, i] = own[i]['acc'], own[i]['p']
            for j in range(n):
                if i == j:
                    continue
                ids, ci, cj = X.common_columns(sessions[i].electrode_ids, sessions[j].electrode_ids)
                xi, xj = data[i]['x'][:, ci], data[j]['x'][:, cj]
                scaler, model = X.common_direction_model(ctx, i, j, xi, data[i]['y'], p['lam'])
                n_common[i, j] = scaler['keep'].sum()
                if model is None:
                    continue
                row = dict(train=sessions[i].label, test=sessions[j].label,
                           days_apart=X.day_number(sessions[j]) - X.day_number(sessions[i]), forward=j > i,
                           common_electrodes=int(scaler['keep'].sum()),
                           test_within_day_cv=round(own[j]['acc'], 4))
                for m in modes:
                    pred = X.predict_classifier(model, X.transform(scaler, xj, m))
                    yj = data[j]['y']
                    a = A.balanced_accuracy(yj, pred)
                    null = np.array([A.balanced_accuracy(A.permute_within_block(rng, yj, data[j]['blocks']), pred)
                                     for _ in range(p['n_perm'])])
                    pv = float((1 + np.sum(null >= a - 1e-12)) / (1 + len(null))) if len(null) else np.nan
                    acc[m][i, j], pval[m][i, j] = a, pv
                    row[f'accuracy_{m}'] = round(a, 4)
                    row[f'perm_p_{m}'] = round(pv, 4) if np.isfinite(pv) else ''
                rows.append(row)

        day_rows = [dict(recording=s.label, date=s.recording.date, trials=len(data[k]['x']),
                         tested_trials=own[k]['n_tested'], blocks=int(len(np.unique(data[k]['blocks']))),
                         balanced_accuracy=round(own[k]['acc'], 4), chance=0.125,
                         null_p95=round(float(np.percentile(own[k]['null'], 95)), 4) if len(own[k]['null']) else '',
                         perm_p=round(own[k]['p'], 4) if np.isfinite(own[k]['p']) else '', n_perm=p['n_perm'])
                    for k, s in enumerate(sessions)]
        ctx.progress('figures')
        crit = ctx.criteria
        base = (f'Inputs: included trials of each recording (topbar filter); features = mean rate (Hz) of every '
                f'stored column in the window {crit.window_start:g}-{crit.window_end:g} s after T_TRIAL_START '
                f'(trials without neural frames dropped); label = eight target directions. Decoder: one-hot ridge '
                f'classifier on standardized features (z = (x - mean) / SD of the training trials; constant '
                f'columns dropped), weights = (Z^T Z + n_train * lambda I)^-1 Z^T (Y - mean Y), lambda = '
                f'{p["lam"]:g}; prediction = argmax score. Metric: balanced accuracy = mean per-direction recall; '
                f'chance = 1/8. Within-day: {p["n_folds"]} contiguous chronological block folds, training excludes '
                f'blocks within +-1 of the test blocks; null = {p["n_perm"]} within-block permutations of the '
                f'direction labels with the whole CV refitted exactly through the linear CV operator; '
                f'p = (1 + #null >= observed) / (1 + n_perm), seed {p["seed"]}. Cross-day (train i, test j): fitted '
                f'on all trials of recording i using only electrode ids transmitted by both recordings (aligned by '
                f'id, never by column); test features standardized with the training-day mean/SD (fixed) and/or '
                f'with the test-day own mean/SD (re-standardized); cross-day null = test labels permuted within '
                f'blocks with predictions fixed, {p["n_perm"]} permutations.')
        result.add_figure(
            'Within-day direction decoding', self.within_figure(own, labels, p),
            method=base + ' Top: within-day block-CV balanced accuracy per recording (all transmitted electrodes), '
                          'red tick = 95th percentile of the permutation null, dashed = chance. Bottom: confusion '
                          'matrices of held-out predictions, rows (true direction) normalized to 1.',
            data={'per_recording': day_rows,
                  'confusion': [dict(recording=labels[k], true_deg=t * 45, predicted_deg=q * 45,
                                     fraction=float(own[k]['conf'][t, q]))
                                for k in range(n) for t in range(8) for q in range(8)],
                  'null_distribution': [dict(recording=labels[k], permutation=b, balanced_accuracy=float(v))
                                        for k in range(n) for b, v in enumerate(own[k]['null'])]})
        mats = {f'accuracy_{m}': acc[m] for m in modes} | {f'perm_p_{m}': pval[m] for m in modes}
        mats['common_electrodes'] = n_common
        result.add_figure(
            'Cross-day accuracy matrices', self.matrices(acc, pval, n_common, labels, p),
            method=base + ' Rows = training recording i, columns = test recording j; one matrix per test '
                          'standardization; diagonal = within-day block CV on all transmitted electrodes of that '
                          'recording. Red box = permutation p >= 0.05. Right: common electrodes non-constant in the '
                          'training recording (diagonal = own non-constant electrodes). Shared colour scale from '
                          'chance (0.125).',
            data={'matrices': X.matrix_rows(mats, labels, 'train', 'test')})
        result.add_figure(
            'Accuracy vs days apart', self.vs_days(rows, own, sessions, modes),
            method=base + ' Every ordered pair (i, j), i != j: cross-day balanced accuracy vs |calendar days '
                          'between the recording dates| (0 = same day; small jitter). Blue = train earlier / test '
                          'later in selection order, orange = reverse; dashed = least-squares line; green diamonds '
                          'at x = -0.4 = within-day CV accuracy of each recording. One panel per standardization.',
            data={'pairs': rows, 'within_day': day_rows})
        result.tables['per_recording'] = day_rows
        result.tables['pairwise'] = rows
        off = ~np.eye(n, dtype=bool)
        result.summary.update(recordings=labels, within_day_accuracy=[round(o['acc'], 3) for o in own],
                              within_day_perm_p=[round(o['p'], 4) if np.isfinite(o['p']) else None for o in own],
                              **{f'mean_cross_day_accuracy_{m}': round(float(np.nanmean(acc[m][off])), 3) for m in modes},
                              n_perm=p['n_perm'])
        result.notes.append('对角线 = 当天全部传输电极的 block 交叉验证；非对角 = 第 i 天全部 trial 训练、只用两天共同传输的电极。'
                            f'机会水平 1/8；置换次数 {p["n_perm"]}，p 的下限为 1/(n_perm+1)。')
        return result

    # ------------------------------------------------------------------ figures
    def within_figure(self, own, labels, p):
        n = len(labels)
        cols = min(n, 6)
        rows = int(np.ceil(n / cols))
        fig = new_figure(max(11, 2.1 * cols), 4 + 2.2 * rows)
        grid = fig.add_gridspec(1 + rows, cols, height_ratios=[2.2] + [1.6] * rows)
        ax = fig.add_subplot(grid[0, :])
        x = np.arange(n)
        a = np.array([o['acc'] for o in own])
        ax.bar(x, a, color='#2176ae', width=0.6)
        for k, o in enumerate(own):
            if len(o['null']):
                ax.plot([k - 0.3, k + 0.3], [np.percentile(o['null'], 95)] * 2, color='#c76d4b', lw=1.4,
                        label='null 95th pct' if k == 0 else None)
            pt = f"p={o['p']:.3f}" if np.isfinite(o['p']) else ''
            ax.text(k, a[k] + 0.01, f"{a[k]:.3f}\n{pt}", ha='center', va='bottom', fontsize=6.5)
        ax.axhline(1 / 8, color=style.INK2, ls='--', lw=0.8, label='chance 1/8')
        ax.set_ylim(0, min(1.0, max(a.max() * 1.3, 0.3)))
        X.recording_ticks(ax, x, labels)
        ax.set_ylabel('balanced accuracy')
        ax.set_title(f'Within-day block-CV ({p["n_folds"]} folds, λ={p["lam"]:g}) · within-block permutation '
                     f'null ({p["n_perm"]})')
        ax.legend(frameon=False, loc='upper right')
        for k, o in enumerate(own):
            ax = fig.add_subplot(grid[1 + k // cols, k % cols])
            ax.imshow(o['conf'], cmap='Blues', vmin=0, vmax=1)
            ax.set_xticks(range(0, 8, 2))
            ax.set_xticklabels([f'{d * 45}' for d in range(0, 8, 2)], fontsize=5.5)
            ax.set_yticks(range(0, 8, 2))
            ax.set_yticklabels([f'{d * 45}' for d in range(0, 8, 2)], fontsize=5.5)
            ax.set_title(labels[k], fontsize=7, color=X.session_label_color(labels[k]))
            ax.set_xlabel('predicted', fontsize=6)
            if k % cols == 0:
                ax.set_ylabel('true', fontsize=6)
        fig.suptitle('Eight-direction decoding within each recording (confusion rows normalized)', fontsize=9)
        return fig

    def matrices(self, acc, pval, n_common, labels, p):
        modes = list(acc)
        fig = new_figure(4.6 * (len(modes) + 1), 4.8)
        axes = fig.subplots(1, len(modes) + 1)
        top = max(0.5, max(np.nanmax(acc[m]) for m in modes))
        for ax, m in zip(axes, modes):
            X.heatmap(ax, acc[m], labels, cmap='viridis', vmin=0.125, vmax=top,
                      title=f'Balanced accuracy · test features {X.STANDARDIZE_LABELS[m]}')
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if np.isfinite(pval[m][i, j]) and pval[m][i, j] >= 0.05:
                        ax.add_patch(_box(i, j))
        X.heatmap(axes[-1], n_common, labels, fmt='{:.0f}', cmap='Blues', vmin=0, title='Common non-constant electrodes (diag = own)',
                  ylabel='')
        fig.suptitle('Train on recording i (rows), test on recording j (columns) · diagonal = within-day CV · '
                     'red box = permutation p ≥ 0.05 · chance 0.125', fontsize=9)
        return fig

    def vs_days(self, rows, own, sessions, modes):
        fig = new_figure(5.2 * len(modes), 4.4)
        axes = np.atleast_1d(fig.subplots(1, len(modes)))
        for ax, m in zip(axes, modes):
            if rows:
                d = np.array([r['days_apart'] for r in rows], float)
                v = np.array([r.get(f'accuracy_{m}', np.nan) for r in rows], float)
                ceiling = np.array([r['test_within_day_cv'] for r in rows], float)
                fwd = np.array([r['forward'] for r in rows], bool)
                ax.scatter(np.abs(d[fwd]) + X.jitter(fwd.sum(), 0.12, 1), v[fwd], s=18, color='#2176ae',
                           label='train earlier → test later')
                ax.scatter(np.abs(d[~fwd]) + X.jitter((~fwd).sum(), 0.12, 2), v[~fwd], s=18, color='#e07b39',
                           marker='s', label='train later → test earlier')
                ad = np.abs(d)
                if len(np.unique(ad)) > 1:
                    k, b = np.polyfit(ad, v, 1)
                    xs = np.array([ad.min(), ad.max()])
                    ax.plot(xs, k * xs + b, color=style.INK2, ls='--', lw=0.8, label=f'slope {k:.3g}/day')
                ax.scatter(np.zeros(len(own)) - 0.4, [o['acc'] for o in own], marker='D', s=20, color='#4f9d69',
                           label='within-day CV (x = -0.4)')
                ax.text(0.98, 0.02, f'mean test-day CV {np.nanmean(ceiling):.3f}', transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=6.5)
            ax.axhline(1 / 8, color=style.INK2, ls=':', lw=0.8)
            ax.set_xlabel('|days apart| (0 = same day)')
            ax.set_ylabel('balanced accuracy')
            ax.set_ylim(0, 1)
            ax.set_title(f'test features {X.STANDARDIZE_LABELS[m]}', fontsize=8)
            ax.legend(frameon=False, fontsize=6, loc='upper right')
        fig.suptitle('Cross-day direction decoding vs elapsed days', fontsize=9)
        return fig


def _box(i, j):
    from matplotlib.patches import Rectangle
    return Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='#ef2727', lw=1.4)


SECTION = CrossDayDirectionDecoding()
