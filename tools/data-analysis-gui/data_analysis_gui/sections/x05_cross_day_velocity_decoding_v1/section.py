"""X05 v1 — cross-day cursor-velocity decoding (R², RMSE) with Ridge and Vel-KF.

Bins follow core `velocity_bins` (cursor-displacement velocity, trailing neural
window), as in neural_behavior `compare_velocity_records.py`.

  * within-day: Ridge and Vel-KF (`data_analysis_gui.decoders`), five
    chronological block folds with ±1-block purge (`decoders.section_cv`)
  * train day i / test day j: fitted on all bins of day i using only the
    electrodes both recordings transmitted; each decoder normalises with day i
    statistics; option to re-centre or re-standardize the test day
    (`decoders.align_features`, identical to core.crossday.transform)
R² is the mean over x and y; RMSE is the vector RMSE (task units / s).
"""
from __future__ import annotations

import numpy as np

from ... import decoders as D
from ...core import crossday as X
from ...core import analysis as A
from ...core import style
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure
from ..s10_velocity_decoding_v1.section import block_bootstrap, integrate


def spec_of(ctx, p) -> D.BinSpec:
    # Velocity-bin quality thresholds and policy come from the cohort TrialFilter (None fields).
    return D.BinSpec(offset=float(p['offset']), bin_s=float(p['bin']), n_bins=int(p['n_bins']), until='window')


def transfer(ctx, k_i, k_j, ci, cj, names, spec, p):
    """Fit on day i (columns ci), predict day j (columns cj); scores per decoder."""
    sessions, cohorts = ctx.sessions, ctx.cohorts
    tr = D.build_data(cohorts[k_i], spec, cache=ctx.cached).columns(ci)
    te = D.build_data(cohorts[k_j], spec, cache=ctx.cached).columns(cj)
    n_el = int((tr.features.std(0) > 1e-12).sum())
    if n_el < 5:
        return None
    out = dict(n_electrodes=n_el)
    for m in names:
        dec = D.get(m)
        dp = D.decoder_params(dec, p)
        key = ('xday_transfer', id(cohorts[k_i]), id(cohorts[k_j]), m, spec.key(cohorts[k_i].criteria),
               tuple(sorted(dp.items())),
               p['test_standardize'], tuple(map(int, ci)), tuple(map(int, cj)))

        def compute(dec=dec, dp=dp):
            kw = {}
            if D.needs_native(dec, te):
                kw = dict(train_native=D.build_data(cohorts[k_i], spec, dec.BIN_SECONDS, cache=ctx.cached).columns(ci),
                          test_native=D.build_data(cohorts[k_j], spec, dec.BIN_SECONDS, cache=ctx.cached).columns(cj))
            pred, info, _ = D.fit_predict(dec, tr, te, dp, test_mode=p['test_standardize'], **kw)
            return dict(X.velocity_scores(te.velocity, pred), coverage=float(np.isfinite(pred).all(1).mean()),
                        seconds=info['fit_seconds'] + info['predict_seconds'])
        ctx.check_cancel()
        ctx.progress(f'{D.label(m)}: {sessions[k_i].label} → {sessions[k_j].label}')
        out[m] = ctx.cached(key, compute)
    return out


class CrossDayVelocityDecoding(Section):
    id = 'x05'
    mode = 'cross'
    cache_code_dependencies = ('sections/s10_velocity_decoding_v1',)
    title = '跨天速度解码准确性 r2 rmse'
    description = ('每天 Ridge 与 Vel-KF 的 block 交叉验证 R²/RMSE；共同电极上第 i 天训练、第 j 天测试的 R² 与 RMSE 矩阵')
    params = (
        Param('bin', 'bin 宽 / s', 'float', 0.1, minimum=0.02, maximum=0.5),
        Param('offset', '首个 bin 相对 T_TRIAL_START / s', 'float', 0.2, minimum=0.0),
        Param('n_bins', '每 trial bin 数', 'int', 10, minimum=1, maximum=200),
        Param('lam_ridge', 'Ridge λ（惩罚 n·λ）', 'float', 1.0, minimum=1e-8),
        Param('lam_kf', 'KF 最小二乘 λ', 'float', 1e-4, minimum=0.0),
        Param('q_shrinkage', 'KF 观测协方差收缩 α', 'float', 0.5, minimum=0.0, maximum=1.0),
        Param('test_standardize', '测试日特征标准化', 'choice', 'train', choices=X.STANDARDIZE,
              help='train = 训练日均值/SD；recentre = 测试日均值；test_own = 测试日均值/SD'),
    )

    def run(self, ctx: AnalysisContext) -> SectionResult:
        p = ctx.params
        names = ('ridge', 'velocity_kf')
        sessions, n = ctx.sessions, len(ctx.sessions)
        labels = X.recording_labels(s.recording for s in sessions)
        dates = [s.recording.date for s in sessions]
        spec = spec_of(ctx, p)
        result = SectionResult()
        bins, own, overview_rows = [], [], []
        for k, s in enumerate(sessions):
            ctx.check_cancel()
            ctx.progress(f'within-day velocity CV {labels[k]}')
            vb = D.build_data(ctx.cohorts[k], spec, cache=ctx.cached)
            if vb.n < 50 or len(np.unique(vb.block)) < 3:
                raise ValueError(f'{s.label}: 有效速度 bin 太少（{vb.n}）或 block 不足')
            bins.append(vb)
            cv = D.section_cv(ctx, ctx.cohorts[k], names, spec, p, data=vb)
            predictions = {m: cv['results'][m]['pred'] for m in names}
            overview_rows.extend(self.within_overview_rows(vb, predictions, spec.bin_s, labels[k]))
            row = dict(n_bins=int(vb.n))
            for m in names:
                pr = cv['results'][m]['pred']
                t = np.isfinite(pr).all(1)
                row[m] = dict(X.velocity_scores(vb.velocity[t], pr[t]), n_tested=int(t.sum()),
                              seconds=cv['results'][m]['seconds'])
            row['n_tested'] = int(max(row[m]['n_tested'] for m in names))
            own.append(row)
        mats = {(m, q): np.full((n, n), np.nan) for m in names for q in ('r2', 'rmse')}
        n_common = np.full((n, n), np.nan)
        rows = []
        for i in range(n):
            ctx.check_cancel()
            n_common[i, i] = int((bins[i].features.std(0) > 1e-12).sum())
            for m in names:
                mats[m, 'r2'][i, i], mats[m, 'rmse'][i, i] = own[i][m]['r2'], own[i][m]['rmse']
            for j in range(n):
                if i == j:
                    continue
                _, ci, cj = X.common_columns(sessions[i].electrode_ids, sessions[j].electrode_ids)
                out = transfer(ctx, i, j, ci, cj, names, spec, p)
                if out is None:
                    continue
                n_common[i, j] = out['n_electrodes']
                row = dict(train=sessions[i].label, test=sessions[j].label,
                           days_apart=X.day_number(sessions[j]) - X.day_number(sessions[i]),
                           common_electrodes=out['n_electrodes'], test_standardize=p['test_standardize'])
                for m in names:
                    mats[m, 'r2'][i, j], mats[m, 'rmse'][i, j] = out[m]['r2'], out[m]['rmse']
                    row.update({f'{m}_r2': round(out[m]['r2'], 4), f'{m}_rmse': round(out[m]['rmse'], 4),
                                f'{m}_r': round(out[m]['r'], 4), f'{m}_coverage': round(out[m]['coverage'], 4),
                                f'{m}_test_within_day_r2': round(own[j][m]['r2'], 4)})
                rows.append(row)
        day_rows = [dict(recording=s.label, date=s.recording.date, bins=own[k]['n_bins'], tested_bins=own[k]['n_tested'],
                         speed_rms=round(float(np.sqrt(np.mean((bins[k].velocity ** 2).sum(1)))), 4),
                         **{f'{m}_{q}': round(own[k][m][q], 4) for m in names for q in ('r2', 'rmse', 'r')},
                         **{f'{m}_cv_seconds': own[k][m]['seconds'] for m in names})
                    for k, s in enumerate(sessions)]
        ctx.progress('figures')
        base = self.base_method(spec, names, p, bins[0]) + ' ' + D.cohort_rule_text(ctx.cohorts[0], spec)
        result.add_figure('Within-day velocity and position',
                          self.within_overview(overview_rows, labels, dates, ('Velocity', 'Position')),
                          method=base + ' Dots are within-recording held-out mean x/y R² for Ridge and Vel-KF on '
                          'matched predictions. Position is integrated per uninterrupted trial from the true '
                          'first-bin position; only predicted endpoints are scored. Bars: 95% block-bootstrap '
                          'percentile intervals over the held-out predictions.',
                          data=dict(scores=[r for r in overview_rows if r['metric'] in ('Velocity', 'Position')]))
        result.add_figure('Within-day velocity by axis',
                          self.within_overview(overview_rows, labels, dates, ('Velocity X', 'Velocity Y')),
                          method=base + ' Within-recording held-out R² for each velocity axis on the same '
                          'bins for both decoders. Bars: 95% block-bootstrap percentile intervals.',
                          data=dict(scores=[r for r in overview_rows if r['metric'] in ('Velocity X', 'Velocity Y')]))
        result.add_figure('Within-day velocity decoding', self.within_figure(own, bins, labels, names, p),
                          method=base + ' Left/middle: within-day held-out R² (mean of per-axis R² = 1 − SSE/SST) and '
                          'vector RMSE = sqrt(mean over axes of MSE) per recording and decoder, from 5 chronological '
                          'block folds with ±1 block purge, all transmitted electrodes, bins each decoder predicted. '
                          'Right: RMS cursor speed sqrt(mean ‖v‖²) of the evaluation bins (scale for RMSE).',
                          data=dict(per_recording=day_rows))
        result.add_figure('Cross-day R2 and RMSE matrices', self.matrices(mats, n_common, labels, names, p),
                          method=base + ' Matrices: row = training recording i (model fitted on all its bins, '
                          'electrodes shared with j), column = test recording j (all its bins); diagonal = within-day '
                          'block CV on all transmitted electrodes. Test features: '
                          f'{X.STANDARDIZE_LABELS[p["test_standardize"]]}. Common electrodes = electrode ids both '
                          'recordings transmitted that are non-constant on day i.',
                          data=dict(pairwise=rows, **{f'{m}_{q}_matrix': self.matrix_rows(mats[m, q], labels)
                                                      for m in names for q in ('r2', 'rmse')},
                                    common_electrodes=self.matrix_rows(n_common, labels)))
        result.add_figure('Velocity transfer vs days apart', self.vs_days(rows, own, names),
                          method=base + ' Each ordered pair (i→j) at |days apart| (calendar dates; jittered ±0.12 for '
                          'visibility, decoders offset by 0.15); hollow diamonds at x = −0.5 = within-day CV per '
                          'recording.', data=dict(pairwise=rows, per_recording=day_rows))
        cmp_fig, cmp_data = self.comparison(own, rows, mats, labels, names)
        result.add_figure('Decoder comparison', cmp_fig,
                          method=base + ' Decoder comparison: (a) within-day block-CV R² (mean of x/y R²), decoder × '
                          'recording; (b) per decoder, mean over all ordered pairs i≠j of the transfer R² (bar) with '
                          'every pair as a dot, beside the mean within-day R² (hollow bar); (c) transfer R² averaged '
                          'per |days apart| (0 = mean within-day CV) per decoder; (d) same for RMSE. Transfer = fitted '
                          f'on all bins of day i, common electrodes, test features {X.STANDARDIZE_LABELS[p["test_standardize"]]}.',
                          data=cmp_data)
        result.tables['decoder_comparison'] = cmp_data['summary']
        result.tables['per_recording'] = day_rows
        result.tables['pairwise'] = rows
        off = ~np.eye(n, dtype=bool)
        result.summary.update(
            recordings=labels, decoders=list(names),
            **{f'within_day_{m}_r2': [round(o[m]['r2'], 3) for o in own] for m in names},
            **{f'within_day_{m}_rmse': [round(o[m]['rmse'], 4) for o in own] for m in names},
            **{f'mean_cross_day_{m}_r2': round(float(np.nanmean(mats[m, 'r2'][off])), 3) for m in names},
            test_standardize=p['test_standardize'])
        result.notes.append('对角线 = 当天全部传输电极的 block 交叉验证；非对角 = 第 i 天全部 bin 训练、只用两天共同传输的电极。'
                            'R2 可为负（比预测常数均值更差）。')
        return result

    @staticmethod
    def within_overview_rows(vb, predictions, bin_s, label):
        common = np.logical_and.reduce([np.isfinite(pred).all(axis=1) for pred in predictions.values()])
        common &= np.isfinite(vb.velocity).all(axis=1)
        truth_pos = integrate(vb, vb.velocity, bin_s)
        predicted_pos = {m: integrate(vb, np.where(common[:, None], pred, np.nan), bin_s)
                         for m, pred in predictions.items()}
        trials = sorted(set(truth_pos).intersection(*(set(pos) for pos in predicted_pos.values())))
        block_by_trial = dict(zip(vb.trial, vb.block))
        position_blocks = np.repeat([block_by_trial[t] for t in trials],
                                    [len(truth_pos[t]) - 1 for t in trials])
        rows = []

        def add(metric, model, truth, pred, blocks, n_trials):
            if len(truth) < 2:
                value, lo, hi = np.nan, np.nan, np.nan
            else:
                value = float(np.mean(A.r2_score(truth, pred)))
                lo, hi = (block_bootstrap(truth, pred, blocks, 200)
                          if len(np.unique(blocks)) >= 2 else (np.nan, np.nan))
            rows.append(dict(recording=label, metric=metric, decoder=model, r2=value,
                             ci_low=float(lo), ci_high=float(hi), n_trials=n_trials, n_points=len(truth)))

        y, blocks = vb.velocity[common], vb.block[common]
        n_trials = len(np.unique(vb.trial[common]))
        for model, prediction in predictions.items():
            pred = prediction[common]
            add('Velocity', model, y, pred, blocks, n_trials)
            for axis, metric in enumerate(('Velocity X', 'Velocity Y')):
                add(metric, model, y[:, axis:axis + 1], pred[:, axis:axis + 1], blocks, n_trials)
            # Skip the seeded position, which would contribute an artificially perfect prediction.
            actual = np.concatenate([truth_pos[t][1:] for t in trials]) if trials else np.empty((0, 2))
            estimated = (np.concatenate([predicted_pos[model][t][1:] for t in trials])
                         if trials else np.empty((0, 2)))
            add('Position', model, actual, estimated, position_blocks, len(trials))
        return rows

    @staticmethod
    def within_overview(rows, labels, dates, metrics):
        fig = new_figure(max(11, 1.5 * len(labels)), 5.2)
        axes = fig.subplots(1, 2, sharey=True)
        x = np.arange(len(labels))
        day_boundaries = [i - .5 for i in range(1, len(dates)) if dates[i] != dates[i - 1]]
        for ax, metric in zip(axes, metrics):
            for boundary in day_boundaries:
                ax.axvline(boundary, color='#c8c8c8', linestyle='--', linewidth=.9, zorder=1)
            metric_rows = [row for row in rows if row['metric'] == metric]
            for model, offset, color, name in (('velocity_kf', -.16, '#c0392b', 'Vel-KF'),
                                                ('ridge', .16, D.COLORS['ridge'], 'Ridge')):
                points = [row for row in metric_rows if row['decoder'] == model]
                y = np.array([row['r2'] for row in points])
                lo = np.array([row['ci_low'] for row in points])
                hi = np.array([row['ci_high'] for row in points])
                ok = np.isfinite(lo) & np.isfinite(hi)
                ax.errorbar(x[ok] + offset, (lo[ok] + hi[ok]) / 2, yerr=(hi[ok] - lo[ok]) / 2,
                            fmt='none', ecolor=color, capsize=4)
                ax.scatter(x + offset, y, color=color, label=name, zorder=3)
            n = [row['n_trials'] for row in metric_rows if row['decoder'] == 'ridge']
            X.recording_ticks(ax, x, [f'{label} · n={count}' for label, count in zip(labels, n)], fontsize=7)
            ax.grid(axis='y', alpha=.25)
            ax.set_title(metric)
        # Mark zero only when it falls within the data-driven view.
        if axes[0].get_ylim()[0] < 0 < axes[0].get_ylim()[1]:
            for ax in axes:
                ax.axhline(0, color=style.INK2, lw=.6)
        axes[0].set_ylabel('Held-out R²')
        axes[0].legend(frameon=False, loc='best')
        fig.suptitle('Within-recording block CV · n = evaluated trials · bars = 95% block-bootstrap CI',
                     fontsize=10)
        return fig

    @staticmethod
    def base_method(spec, names, p, vb):
        return (f'Inputs: velocity cohort of each recording; {spec.n_bins} bins of {spec.bin_s * 1000:g} ms from '
                f'T_TRIAL_START+{spec.offset:g} s; GT = cursor-displacement velocity over the bin; features = mean '
                f'rate (Hz) of the frames in the trailing {vb.extra["neural_window"] * 1000:g} ms. Decoders: '
                + D.describe(names, p, spec))

    @staticmethod
    def matrix_rows(matrix, labels):
        return [dict(train=labels[i], **{labels[j]: (None if not np.isfinite(matrix[i, j]) else float(matrix[i, j]))
                                         for j in range(len(labels))}) for i in range(len(labels))]

    # ------------------------------------------------------------------ figures
    def within_figure(self, own, bins, labels, names, p):
        n = len(labels)
        x = np.arange(n)
        k = len(names)
        width = 0.8 / k
        fig = new_figure(12 + 0.5 * k, 4.4)
        axes = fig.subplots(1, 3)
        for ax, q, title in ((axes[0], 'r2', 'R² (mean of x, y)'), (axes[1], 'rmse', 'RMSE / (units/s)')):
            for j, m in enumerate(names):
                off = (j - (k - 1) / 2) * width
                v = [o[m][q] for o in own]
                ax.bar(x + off, v, width=width, color=D.COLORS[m], label=D.label(m))
                for xi, vi in zip(x + off, v):
                    ax.text(xi, vi if vi >= 0 else 0, f'{vi:.2f}', ha='center', va='bottom', fontsize=5)
            ax.set_ylabel(title)
            ax.legend(frameon=False, fontsize=6)
        axes[0].axhline(0, color=style.INK2, lw=0.6)
        axes[0].set_title('Within-day block-CV R²')
        axes[1].set_title('Within-day block-CV RMSE')
        speed = [np.sqrt(np.mean((b.velocity ** 2).sum(1))) for b in bins]
        axes[2].bar(x, speed, color=style.GREY, width=0.6)
        axes[2].set_ylabel('RMS cursor speed / (units/s)')
        axes[2].set_title('Velocity scale (RMSE reference)')
        for ax in axes:
            X.recording_ticks(ax, x, labels)
        fig.suptitle(f'Velocity decoding per recording · bins {p["bin"] * 1000:.0f} ms × {p["n_bins"]} from '
                     f'+{p["offset"]:g} s', fontsize=9)
        return fig

    def matrices(self, mats, n_common, labels, names, p):
        k = len(names)
        fig = new_figure(14, 4.5 * max(2, k))
        axes = fig.subplots(max(2, k), 3)
        r2max = max(0.1, np.nanmax([np.nanmax(mats[m, 'r2']) for m in names]))
        rmax = np.nanmax([np.nanmax(mats[m, 'rmse']) for m in names])
        for r, m in enumerate(names):
            X.heatmap(axes[r, 0], mats[m, 'r2'], labels, cmap='viridis', vmin=0,
                      vmax=r2max, title=f'{D.label(m)} R² (negative shown below colour range)')
            X.heatmap(axes[r, 1], mats[m, 'rmse'], labels, fmt='{:.3g}', cmap='viridis_r', vmin=0, vmax=rmax,
                      title=f'{D.label(m)} RMSE', ylabel='')
        for r in range(k, max(2, k)):
            axes[r, 0].axis('off')
            axes[r, 1].axis('off')
        X.heatmap(axes[0, 2], n_common, labels, fmt='{:.0f}', cmap='Blues', vmin=0,
                  title='Common non-constant electrodes (diag = own)', ylabel='')
        for r in range(1, max(2, k)):
            axes[r, 2].axis('off')
        axes[1, 2].text(0, 1, '\n'.join([
            'Rows: training recording i (all bins).',
            'Columns: test recording j.',
            'Diagonal: within-day block CV,',
            '  all transmitted electrodes.',
            'Off-diagonal: electrodes both',
            '  recordings transmitted.',
            f'Test features: {X.STANDARDIZE_LABELS[p["test_standardize"]]}.',
            'R² < 0 = worse than predicting',
            '  the training mean velocity.']), va='top', fontsize=8, transform=axes[1, 2].transAxes)
        fig.suptitle('Cross-day velocity decoding', fontsize=9)
        return fig

    def vs_days(self, rows, own, names):
        fig = new_figure(11, 4.2)
        axes = fig.subplots(1, 2)
        markers = 'os^vDP'
        k = len(names)
        for ax, q, title in ((axes[0], 'r2', 'R²'), (axes[1], 'rmse', 'RMSE')):
            for j, m in enumerate(names):
                off = (j - (k - 1) / 2) * 0.15
                if rows:
                    d = np.abs(np.array([r['days_apart'] for r in rows], float))
                    v = np.array([r[f'{m}_{q}'] for r in rows], float)
                    ax.scatter(d + X.jitter(len(d), 0.12, j) + off, v, s=16, color=D.COLORS[m],
                               label=f'{D.label(m)} cross-day', marker=markers[j % len(markers)])
                ax.scatter(np.full(len(own), -0.5) + off, [o[m][q] for o in own], marker='D', s=20,
                           facecolors='none', edgecolors=D.COLORS[m], label=f'{D.label(m)} within-day (x=-0.5)')
            if q == 'r2':
                ax.axhline(0, color=style.INK2, lw=0.6)
            ax.set_xlabel('|days apart| (0 = same day)')
            ax.set_ylabel(title)
            ax.legend(frameon=False, fontsize=5.5)
        fig.suptitle('Velocity decoding transfer vs elapsed days (every ordered pair)', fontsize=9)
        return fig


    def comparison(self, own, rows, mats, labels, names):
        k, n = len(names), len(labels)
        fig = new_figure(15, 9)
        grid = fig.add_gridspec(2, 2)
        colors = [D.COLORS[m] for m in names]
        dlabels = [D.label(m) for m in names]
        # (a) decoder × recording heatmap
        ax = fig.add_subplot(grid[0, 0])
        heat = np.array([[o[m]['r2'] for o in own] for m in names])
        X.heatmap(ax, heat, dlabels, col_labels=labels, cmap='viridis', vmin=0, vmax=max(0.1, np.nanmax(heat)),
                  title='(a) within-day block-CV R²', xlabel='recording', ylabel='')
        # (b) mean off-diagonal transfer R²
        ax = fig.add_subplot(grid[0, 1])
        off = ~np.eye(n, dtype=bool)
        summary = []
        for j, m in enumerate(names):
            pairs = mats[m, 'r2'][off]
            pairs = pairs[np.isfinite(pairs)]
            within = float(np.nanmean([o[m]['r2'] for o in own]))
            mean_t = float(np.mean(pairs)) if len(pairs) else np.nan
            ax.bar(j - 0.2, within, 0.38, facecolor='none', edgecolor=colors[j], lw=1.2,
                   label='mean within-day' if j == 0 else None)
            ax.bar(j + 0.2, mean_t, 0.38, color=colors[j], label='mean cross-day' if j == 0 else None)
            ax.scatter(np.full(len(pairs), j + 0.2) + X.jitter(len(pairs), 0.08, j), pairs, s=10, color=style.INK,
                       zorder=3)
            ax.text(j + 0.2, mean_t if np.isfinite(mean_t) else 0, f'{mean_t:.2f}', ha='center', va='bottom',
                    fontsize=6)
            summary.append(dict(decoder=m, label=D.label(m), mean_within_day_r2=within, mean_cross_day_r2=mean_t,
                                n_pairs=int(len(pairs)),
                                mean_cross_day_rmse=float(np.nanmean(mats[m, 'rmse'][off])) if n > 1 else np.nan))
        ax.axhline(0, color=style.INK2, lw=0.6)
        ax.set_xticks(range(k), dlabels, rotation=20, ha='right', fontsize=7)
        ax.set_ylabel('R²')
        ax.set_title('(b) within-day vs transfer (dots = ordered pairs)', fontsize=8)
        ax.legend(frameon=False, fontsize=6)
        # (c, d) by days apart
        by_days = []
        for col, q, title in ((0, 'r2', '(c) R² vs |days apart|'), (1, 'rmse', '(d) RMSE vs |days apart|')):
            ax = fig.add_subplot(grid[1, col])
            for j, m in enumerate(names):
                days = sorted({abs(r['days_apart']) for r in rows})
                xs, ys = [0], [float(np.nanmean([o[m][q] for o in own]))]
                for d in days:
                    vals = [r[f'{m}_{q}'] for r in rows if abs(r['days_apart']) == d]
                    xs.append(d)
                    ys.append(float(np.nanmean(vals)))
                    if col == 0:
                        by_days.append(dict(decoder=m, days_apart=d, n_pairs=len(vals), mean_r2=ys[-1]))
                    else:
                        next(b for b in by_days if b['decoder'] == m and b['days_apart'] == d)['mean_rmse'] = ys[-1]
                ax.plot(xs, ys, marker='os^vD'[j % 5], color=colors[j], label=dlabels[j], lw=1.2)
            if q == 'r2':
                ax.axhline(0, color=style.INK2, lw=0.6)
            ax.set_xlabel('|days apart| (0 = within-day CV)')
            ax.set_ylabel(q.upper() if q == 'rmse' else 'R²')
            ax.set_title(title, fontsize=8)
            ax.legend(frameon=False, fontsize=6)
        note = '' if k > 1 else ' · only one decoder selected'
        fig.suptitle(f'Decoder comparison across recordings ({n} recordings){note}', fontsize=10)
        return fig, dict(summary=summary, within_day=[dict(decoder=m, **{labels[i]: own[i][m]['r2'] for i in range(n)})
                                                      for m in names], by_days_apart=by_days)


SECTION = CrossDayVelocityDecoding()
