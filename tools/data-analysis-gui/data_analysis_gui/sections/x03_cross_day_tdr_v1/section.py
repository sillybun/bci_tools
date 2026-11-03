"""X03 v1 — cross-day TDR direction plane and variance.

Per recording: Mante-style TDR on the topbar window rates as in neural_behavior
`compare_tdr_records.py` / `population_tdr.py` (per-channel OLS on
[const, cos θ, sin θ, gain, time], QR over gain → time → cos → sin), with block
cross-validated projections (axes fitted without the projected trial).

Between recordings i and j (electrodes transmitted by both, non-constant on both):
  * principal angles between the two direction planes (fitted on all trials of each)
  * transfer: day j trials z-scored (own statistics by default) and projected on
    day i's plane; the angle of the projection is read out against the target
    (Fisher–Lee circular r, rotation invariant; median |error| with no offset fitted,
    since both planes are anchored to the cos/sin task regressors)
  * share of day j's total z-scored variance captured by day i's plane.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

from ...core import analysis as A
from ...core import crossday as X
from ...core import style
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure


def centroids(scores, y):
    out = np.full((8, 2), np.nan)
    for d in range(8):
        sel = (y == d) & np.isfinite(scores).all(1)
        if sel.any():
            out[d] = scores[sel].mean(0)
    return out


def fit_plane(x, y):
    """All-trial TDR on x (trials × electrodes, all non-constant); returns plane and z stats."""
    z, mu, sd = A.zscore(x)
    design, names = A.tdr_design(y, x.sum(1), np.arange(len(x)))
    fit = A.tdr_axes(z, design, names)
    cols = [fit['axis_names'].index('cos'), fit['axis_names'].index('sin')]
    total = z.var(0, ddof=1).sum()
    var = {n: float((z @ fit['Q'][:, i]).var(ddof=1) / total) for i, n in enumerate(fit['axis_names'])}
    return dict(P=fit['Q'][:, cols], Q=fit['Q'], names=fit['axis_names'], mu=mu, sd=sd, var=var)


class CrossDayTDR(Section):
    id = 'x03'
    mode = 'cross'
    title = '跨天 TDR + variance'
    description = ('每份记录交叉验证 TDR（方向平面方差、圆周相关），共同电极上的 TDR 轴与方向平面主角度，'
                   '把第 j 天 trial 投影到第 i 天的轴上（跨天迁移）的圆周相关矩阵，以及方差解释趋势')
    params = (
        Param('n_folds', '交叉验证 block 折数', 'int', 5, minimum=2, maximum=10),
        Param('test_standardize', '迁移时第 j 天特征标准化', 'choice', 'test_own', choices=('test_own', 'train'),
              help='test_own = 用第 j 天自身均值/SD z-score；train = 用第 i 天统计量'),
        Param('projection_ref', '投影小图使用的参考记录序号（1 = 第一份）', 'int', 1, minimum=1),
    )

    def run(self, ctx: AnalysisContext) -> SectionResult:
        p = ctx.params
        sessions, n = ctx.sessions, len(ctx.sessions)
        labels = X.recording_labels(s.recording for s in sessions)
        result = SectionResult()
        data, own = [], []
        for k, s in enumerate(sessions):
            ctx.check_cancel()
            ctx.progress(f'TDR {labels[k]}')
            d = X.direction_data(ctx, k)
            if len(d['x']) < 24 or len(np.unique(d['blocks'])) < 3 or len(np.unique(d['y'])) < 8:
                raise ValueError(f'{s.label}: trial 太少或方向/block 不全（{len(d["x"])} trials），无法做 TDR')
            data.append(d)
            own.append(ctx.cached(('xday_tdr_own', id(ctx.cohorts[k]), p['n_folds']),
                                  lambda d=d: self.within_day(d, p['n_folds'])))
        ref = min(max(p['projection_ref'], 1), n) - 1

        # ---------------------------------------------------------- pairwise
        ctx.progress('pairwise planes')
        angle1, angle2 = np.full((n, n), np.nan), np.full((n, n), np.nan)
        tr_r, tr_err, tr_var, n_common = (np.full((n, n), np.nan) for _ in range(4))
        ref_scores = [None] * n
        rows = []
        for i in range(n):
            ctx.check_cancel()
            tr_r[i, i] = own[i]['cv']['circular_r']
            n_common[i, i] = own[i]['n_active']
            tr_err[i, i] = own[i]['cv']['median_abs_error_deg']
            tr_var[i, i] = 100 * (own[i]['var']['cos'] + own[i]['var']['sin'])
            for j in range(n):
                if i == j:
                    continue
                ids, ci, cj = X.common_columns(sessions[i].electrode_ids, sessions[j].electrode_ids)
                xi, xj = data[i]['x'][:, ci], data[j]['x'][:, cj]
                ok = (xi.std(0) > 1e-12) & (xj.std(0) > 1e-12)
                n_common[i, j] = ok.sum()
                if ok.sum() < 10:
                    continue
                xi, xj = xi[:, ok], xj[:, ok]
                fi = self._plane(ctx, i, j, xi, data[i]['y'])
                fj = self._plane(ctx, j, i, xj, data[j]['y'])
                ang = np.degrees(linalg.subspace_angles(fi['P'], fj['P']))
                angle1[i, j], angle2[i, j] = ang.min(), ang.max()
                if p['test_standardize'] == 'train':
                    zj = (xj - fi['mu']) / np.where(fi['sd'] > 1e-12, fi['sd'], 1)
                else:
                    zj, _, _ = A.zscore(xj)
                scores = zj @ fi['P']
                read = X.plane_readout(scores, data[j]['y'])
                zown, _, _ = A.zscore(xj)
                var = float((zown @ fi['P']).var(0, ddof=1).sum() / zown.var(0, ddof=1).sum())
                tr_r[i, j], tr_err[i, j], tr_var[i, j] = read['circular_r'], read['median_abs_error_deg'], 100 * var
                if i == ref:
                    ref_scores[j] = scores
                rows.append(dict(axes_from=sessions[i].label, projected=sessions[j].label,
                                 days_apart=X.day_number(sessions[j]) - X.day_number(sessions[i]), forward=j > i,
                                 common_electrodes=int(ok.sum()),
                                 principal_angle_1_deg=round(float(ang.min()), 2),
                                 principal_angle_2_deg=round(float(ang.max()), 2),
                                 transfer_circular_r=round(read['circular_r'], 4),
                                 transfer_median_abs_error_deg=round(read['median_abs_error_deg'], 2),
                                 transfer_within_45deg=round(read['within_45deg'], 4),
                                 variance_of_projected_day_captured_pct=round(100 * var, 3),
                                 own_plane_variance_pct=round(100 * (fj['var']['cos'] + fj['var']['sin']), 3)))
        ref_scores[ref] = own[ref]['held']

        day_rows = []
        for k, s in enumerate(sessions):
            o = own[k]
            day_rows.append(dict(recording=s.label, date=s.recording.date, trials=len(data[k]['x']),
                                 active_electrodes=o['n_active'],
                                 cv_circular_r=round(o['cv']['circular_r'], 4),
                                 cv_median_abs_error_deg=round(o['cv']['median_abs_error_deg'], 2),
                                 cv_within_45deg=round(o['cv']['within_45deg'], 4),
                                 **{f'variance_{nm}_pct': round(100 * v, 3) for nm, v in o['var'].items()},
                                 plane_variance_pct=round(100 * (o['var']['cos'] + o['var']['sin']), 3),
                                 cv_plane_variance_pct=round(100 * o['cv_var'], 3)))

        ctx.progress('figures')
        lim = np.nanpercentile(np.abs(np.vstack([o['held'] for o in own])), 99) * 1.25
        crit = ctx.criteria
        base = (f'Inputs: included trials of each recording (topbar filter), mean rate (Hz) per stored column in '
                f'the window {crit.window_start:g}-{crit.window_end:g} s after T_TRIAL_START; trials without '
                f'neural frames in the window are dropped. TDR (Mante et al. 2013, as neural_behavior '
                f'population_tdr.py): electrodes z-scored over trials; per-electrode OLS on [const, cos(theta), '
                f'sin(theta), gain = z(total rate over the electrodes used), time = z(trial sequence index)]; the '
                f'coefficient vectors are orthonormalized by QR in the order gain, time, cos, sin and the direction '
                f'plane is the (cos, sin) pair. Within-day cross-validation: {p["n_folds"]} contiguous chronological '
                f'block folds, training excludes blocks within +-1 of the test blocks; z-score statistics and axes '
                f'are fitted on training trials only. Angular readout: angle = atan2(sin score, cos score) vs target '
                f'angle; circular r = Fisher-Lee (rotation invariant); median |error| uses the raw angle difference '
                f'(no offset fitted, axes are anchored to the cos/sin regressors). Variance captured = variance of '
                f'the projection / total variance of the z-scored electrodes. Cross-day pairs (i, j) use only '
                f'electrode ids transmitted by both recordings and non-constant on both (aligned by id, not by '
                f'column); each day plane is fitted on all of its trials on that common set.')

        def proj_rows(scores_list):
            out = []
            for k in range(n):
                sc = scores_list[k]
                if sc is None:
                    continue
                for t in range(len(sc)):
                    out.append(dict(recording=labels[k], trial=t, direction_deg=int(data[k]['y'][t]) * 45,
                                    block=int(data[k]['blocks'][t]),
                                    cos_axis=float(sc[t, 0]) if np.isfinite(sc[t, 0]) else '',
                                    sin_axis=float(sc[t, 1]) if np.isfinite(sc[t, 1]) else ''))
            return out

        def centroid_rows(scores_list):
            out = []
            for k in range(n):
                if scores_list[k] is None:
                    continue
                cen = centroids(scores_list[k], data[k]['y'])
                out += [dict(recording=labels[k], direction_deg=d * 45, cos_axis=float(cen[d, 0]),
                             sin_axis=float(cen[d, 1])) for d in range(8) if np.isfinite(cen[d]).all()]
            return out

        held = [o['held'] for o in own]
        result.add_figure(
            'Within-day cross-validated TDR planes',
            self.projections(held, data, labels, lim,
                             [f"CV r={o['cv']['circular_r']:.2f} · |err| {o['cv']['median_abs_error_deg']:.0f}° · "
                              f"plane {100 * (o['var']['cos'] + o['var']['sin']):.2f}%" for o in own],
                             'Block-CV TDR projection per recording (axes fitted without the projected trials)'),
            method=base + ' Each panel: held-out (cross-validated) projection of every trial of one recording on '
                          'its own direction plane, all non-constant transmitted electrodes; large dots = direction '
                          'centroids; ellipses = 1 SD. ' + style.ELLIPSE_METHOD +
                          'Title = CV circular r, CV median |error|, in-sample plane variance. Shared '
                          'axis limits = 1.25 x 99th percentile of |score| over all panels.',
            data={'trial_scores': proj_rows(held), 'centroids': centroid_rows(held), 'per_recording': day_rows})
        result.add_figure(
            'Variance explained and readout trends', self.trends(own, sessions, labels),
            method=base + ' Panels: (1) % variance captured by the direction plane fitted on all trials '
                          '(in-sample) and by the held-out CV projections; (2) % variance per TDR axis (gain, time, '
                          'cos, sin; in-sample, all trials); (3) within-day CV circular r (left axis) and median '
                          '|angle error| in degrees (right axis).',
            data={'per_recording': day_rows})
        result.add_figure(
            'Principal angles between direction planes', self.angles(angle1, angle2, n_common, labels),
            method=base + ' Principal angles (scipy.linalg.subspace_angles) between the direction planes of '
                          'recordings i and j, both fitted on the common electrode set; smallest and largest of the '
                          'two angles in degrees. Right: number of common non-constant electrodes (diagonal = '
                          'active electrodes of the recording itself). Matrix is symmetric.',
            data={'matrices': X.matrix_rows({'principal_angle_1_deg': angle1, 'principal_angle_2_deg': angle2,
                                             'common_electrodes': n_common}, labels, 'recording_i', 'recording_j')})
        std_text = ('day j z-scored with its own mean/SD' if p['test_standardize'] == 'test_own'
                    else 'day j z-scored with day i mean/SD')
        result.add_figure(
            'Cross-day transfer matrices', self.transfer(tr_r, tr_err, tr_var, labels, p),
            method=base + f' Row i = recording whose plane (axes) is used, column j = recording whose trials are '
                          f'projected ({std_text}). Panels: circular r and median |angle error| of the projection '
                          f'angle vs target; % of the total z-scored variance of day j (own statistics) captured by '
                          f'the plane of day i. Diagonal = within-day block-CV values on all active electrodes of '
                          f'that day (third panel diagonal: in-sample plane variance).',
            data={'matrices': X.matrix_rows({'circular_r': tr_r, 'median_abs_error_deg': tr_err,
                                             'variance_captured_pct': tr_var}, labels, 'axes_from', 'projected')})
        lim2 = np.nanpercentile(np.abs(np.vstack([s for s in ref_scores if s is not None])), 99) * 1.25
        result.add_figure(
            f'Projection on {labels[ref]} axes',
            self.projections(ref_scores, data, labels, lim2,
                             ['own CV (reference)' if k == ref else
                              f"r={tr_r[ref, k]:.2f} · |err| {tr_err[ref, k]:.0f}° · {int(n_common[ref, k])} el."
                              for k in range(n)],
                             f'All recordings projected on the direction plane of {labels[ref]} '
                             f'(common electrodes, z-score: {p["test_standardize"]})'),
            method=base + f' Reference = {labels[ref]}. Each panel j: trials of recording j projected on the '
                          f'reference plane fitted on all reference trials over the electrodes common to both '
                          f'({std_text}); the reference panel shows its own held-out CV projection. Title: circular '
                          f'r, median |error|, number of common electrodes.' + style.ELLIPSE_METHOD,
            data={'trial_scores': proj_rows(ref_scores), 'centroids': centroid_rows(ref_scores)})
        result.add_figure(
            'Transfer vs days apart', self.vs_days(rows, labels),
            method=base + ' Every ordered pair (i, j), i != j: transfer circular r, median |error| and smallest '
                          'principal angle vs |calendar days between the recording dates| (0 = same day; small '
                          'horizontal jitter). Blue = later recording projected on earlier axes, orange = reverse. '
                          'Dashed = least-squares line over all pairs.',
            data={'pairs': rows})
        result.tables['per_recording'] = day_rows
        result.tables['pairwise'] = rows
        off = ~np.eye(n, dtype=bool)
        result.summary.update(
            recordings=labels,
            cv_circular_r=[round(o['cv']['circular_r'], 3) for o in own],
            plane_variance_pct=[round(100 * (o['var']['cos'] + o['var']['sin']), 2) for o in own],
            mean_principal_angle_1_deg=round(float(np.nanmean(angle1[off])), 1),
            mean_transfer_circular_r=round(float(np.nanmean(tr_r[off])), 3),
            mean_transfer_median_abs_error_deg=round(float(np.nanmean(tr_err[off])), 1),
            test_standardize=p['test_standardize'])
        result.notes.append('对角线为当天全部活跃电极的 block 交叉验证结果；非对角元素只用两份记录共同传输且均非恒定的电极，'
                            '轴在第 i 天全部 trial 上拟合，第 j 天的 trial 未参与拟合。主角度：0° = 同一平面，90° = 正交。')
        return result

    # ------------------------------------------------------------------ computation
    @staticmethod
    def within_day(d, n_folds):
        x = d['x']
        active = x.std(0) > 1e-12
        held, full = A.tdr_cv(x[:, active], d['y'], d['blocks'], np.arange(len(x)), n_folds)
        cols = [full['axis_names'].index('cos'), full['axis_names'].index('sin')]
        held = held[:, cols]
        z, _, _ = A.zscore(x[:, active])
        ok = np.isfinite(held).all(1)
        cv_var = float(held[ok].var(0, ddof=1).sum() / z.var(0, ddof=1).sum())
        return dict(held=held, var=full['variance_captured'], cv=X.plane_readout(held, d['y']),
                    n_active=int(active.sum()), cv_var=cv_var)

    @staticmethod
    def _plane(ctx, i, j, x, y):
        key = ('xday_tdr_plane', id(ctx.cohorts[i]), id(ctx.cohorts[j]))
        return ctx.cached(key, lambda: fit_plane(x, y))

    # ------------------------------------------------------------------ figures
    def projections(self, scores, data, labels, lim, subtitles, title):
        n = len(labels)
        cols = n if n <= 5 else min(4, int(np.ceil(n / 2)))
        rows = int(np.ceil(n / cols))
        fig = new_figure(3.3 * cols, 3.4 * rows + 0.5)
        axes = np.atleast_1d(fig.subplots(rows, cols, squeeze=False)).ravel()
        for ax in axes[n:]:
            ax.axis('off')
        for k in range(n):
            ax = axes[k]
            s, y = scores[k], data[k]['y']
            if s is None:
                ax.text(0.5, 0.5, 'too few common electrodes', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(labels[k], fontsize=8, color=X.session_label_color(labels[k]))
                continue
            style.direction_ellipses(ax, s, y)
            for d in range(8):
                sel = y == d
                ax.scatter(s[sel, 0], s[sel, 1], s=6, color=style.direction_color(d), alpha=0.45, lw=0,
                           label=f'{d * 45}°' if k == 0 else None)
            cen = centroids(s, y)
            ring = np.vstack([cen, cen[:1]])
            ax.plot(ring[:, 0], ring[:, 1], color=style.INK, lw=0.9, alpha=0.6)
            ax.scatter(cen[:, 0], cen[:, 1], s=40, c=list(style.DIRECTION_COLORS), edgecolors=style.INK, lw=0.8,
                       zorder=4)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal', adjustable='box')
            ax.axhline(0, color=style.GREY, lw=0.4)
            ax.axvline(0, color=style.GREY, lw=0.4)
            ax.set_title(f'{labels[k]}\n{subtitles[k]}', fontsize=7.5,
                         color=X.session_label_color(labels[k]))
            ax.set_xlabel('cos axis')
            ax.set_ylabel('sin axis')
        fig.legend(*axes[0].get_legend_handles_labels(), ncol=8, frameon=False, loc='outside lower center',
                   markerscale=2.5)
        fig.suptitle(title + '\nLarge dots = centroids; ellipses = 1 SD', fontsize=9)
        return fig

    def trends(self, own, sessions, labels):
        n = len(labels)
        x = np.arange(n)
        fig = new_figure(12, 4.6)
        axes = fig.subplots(1, 3)
        ax = axes[0]
        plane = np.array([100 * (o['var']['cos'] + o['var']['sin']) for o in own])
        cvp = np.array([100 * o['cv_var'] for o in own])
        ax.bar(x - 0.18, plane, width=0.36, color='#2176ae', label='in-sample plane')
        ax.bar(x + 0.18, cvp, width=0.36, color='#9ecae1', label='CV plane')
        for xi, v in zip(x, plane):
            ax.text(xi - 0.18, v, f'{v:.2f}', ha='center', va='bottom', fontsize=6)
        ax.set_ylabel('% of total z-scored variance')
        ax.set_title('Direction-plane variance')
        ax.legend(frameon=False)
        ax = axes[1]
        for name, color in (('gain', style.GREY), ('time', '#30638e'), ('cos', '#d1495b'), ('sin', '#edae49')):
            ax.plot(x, [100 * o['var'].get(name, np.nan) for o in own], 'o-', color=color, label=name, ms=4)
        ax.set_ylim(bottom=0)
        ax.set_ylabel('% of total z-scored variance')
        ax.set_title('Variance captured per TDR axis')
        ax.legend(frameon=False, ncol=2)
        ax = axes[2]
        r = [o['cv']['circular_r'] for o in own]
        err = [o['cv']['median_abs_error_deg'] for o in own]
        ax.plot(x, r, 'o-', color='#2176ae', label='CV circular r')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('Fisher–Lee circular r', color='#2176ae')
        tw = ax.twinx()
        tw.plot(x, err, 's--', color='#e07b39', label='median |error|')
        tw.set_ylabel('median |angle error| / °', color='#e07b39')
        tw.set_ylim(0, 100)
        tw.spines['right'].set_visible(True)
        ax.set_title('Within-day CV angular readout')
        for a in axes:
            X.recording_ticks(a, x, labels)
        fig.suptitle('Within-day TDR across recordings', fontsize=9)
        return fig

    def angles(self, a1, a2, n_common, labels):
        fig = new_figure(13, 4.6)
        axes = fig.subplots(1, 3)
        X.heatmap(axes[0], a1, labels, fmt='{:.0f}', cmap='viridis_r', vmin=0, vmax=90,
                  title='Smallest principal angle / °', xlabel='', ylabel='')
        X.heatmap(axes[1], a2, labels, fmt='{:.0f}', cmap='viridis_r', vmin=0, vmax=90,
                  title='Largest principal angle / °', xlabel='', ylabel='')
        X.heatmap(axes[2], n_common, labels, fmt='{:.0f}', cmap='Blues', vmin=0,
                  title='Common non-constant electrodes (diag = own active)', xlabel='', ylabel='')
        fig.suptitle('Direction planes fitted per recording on the electrodes both recordings transmitted '
                     '(0° = identical, 90° = orthogonal)', fontsize=9)
        return fig

    def transfer(self, r, err, var, labels, p):
        fig = new_figure(13, 4.8)
        axes = fig.subplots(1, 3)
        X.heatmap(axes[0], r, labels, cmap='viridis', vmin=0, vmax=1, title='Circular r (Fisher–Lee)',
                  xlabel='projected recording (j)', ylabel='axes from recording (i)')
        X.heatmap(axes[1], err, labels, fmt='{:.0f}', cmap='viridis_r', vmin=0, vmax=90,
                  title='Median |angle error| / ° (no offset fitted)', xlabel='projected recording (j)', ylabel='')
        X.heatmap(axes[2], var, labels, fmt='{:.2f}', cmap='viridis', vmin=0,
                  title="% of day j variance in day i's plane", xlabel='projected recording (j)', ylabel='')
        fig.suptitle(f'Cross-day transfer of the TDR direction plane · diagonal = within-day CV · '
                     f'test z-score: {p["test_standardize"]}', fontsize=9)
        return fig

    def vs_days(self, rows, labels):
        fig = new_figure(12, 4.2)
        axes = fig.subplots(1, 3)
        if rows:
            d = np.array([r['days_apart'] for r in rows], float)
            for ax, key, name in ((axes[0], 'transfer_circular_r', 'transfer circular r'),
                                  (axes[1], 'transfer_median_abs_error_deg', 'median |error| / °'),
                                  (axes[2], 'principal_angle_1_deg', 'smallest principal angle / °')):
                v = np.array([r[key] for r in rows], float)
                fwd = np.array([r['forward'] for r in rows], bool)
                ax.scatter(np.abs(d[fwd]) + X.jitter(fwd.sum(), 0.12, 1), v[fwd], s=18, color='#2176ae',
                           label='later day on earlier axes')
                ax.scatter(np.abs(d[~fwd]) + X.jitter((~fwd).sum(), 0.12, 2), v[~fwd], s=18, color='#e07b39',
                           marker='s', label='earlier day on later axes')
                ad = np.abs(d)
                if len(np.unique(ad)) > 1 and np.isfinite(v).sum() > 2:
                    ok = np.isfinite(v)
                    k, b = np.polyfit(ad[ok], v[ok], 1)
                    xs = np.array([ad.min(), ad.max()])
                    ax.plot(xs, k * xs + b, color=style.INK2, lw=0.8, ls='--', label=f'slope {k:.3g}/day')
                ax.set_xlabel('|days apart| (0 = same day)')
                ax.set_ylabel(name)
                ax.legend(frameon=False, fontsize=6)
        fig.suptitle('Cross-day TDR transfer vs elapsed time (every ordered pair)', fontsize=9)
        return fig


SECTION = CrossDayTDR()
