"""S04 v1 — dimensionality reduction (PCA, TDR and UMAP) of the topbar analysis window.

PCA describes variance without assuming that any particular PC is a gain axis.
The first three result subsections are PCA, UMAP and TDR; detailed projections
and loading maps follow as supplementary views.

TDR follows Mante et al. (2013) as re-implemented in neural_behavior
`population_tdr.py`: per-channel regression on [const, cos θ, sin θ, gain, time],
QR over the axis order gain → time → cos → sin, projection on the orthogonalized
direction plane. Evidential projections are cross-validated over chronological
block folds (axes fitted without the projected trial); the null shuffles
directions within blocks and refits everything.
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

from ...core import analysis as A
from ...core import style
from ...core.channels import draw_channel_map
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure
from . import diagnostics as DG

DIRS = np.arange(8)


def centroids(scores, directions):
    out = np.full((8, scores.shape[1]), np.nan)
    for d in DIRS:
        sel = directions == d
        if sel.any():
            out[d] = np.nanmean(scores[sel], 0)
    return out


def direction_F(scores, directions):
    """Direction one-way F per score column (how much direction each axis carries)."""
    ok = np.isfinite(scores).all(1)
    if ok.sum() < 10:
        return np.full(scores.shape[1], np.nan)
    return A.oneway_anova(scores[ok], directions[ok])['F']


def fisher_lee(a, b):
    """Rotation-invariant circular correlation (Fisher & Lee 1983): +1 rotation, -1 reflection.

    The Jammalamadaka form (`analysis.circular_corr`) centres on each sample's mean
    direction, which is undefined for balanced eight-direction targets, so its sign
    and size flip with tiny imbalances. Fisher–Lee needs no mean direction.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    A_, B_ = np.sum(np.cos(a) * np.cos(b)), np.sum(np.sin(a) * np.sin(b))
    C_, D_ = np.sum(np.cos(a) * np.sin(b)), np.sum(np.sin(a) * np.cos(b))
    den = (n ** 2 - np.sum(np.cos(2 * a)) ** 2 - np.sum(np.sin(2 * a)) ** 2) *           (n ** 2 - np.sum(np.cos(2 * b)) ** 2 - np.sum(np.sin(2 * b)) ** 2)
    return float(4 * (A_ * B_ - C_ * D_) / np.sqrt(den)) if den > 0 else float('nan')


def angular_readout(scores, directions):
    """Circular agreement of the plane angle with the target angle (rotation offset estimated)."""
    ok = np.isfinite(scores).all(1)
    a = np.arctan2(scores[ok, 1], scores[ok, 0])
    t = np.radians(directions[ok] * 45.0)
    if len(a) < 3:
        return dict(circular_r=np.nan, circular_r_js=np.nan, resultant=np.nan, offset_deg=np.nan,
                    median_abs_error_deg=np.nan, within_45deg_frac=np.nan, angle=a, target=t, error=np.zeros(0))
    offset = np.angle(np.exp(1j * (a - t)).mean())
    err = np.angle(np.exp(1j * (a - t - offset)))
    return dict(circular_r=fisher_lee(t, a), circular_r_js=A.circular_corr(t, a),
                resultant=float(np.abs(np.exp(1j * err).mean())), offset_deg=float(np.degrees(offset) % 360),
                median_abs_error_deg=float(np.degrees(np.median(np.abs(err)))),
                within_45deg_frac=float((np.abs(np.degrees(err)) <= 45).mean()),
                angle=a, target=t, error=err)


def cv_pca(x, blocks, reference, k, n_folds):
    """Held-out PC scores: components fitted on training folds, signs aligned to `reference`."""
    out = np.full((len(x), k), np.nan)
    for _, tr, te in A.block_folds(blocks, n_folds):
        fit = A.pca(x[tr], k)
        comps = fit['components']
        kk = min(len(comps), k)
        sign = np.sign(np.sum(comps[:kk] * reference[:kk], 1))
        sign[sign == 0] = 1
        z, _, _ = A.zscore(x[te], fit['mu'], fit['sd'])
        out[te, :kk] = z @ (comps[:kk] * sign[:, None]).T
    return out


def umap_embedding(x, n_neighbors=15, min_dist=0.1, seed=20261029):
    """Unsupervised full-cohort embedding; labels are never passed to UMAP."""
    from importlib.metadata import version
    from umap import UMAP

    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError('UMAP 需要至少 3 个 trial 的有限二维特征矩阵')
    active = x.std(0) > 1e-12
    if not active.any():
        raise ValueError('UMAP 没有可用的非恒定通道')
    z, _, _ = A.zscore(x[:, active])
    neighbors = min(int(n_neighbors), len(z) - 1)
    model = UMAP(n_components=2, n_neighbors=neighbors, min_dist=float(min_dist),
                 metric='euclidean', init='random', random_state=int(seed), n_jobs=1)
    scores = model.fit_transform(z)
    if scores.shape != (len(z), 2) or not np.isfinite(scores).all():
        raise ValueError('UMAP 未生成有效的二维坐标')
    return dict(scores=scores, n_neighbors=neighbors, min_dist=float(min_dist), seed=int(seed),
                active_channels=int(active.sum()), version=version('umap-learn'))


class Dimensionality(Section):
    id = 's04'
    title = '降维分析 PCA UMAP TDR'
    description = ('PCA 解释变异与增益控制；UMAP 探索局部结构与状态漂移；'
                   'TDR 检验任务结构、折间稳定性及实测速度控制。首图两行三列总览，随后各方法详细主图与补充诊断。')
    params = (
        Param('n_components', 'PCA 主成分数', 'int', 10, minimum=3, maximum=50),
        Param('n_folds', '交叉验证 block 折数', 'int', 5, minimum=2, maximum=10),
        Param('n_perm', 'PCA / TDR 置换次数（block 内打乱方向）', 'int', 100, minimum=0, maximum=2000),
        Param('seed', '随机种子（TDR / UMAP）', 'int', 20261029, minimum=0, maximum=4294967295),
        Param('umap_enabled', '计算 UMAP', 'bool', True),
        Param('umap_neighbors', 'UMAP 邻居数', 'int', 15, minimum=2, maximum=200),
        Param('umap_min_dist', 'UMAP 最小距离', 'float', 0.1, minimum=0, maximum=1),
    )

    def run(self, ctx: AnalysisContext, *, overview_target=None) -> SectionResult:
        p, cohort, session = ctx.params, ctx.cohort, ctx.session
        result = SectionResult()
        x = cohort.window_rates
        ok = np.isfinite(x).all(1)
        if ok.sum() < 16:
            raise ValueError(f'保留的 trial 太少（{int(ok.sum())}），无法做降维分析')
        if (~ok).any():
            result.notes.append(f'{int((~ok).sum())} 个 trial 分析窗口无神经帧，已剔除。')
        x, dirs, blocks = x[ok], cohort.directions[ok], cohort.blocks[ok]
        trial_orders = cohort.orders[ok]
        orders = cohort.starts[ok] - cohort.starts[ok][0]  # actual elapsed time, not order within block
        if len(np.unique(blocks)) < 3:
            raise ValueError('S04 的 block 交叉验证需要至少 3 个有效 block')
        active = x.std(0) > 1e-12
        xa = x[:, active]
        if xa.shape[1] < 4:
            raise ValueError('PCA / TDR 需要至少 4 个非恒定通道')
        total = xa.sum(1)
        win = f'{cohort.criteria.window_start:g}–{cohort.criteria.window_end:g} s'
        banner = (f'{session.label} · {len(x)} trials · {int(active.sum())} active channels · '
                  f'window {win} after T_TRIAL_START')

        # ------------------------------------------------------------- PCA
        ctx.progress('PCA')
        k = min(p['n_components'], xa.shape[1], len(xa) - 1)
        fit = A.pca(xa, k)
        # PC signs are display conventions, never a declaration of what a PC encodes.
        r_tot = DG.correlations(fit['scores'], total)
        sign = np.where(r_tot < 0, -1.0, 1.0)
        fit['scores'] *= sign
        fit['components'] *= sign[:, None]
        r_tot *= sign
        full_explained = self._full_spectrum(xa)
        cv_scores = cv_pca(xa, blocks, fit['components'], k, p['n_folds'])
        F_in, F_cv = direction_F(fit['scores'], dirs), direction_F(cv_scores, dirs)
        win_txt = (f'Input: per included trial, mean firing rate (spike count / frame duration, Hz) of every stored '
                   f'channel in the topbar window T_TRIAL_START + [{cohort.criteria.window_start:g}, '
                   f'{cohort.criteria.window_end:g}) s; trials without neural frames dropped; active channels = '
                   f'nonzero across-trial SD ({int(active.sum())} of {x.shape[1]}). ')
        pca_txt = ('PCA: each active channel z-scored across trials (mean 0, SD 1), SVD of the trials x channels matrix; '
                   'explained variance = squared singular value / sum. Each PC sign is oriented toward positive '
                   'total-rate correlation when defined; this is a display convention, not a gain-axis assumption. ')
        cv_txt = (f'Cross-validated PCA: {p["n_folds"]} contiguous chronological block folds; training trials exclude '
                  'blocks within +-1 of the test blocks (purge); channel z-score parameters and PCs fitted on '
                  'training trials only, test trials projected; PC signs of each fold aligned to the full-data PCs '
                  'by the sign of the loading dot product. ')
        ks = np.arange(1, k + 1)
        tid = cohort.trial_ids[ok]
        result.add_figure(
            'PCA variance and gain axis',
            self.pca_overview(fit, full_explained, total, dirs, r_tot, F_in, F_cv, active, ctx, banner),
            method=win_txt + pca_txt + cv_txt + (
                'Panels: (a) scree = % variance per PC (first 20) with cumulative % on the right axis; '
                '(b) score of the PC with strongest absolute total-rate correlation vs total rate; '
                f'(c) one-way ANOVA F across the 8 directions of each PC score, in-sample and cross-validated, '
                f'dashed line = F critical value at p = 0.05/{k} (Bonferroni over PCs), df = (7, n_trials - 8); '
                '(d) Pearson r of each PC score with the trial total rate; '
                '(e) loadings of that most gain-correlated PC on the physical channel layout, grey = inactive or '
                'untransmitted.'),
            data=dict(
                spectrum=dict(pc=np.arange(1, len(full_explained) + 1), explained_pct=100 * full_explained,
                              cumulative_pct=100 * np.cumsum(full_explained)),
                per_pc=dict(pc=ks, explained_pct=100 * fit['explained'], direction_F_in_sample=F_in,
                            direction_F_cv=F_cv, r_with_total_rate=r_tot),
                gain_pc_vs_total=dict(trial_id=tid, direction_deg=dirs * 45, total_rate_hz=total,
                                     score=fit['scores'][:, int(np.nanargmax(np.abs(r_tot))) if np.isfinite(r_tot).any() else 0]),
                gain_pc_loadings=dict(electrode=session.electrode_ids[active], stored_column=np.flatnonzero(active) + 1,
                                     loading=fit['components'][int(np.nanargmax(np.abs(r_tot))) if np.isfinite(r_tot).any() else 0])))
        cen_in, cen_cv = centroids(fit['scores'][:, :3], dirs), centroids(cv_scores[:, :3], dirs)
        result.add_figure(
            'PCA projections', self.pca_scatter(fit['scores'], cv_scores, dirs, fit['explained'], banner),
            method=win_txt + pca_txt + cv_txt + (
                'Panels: top row in-sample scores PC1-PC2 and PC2-PC3; bottom row the same from cross-validated '
                'projections. Small dots = trials coloured by target direction; large dots = per-direction mean '
                '(centroid) of the plotted scores, joined in direction order. ' + style.ELLIPSE_METHOD +
                'Axes clipped at 1.1 x the 99.5th '
                'percentile of |score|; the number of hidden trials is printed. Axis labels give in-sample % variance.'),
            data=dict(trials=dict(trial_id=tid, block=blocks, direction_deg=dirs * 45,
                                  PC1=fit['scores'][:, 0], PC2=fit['scores'][:, 1], PC3=fit['scores'][:, 2],
                                  PC1_cv=cv_scores[:, 0], PC2_cv=cv_scores[:, 1], PC3_cv=cv_scores[:, 2]),
                      centroids=dict(direction_deg=DIRS * 45,
                                     PC1_in_sample=cen_in[:, 0], PC2_in_sample=cen_in[:, 1], PC3_in_sample=cen_in[:, 2],
                                     PC1_cv=cen_cv[:, 0], PC2_cv=cen_cv[:, 1], PC3_cv=cen_cv[:, 2])))

        # ------------------------------------------------------------- TDR
        ctx.progress('TDR')
        tdr = DG.tdr_fit_cv(xa, dirs, blocks, orders, p['n_folds'])
        held, full = tdr['scores'], tdr['full']
        names = full['axis_names']
        cols = [names.index('cos'), names.index('sin')]
        obs = angular_readout(held[:, cols], dirs)
        ins = angular_readout(full['scores'][:, cols], dirs)
        z, _, _ = A.zscore(xa)
        tested = np.isfinite(held[:, cols]).all(1)
        cv_var = {n: float(np.nanvar(held[tested, i], ddof=1) / z[tested].var(0, ddof=1).sum())
                  for i, n in enumerate(names)}
        null = np.zeros(0)
        if p['n_perm'] > 0:
            rng = np.random.default_rng(p['seed'])
            null = np.empty(p['n_perm'])
            for i in range(p['n_perm']):
                if i % 20 == 0:
                    ctx.check_cancel()
                    ctx.progress(f'TDR permutation {i}/{p["n_perm"]}')
                perm = A.permute_within_block(rng, dirs, blocks)
                s = DG.tdr_fit_cv(xa, perm, blocks, orders, p['n_folds'])['scores']
                null[i] = abs(angular_readout(s[:, cols], perm)['circular_r'])
        valid_null = null[np.isfinite(null)]
        p_perm = (float((1 + np.sum(valid_null >= abs(obs['circular_r']))) / (1 + len(valid_null)))
                  if len(valid_null) == len(null) and len(null) and np.isfinite(obs['circular_r']) else np.nan)
        ok_obs = np.isfinite(held[:, cols]).all(1)

        def readout_row(kind, r, perm_p):
            return dict(kind=kind, fisher_lee_r=r['circular_r'], jammalamadaka_r=r['circular_r_js'],
                        offset_deg=r['offset_deg'], median_abs_error_deg=r['median_abs_error_deg'],
                        within_45deg_frac=r['within_45deg_frac'], permutation_p=perm_p)
        result.add_figure(
            'TDR direction plane',
            self.tdr_figure(held[:, cols], full['scores'][:, cols], dirs, obs, ins, full, cv_var, null, p_perm, banner),
            method=win_txt + (
                'TDR (Mante et al. 2013, as neural_behavior population_tdr.py): channels z-scored across trials; '
                'per channel OLS of z on [const, cos(theta), sin(theta), gain = z-scored trial total rate, '
                'time = z-scored elapsed seconds since the first included trial]; coefficient vectors for gain, time, cos, sin '
                '(in that order) are orthonormalised by QR, each axis signed to agree with its own coefficient vector; '
                'trials are projected onto the resulting axes. Held-out projections: '
                f'{p["n_folds"]} chronological block folds with +-1 block purge; z-score parameters and axes fitted '
                'on training trials, test trials projected. In-sample = fitted on all trials (reference only, '
                'not evidence). '
                'Angular readout: decoded angle = atan2(sin-axis score, cos-axis score); rotation offset = circular '
                'mean of (decoded - target); error = wrapped (decoded - target - offset). Circular r = Fisher-Lee '
                '(1983) rotation-invariant circular correlation between target and decoded angle (+1 rotation, -1 '
                'reflection); the Jammalamadaka form is listed in readout_summary for comparison. '
                'Variance captured = variance of the axis projection / total variance of the z-scored channels '
                '(held-out: variance of held-out projections over the same total). '
                f'Permutation null: {len(null)} within-block shuffles of direction labels (seed {p["seed"]}), each '
                'refitting the complete held-out TDR and recomputing |Fisher-Lee r|; p = (1 + #null >= observed) / '
                '(1 + n_perm). '
                + style.ELLIPSE_METHOD +
                'Panels: (a) held-out cos/sin plane with direction centroids; (b) in-sample plane; (c) variance '
                'captured per axis; (d) target vs decoded angle minus offset, unwrapped to within 180 deg of the '
                'target, x jittered by N(0, 2.5 deg) for display; (e) histogram of angular error (15 deg bins); '
                '(f) permutation null histogram with observed |r|.'),
            data=dict(
                trials=dict(trial_id=tid, block=blocks, direction_deg=dirs * 45,
                            **{f'{n}_cv': held[:, i] for i, n in enumerate(names)},
                            **{f'{n}_in_sample': full['scores'][:, i] for i, n in enumerate(names)}),
                angular_readout=dict(trial_id=tid[ok_obs], target_deg=np.degrees(obs['target']),
                                     decoded_deg=np.degrees(obs['angle']) % 360, error_deg=np.degrees(obs['error'])),
                variance_captured=dict(axis=names, in_sample_pct=[100 * full['variance_captured'][n] for n in names],
                                       held_out_pct=[100 * cv_var[n] for n in names]),
                permutation_null=dict(perm=np.arange(len(null)), abs_fisher_lee_r=null),
                readout_summary=[readout_row('held_out', obs, p_perm), readout_row('in_sample', ins, '')]))

        # ------------------------------------------------------------- UMAP
        embedding = None
        if p['umap_enabled']:
            ctx.check_cancel()
            ctx.progress('UMAP（首次运行需要编译，可能稍慢）')
            embedding = umap_embedding(xa, p['umap_neighbors'], p['umap_min_dist'], p['seed'])
            ctx.check_cancel()
            scores = embedding['scores']
            fig = new_figure(12, 5.5)
            left, right = fig.subplots(1, 2, sharex=True, sharey=True)
            style.direction_ellipses(left, scores, dirs, fill=False)
            for d in DIRS:
                sel = dirs == d
                if sel.any():
                    left.scatter(scores[sel, 0], scores[sel, 1], s=15, alpha=0.75,
                                 color=style.direction_color(d), linewidths=0, label=f'{d * 45}°')
            centres = centroids(scores, dirs)
            present = np.isfinite(centres).all(1)
            if present.sum() > 1:
                ring = centres[present]
                if len(ring) > 2:
                    ring = np.vstack([ring, ring[:1]])
                left.plot(ring[:, 0], ring[:, 1], color=style.INK, lw=1, alpha=0.65, zorder=3)
            left.scatter(centres[present, 0], centres[present, 1], s=90,
                         c=[style.direction_color(d) for d in DIRS[present]],
                         edgecolors='black', linewidths=1, zorder=4, label='Direction centroid')
            left.legend(frameon=False, fontsize=7, ncol=2)
            left.set_title('Target direction + centroids + 1 SD ellipses')
            points = right.scatter(scores[:, 0], scores[:, 1], c=blocks, cmap='viridis',
                                   s=15, alpha=0.75, linewidths=0)
            fig.colorbar(points, ax=right, label='Block')
            right.set_title('Recording block (same embedding)')
            for ax in (left, right):
                ax.set_xlabel('UMAP 1 / arbitrary units')
                ax.set_ylabel('UMAP 2 / arbitrary units')
                ax.set_aspect('equal', adjustable='box')
            fig.suptitle(banner + '\nUnsupervised full-cohort UMAP · exploratory view', fontsize=9)
            rows = [dict(trial_id=int(t), block=int(b), order=int(o), direction_deg=int(d * 45),
                         UMAP1=float(s[0]), UMAP2=float(s[1]))
                    for t, b, o, d, s in zip(tid, blocks, trial_orders, dirs, scores)]
            centroid_rows = [dict(direction_deg=int(d * 45), n_trials=int((dirs == d).sum()),
                                  UMAP1=float(centres[d, 0]), UMAP2=float(centres[d, 1]))
                             for d in DIRS[present]]
            result.add_figure(
                'UMAP embedding', fig,
                method=win_txt + (
                    'Unsupervised UMAP on all included trials, using the same active-channel window rates '
                    'as PCA, z-scored per channel over the full cohort; no PCA pre-reduction. '
                    'Direction and block labels are used only for display, never for fitting. '
                    'Left panel: large black-edged points are arithmetic mean embedding coordinates per '
                    'direction, joined in ascending direction order and closed when at least three directions '
                    'are present; absent directions are omitted. These are embedding-space centroids. ' +
                    style.ELLIPSE_METHOD +
                    f'umap-learn {embedding["version"]}; n_components=2, metric=euclidean, '
                    f'n_neighbors={embedding["n_neighbors"]} (requested {p["umap_neighbors"]}, capped at n_trials-1), '
                    f'min_dist={embedding["min_dist"]}, spread=1, init=random, random_state={p["seed"]}, '
                    'n_jobs=1, other settings use library defaults. Both panels show identical coordinates; '
                    'all trials are displayed. This is a full-cohort exploratory embedding, not a held-out '
                    'decoding score; axes have arbitrary units and carry no explained-variance percentage.'),
                data={'trials': rows, 'centroids': centroid_rows})
            result.tables['umap_projections'] = rows
            result.tables['umap_centroids'] = centroid_rows
            result.summary['umap'] = {key: value for key, value in embedding.items() if key != 'scores'}
            result.notes.append('UMAP 为全队列无监督探索图，方向和 block 仅用于着色；分群不等于解码性能。'
                                '坐标轴无物理单位，不给出解释方差，跨记录的独立嵌入坐标不可直接比较。')
            if embedding['n_neighbors'] != p['umap_neighbors']:
                result.notes.append(f'UMAP 邻居数由 {p["umap_neighbors"]} 自动限制为 '
                                    f'{embedding["n_neighbors"]}（trial 数 − 1）。')

        # ------------------------------------------------------------- outputs
        result.summary.update(
            trials=int(len(x)), active_channels=int(active.sum()),
            pc_explained_pct=[round(float(100 * v), 2) for v in fit['explained'][:5]],
            pc1_vs_total_rate_r=round(float(r_tot[0]), 3),
            pcs_for_80pct=int(np.searchsorted(np.cumsum(full_explained), 0.8) + 1),
            tdr_cv_circular_r=round(float(obs['circular_r']), 3),
            tdr_cv_circular_r_jammalamadaka=round(float(obs['circular_r_js']), 3),
            tdr_cv_error_resultant_length=round(float(obs['resultant']), 3),
            tdr_cv_median_abs_error_deg=round(float(obs['median_abs_error_deg']), 1),
            tdr_cv_within_45deg=round(float(obs['within_45deg_frac']), 3),
            tdr_in_sample_circular_r=round(float(ins['circular_r']), 3),
            tdr_variance_captured_pct={n: round(100 * v, 2) for n, v in full['variance_captured'].items()},
            tdr_perm_n=int(len(null)), tdr_perm_p=p_perm,
            tdr_null_abs_r_p95=round(float(np.percentile(null, 95)), 3) if len(null) else None)
        ids = session.electrode_ids[active]
        result.tables['pca_loadings'] = [
            dict(electrode=int(e), **{f'PC{i + 1}': round(float(fit['components'][i, j]), 5) for i in range(min(k, 5))})
            for j, e in enumerate(ids)]
        result.tables['trial_projections'] = [
            dict(trial_id=int(t), block=int(b), direction_deg=int(d) * 45,
                 **{f'PC{i + 1}': round(float(fit['scores'][j, i]), 4) for i in range(3)},
                 **{f'PC{i + 1}_cv': (round(float(cv_scores[j, i]), 4) if np.isfinite(cv_scores[j, i]) else '')
                    for i in range(3)},
                 **{f'tdr_{n}_cv': (round(float(held[j, c]), 4) if np.isfinite(held[j, c]) else '')
                    for c, n in enumerate(names)})
            for j, (t, b, d) in enumerate(zip(cohort.trial_ids[ok], blocks, dirs))]
        result.notes.append('PCA 散点为样本内拟合；CV 版本的主成分只用训练 block 拟合。TDR 方向平面的旋转是任意的，'
                            '角度读出先估计并扣除一个整体偏移再计算误差。')
        # --------------------------------------------------------- three main subsections
        ctx.progress('PCA gain and time controls')
        controls = DG.pca_controls(xa, fit, dirs, blocks, orders, p['n_perm'], p['seed'],
                                   ctx.progress, ctx.check_cancel)
        result.tables['pca_gain_control'] = controls['comparison']
        result.tables['pca_variable_correlations'] = controls['per_pc']
        result.tables['pca_control_null'] = [dict(permutation=i, original_max_F=a, residual_max_F=b)
                                           for i, (a, b) in enumerate(controls['nulls'])]
        residual_fit = controls['adjusted']
        from .residual_cv import append_heldout
        ctx.progress('held-out residual PCA (training-only adjustment)')
        append_heldout(result,x,dirs,blocks,tid,session.electrode_ids,p,self._scatter,banner)
        result.tables['pca_gain_residual_scores'] = [
            dict(trial_id=int(t), direction_deg=int(d * 45),
                 **{f'PC{j + 1}': float(s[j]) for j in range(min(3, len(s)))})
            for t, d, s in zip(tid, dirs, residual_fit['scores'])] if residual_fit else []
        result.add_figure('PCA', DG.pca_main(fit, full_explained, dirs, controls, self._scatter, banner),
                          method=win_txt + pca_txt + style.ELLIPSE_METHOD +
                          'Main PCA: spectrum and full-cohort PC1-PC2 / PC2-PC3; correlations of every '
                          'requested PC with total rate and actual elapsed seconds. Gain control regresses '
                          'intercept and total active-channel rate out of every channel on the full cohort, '
                          'then fits a new z-scored PCA. Compare direction ANOVA F sorted separately by F '
                          '(PC ranks are not corresponding axes); residual PC1-PC2 is a new basis. '
                          f'{p["n_perm"]} shared within-block label shuffles give separate 95th percentiles '
                          'of maximum F across all displayed PCs in each basis. Dashed lines are empirical '
                          'null thresholds; when permutations=0 no empirical threshold is shown. This '
                          'is a descriptive sensitivity analysis, not held-out gain control.',
                          data={'per_pc': controls['per_pc'], 'gain_control': controls['comparison'],
                                'residual_trials': result.tables['pca_gain_residual_scores']})
        result.tables['tdr_axis_stability'] = tdr['stability']
        result.tables['tdr_fold_designs'] = tdr['designs']
        ctx.progress('TDR matched-cohort kinematics controls')
        kin, kin_reason = DG.measured_kinematics(cohort, ok)
        kin_ok = np.isfinite(kin).all(1)
        kin_rows, kin_predictions, kin_effects, kin_scores = [], [], [], []
        if kin_ok.sum() >= 80 and len(np.unique(blocks[kin_ok])) >= 3:
            for model, kinematics in (('direction', None), ('+ kinematics', kin[kin_ok])):
                ctx.check_cancel()
                matched = DG.tdr_fit_cv(xa[kin_ok], dirs[kin_ok], blocks[kin_ok], orders[kin_ok],
                                        p['n_folds'], kin=kinematics)
                readout = angular_readout(matched['scores'][:, 2:4], dirs[kin_ok])
                kin_scores.append(matched['scores'][:, 2:4])
                matched_fit = matched['full']
                design, predictor_names = A.tdr_design(dirs[kin_ok], total[kin_ok], orders[kin_ok])
                if kinematics is not None:
                    design = np.column_stack([design, A.zscore(kinematics)[0]])
                    predictor_names += ['vx', 'vy', 'speed']
                # beta norm alone is scale-dependent; multiply by each predictor's SD.
                effects = {name: float(np.linalg.norm(matched_fit['beta'][j]) * design[:, j].std())
                           for j, name in enumerate(predictor_names) if name != 'const'}
                kin_rows.append(dict(model=model, n_trials=int(kin_ok.sum()),
                                     tested_trials=int(np.isfinite(matched['scores'][:, 2:4]).all(1).sum()),
                                     circular_r=readout['circular_r'], median_error_deg=readout['median_abs_error_deg'],
                                     design_rank=matched_fit['rank'], design_columns=matched_fit['columns'],
                                     **{f'effect_{name}': effects[name] for name in ('gain', 'time', 'cos', 'sin')}))
                kin_effects += [dict(model=model, predictor=name, standardized_effect=value)
                                for name, value in effects.items()]
                kin_predictions += [dict(model=model, trial_id=int(t), block=int(b), direction_deg=int(d * 45),
                                         cos_score=float(s[2]), sin_score=float(s[3]))
                                    for t, b, d, s in zip(tid[kin_ok], blocks[kin_ok], dirs[kin_ok], matched['scores'])]
                if any(row['design_rank'] < row['design_columns'] for row in matched['designs']):
                    result.notes.append(f'TDR {model} 存在秩亏的训练折，预测变量效应不可作唯一归因。')
            paired = np.isfinite(kin_scores[0]).all(1) & np.isfinite(kin_scores[1]).all(1)
            for row, prediction in zip(kin_rows, kin_scores):
                paired_readout = angular_readout(prediction[paired], dirs[kin_ok][paired])
                row.update(tested_trials=int(paired.sum()), circular_r=paired_readout['circular_r'],
                           median_error_deg=paired_readout['median_abs_error_deg'])
            paired_ids = set(tid[kin_ok][paired].tolist())
            for row in kin_predictions:
                row['paired_evaluation'] = row['trial_id'] in paired_ids
        else:
            result.notes.append('TDR 实测速度控制未计算：' + (kin_reason or
                                f'完整速度 trial {int(kin_ok.sum())} 个；需要至少 80 个及 3 个有效 block。'))
        result.tables['tdr_kinematics_comparison'] = kin_rows
        result.tables['tdr_kinematics_effects'] = kin_effects
        result.tables['tdr_kinematics_predictions'] = kin_predictions
        result.tables['tdr_measured_kinematics'] = [dict(trial_id=int(t), vx=float(v[0]), vy=float(v[1]),
                                                       mean_speed=float(v[2])) for t, v in zip(tid[kin_ok], kin[kin_ok])]
        result.add_figure('TDR', DG.tdr_main(held[:, cols], dirs, obs, null, p_perm, tdr['stability'],
                                           kin_rows, self._scatter, banner),
                          method=win_txt + style.ELLIPSE_METHOD +
                          'TDR: regression on cos/sin direction, z-scored total rate and actual elapsed time, '
                          'QR ordered gain,time,cos,sin. Block CV purges adjacent blocks; neural and kinematic '
                          'standardizers are fitted on training trials only. Constant training channels are '
                          'dropped in that fold; undefined/dependent axes are not plotted as meaningful axes. '
                          'Direction labels enter fitting, so the main plane is held-out. Angular offset is '
                          'estimated from pooled held-out labels for descriptive error display, not an unbiased '
                          f'decoding score. Null uses {len(valid_null)} finite complete-refit within-block shuffles. '
                          'Axis stability is the signed dot product with the full-cohort reference, not an '
                          'independent significance test. Kinematics comparison uses identical complete quality-'
                          'checked 100-ms-bin trials in both models (>=80 trials); window-mean vx, vy and mean '
                          'speed are additional regressors. Effects = coefficient-vector norm × predictor SD '
                          'on the full matched cohort; correlated predictors prevent causal attribution. '
                          'This tests conditional associations, not pure motor intention.',
                          data={'stability': tdr['stability'], 'kinematics_comparison': kin_rows,
                                'kinematics_effects': kin_effects,
                                'held_out_trials': [dict(trial_id=int(t), direction_deg=int(d * 45),
                                                        cos_axis=float(s[0]), sin_axis=float(s[1]))
                                                    for t, d, s in zip(tid, dirs, held[:, cols])]})
        if p['umap_enabled']:
            ctx.progress('UMAP neighborhood quality')
            quality, subset = DG.neighborhood_quality(xa, scores, fit['scores'], p['seed'])
            umap_index = next(i for i, (title, _) in enumerate(result.figures) if title == 'UMAP embedding')
            DG.umap_main(result.figures[umap_index][1], scores, blocks, total, quality)
            result.figure_data[umap_index].update(
                neighborhood_quality=quality,
                quality_trials=[dict(trial_id=int(tid[i])) for i in subset],
                state_colours=[dict(trial_id=int(t), block=int(b), total_rate_hz=float(r),
                                   elapsed_s=float(tm)) for t, b, r, tm in zip(tid, blocks, total, orders)])
            result.methods[umap_index] += (
                ' Additional panels: same embedding coloured by total active-channel rate; neighborhood '
                'trustworthiness at k=3,5,10,15 where k<n/2, using standardized original features and PCA2 '
                'as comparison. Evaluation uses all trials up to 2000, otherwise a seeded sample whose '
                'trial IDs are exported. Embedding itself uses all trials. Trustworthiness is descriptive '
                'local-neighbor preservation, not explained variance or evidence of nonlinear coding.')
            result.tables['umap_neighborhood_quality'] = quality
        gain_pc = int(np.nanargmax(np.abs(r_tot))) + 1 if np.isfinite(r_tot).any() else None
        result.summary.update(gain_correlated_pc=gain_pc, time_variable='elapsed_seconds',
                              tdr_kinematics_trials=int(kin_ok.sum()),
                              tdr_kinematics_computed=bool(kin_rows),
                              tdr_valid_permutations=len(valid_null))
        result.notes.append('PCA 不预设任何 PC 为增益轴；时间项使用真实经过秒数。'
                            '去增益后的 PCA 为全队列敏感性分析，可能同时移除与增益相关的任务信号。'
                            'TDR 的角度误差扣除了由测试标签估计的偏移，属于描述性读出。')
        if len(valid_null) != len(null):
            result.notes.append(f'TDR 有 {len(null) - len(valid_null)} 次置换的方向轴不可定义，'
                                '置换 p 值不报告，不能把无效重复当作零统计量。')
        if session.decoder_velocity is not None or session.recording.online:
            result.notes.append('在线记录中的 cursor 是闭环输出，速度控制不能解释为控制了独立测量的动物运动。')
        from .overview import overview
        result.add_figure('Dimensionality overview',
            overview(fit,residual_fit,full_explained,dirs,blocks,
                     embedding,held[:,cols],obs,self._scatter,banner,figure=overview_target),
            method=win_txt+pca_txt+cv_txt+(
                'The first two rows show PCA / UMAP / TDR by column. '
                'Top row: full-cohort PC1-PC2; full-cohort UMAP by direction; held-out TDR cos/sin. '
                'Bottom PCA: full-cohort residual PCA PC1/PC2 after subtracting each channel intercept '
                'and its linear association with centered trial total rate, then restandardizing residual '
                'channels and refitting PCA. Each original/residual PC axis includes its own explained '
                'variance fraction; residual fractions refer to residual-feature variance, a different '
                'denominator. This panel is in-sample; train-only held-out residual projections are in '
                'Supplementary and displayed by fold, not pooled across different PCA bases. '
                'Bottom UMAP: identical embedding and limits, coloured by recording block; no label '
                'used during embedding. UMAP origin is arbitrary; no explained-variance interpretation. '
                'Bottom TDR: the held-out angular-error scatter from the TDR page. '
                'Angular error has a target-label-derived rotation offset, thus is descriptive. '
                'Projection centroids and unfilled 1 SD covariance ellipses describe trial spread, '
                'not confidence of the mean. Controls remain on subsequent PCA/UMAP/TDR pages.'),
            data=dict(pca_spectrum=dict(pc=np.arange(1,len(full_explained)+1),explained_fraction=full_explained),
                      pca_correlations=dict(pc=np.arange(1,len(r_tot)+1),r_total_rate=r_tot,
                                            r_elapsed=DG.correlations(fit['scores'],orders)),
                      residual_pca=[] if residual_fit is None else result.tables['pca_gain_residual_scores'],
                      residual_variance=[] if residual_fit is None else [dict(pc=i+1,explained_fraction=float(v))
                          for i,v in enumerate(residual_fit['explained'])],
                      projections=dict(trial_id=tid,direction_deg=dirs*45,block=blocks,
                                       pc1=fit['scores'][:,0],pc2=fit['scores'][:,1],
                                       tdr_cos=held[:,cols[0]],tdr_sin=held[:,cols[1]]),
                      umap=[] if embedding is None else [dict(trial_id=int(t),block=int(b),direction_deg=int(d*45),
                             umap1=float(s[0]),umap2=float(s[1])) for t,b,d,s in zip(tid,blocks,dirs,embedding['scores'])],
                      tdr_null=dict(permutation=np.arange(len(null)),abs_circular_r=null),
                      tdr_angular_error=dict(trial_id=tid[ok_obs][:len(obs['error'])],
                                             target_deg=np.degrees(obs['target'][:len(obs['error'])]),
                                             error_deg=np.degrees(obs['error'])),
                      tdr_statistics=[dict(circular_r=obs['circular_r'],p=p_perm,
                                           median_abs_error_deg=obs['median_abs_error_deg'])]))
        # Overview first; each method retains one detailed subsection afterwards.
        primary = ['Dimensionality overview','PCA', 'UMAP embedding', 'TDR']
        order = [i for title in primary for i, (name, _) in enumerate(result.figures) if name == title]
        order += [i for i, (name, _) in enumerate(result.figures) if name not in primary]
        result.figures = [result.figures[i] for i in order]
        result.methods = [result.methods[i] for i in order]
        result.figure_data = [result.figure_data[i] for i in order]
        supplementary = {'PCA variance and gain axis': 'Supplementary PCA loadings and gain',
                         'PCA projections': 'Supplementary PCA held-out projections',
                         'TDR direction plane': 'Supplementary TDR in-sample and readout'}
        result.figures = [(supplementary.get(title, title), fig) for title, fig in result.figures]
        return result

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _full_spectrum(xa):
        z, _, _ = A.zscore(xa)
        s = np.linalg.svd(z - z.mean(0), compute_uv=False)
        var = s ** 2
        return var / var.sum() if var.sum() > 0 else var

    @staticmethod
    def _scatter(ax, s, dirs, xlabel, ylabel, title, legend=False):
        style.direction_ellipses(ax, s, dirs, fill=False)
        ok = np.isfinite(s).all(1)
        for d in DIRS:
            m = ok & (dirs == d)
            ax.scatter(s[m, 0], s[m, 1], s=9, color=style.direction_color(d), alpha=0.5, linewidths=0,
                       label=f'{d * 45}°' if legend else None)
        cen = centroids(s[ok], dirs[ok])
        good = np.isfinite(cen).all(1)
        ring = np.vstack([cen[good], cen[good][:1]])
        ax.plot(ring[:, 0], ring[:, 1], color=style.INK, lw=0.8, alpha=0.5, zorder=4)
        ax.scatter(cen[good, 0], cen[good, 1], s=60, c=[style.direction_color(d) for d in DIRS[good]],
                   edgecolors='black', linewidths=0.8, zorder=5)
        lim = np.nanpercentile(np.abs(s[ok]), 99.5) * 1.1 if ok.any() else 1
        hidden = int((np.abs(s[ok]) > lim).any(1).sum())
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal', adjustable='box')
        ax.axhline(0, color=style.GREY, lw=0.5, ls=':')
        ax.axvline(0, color=style.GREY, lw=0.5, ls=':')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8)
        if hidden:
            ax.text(0.99, 0.01, f'{hidden} outside view', transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=6, color=style.INK2)
        if legend:
            ax.legend(fontsize=6, frameon=False, ncol=2, loc='upper left', handletextpad=0.1, columnspacing=0.5,
                      markerscale=1.5)

    # ------------------------------------------------------------------ figures
    def pca_overview(self, fit, full_explained, total, dirs, r_tot, F_in, F_cv, active, ctx, banner):
        fig = new_figure(13, 7.5)
        grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.75])
        k = len(fit['explained'])
        ax = fig.add_subplot(grid[0, 0])
        n_show = min(20, len(full_explained))
        x = np.arange(1, n_show + 1)
        ax.bar(x, 100 * full_explained[:n_show], color='#30638e', label='per PC')
        ax2 = ax.twinx()
        ax2.plot(x, 100 * np.cumsum(full_explained)[:n_show], color='#c76d4b', marker='o', ms=3, label='cumulative')
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('cumulative variance / %', color='#c76d4b')
        ax2.spines['right'].set_visible(True)
        ax.set_xlabel('principal component')
        ax.set_ylabel('variance explained / %')
        ax.set_xticks(x[::2] if n_show > 10 else x)
        ax.set_title('Scree (z-scored window rates)')

        ax = fig.add_subplot(grid[0, 1])
        gain_index = int(np.nanargmax(np.abs(r_tot))) if np.isfinite(r_tot).any() else 0
        pc1 = fit['scores'][:, gain_index]
        for d in DIRS:
            m = dirs == d
            ax.scatter(total[m], pc1[m], s=8, color=style.direction_color(d), alpha=0.6, linewidths=0)
        ax.set_xlabel('total rate over active channels / Hz')
        ax.set_ylabel(f'PC{gain_index + 1} score')
        ax.set_title(f'Most gain-correlated: PC{gain_index + 1}, r = {r_tot[gain_index]:.2f}')

        ax = fig.add_subplot(grid[1, 0])
        xs = np.arange(1, k + 1)
        ax.bar(xs - 0.2, F_in, width=0.4, color=style.GREY, label='in-sample')
        ax.bar(xs + 0.2, F_cv, width=0.4, color='#00798c', label='cross-validated')
        df_w = max(len(dirs) - 8, 1)
        crit = stats.f.ppf(1 - 0.05 / k, 7, df_w)
        ax.axhline(crit, color='#c76d4b', ls='--', lw=0.8, label=f'p = 0.05/{k}')
        ax.set_xticks(xs)
        ax.set_xlabel('principal component')
        ax.set_ylabel('direction one-way F')
        ax.set_title('Direction information per PC')
        ax.legend(frameon=False, fontsize=6)

        ax = fig.add_subplot(grid[1, 1])
        ax.bar(xs, r_tot, color=['#c76d4b' if abs(v) > 0.5 else style.GREY for v in r_tot])
        ax.axhline(0, color=style.INK, lw=0.5)
        ax.set_ylim(-1, 1)
        ax.set_xticks(xs)
        ax.set_xlabel('principal component')
        ax.set_ylabel('r with total rate')
        ax.set_title('Gain correlation per PC')

        ax = fig.add_subplot(grid[:, 2])
        values = np.full(len(active), np.nan)
        values[active] = fit['components'][gain_index]
        lim = np.nanmax(np.abs(values)) or 1
        draw_channel_map(ax, ctx.session.electrode_ids, values, layout=ctx.layout, cmap='RdBu_r',
                         norm=TwoSlopeNorm(0, -lim, lim), title=f'PC{gain_index + 1} loadings (grey = inactive)',
                         cbar_label='loading')
        frac = float(np.mean(fit['components'][gain_index] > 0))
        ax.text(0.5, -0.06, f'{100 * frac:.0f}% of loadings positive', transform=ax.transAxes, ha='center',
                va='top', fontsize=7)
        fig.suptitle(banner + ' · PCA', fontsize=9)
        return fig

    def pca_scatter(self, scores, cv, dirs, explained, banner):
        fig = new_figure(11, 10)
        axes = fig.subplots(2, 2)
        e = 100 * explained
        for row, (s, tag) in enumerate([(scores, 'in-sample'), (cv, 'cross-validated (train-fold PCs)')]):
            self._scatter(axes[row, 0], s[:, [0, 1]], dirs, f'PC1 ({e[0]:.1f}%)', f'PC2 ({e[1]:.1f}%)',
                          f'PC1–PC2 · {tag}', legend=row == 0)
            self._scatter(axes[row, 1], s[:, [1, 2]], dirs, f'PC2 ({e[1]:.1f}%)', f'PC3 ({e[2]:.1f}%)',
                          f'PC2–PC3 · {tag}')
        fig.suptitle(banner + ' · large dots = centroids; ellipses = 1 SD', fontsize=9)
        return fig

    def tdr_figure(self, held, ins_scores, dirs, obs, ins, full, cv_var, null, p_perm, banner):
        fig = new_figure(14, 9)
        grid = fig.add_gridspec(2, 3)
        ax = fig.add_subplot(grid[0, 0])
        self._scatter(ax, held, dirs, 'direction axis 1 (b_cos ⊥)', 'direction axis 2 (b_sin ⊥)',
                      f'Cross-validated projection · Fisher–Lee r = {obs["circular_r"]:.2f}', legend=True)
        ax = fig.add_subplot(grid[0, 1])
        self._scatter(ax, ins_scores, dirs, 'direction axis 1 (b_cos ⊥)', 'direction axis 2 (b_sin ⊥)',
                      f'In-sample (not evidence) · Fisher–Lee r = {ins["circular_r"]:.2f}')

        ax = fig.add_subplot(grid[0, 2])
        names = full['axis_names']
        xs = np.arange(len(names))
        ax.bar(xs - 0.2, [100 * full['variance_captured'][n] for n in names], width=0.4, color=style.GREY,
               label='in-sample')
        ax.bar(xs + 0.2, [100 * cv_var[n] for n in names], width=0.4, color='#00798c', label='held-out')
        ax.set_xticks(xs)
        ax.set_xticklabels(names)
        ax.set_ylabel('share of total channel variance / %')
        ax.set_title('Variance captured per TDR axis')
        ax.legend(frameon=False, fontsize=6)

        ax = fig.add_subplot(grid[1, 0])
        if len(obs['angle']):
            jitter = np.random.default_rng(0).normal(0, 2.5, len(obs['target']))
            # target + wrapped error: no 0/360 seam at the diagonal
            shifted = np.degrees(obs['target'] + obs['error'])
            ax.scatter(np.degrees(obs['target']) + jitter, shifted, s=8,
                       c=[style.direction_color(int(round(t / 45)) % 8) for t in np.degrees(obs['target'])],
                       alpha=0.6, linewidths=0)
            ax.plot([0, 315], [0, 315], color=style.INK, lw=0.6, ls='--')
        ax.set_xticks(np.arange(0, 360, 45))
        ax.set_yticks(np.arange(-180, 541, 90))
        if len(obs['angle']):
            ax.set_ylim(min(-45, shifted.min() - 10), max(360, shifted.max() + 10))
        ax.set_xlabel('target direction / °')
        ax.set_ylabel('decoded angle − offset (unwrapped near target) / °')
        ax.set_title(f'Held-out angle (offset {obs["offset_deg"]:.0f}° removed)')

        ax = fig.add_subplot(grid[1, 1])
        if len(obs['error']):
            ax.hist(np.degrees(obs['error']), bins=np.arange(-180, 181, 15), color='#30638e')
        ax.axvline(0, color=style.INK, lw=0.6)
        ax.set_xticks(np.arange(-180, 181, 90))
        ax.set_xlabel('angular error / °')
        ax.set_ylabel('trials')
        ax.set_title(f'Median |error| {obs["median_abs_error_deg"]:.0f}° · '
                     f'{100 * obs["within_45deg_frac"]:.0f}% within ±45°')

        ax = fig.add_subplot(grid[1, 2])
        if len(null):
            ax.hist(null, bins=20, color=style.GREY, label=f'null (n={len(null)})')
            ax.axvline(abs(obs['circular_r']), color='#c76d4b', lw=1.5, label=f'observed |r| = {abs(obs["circular_r"]):.2f}')
            ax.set_title(f'Within-block permutation · p = {p_perm:.3g}')
            ax.legend(frameon=False, fontsize=6)
        else:
            ax.text(0.5, 0.5, 'n_perm = 0: null skipped', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Within-block permutation')
        ax.set_xlabel('|Fisher–Lee circular r| (held-out, refitted per shuffle)')
        ax.set_ylabel('count')
        fig.suptitle(banner + ' · TDR (gain → time → cos → sin, QR-orthogonalized)\nEllipses = 1 SD', fontsize=9)
        return fig


SECTION = Dimensionality()
