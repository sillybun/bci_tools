"""Source-backed diagnostics for the three S04 overview subsections."""
from __future__ import annotations

import numpy as np

from ...core import analysis as A, style
from ...core.features import velocity_bins
from ..base import new_figure


def correlations(scores, variable):
    """Pearson correlations, leaving undefined constant-variable correlations as NaN."""
    return np.array([np.corrcoef(s, variable)[0, 1]
                     if np.std(s) > 1e-12 and np.std(variable) > 1e-12 else np.nan
                     for s in scores.T])


def gain_residuals(x):
    """Remove intercept and trial total rate from each channel (full-cohort control)."""
    total = x.sum(1)
    centred = total - total.mean()
    beta = centred @ (x - x.mean(0)) / max(float(centred @ centred), 1e-12)
    return x - x.mean(0) - centred[:, None] * beta


def pca_controls(x, fit, directions, blocks, elapsed, n_perm, seed, progress, cancel):
    residual = gain_residuals(x)
    active = residual.std(0) > max(float(x.std(0).max()) * 1e-10, 1e-12)
    k = fit['scores'].shape[1]
    adjusted = A.pca(residual[:, active], min(k, int(active.sum()), len(x) - 1)) if active.sum() >= 3 else None
    raw_f = A.oneway_anova(fit['scores'], directions)['F']
    adjusted_f = A.oneway_anova(adjusted['scores'], directions)['F'] if adjusted else np.zeros(0)
    rng = np.random.default_rng(seed)
    nulls = np.full((n_perm, 2), np.nan)
    for i in range(n_perm):
        if i % 20 == 0:
            cancel()
            progress(f'PCA gain-control permutation {i}/{n_perm}')
        labels = A.permute_within_block(rng, directions, blocks)
        nulls[i, 0] = np.nanmax(A.oneway_anova(fit['scores'], labels)['F'])
        if adjusted:
            nulls[i, 1] = np.nanmax(A.oneway_anova(adjusted['scores'], labels)['F'])
    r_gain = correlations(fit['scores'], x.sum(1))
    r_time = correlations(fit['scores'], elapsed)
    per_pc = [dict(pc=i + 1, direction_F=float(raw_f[i]), r_total_rate=float(r_gain[i]),
                   r_elapsed_time=float(r_time[i])) for i in range(k)]
    comparison = []
    for j, (name, values) in enumerate((('original', raw_f), ('gain_removed', adjusted_f))):
        threshold = float(np.percentile(nulls[:, j], 95)) if n_perm and len(values) else np.nan
        probability = float((1 + (nulls[:, j] >= np.max(values)).sum()) / (1 + n_perm)) if n_perm and len(values) else np.nan
        for rank, pc in enumerate(np.argsort(-values), start=1):
            comparison.append(dict(condition=name, rank=rank, pc=int(pc + 1), direction_F=float(values[pc]),
                                   null_max_F_p95=threshold, p_max_F=probability))
    return dict(adjusted=adjusted, per_pc=per_pc, comparison=comparison, nulls=nulls)


def pca_main(fit, spectrum, dirs, controls, scatter, banner):
    fig = new_figure(14, 8.5)
    axes = fig.subplots(2, 3)
    ax = axes[0, 0]
    n = min(len(spectrum), 20)
    pcs = np.arange(1, n + 1)
    ax.bar(pcs, 100 * spectrum[:n], color='#30638e')
    twin = ax.twinx()
    twin.plot(pcs, 100 * np.cumsum(spectrum[:n]), 'o-', color='#c76d4b', ms=3)
    twin.set_ylim(0, 100)
    twin.set_ylabel('Cumulative / %')
    ax.set(xlabel='PC', ylabel='Variance / %', title='A  PCA explained-variance spectrum')
    for ax, pair, letter in ((axes[0, 1], (0, 1), 'B'), (axes[0, 2], (1, 2), 'C')):
        i, j = pair
        scatter(ax, fit['scores'][:, [i, j]], dirs,
                f'PC{i + 1} ({100 * fit["explained"][i]:.1f}%)',
                f'PC{j + 1} ({100 * fit["explained"][j]:.1f}%)',
                f'{letter}  Original PCA: PC{i + 1}–PC{j + 1} (in-sample)', legend=(letter == 'B'))
    ax = axes[1, 0]
    rows = controls['per_pc']
    pcs = np.array([r['pc'] for r in rows])
    ax.bar(pcs - .2, [r['r_total_rate'] for r in rows], width=.4, label='Total rate')
    ax.bar(pcs + .2, [r['r_elapsed_time'] for r in rows], width=.4, label='Elapsed time')
    ax.axhline(0, color=style.GREY, lw=.5)
    ax.set(ylim=(-1, 1), xlabel='PC', ylabel='Pearson r', title='D  PC associations with total rate and elapsed time')
    ax.legend(frameon=False)
    ax = axes[1, 1]
    for name, shift, colour in (('original', -.2, '#2176ae'), ('gain_removed', .2, '#c0392b')):
        selected = [r for r in controls['comparison'] if r['condition'] == name]
        if not selected:
            continue
        ax.bar(np.array([r['rank'] for r in selected]) + shift, [r['direction_F'] for r in selected],
               width=.4, color=colour, label='Original PCA' if name=='original' else 'Total-rate-adjusted PCA')
        if np.isfinite(selected[0]['null_max_F_p95']):
            ax.axhline(selected[0]['null_max_F_p95'], color=colour, ls='--', lw=.8)
    ax.set(xlabel='Rank by direction F (different PC bases)', ylabel='Direction ANOVA F',
           title='E  Direction association before/after total-rate adjustment')
    ax.legend(frameon=False, fontsize=6)
    if controls['adjusted']:
        adjusted=controls['adjusted']
        scatter(axes[1, 2], adjusted['scores'][:, :2], dirs,
                f'Residual PC1 ({100*adjusted["explained"][0]:.1f}% residual var.)',
                f'Residual PC2 ({100*adjusted["explained"][1]:.1f}% residual var.)',
                'F  PCA after total-rate linear adjustment (in-sample)')
    else:
        style.text_panel(axes[1, 2], ['Residual feature rank too small', 'after removing total rate.'])
    fig.suptitle(banner + '\nPCA · variance, gain and direction structure · ellipses = 1 SD', fontsize=9)
    return fig


def neighborhood_quality(x, scores, pca_scores, seed, cap=2000):
    """Trustworthiness on a fixed deterministic subset; no labels enter evaluation."""
    from sklearn.manifold import trustworthiness

    n = len(x)
    subset = np.sort(np.random.default_rng(seed).choice(n, min(n, cap), replace=False))
    z, _, _ = A.zscore(x)
    ks = sorted({k for k in (3, 5, 10, 15) if k < len(subset) / 2})
    return [dict(neighbors=k, n_evaluated=len(subset), n_total=n,
                 umap=float(trustworthiness(z[subset], scores[subset], n_neighbors=k)),
                 pca2=float(trustworthiness(z[subset], pca_scores[subset, :2], n_neighbors=k))) for k in ks], subset


def umap_main(fig, scores, blocks, total, quality):
    """Expand the existing direction/block plot without refitting its embedding."""
    left, right, _ = fig.axes
    right.collections[0].colorbar.remove()
    grid = fig.add_gridspec(2, 2)
    left.set_subplotspec(grid[0, 0])
    right.set_subplotspec(grid[0, 1])
    # Recreate the colour bar in the new layout instead of retaining its old gridspec.
    fig.colorbar(right.collections[0], ax=right, label='Block')
    ax = fig.add_subplot(grid[1, 0], sharex=left, sharey=left)
    dots = ax.scatter(scores[:, 0], scores[:, 1], c=total, cmap='magma', s=12, alpha=.75, lw=0)
    fig.colorbar(dots, ax=ax, label='Total rate / Hz')
    ax.set(xlabel='UMAP 1 / arbitrary units', ylabel='UMAP 2 / arbitrary units',
           title='Total population rate (same embedding)', aspect='equal')
    ax = fig.add_subplot(grid[1, 1])
    ks = [r['neighbors'] for r in quality]
    ax.plot(ks, [r['umap'] for r in quality], 'o-', label='UMAP')
    ax.plot(ks, [r['pca2'] for r in quality], 's--', label='PCA 2D')
    ax.set(xlabel='Evaluation neighbors k', ylabel='Trustworthiness', ylim=(0, 1.02),
           title=f'Local-neighborhood preservation (n={quality[0]["n_evaluated"]})')
    ax.legend(frameon=False)
    fig.set_size_inches(12, 10)


def tdr_fit_cv(x, dirs, blocks, elapsed, folds, kin=None):
    """TDR with fold-local standardization and optional measured kinematics controls."""
    axes = ('gain', 'time', 'cos', 'sin')

    def fit(indices):
        values = x[indices]
        active = values.std(0) > 1e-12
        z, mu, sd = A.zscore(values)
        z[:, ~active] = 0
        design, names = A.tdr_design(dirs[indices], values.sum(1), elapsed[indices])
        if kin is not None:
            kin_z, _, _ = A.zscore(kin[indices])
            design = np.column_stack([design, kin_z])
            names += ['vx', 'vy', 'speed']
        fitted = A.tdr_axes(z, design, names, axis_order=axes)
        norms = np.linalg.norm(fitted['beta'], axis=1)
        valid = np.array([norms[names.index(a)] > 1e-10 for a in axes])
        # QR can invent arbitrary directions for dependent coefficient vectors.
        for j in range(len(axes)):
            vectors = np.column_stack([fitted['beta'][names.index(a)] for a in axes[:j + 1]])
            valid[j] &= np.linalg.matrix_rank(vectors) == j + 1
        fitted.update(mu=mu, sd=sd, active=active, valid=valid, names=names,
                      rank=int(np.linalg.matrix_rank(design)), columns=design.shape[1])
        return fitted

    full = fit(np.arange(len(x)))
    z, _, _ = A.zscore(x)
    full['scores'] = z @ full['Q']
    full['scores'][:, ~full['valid']] = np.nan
    denominator = z.var(0, ddof=1).sum()
    full['variance_captured'] = {name: float(np.nanvar(full['scores'][:, j], ddof=1) / denominator)
                                if full['valid'][j] else np.nan for j, name in enumerate(axes)}
    scores = np.full((len(x), 4), np.nan)
    stability, designs = [], []
    for fold, tr, te in A.block_folds(blocks, folds):
        local = fit(tr)
        z, _, _ = A.zscore(x[te], local['mu'], local['sd'])
        z[:, ~local['active']] = 0
        scores[te] = z @ local['Q']
        scores[np.ix_(te, ~local['valid'])] = np.nan
        designs.append(dict(fold=fold, train_trials=len(tr), test_trials=len(te),
                            design_rank=local['rank'], design_columns=local['columns']))
        for j, name in enumerate(axes):
            cosine = float(local['Q'][:, j] @ full['Q'][:, j]) if local['valid'][j] and full['valid'][j] else np.nan
            stability.append(dict(fold=fold, axis=name, cosine_to_full=cosine))
    return dict(scores=scores, full=full, stability=stability, designs=designs)


def measured_kinematics(cohort, ok):
    """Window-mean observed cursor velocity on complete quality-checked 100-ms bins."""
    width = cohort.criteria.window_end - cohort.criteria.window_start
    count = int(round(width / .1))
    if count < 1 or not np.isclose(count * .1, width):
        return np.full((int(ok.sum()), 3), np.nan), 'Analysis window must contain whole 100-ms bins.'
    bins = velocity_bins(cohort, n_bins=count, until='window', policy='drop_trial')
    means = np.full((cohort.n, 3), np.nan)
    for trial in np.unique(bins.trial):
        velocities = bins.velocity[bins.trial == trial]
        if len(velocities) == count:
            means[trial] = [*velocities.mean(0), np.linalg.norm(velocities, axis=1).mean()]
    return means[ok], ''


def plot_angular_error(ax, obs, label):
    median = obs['median_abs_error_deg']
    title = f'{label}  Median |error| {median:.1f}°; descriptive offset removed'
    if len(obs['error']):
        target = np.degrees(obs['target'])
        error = np.degrees(obs['error'])
        jitter = np.random.default_rng(0).normal(0, 2.5, len(target))
        ax.scatter(target + jitter, error, c=[style.direction_color(int(round(t / 45)) % 8) for t in target], s=8)
    else:
        title = f'{label}  Angular error unavailable'
    ax.axhline(0, color=style.INK, lw=.7)
    for value in (-45, 45):
        ax.axhline(value, color=style.GREY, ls='--', lw=.8)
    ax.set(xlabel='Target direction / deg', ylabel='Angular error / deg', ylim=(-180, 180),
           title=title)


def tdr_main(held, dirs, obs, null, p_perm, stability, kin_rows, scatter, banner):
    fig = new_figure(14, 8.5)
    axes = fig.subplots(2, 3)
    scatter(axes[0, 0], held, dirs, 'Direction cos axis', 'Direction sin axis',
            'A  Held-out direction plane', legend=True)
    plot_angular_error(axes[0, 1], obs, 'B')
    ax = axes[0, 2]
    if len(null):
        ax.hist(null[np.isfinite(null)], bins=20, color=style.GREY)
    ax.axvline(abs(obs['circular_r']), color='#c0392b', label='Observed')
    ax.set(xlabel='|Fisher–Lee r|', ylabel='Permutations', title=f'C  Within-block null; p={p_perm:.4g}')
    ax = axes[1, 0]
    for j, name in enumerate(('gain', 'time', 'cos', 'sin')):
        rows = [r for r in stability if r['axis'] == name]
        ax.plot([r['fold'] for r in rows], [r['cosine_to_full'] for r in rows], 'o-', label=name, ms=3)
    ax.set(xlabel='Fold', ylabel='cos(fold axis, full-cohort axis)', ylim=(-1.05, 1.05),
           title='D  Axis stability (descriptive reference)')
    ax.legend(frameon=False, ncol=2, fontsize=6)
    ax = axes[1, 1]
    if kin_rows:
        ax.bar([0, 1], [abs(r['circular_r']) for r in kin_rows], color=['#2176ae', '#e07b39'])
        ax.set_xticks([0, 1], ['Direction model', '+ vx, vy, speed'])
        ax.set(ylim=(0, 1), ylabel='Held-out |Fisher–Lee r|',
               title=f'E  Same velocity-eligible cohort (n={kin_rows[0]["n_trials"]})')
        for i, row in enumerate(kin_rows):
            ax.text(i, abs(row['circular_r']) + .025, f'{abs(row["circular_r"]):.2f}', ha='center')
    else:
        style.text_panel(ax, ['E  Kinematics comparison unavailable', 'See notes for cohort size / window reason.',
                              'Online output is not independent movement.'])
    ax = axes[1, 2]
    if kin_rows:
        names = ('cos', 'sin', 'gain', 'time')
        xs = np.arange(len(names))
        for row, shift, colour in zip(kin_rows, (-.18, .18), ('#2176ae', '#e07b39')):
            ax.bar(xs + shift, [row[f'effect_{n}'] for n in names], width=.36,
                   color=colour, label=row['model'])
        ax.set_xticks(xs, names)
        ax.set(ylabel='||beta|| × SD(predictor)', title='F  Conditional predictor effect sizes')
        ax.legend(frameon=False, fontsize=6)
    else:
        style.text_panel(ax, ['F  No matched-cohort effect comparison.',
                              'Direction structure is an association,', 'not proof of motor intention.'])
    fig.suptitle(banner + '\nTDR · held-out structure, stability and measured-kinematics controls · ellipses = 1 SD', fontsize=9)
    return fig
