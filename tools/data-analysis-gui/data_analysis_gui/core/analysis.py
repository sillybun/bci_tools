"""Shared statistics, cross-validation and decoders (numpy/scipy only).

Conventions follow neural_behavior/independent_recheck_v1:
  * five chronological block folds; training drops blocks within ±1 of the test range
  * ridge on standardized features, lambda scaled by n_train, intercept = train mean
  * within-block permutations for nulls
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, stats


# ---------------------------------------------------------------- folds
def block_folds(blocks: np.ndarray, n_folds: int = 5, purge: int = 1):
    """Yield (fold, train_idx, test_idx) over contiguous chronological block groups."""
    blocks = np.asarray(blocks)
    unique = np.unique(blocks)
    n_folds = max(2, min(n_folds, len(unique)))
    groups = np.array_split(unique, n_folds)
    for f, group in enumerate(groups):
        test = np.flatnonzero(np.isin(blocks, group))
        lo, hi = group.min(), group.max()
        train = np.flatnonzero((blocks < lo - purge) | (blocks > hi + purge))
        if len(train) and len(test):
            yield f, train, test


def fold_labels(blocks, n_folds=5):
    out = np.full(len(blocks), -1)
    for f, _, test in block_folds(blocks, n_folds, purge=0):
        out[test] = f
    return out


def permute_within_block(rng, labels, blocks):
    out = labels.copy()
    for b in np.unique(blocks):
        ii = np.flatnonzero(blocks == b)
        out[ii] = rng.permutation(labels[ii])
    return out


# ---------------------------------------------------------------- stats
def bh_fdr(p: np.ndarray, family: np.ndarray | None = None) -> np.ndarray:
    """Benjamini–Hochberg q-values; channels outside `family` get q = 1."""
    p = np.asarray(p, float)
    q = np.ones_like(p)
    idx = np.arange(len(p)) if family is None else np.asarray(family)
    if not len(idx):
        return q
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    return q


def oneway_anova(values: np.ndarray, groups: np.ndarray, n_groups: int = 8):
    """Vectorized one-way ANOVA per column. Returns F, p, partial eta², group means, SEM."""
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    present = [g for g in range(n_groups) if np.any(groups == g)]
    counts = np.array([np.sum(groups == g) for g in range(n_groups)])
    mean = np.full((n_groups, values.shape[1]), np.nan)
    sem = np.full_like(mean, np.nan)
    for g in present:
        v = values[groups == g]
        mean[g] = v.mean(0)
        sem[g] = v.std(0, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
    grand = values.mean(0)
    ssb = np.nansum(counts[:, None] * (mean - grand) ** 2, axis=0)
    sst = ((values - grand) ** 2).sum(0)
    ssw = np.maximum(sst - ssb, 0)
    df_b, df_w = len(present) - 1, len(values) - len(present)
    valid = (ssw > 1e-12) & (df_w > 0) & (df_b > 0)
    F = np.zeros(values.shape[1])
    F[valid] = (ssb[valid] / df_b) / (ssw[valid] / df_w)
    p = np.ones(values.shape[1])
    p[valid] = stats.f.sf(F[valid], df_b, df_w)
    eta = np.divide(ssb, sst, out=np.zeros_like(sst), where=sst > 1e-12)
    return dict(F=F, p=p, eta2=eta, mean=mean, sem=sem, counts=counts, df=(df_b, df_w))


def balanced_accuracy(y, pred, n_classes=8):
    recalls = [np.mean(pred[y == k] == k) for k in range(n_classes) if np.any(y == k)]
    return float(np.mean(recalls)) if recalls else float('nan')


def r2_score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean(0)) ** 2).sum(0)
    return 1 - ((y - p) ** 2).sum(0) / np.where(ss > 0, ss, np.nan)


def rmse(y, p):
    return np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2, axis=0))


def pearson(y, p):
    out = []
    for k in range(y.shape[1]):
        out.append(float(np.corrcoef(y[:, k], p[:, k])[0, 1]) if np.std(p[:, k]) > 1e-12 else float('nan'))
    return np.array(out)


# ---------------------------------------------------------------- features
def standardize_fit(x):
    mu, sd = x.mean(0), x.std(0)
    keep = sd > 1e-12
    return mu, np.where(keep, sd, 1.0), keep


def ridge_fit(x, y, lam=1.0):
    """Standardized ridge, penalty n*lam; returns a predictor closure and coefficients."""
    mu, sd, keep = standardize_fit(x)
    z = (x[:, keep] - mu[keep]) / sd[keep]
    coef = linalg.solve(z.T @ z + len(z) * lam * np.eye(z.shape[1]), z.T @ (y - y.mean(0)), assume_a='pos')
    intercept = y.mean(0)

    def predict(xn):
        return ((xn[:, keep] - mu[keep]) / sd[keep]) @ coef + intercept
    return predict, dict(mu=mu, sd=sd, keep=keep, coef=coef, intercept=intercept)


def ridge_classifier_cv(x, y, blocks, lam=1.0, n_classes=8, n_folds=5):
    """Held-out class scores via one-hot ridge; also returns the linear CV operator."""
    n = len(y)
    operator = np.zeros((n, n))
    folds = np.full(n, -1)
    for f, tr, te in block_folds(blocks, n_folds):
        mu, sd, keep = standardize_fit(x[tr])
        a = (x[tr][:, keep] - mu[keep]) / sd[keep]
        b = (x[te][:, keep] - mu[keep]) / sd[keep]
        # Kernel form of (AᵀA + nλI)⁻¹Aᵀ; columns of A are centred so labels need no
        # centring. scores = operator @ onehot keeps exact permutation refits cheap.
        gram = a @ a.T + len(a) * lam * np.eye(len(a))
        h = b @ a.T @ linalg.solve(gram, np.eye(len(a)), assume_a='pos')
        operator[np.ix_(te, tr)] = h + 1 / len(tr)
        folds[te] = f
    scores = operator @ np.eye(n_classes)[y]
    tested = folds >= 0
    return scores, operator, folds, tested


def permutation_null_accuracy(operator, y, blocks, tested, n_perm=200, seed=20261029, n_classes=8):
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        labels = permute_within_block(rng, y, blocks)
        pred = (operator @ np.eye(n_classes)[labels]).argmax(1)
        null[i] = balanced_accuracy(labels[tested], pred[tested], n_classes)
    return null


# ---------------------------------------------------------------- Kalman filter / velocity CV
# The decoders live in `data_analysis_gui.decoders` (one module per decoder); these wrappers keep
# the historical API used by other sections. Imports are lazy: data_analysis_gui.decoders imports
# this module for `block_folds`.
def kf_fit(z, v, sequence, lam=1e-4, q_shrinkage=0.5):
    """Velocity Kalman filter fit; see `data_analysis_gui.decoders.velocity_kf.kf_fit`."""
    from ..decoders.velocity_kf import kf_fit as fit
    return fit(z, v, sequence, lam, q_shrinkage)


def kf_predict(par, z, sequence):
    """Batch KF replay; see `data_analysis_gui.decoders.velocity_kf.kf_predict`."""
    from ..decoders.velocity_kf import kf_predict as predict
    return predict(par, z, sequence)


def cv_velocity(features, velocity, trial_of_bin, sequence, blocks_of_bin, models=('ridge', 'kf'),
                lam_ridge=1.0, lam_kf=1e-4, q_shrinkage=0.5, n_folds=5):
    """Block-CV velocity decoding on bins. Returns predictions per model and fold labels.

    Delegates to `data_analysis_gui.decoders.cv_decode` with the local `ridge` and `velocity_kf`
    decoders ('kf' is the historical key for velocity_kf); numerics are unchanged.
    """
    from .. import decoders as D
    velocity = np.asarray(velocity, float)
    data = D.DecoderData(features=np.asarray(features), velocity=velocity, position=np.zeros_like(velocity),
                         sequence=np.asarray(sequence), trial=np.asarray(trial_of_bin), bin_seconds=None,
                         electrode_ids=np.arange(1, np.shape(features)[1] + 1), block=np.asarray(blocks_of_bin))
    params = dict(lam_ridge=lam_ridge, lam_kf=lam_kf, q_shrinkage=q_shrinkage)
    pred, folds, rows = {}, np.full(len(velocity), -1), {}
    for m in models:
        res = D.cv_decode(D.get('velocity_kf' if m == 'kf' else m), data, params, n_folds)
        pred[m] = res['pred']
        folds = res['folds']
        for r in res['info']:
            row = rows.setdefault(r['fold'], dict(fold=r['fold'], n_train_bins=r['n_train_bins'],
                                                  n_test_bins=r['n_test_bins']))
            if m == 'kf':
                row.update(kf_lambda=r['kf_lambda'], kf_q_shrinkage=r['kf_q_shrinkage'],
                           kf_spectral_radius=r['kf_spectral_radius'])
    return pred, folds, [rows[f] for f in sorted(rows)]


# ---------------------------------------------------------------- dimensionality reduction
def zscore(x, mu=None, sd=None):
    if mu is None:
        mu, sd = x.mean(0), x.std(0)
    return (x - mu) / np.where(sd > 1e-12, sd, 1.0), mu, sd


def pca(x, n_components=10):
    z, mu, sd = zscore(x)
    u, s, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
    var = s ** 2 / max(len(z) - 1, 1)
    explained = var / var.sum() if var.sum() > 0 else var
    k = min(n_components, len(s))
    return dict(scores=(z - z.mean(0)) @ vt[:k].T, components=vt[:k], explained=explained[:k],
                mu=mu, sd=sd)


def tdr_design(directions, total_rate=None, order=None):
    theta = np.radians(np.asarray(directions) * 45.0)
    cols = {'const': np.ones(len(theta)), 'cos': np.cos(theta), 'sin': np.sin(theta)}
    if total_rate is not None:
        cols['gain'] = stats.zscore(total_rate) if np.std(total_rate) > 0 else np.zeros(len(theta))
    if order is not None:
        o = np.asarray(order, float)
        cols['time'] = stats.zscore(o) if np.std(o) > 0 else np.zeros(len(theta))
    names = list(cols)
    return np.column_stack([cols[n] for n in names]), names


def tdr_axes(z, design, names, axis_order=('gain', 'time', 'cos', 'sin')):
    beta = linalg.lstsq(design, z)[0]
    use = [n for n in axis_order if n in names]
    M = np.column_stack([beta[names.index(n)] for n in use])
    Q, R = np.linalg.qr(M)
    Q = Q * np.sign(np.where(np.abs(np.diag(R)) < 1e-12, 1.0, np.diag(R)))
    return dict(beta=beta, axis_names=use, Q=Q)


def tdr_cv(x, directions, blocks, orders=None, n_folds=5):
    """Cross-validated TDR projections: each trial projected on axes fitted without it."""
    active = x.std(0) > 1e-12
    x = x[:, active]
    total = x.sum(1)
    out = None
    for f, tr, te in block_folds(blocks, n_folds):
        ztr, mu, sd = zscore(x[tr])
        design, names = tdr_design(directions[tr], total[tr], None if orders is None else orders[tr])
        fit = tdr_axes(ztr, design, names)
        zte, _, _ = zscore(x[te], mu, sd)
        if out is None:
            out = np.full((len(x), len(fit['axis_names'])), np.nan)
        out[te] = zte @ fit['Q']
    z, _, _ = zscore(x)
    design, names = tdr_design(directions, total, orders)
    full = tdr_axes(z, design, names)
    full['scores'] = z @ full['Q']
    full['variance_captured'] = {n: float((z @ full['Q'][:, i]).var(ddof=1) / z.var(0, ddof=1).sum())
                                 for i, n in enumerate(full['axis_names'])}
    full['active'] = active
    return out, full


def circular_corr(a, b):
    """Jammalamadaka–SenGupta circular correlation.

    Centres each variable on its own mean direction, which is undefined for a
    balanced eight-target design; prefer a rotation-invariant Fisher–Lee r
    (see core.crossday.fisher_lee) for target-vs-decoded angle readouts.
    """
    a, b = np.asarray(a), np.asarray(b)
    sa, sb = np.sin(a - np.angle(np.exp(1j * a).mean())), np.sin(b - np.angle(np.exp(1j * b).mean()))
    den = np.sqrt((sa ** 2).sum() * (sb ** 2).sum())
    return float((sa * sb).sum() / den) if den > 0 else float('nan')
