"""Grid-task target decoding: target position regression and target-identity classification.

Eight-direction confusion matrices are not meaningful for the Grid task (targets
lie on a lattice, starts vary). Instead:
  * target position (x, y) and reach displacement (target − start) are regressed
    from the analysis-window rates with standardized ridge (penalty n·λ) over the
    shared chronological block folds (±1 block purge);
  * target identity over the unique lattice targets is classified with one-hot
    ridge (argmax) on the same folds, scored by balanced accuracy vs 1/K chance,
    optionally with a within-block permutation null.
"""
from __future__ import annotations

import numpy as np

from . import analysis as A

MAX_TARGETS = 64          # identity classification only when the lattice is this small
MIN_PER_TARGET = 3


def target_labels(cohort, decimals=3):
    """Unique (rounded) target positions and the class index of every cohort trial."""
    xy = np.c_[cohort.array('target_x'), cohort.array('target_y')]
    unique, labels = np.unique(np.round(xy, decimals), axis=0, return_inverse=True)
    return unique, labels.ravel(), xy


def cv_ridge_regression(x, y, blocks, lam=1.0, n_folds=5):
    pred = np.full(y.shape, np.nan)
    folds = np.full(len(y), -1)
    for f, tr, te in A.block_folds(blocks, n_folds):
        predict, _ = A.ridge_fit(x[tr], y[tr], lam)
        pred[te] = predict(x[te])
        folds[te] = f
    return pred, folds


def _metrics(y, pred):
    ok = np.isfinite(pred).all(1)
    y, pred = y[ok], pred[ok]
    r2 = A.r2_score(y, pred)
    err = np.linalg.norm(pred - y, axis=1)
    return dict(n=int(ok.sum()), r2_x=float(r2[0]), r2_y=float(r2[1]),
                r2_xy=float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean(0)) ** 2)),
                median_error=float(np.median(err)), mean_error=float(err.mean()),
                baseline_median_error=float(np.median(np.linalg.norm(y - y.mean(0), axis=1))))


def grid_target_decoding(cohort, lam=1.0, n_perm=0, seed=20261029):
    x = np.asarray(cohort.window_rates, float)
    ok = np.isfinite(x).all(1)
    x, blocks = x[ok], cohort.blocks[ok]
    x = x[:, x.std(0) > 1e-12]
    unique, labels, target = target_labels(cohort)
    labels, target = labels[ok], target[ok]
    start = np.c_[cohort.array('start_x'), cohort.array('start_y')][ok]
    out = dict(n_trials=int(ok.sum()), n_targets=int(len(unique)), targets=unique, target=target,
               displacement=target - start, rows=np.flatnonzero(ok), lam=lam)
    pos_pred, folds = cv_ridge_regression(x, target, blocks, lam)
    disp_pred, _ = cv_ridge_regression(x, target - start, blocks, lam)
    out.update(position_pred=pos_pred, displacement_pred=disp_pred, folds=folds,
               position=_metrics(target, pos_pred), displacement_metrics=_metrics(target - start, disp_pred))
    present, counts = np.unique(labels, return_counts=True)
    out['identity'] = None
    if len(unique) <= MAX_TARGETS and counts.min() >= MIN_PER_TARGET and len(unique) >= 2:
        k = len(unique)
        scores, operator, cfolds, tested = A.ridge_classifier_cv(x, labels, blocks, lam, n_classes=k)
        pred = scores.argmax(1)
        ba = A.balanced_accuracy(labels[tested], pred[tested], k)
        null = A.permutation_null_accuracy(operator, labels, blocks, tested, n_perm, seed, n_classes=k) \
            if n_perm else np.zeros(0)
        cm = np.zeros((k, k))
        np.add.at(cm, (labels[tested], pred[tested]), 1)
        err = np.linalg.norm(unique[pred[tested]] - unique[labels[tested]], axis=1)
        out['identity'] = dict(
            balanced_accuracy=ba, accuracy=float(np.mean(pred[tested] == labels[tested])), chance=1 / k,
            null=null, p=float((1 + np.sum(null >= ba)) / (1 + len(null))) if len(null) else float('nan'),
            confusion=cm / np.maximum(cm.sum(1, keepdims=True), 1), labels=labels, pred=pred, tested=tested,
            median_class_error=float(np.median(err)), min_per_target=int(counts.min()))
    return out
