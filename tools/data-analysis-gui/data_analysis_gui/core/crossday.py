"""Helpers shared by the cross-day (x*) sections.

Different recordings transmit different 512-of-1024 electrode subsets, so every
comparison between two recordings is made on the electrode ids both transmitted
(`common_columns`), never on stored column positions.

Feature standardization for a model trained on day i and applied to day j:
  * 'train'      – day i mean and SD (the model is applied fixed)
  * 'recentre'   – day j mean, day i SD (re-centre features per day)
  * 'test_own'   – day j mean and SD (re-standardize the test day)
"""
from __future__ import annotations

from collections import Counter
from datetime import date as _date
from hashlib import sha256
import re

import numpy as np
from matplotlib import colormaps
from scipy import linalg

from . import analysis as A
from .features import velocity_bins

STANDARDIZE = ('train', 'recentre', 'test_own')
STANDARDIZE_LABELS = {'train': 'fixed (train-day mean/SD)', 'recentre': 're-centred (test-day mean)',
                      'test_own': 're-standardized (test-day mean/SD)'}


# ---------------------------------------------------------------- identity
SESSION_COLORS = {
    ('online', 'CO8'): '#bd5021',
    ('online', 'Grid'): '#11785f',
    ('offline', 'Grid'): '#2166ac',
}
SESSION_STYLES = {
    'CO8': ('Center Out 8, Offline', '#808080', 's'),
    'Grid': ('Grid, Offline', SESSION_COLORS['offline', 'Grid'], 'o'),
    'CO8 online': ('Center Out 8, Online', SESSION_COLORS['online', 'CO8'], 's'),
    'Grid online': ('Grid, Online', SESSION_COLORS['online', 'Grid'], 'o'),
}
OTHER_SESSION_STYLE = ('Other', '#808080', 'D')
CONDITION_STYLES = {
    'restrained': ('CO8, Online Restrained', '#7b3294', 's'),
}


def recording_style(recording):
    return next((style for key, style in CONDITION_STYLES.items()
                 if key in recording_label(recording)),
                SESSION_STYLES.get(recording.data_type, OTHER_SESSION_STYLE))


def session_label_color(label):
    for key, (_, color, _) in CONDITION_STYLES.items():
        if key in label:
            return color
    parts = label.split(' · ', 3)
    return SESSION_COLORS.get(tuple(parts[1:3]), '#000000')


def _recording_suffix(recording):
    return sha256(recording.key.encode('utf-8')).hexdigest()[:7]


def recording_label(recording) -> str:
    mode = 'online' if recording.online else 'offline'
    task = {'centerout': 'CO8', 'grid': 'Grid'}.get(recording.task_mode, '?')
    # Legacy files used run-th70 (etc.) as the condition without a run number.
    run = (f'rec-{_recording_suffix(recording)}' if re.fullmatch(r'th-?\d+(?:\.\d+)?', recording.run, re.IGNORECASE)
           else f'run{recording.run or "—"}')
    condition = getattr(recording, 'condition', '')
    explicit_tag = getattr(recording, 'header_info', {}).get('condition_tag')
    if explicit_tag is not None:
        if explicit_tag:
            run += f' [{explicit_tag}]'
    elif condition:
        speed = re.search(r'初始辅助 ([\d.]+) px/s', condition)
        tag = (f'restrained assist{speed[1]}' if speed else
               'arm free' if '自由手' in condition else 'manual')
        run += f' [{tag}]'
    return f'{recording.date[5:] or "—"} · {mode} · {task} · {run}'


def recording_labels(recordings):
    recordings = list(recordings)
    labels = [recording_label(r) for r in recordings]
    counts = Counter(labels)
    return [f'{label} / {_recording_suffix(r)}' if counts[label] > 1 else label
            for r, label in zip(recordings, labels)]


def recording_label_lines(label):
    parts = label.split(' · ', 3)
    display = ' · '.join(parts[:2]) + '\n' + ' · '.join(parts[2:]) if len(parts) == 4 else label
    return re.sub(r' \[restrained assist\s*([\d.]+)\]', r' [assist \1]', display)


def recording_ticks(ax, positions, labels, *, fontsize=6.5):
    labels = list(labels)
    ax.set_xticks(positions, [recording_label_lines(label) for label in labels],
                  rotation=45, rotation_mode='anchor', ha='right',
                  fontstyle='normal', fontsize=fontsize)
    for tick, label in zip(ax.get_xticklabels(), labels):
        tick.set_color(session_label_color(label))


def recording_yticks(ax, labels):
    ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
    for tick, label in zip(ax.get_yticklabels(), labels):
        tick.set_color(session_label_color(label))


def day_number(session) -> int:
    try:
        return _date.fromisoformat(session.recording.date).toordinal()
    except (TypeError, ValueError):
        return 0


def days_elapsed(sessions) -> np.ndarray:
    d = np.array([day_number(s) for s in sessions])
    return d - d.min()


def common_columns(ids_a, ids_b):
    """Shared electrode ids (sorted) and their stored columns in a and b."""
    ids, ca, cb = np.intersect1d(np.asarray(ids_a), np.asarray(ids_b), return_indices=True)
    return ids, ca, cb


def full_vector(electrode_ids, values, total=1024) -> np.ndarray:
    out = np.full(total, np.nan)
    out[np.asarray(electrode_ids) - 1] = values
    return out


# ---------------------------------------------------------------- per-day data (cached)
def direction_data(ctx, k):
    """Window-rate features, directions and blocks of included trials with finite rates."""
    cohort = ctx.cohorts[k]

    def compute():
        x = cohort.window_rates
        ok = np.isfinite(x).all(1)
        return dict(x=x[ok], y=cohort.directions[ok], blocks=cohort.blocks[ok], orders=cohort.orders[ok],
                    n_dropped=int((~ok).sum()))
    return ctx.cached(('xday_direction', id(cohort)), compute)


def velocity_data(ctx, k, offset=0.2, bin_s=0.1, n_bins=10, neural_window=0.2):
    cohort = ctx.cohorts[k]
    crit = cohort.criteria
    key = ('xday_velocity', id(cohort), offset, bin_s, n_bins, neural_window)
    return ctx.cached(key, lambda: velocity_bins(cohort, offset=offset, bin_s=bin_s, n_bins=n_bins,
                                                 neural_window=neural_window, min_frames=crit.min_frames,
                                                 max_gap=crit.velocity_max_gap))


# ---------------------------------------------------------------- standardization
def fit_scaler(x):
    mu, sd, keep = A.standardize_fit(x)
    return dict(mu=mu, sd=sd, keep=keep)


def transform(scaler, x, mode='train'):
    keep = scaler['keep']
    xs = x[:, keep]
    mu, sd = scaler['mu'][keep], scaler['sd'][keep]
    if mode in ('recentre', 'test_own'):
        mu = xs.mean(0)
    if mode == 'test_own':
        s = xs.std(0)
        sd = np.where(s > 1e-12, s, 1.0)
    return (xs - mu) / sd


# ---------------------------------------------------------------- models on standardized features
def fit_classifier(z, y, lam=1.0, n_classes=8):
    onehot = np.eye(n_classes)[y]
    centre = onehot.mean(0)
    coef = linalg.solve(z.T @ z + len(z) * lam * np.eye(z.shape[1]), z.T @ (onehot - centre), assume_a='pos')
    return dict(coef=coef, intercept=centre)


def predict_classifier(model, z):
    return (z @ model['coef'] + model['intercept']).argmax(1)


def common_direction_model(ctx, i, j, x, y, lam):
    """Fit on the shared stored columns of an ordered recording pair."""
    def compute():
        scaler = fit_scaler(x)
        model = fit_classifier(transform(scaler, x), y, lam) if scaler['keep'].sum() >= 5 else None
        return scaler, model

    key = ('xday_direction_model', id(ctx.cohorts[i]), id(ctx.cohorts[j]), lam)
    return ctx.cached(key, compute)


def fit_ridge(z, v, lam=1.0):
    coef = linalg.solve(z.T @ z + len(z) * lam * np.eye(z.shape[1]), z.T @ (v - v.mean(0)), assume_a='pos')
    return dict(coef=coef, intercept=v.mean(0))


def predict_ridge(model, z):
    return z @ model['coef'] + model['intercept']


def fit_kf(z, v, sequence, lam=1e-4, q_shrinkage=0.5):
    scale = float(np.sqrt(np.mean((v ** 2).sum(1)))) or 1.0
    par = A.kf_fit(z, v / scale, sequence, lam, q_shrinkage)
    par['scale'] = scale
    return par


def predict_kf(par, z, sequence):
    return A.kf_predict(par, z, sequence) * par['scale']


def velocity_scores(v, pred):
    """Mean R² over x/y, vector RMSE (sqrt of mean squared error per component), mean Pearson r."""
    ok = np.isfinite(pred).all(1)
    if ok.sum() < 3:
        return dict(r2=np.nan, rmse=np.nan, r=np.nan)
    return dict(r2=float(np.nanmean(A.r2_score(v[ok], pred[ok]))),
                rmse=float(np.sqrt(np.mean(A.rmse(v[ok], pred[ok]) ** 2))),
                r=float(np.nanmean(A.pearson(v[ok], pred[ok]))))


# ---------------------------------------------------------------- circular readout
def fisher_lee(a, b):
    """Rotation-invariant circular correlation (Fisher & Lee 1983)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    A_, B_ = np.sum(np.cos(a) * np.cos(b)), np.sum(np.sin(a) * np.sin(b))
    C_, D_ = np.sum(np.cos(a) * np.sin(b)), np.sum(np.sin(a) * np.cos(b))
    den = (n ** 2 - np.sum(np.cos(2 * a)) ** 2 - np.sum(np.sin(2 * a)) ** 2) * \
          (n ** 2 - np.sum(np.cos(2 * b)) ** 2 - np.sum(np.sin(2 * b)) ** 2)
    return float(4 * (A_ * B_ - C_ * D_) / np.sqrt(den)) if den > 0 else float('nan')


def plane_readout(scores, directions):
    """Agreement of the (cos, sin) plane angle with the target angle.

    TDR axes are anchored to the task regressors, so no rotation offset is fitted:
    `median_abs_error_deg` is the raw angular error. `circular_r` is rotation invariant.
    """
    ok = np.isfinite(scores).all(1)
    if ok.sum() < 3:
        return dict(circular_r=np.nan, median_abs_error_deg=np.nan, within_45deg=np.nan)
    a = np.arctan2(scores[ok, 1], scores[ok, 0])
    t = np.radians(np.asarray(directions)[ok] * 45.0)
    err = np.angle(np.exp(1j * (a - t)))
    return dict(circular_r=fisher_lee(t, a), median_abs_error_deg=float(np.degrees(np.median(np.abs(err)))),
                within_45deg=float(np.mean(np.abs(np.degrees(err)) <= 45)))


# ---------------------------------------------------------------- drawing
def heatmap(ax, matrix, labels, *, fmt='{:.2f}', cmap='viridis', vmin=None, vmax=None, title='',
            cbar_label='', xlabel='test recording', ylabel='train recording', fontsize=6.5, colorbar=True,
            col_labels=None, recording_columns=True):
    matrix = np.asarray(matrix, float)
    cm = colormaps[cmap].copy() if isinstance(cmap, str) else cmap.copy()
    cm = cm.with_extremes(bad='#d6d4cf')
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cm, vmin=vmin, vmax=vmax, aspect='auto',
                      interpolation='nearest')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            text = fmt.format(v) if np.isfinite(v) else 'NA'
            color = 'black'
            if np.isfinite(v):
                r, g, b, _ = cm(image.norm(v))
                color = 'white' if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=fontsize, color=color)
    col_labels = labels if col_labels is None else col_labels
    if recording_columns:
        recording_ticks(ax, range(matrix.shape[1]), col_labels)
    else:
        ax.set_xticks(range(matrix.shape[1]), col_labels, rotation=40, ha='right', fontsize=6.5)
    recording_yticks(ax, labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[:].set_visible(False)
    if title:
        ax.set_title(title, fontsize=8)
    if colorbar:
        cb = ax.figure.colorbar(image, ax=ax, fraction=0.05, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=7)
    return image


def jitter(n, width=0.25, seed=0):
    return np.random.default_rng(seed).uniform(-width, width, n)


# ---------------------------------------------------------------- figure data export
def matrix_rows(matrices: dict, labels, row_name='row_recording', col_name='col_recording', col_labels=None):
    """Long-format rows for one or several same-shape matrices {value_name: matrix}."""
    col_labels = labels if col_labels is None else col_labels
    first = next(iter(matrices.values()))
    rows = []
    for i in range(first.shape[0]):
        for j in range(first.shape[1]):
            row = {row_name: labels[i], col_name: col_labels[j]}
            for name, m in matrices.items():
                v = m[i, j]
                row[name] = float(v) if np.isfinite(v) else ''
            rows.append(row)
    return rows
