"""Velocity Kalman filter (local numpy).

Decoder:  velocity_kf
Source:   local numpy implementation (moved from `core.analysis.kf_fit/kf_predict`).
          Not the Decoder SDK ReFIT-KF; see refit_kf_filter.py for that.
Fits:     state v_t (2-D velocity, divided by the training RMS speed),
          v_t = A v_{t−1} + w,  z_t = C v_t + b + q, on features z-scored with
          training-row mean/SD (constant training channels dropped). A, C, b by
          least squares with a small ridge λ; transitions never cross sequence
          breaks. Observation covariance Q is shrunk toward its diagonal,
          Q ← (1−α)Q + α·diag Q (with a few thousand bins the raw 512×512
          estimate overfits).
Bins:     any bin period (BIN_SECONDS = None); A is per bin step.
Assumes:  state resets to v = 0, P = 0 at each sequence start (trial start or
          after a gap); a spectral radius slightly above 1 is tolerated because
          sequences are short.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

NAME = 'velocity_kf'
LABEL = 'Velocity KF'
SOURCE = dict(kind='local', path='data_analysis_gui/decoders/velocity_kf.py')
BIN_SECONDS = None
DEFAULTS = {'lam_kf': 1e-4, 'q_shrinkage': 0.5}
METHOD = ('Velocity-state Kalman filter (local numpy): least-squares A, C, b with ridge λ on training-z-scored '
          'features, velocity scaled by training RMS speed, observation covariance shrunk toward its diagonal by α; '
          'state reset to zero at every sequence start (trial start or neural/cursor gap).')


def kf_fit(z, v, sequence, lam=1e-4, q_shrinkage=0.5):
    """Velocity Kalman filter: v_t = A v_{t-1} + w, z_t = C v_t + b + q."""
    prev = np.flatnonzero(sequence[:-1] == sequence[1:])
    nxt = prev + 1
    if len(prev) < 3:
        raise ValueError('KF needs consecutive bins within sequences')
    A = linalg.lstsq(np.vstack([v[prev], np.sqrt(lam) * np.eye(2)]),
                     np.vstack([v[nxt], np.zeros((2, 2))]))[0].T
    design = np.column_stack([v, np.ones(len(v))])
    reg = np.diag([np.sqrt(lam), np.sqrt(lam), 0.])
    coef = linalg.lstsq(np.vstack([design, reg]), np.vstack([z, np.zeros((3, z.shape[1]))]))[0]
    vr = v[nxt] - v[prev] @ A.T
    zr = z - design @ coef
    W = vr.T @ vr / len(vr) + lam * np.eye(2)
    Q = zr.T @ zr / len(zr) + lam * np.eye(z.shape[1])
    Q = (1 - q_shrinkage) * Q + q_shrinkage * np.diag(np.diag(Q))
    radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    return dict(A=A, C=coef[:2].T, bias=coef[2], W=W, Q=Q, spectral_radius=radius)


def kf_predict(par, z, sequence):
    """Batch replay; state resets to zero (P=0) at each sequence start."""
    A, Cm, W, Q, bias = par['A'], par['C'], par['W'], par['Q'], par['bias']
    ids, starts, lens = np.unique(sequence, return_index=True, return_counts=True)
    order = np.argsort(starts)
    starts, lens = starts[order], lens[order]
    out = np.zeros((len(z), 2))
    Qi = linalg.cho_solve(linalg.cho_factor(Q), np.eye(len(Q)))
    CtQi = Cm.T @ Qi
    info = CtQi @ Cm
    state = np.zeros((len(starts), 2))
    P = np.zeros((2, 2))
    for step in range(int(lens.max())):
        active = np.flatnonzero(lens > step)
        rows = starts[active] + step
        prior = state[active] @ A.T
        Pp = A @ P @ A.T + W
        P = linalg.inv(linalg.inv(Pp) + info)
        innovation = (z[rows] - bias) @ CtQi.T
        state[active] = (prior @ linalg.inv(Pp).T + innovation) @ P.T
        out[rows] = state[active]
    return out


def fit(train, params):
    lam = float(params.get('lam_kf', DEFAULTS['lam_kf']))
    alpha = float(params.get('q_shrinkage', DEFAULTS['q_shrinkage']))
    x = train.features
    mu, sd = x.mean(0), x.std(0)
    keep = sd > 1e-12
    mu, sd = mu[keep], sd[keep]
    z = (x[:, keep] - mu) / sd
    scale = float(np.sqrt(np.mean((train.velocity ** 2).sum(1)))) or 1.0
    par = kf_fit(z, train.velocity / scale, train.sequence, lam, alpha)
    return dict(par=par, mu=mu, sd=sd, keep=keep, scale=scale,
                info=dict(kf_lambda=lam, kf_q_shrinkage=alpha, kf_spectral_radius=par['spectral_radius'],
                          n_features=int(keep.sum())))


def predict(model, test):
    z = (test.features[:, model['keep']] - model['mu']) / model['sd']
    return kf_predict(model['par'], z, test.sequence) * model['scale']
