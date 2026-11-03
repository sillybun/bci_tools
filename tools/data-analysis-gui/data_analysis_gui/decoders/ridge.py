"""Ridge velocity decoder.

Decoder:  ridge
Source:   local numpy implementation (moved from `core.analysis.cv_velocity`,
          neural_behavior independent_recheck_v1 convention).
Fits:     v = z W + v̄ on standardised features z = (x − μ)/σ (μ, σ from the
          training rows only; channels with σ ≤ 1e-12 in training are dropped),
          penalty n_train · λ · ‖W‖², intercept = training mean velocity.
Bins:     any bin period (BIN_SECONDS = None); each bin is independent.
Assumes:  linear, instantaneous map from the bin's trailing-window rates to the
          bin velocity; no temporal state.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

NAME = 'ridge'
LABEL = 'Ridge'
SOURCE = dict(kind='local', path='data_analysis_gui/decoders/ridge.py')
BIN_SECONDS = None
DEFAULTS = {'lam_ridge': 1.0}
METHOD = ('Ridge regression of bin velocity on features z-scored with training-row mean/SD (constant training '
          'channels dropped); penalty n_train·λ, intercept = training mean velocity.')


def standardizer(x):
    mu, sd = x.mean(0), x.std(0)
    keep = sd > 1e-12
    return mu[keep], sd[keep], keep


def fit(train, params):
    lam = float(params.get('lam_ridge', DEFAULTS['lam_ridge']))
    mu, sd, keep = standardizer(train.features)
    z = (train.features[:, keep] - mu) / sd
    v = train.velocity
    coef = linalg.solve(z.T @ z + len(z) * lam * np.eye(z.shape[1]), z.T @ (v - v.mean(0)), assume_a='pos')
    return dict(mu=mu, sd=sd, keep=keep, coef=coef, intercept=v.mean(0),
                info=dict(lam_ridge=lam, n_features=int(keep.sum())))


def predict(model, test):
    z = (test.features[:, model['keep']] - model['mu']) / model['sd']
    return z @ model['coef'] + model['intercept']
