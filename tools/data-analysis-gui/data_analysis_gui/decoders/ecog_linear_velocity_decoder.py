"""ECoG linear velocity decoder from the BCI Decoder SDK, applied to spike-rate features.

Decoder:  ecog_linear_velocity_decoder  (SDK model_type `ecog_linear_velocity`, implementation 1.0.0)
Source:   github.com/sillybun/bci-decoder-sdk @ 8d79df0475b85ca859a02a8268f5262358a22afa
          src/decoders/ecog_linear_velocity_decoder  (vendored: vendor/sdk_8d79df0/decoders/ecog_linear_velocity_decoder)
          pip spec: bci-decoder-api @ git+ssh://git@github.com/sillybun/bci-decoder-sdk.git@8d79df0475b85ca859a02a8268f5262358a22afa

IMPORTANT — modality mismatch. This decoder was designed for ECoG band power
(`bci.feature.ecog_iir_bandpower_20ms.v1`: log10 IIR band power, (channel, 6
bands) theta…very_high_gamma, 20 ms cadence). Its `EcogLinearVelocityTrainer`,
Artifact validator and Runtime Core accept only that schema, so they cannot take
the 512-electrode spike-rate vectors of these recordings without relabelling
spike rates as band power — which this adapter refuses to do
(`trainer_contract_error()` returns the SDK's own rejection message). What the
contract does allow is its public numerical API, which is modality-free and
exported from the package: `desired_tracking_velocity`, `fit_trial_balanced_ridge`,
`smoothbatch`, `limit_speed`. This adapter runs exactly that update on
spike-rate features (Hz or sqrt Hz). Results describe the SDK's fitting
algorithm on spikes, not ECoG performance and not its Trainer/Runtime lifecycle.

Fits:     normalisation = per-feature training mean and max(SD, 1e-6) (the SDK's
          `EcogFitConfig.std_floor`; the SDK estimates it on a rest block, which
          these recordings do not have, so training rows are used). desired =
          trial-local backward difference of target position / 0.020 s (fixed
          SDK dt) + (target − cursor)/60 ms; offline target = cursor = right-edge
          cursor position, so desired = measured 20 ms velocity. One prepended
          row per sequence (left-edge position) makes the first real row's
          difference valid; that row is then removed (otherwise the SDK estimator
          assigns 0 to the first row of every sequence). Ridge with equal total
          weight per sequence (`fit_trial_balanced_ridge`, penalty `ecog_penalty`,
          sum loss, optional unpenalised intercept), first SmoothBatch update
          against the W=0 seed with first_alpha = 0 (= full optimum, SDK default).
          The SDK default penalty 1.0 (sum loss) is weak for ~500 overlapping-window
          spike features: it is fine on real sessions (~19k native rows) but
          overfits small ones (synthetic 8 blocks: R² 0.02–0.14; penalty 1000: 0.8).
Bins:     BIN_SECONDS = 0.02 (the SDK differencing uses a fixed 20 ms dt);
          native 20 ms bins, predictions averaged onto evaluation bins.
Predicts: v = W z + b, then `limit_speed(v, 1e9)` (no clipping in practice).
"""
from __future__ import annotations

import numpy as np

from . import electrode_layout_id

NAME = 'ecog_linear_velocity_decoder'
LABEL = 'ECoG-linear (SDK, spikes)'
COMMIT = '8d79df0475b85ca859a02a8268f5262358a22afa'
SOURCE = dict(kind='vendored', repo='github.com/sillybun/bci-decoder-sdk', commit=COMMIT,
              path='src/decoders/ecog_linear_velocity_decoder', model_type='ecog_linear_velocity',
              pip_spec=f'bci-decoder-api @ git+ssh://git@github.com/sillybun/bci-decoder-sdk.git@{COMMIT}')
BIN_SECONDS = 0.02
DEFAULTS = {'ecog_penalty': 1.0, 'ecog_intercept': True}
MAX_SPEED = 1e9
METHOD = ('SDK ecog_linear_velocity_decoder numerical API (desired_tracking_velocity, fit_trial_balanced_ridge, '
          'smoothbatch, limit_speed) applied to SPIKE-RATE features — the decoder was designed for ECoG band power and '
          'its Trainer/Runtime reject spike features, so they are not used; normalisation from training rows '
          '(no rest block), desired = measured 20 ms velocity, ridge with equal weight per sequence, first SmoothBatch '
          'update = full optimum.')


def _sdk():
    from .vendor.sdk_8d79df0 import bci_decoder_api as sdk
    from .vendor.sdk_8d79df0.decoders import ecog_linear_velocity_decoder as ecog
    return sdk, ecog


def trainer_contract_error(electrode_ids=(1, 2, 3)) -> str:
    """Message the SDK Trainer raises for a spike-rate (channel,) FeatureSpec."""
    sdk, ecog = _sdk()
    ids = np.asarray(electrode_ids)
    feature = sdk.FeatureSpec(
        schema=sdk.SchemaDefinition('data_analysis_gui.feature.spike_rate_20ms.v1', 'feature', 'float32', ('channel',),
                                    'Per-electrode spike event rate', 'hz'),
        layout=sdk.TensorLayout(electrode_layout_id('data_analysis_gui.electrodes', ids), (len(ids),)))
    action = sdk.ActionSpec(schema=sdk.schema_catalog.load(ecog.CURSOR_VELOCITY_SCHEMA_ID),
                            layout=sdk.TensorLayout('data_analysis_gui.velocity.xy.v1', (2,), {'component': ('x', 'y')}))
    try:
        ecog.EcogLinearVelocityTrainer(feature_spec=feature, action_spec=action, max_speed=MAX_SPEED)
    except ValueError as exc:
        return f'EcogLinearVelocityTrainer rejects spike-rate features: {exc}'
    return ''


def fit(train, params):
    _, ecog = _sdk()
    penalty = float(params.get('ecog_penalty', DEFAULTS['ecog_penalty']))
    intercept = bool(params.get('ecog_intercept', DEFAULTS['ecog_intercept']))
    if abs(train.bin_seconds - BIN_SECONDS) > 1e-9:
        raise ValueError(f'{NAME} requires 20 ms bins (fixed SDK dt), got {train.bin_seconds * 1000:g} ms')
    x = train.features.astype(np.float64)
    keep = x.std(0) > 1e-12
    mean = x[:, keep].mean(0)
    scale = np.maximum(x[:, keep].std(0), 1e-6)
    z = (x[:, keep] - mean) / scale
    # Prepend one row per sequence so the SDK's backward difference is valid on every real row.
    end = train.end_position
    pos, ts, ids, real = [], [], [], []
    stamps = train.timestamps_ns()
    for a, b in train.sequence_bounds():
        pos += [train.position[a], *end[a:b]]
        ts += [stamps[a] - 20_000_000, *stamps[a:b]]
        ids += [int(train.sequence[a])] * (b - a + 1)
        real += [False] + [True] * (b - a)
    pos, ts, ids, real = np.asarray(pos), np.asarray(ts, np.int64), np.asarray(ids, np.int64), np.asarray(real)
    desired = ecog.desired_tracking_velocity(pos, pos, ts, ids, tracking_time_constant_ms=60.0,
                                             max_speed=MAX_SPEED)[real]
    w_opt, b_opt = ecog.fit_trial_balanced_ridge(z, desired, train.sequence, penalty=penalty, fit_intercept=intercept)
    weights, bias = ecog.smoothbatch(np.zeros_like(w_opt), np.zeros(2), w_opt, b_opt, alpha=0.0)
    if not intercept:
        bias = np.zeros(2)
    resid = z @ weights.T + bias - desired
    info = dict(ecog_penalty=penalty, ecog_intercept=intercept, n_features=int(keep.sum()),
                train_rmse=float(np.sqrt(np.mean(np.sum(resid ** 2, 1)))),
                max_label_error=float(np.max(np.abs(desired - train.velocity))))
    return dict(W=weights, b=bias, keep=keep, mean=mean, scale=scale, info=info)


def predict(model, test):
    _, ecog = _sdk()
    z = (test.features[:, model['keep']].astype(np.float64) - model['mean']) / model['scale']
    return ecog.limit_speed(z @ model['W'].T + model['b'], MAX_SPEED)
