"""Madduri SmoothBatch co-adaptive linear decoder from the BCI Decoder SDK (offline).

Decoder:  madduri_coadaptive_decoder  (SDK model_type `madduri_smoothbatch`, implementation 2.0.0)
Source:   github.com/sillybun/bci-decoder-sdk @ b70297c092157a0627fa8d5f97f1694d6043e2bc
          src/decoders/madduri_coadaptive_decoder  (vendored: vendor/sdk_b70297c/decoders/madduri_coadaptive_decoder)
          pip spec: bci-decoder-api @ git+ssh://git@github.com/sillybun/bci-decoder-sdk.git@b70297c092157a0627fa8d5f97f1694d6043e2bc
Fits:     `MadduriSmoothBatchTrainer.fit(make_seed_block(...), MadduriFitConfig(random_seed))`
          (random D ∈ [0, 0.01], b = 0, no data), then `madduri_updates` calls of
          `adapt(block, parent, MadduriAdaptConfig(learning_rate=α, penalty, fit_intercept))`
          on the same training block: D* = argmin ‖X Dᵀ − v_desired‖² + penalty‖D‖²
          (sum loss), D ← α·D_parent + (1−α)·D*. After k updates the seed keeps
          weight αᵏ. v_desired = (target − cursor)/Δ with Δ = the bin period;
          offline we pass cursor = left-edge position and target = right-edge
          position, so v_desired is exactly the measured bin velocity (as the
          macaque_data_analysis adapter does: "observed next endpoint").
Bins:     BIN_SECONDS = None: the core is stateless (v = D u + b per frame) and the
          trainer takes `state_update_period_ns` explicitly; it is set to the
          evaluation bin period (SDK default 20 ms), so no resampling is needed.
Inputs:   features z-scored with training-row mean/SD before the SDK (the SDK does
          not normalise; same as the macaque adapter); channels constant in
          training dropped. fit_intercept default True (macaque adapter; paper
          default False keeps b = 0 — with z-scored features b is the mean velocity).
          max_speed None (no clipping).
Predicts: SDK `create_runtime(artifact).infer()` frame by frame (stateless; reset per sequence).
"""
from __future__ import annotations

import numpy as np

from . import electrode_layout_id
from ._blas import single_thread

NAME = 'madduri_coadaptive_decoder'
LABEL = 'Madduri SmoothBatch (SDK)'
COMMIT = 'b70297c092157a0627fa8d5f97f1694d6043e2bc'
SOURCE = dict(kind='vendored', repo='github.com/sillybun/bci-decoder-sdk', commit=COMMIT,
              path='src/decoders/madduri_coadaptive_decoder', model_type='madduri_smoothbatch',
              pip_spec=f'bci-decoder-api @ git+ssh://git@github.com/sillybun/bci-decoder-sdk.git@{COMMIT}')
BIN_SECONDS = None
DEFAULTS = {'madduri_penalty': 100.0, 'madduri_alpha': 0.5, 'madduri_updates': 5, 'madduri_seed': 20261028,
            'madduri_intercept': True}
METHOD = ('SDK madduri_coadaptive_decoder: random seed decoder (MadduriFitConfig random_seed) followed by k SmoothBatch '
          'adapt() updates on the training block (closed-form ridge D*, sum-loss penalty, blend α·parent + (1−α)·D*), '
          'desired velocity = measured bin velocity (target = right-edge, cursor = left-edge position, Δ = bin '
          'period); features z-scored with training mean/SD before the SDK; held-out inference through the SDK '
          'Runtime (stateless affine map).')


def _sdk():
    from .vendor.sdk_b70297c import bci_decoder_api as sdk
    from .vendor.sdk_b70297c.decoders import madduri_coadaptive_decoder as md
    return sdk, md


def _specs(sdk, data, keep):
    ids = np.asarray(data.electrode_ids)[keep]
    feature = sdk.FeatureSpec(
        schema=sdk.SchemaDefinition('data_analysis_gui.feature.spike_rate_zscored.v1', 'feature', 'float32', ('channel',),
                                    'Per-electrode event rate over a trailing neural window, z-scored with the '
                                    'training-row mean/SD', 'z',
                                    {'bin_period_ns': str(int(round(data.bin_seconds * 1e9)))}),
        layout=sdk.TensorLayout(electrode_layout_id('data_analysis_gui.electrodes', ids), (int(len(ids)),),
                                {'channel': tuple(f'e{int(i)}' for i in ids)}))
    action = sdk.ActionSpec(
        schema=sdk.SchemaDefinition('data_analysis_gui.velocity_2d.v1', 'action', 'float32', ('component',),
                                    'Cursor velocity in task coordinates', 'task_unit_per_s'),
        layout=sdk.TensorLayout('data_analysis_gui.velocity.xy.v1', (2,), {'component': ('x', 'y')}))
    return feature, action


def fit(train, params):
    sdk, md = _sdk()
    p = {k: params.get(k, v) for k, v in DEFAULTS.items()}
    x = train.features
    mu, sd = x.mean(0), x.std(0)
    keep = sd > 1e-12
    mu, sd = mu[keep], sd[keep]
    z = ((x[:, keep] - mu) / sd).astype(np.float32)
    period_ns = int(round(train.bin_seconds * 1e9))
    feature, action = _specs(sdk, train, keep)
    trainer = md.MadduriSmoothBatchTrainer(feature_spec=feature, action_spec=action, max_speed=None,
                                           state_update_period_ns=period_ns)
    source = ('data_analysis_gui.velocity_bins', 'train-rows-only')
    result = trainer.fit(md.make_seed_block(feature_spec=feature, source_ids=source),
                         md.MadduriFitConfig(random_seed=int(p['madduri_seed'])))
    block = md.MadduriInteractionBlock(
        features=z, timestamps_ns=train.timestamps_ns(), trial_ids=train.sequence.astype(np.int64),
        feature_schema_id=feature.schema_id, feature_layout_id=feature.layout_id,
        target_position=train.position + train.velocity * (period_ns / 1e9), cursor_position=train.position,
        source_ids=source)
    config = md.MadduriAdaptConfig(learning_rate=float(p['madduri_alpha']), penalty=float(p['madduri_penalty']),
                                   fit_intercept=bool(p['madduri_intercept']))
    metrics = {}
    for _ in range(int(p['madduri_updates'])):
        result = trainer.adapt(block, result.artifact, config)
        metrics = dict(result.report.metrics)
    info = dict(n_features=int(keep.sum()), updates=int(p['madduri_updates']),
                seed_weight=float(p['madduri_alpha']) ** int(p['madduri_updates']),
                train_rmse=metrics.get('adapt.tracking_velocity_rmse'), model_version=result.artifact.model_version)
    return dict(artifact=result.artifact, keep=keep, mu=mu, sd=sd, info=info)


def predict(model, test):
    sdk, _ = _sdk()
    artifact = model['artifact']
    runtime = sdk.create_runtime(artifact)
    spec = artifact.feature_spec
    mask = np.ones(spec.shape, dtype=bool)
    z = ((test.features[:, model['keep']] - model['mu']) / model['sd']).astype(np.float32)
    ts = test.timestamps_ns()
    out = np.full((test.n, 2), np.nan)
    with single_thread():
        _replay(runtime, sdk, spec, mask, z, ts, test, out)
    return out


def _replay(runtime, sdk, spec, mask, z, ts, test, out):
    for a, b in test.sequence_bounds():
        runtime.reset_state()
        for i in range(a, b):
            result = runtime.infer(sdk.DecoderRequest(features=sdk.FeatureTensor.from_spec(
                spec, values=z[i], valid_mask=mask, timestamp_ns=int(ts[i]), producer_frame_id=i)))
            if result.status is not sdk.InferenceStatus.OK or result.output is None:
                raise RuntimeError(f'{NAME} frame {i}: {result.status.value}: {result.diagnostics.message}')
            out[i] = result.output.action.astype(float)
