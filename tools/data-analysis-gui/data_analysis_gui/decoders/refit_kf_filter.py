"""ReFIT Kalman filter from the BCI Decoder SDK (offline fit + SDK Runtime replay).

Decoder:  refit_kf_filter  (SDK model_type `standalone_refit_kf`, implementation 3.3.0)
Source:   github.com/sillybun/bci-decoder-sdk @ b70297c092157a0627fa8d5f97f1694d6043e2bc
          src/decoders/standalone_refit_kf  (vendored: vendor/sdk_b70297c/decoders/standalone_refit_kf)
          pip spec: bci-decoder-api @ git+ssh://git@github.com/sillybun/bci-decoder-sdk.git@b70297c092157a0627fa8d5f97f1694d6043e2bc
Fits:     `RefitKFTrainer.fit(RefitKFFitData(calibration, initial_tracking))` — the
          SDK's Stage-1 supervised fit. State [position, velocity, 1]; A (velocity
          block), W, C (features on position, velocity, bias), Q, P0 by the SDK's
          mean-loss ridge (transition/observation λ = `refit_reg`) with fixed
          covariance floors 1e-4. Default refit_reg = 1e-2, not the SDK's 1e-4: on
          real 20 ms cursor velocity the 1e-4 transition fit has spectral radius
          ≥ 1 (1.006–1.008 on 2026-10-29 run-001) and the SDK refuses the
          Artifact; 1e-2 gives ≈0.984 (the macaque adapter searched 1e-4…1e-2). Normalisation = SDK "calibration" mean/SD, here
          estimated from the training rows (the recordings have no rest block).
          ReFIT `adapt()` (intention rotation toward the target) is NOT run: it
          needs closed-loop data; this is an offline open-loop evaluation.
Bins:     BIN_SECONDS = 0.02. The SDK fixes the state update to 20 ms and labels
          velocity as a trial-local backward position difference / 0.020 s, so it
          gets its own 20 ms bins (`velocity_bins(bin_s=0.02, neural_window=<native
          window>)`); predictions are averaged onto evaluation bins. Native window
          default 0.2 s trailing (overlapping, = Ridge features; macaque adapter
          precedent). 0.02 s = the SDK's non-overlapping-frame design, but receive-
          time jitter leaves ~38% of 20 ms windows without a frame (2026-10-29
          run-001), shattering sequences into ~2-bin pieces.
Inputs:   features = spike rates (Hz or sqrt Hz) of channels non-constant in
          training; positions/velocities divided by the training RMS speed (as in
          macaque_data_analysis sdk/training/adapter.py) so the SDK's px-scaled
          priors (P0 2500, max_speed) are not mis-scaled; max_speed = 1e9 (off).
          Each sequence gets one prepended initialisation row (position = left
          edge of its first bin, features copied) so the backward difference of
          the first real row is valid; the SDK excludes that row from fitting.
          Sequences shorter than 2 bins are not used for fitting.
Predicts: SDK `create_runtime(artifact).infer()` frame by frame; reset at every
          sequence start with the left-edge cursor position. Position feedback
          (`refit_feedback`): 'decoded' (default) = start position + Σ previous
          decoded velocity × 0.020 s (the cursor the decoder would have produced,
          no GT leak); 'recorded' = recorded cursor at the bin's left edge.
Assumes:  20 ms native bins tile the evaluation bins; SDK log ERRORs about
          non-20 ms timestamp intervals (calibration rows span trial gaps) are
          captured and counted in model info, not shown. Replay runs with numpy's
          OpenBLAS single-threaded (per-frame ~500×500 solves are ~20× faster).
"""
from __future__ import annotations

import logging

import numpy as np

from . import electrode_layout_id
from ._blas import single_thread

NAME = 'refit_kf_filter'
LABEL = 'ReFIT-KF (SDK)'
COMMIT = 'b70297c092157a0627fa8d5f97f1694d6043e2bc'
SOURCE = dict(kind='vendored', repo='github.com/sillybun/bci-decoder-sdk', commit=COMMIT,
              path='src/decoders/standalone_refit_kf', model_type='standalone_refit_kf',
              pip_spec=f'bci-decoder-api @ git+ssh://git@github.com/sillybun/bci-decoder-sdk.git@{COMMIT}')
BIN_SECONDS = 0.02
DEFAULTS = {'refit_reg': 1e-2, 'refit_feedback': 'decoded'}
METHOD = ('SDK standalone_refit_kf Stage-1 supervised fit (state [position, velocity, 1], 20 ms state update, '
          'velocity label = trial-local backward position difference / 0.020 s, SDK normalisation estimated on '
          'training rows, positions/velocities divided by training RMS speed); held-out replay through the SDK '
          'Runtime with state reset at each sequence start and position feedback = integrated decoded velocity '
          '(refit_feedback=decoded) or recorded left-edge cursor (recorded). ReFIT adapt() not used.')


def _sdk():
    from .vendor.sdk_b70297c import bci_decoder_api as sdk
    from .vendor.sdk_b70297c.decoders import standalone_refit_kf as kf
    return sdk, kf


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def _captured(fn):
    handler = _Capture()
    loggers = [logging.getLogger(n) for n in (__package__ + '.vendor.sdk_b70297c',)]
    saved = [(lg, lg.propagate) for lg in loggers]
    for lg in loggers:
        lg.addHandler(handler)
        lg.propagate = False
    try:
        return fn(), handler.records
    finally:
        for lg, prop in saved:
            lg.removeHandler(handler)
            lg.propagate = prop


def _specs(sdk, train, keep):
    ids = np.asarray(train.electrode_ids)[keep]
    window = train.neural_window or train.bin_seconds
    feature = sdk.FeatureSpec(
        schema=sdk.SchemaDefinition(
            'data_analysis_gui.feature.spike_rate_20ms.v1', 'feature', 'float32', ('channel',),
            'Per-electrode quantized event rate (mean 20 ms count / 0.020 s) over a trailing neural window ending '
            'at each 20 ms bin right edge (receive time)', 'sqrt_hz' if train.feature_unit.startswith('sqrt') else 'hz',
            {'bin_period_ns': str(int(round(train.bin_seconds * 1e9))),
             'neural_window_ns': str(int(round(window * 1e9)))}),
        layout=sdk.TensorLayout(electrode_layout_id('data_analysis_gui.electrodes', ids), (int(len(ids)),),
                                {'channel': tuple(f'e{int(i)}' for i in ids)}))

    def vector(role, quantity, unit):
        spec = sdk.ActionSpec if role == 'action' else sdk.FeedbackSpec
        return spec(schema=sdk.SchemaDefinition(f'data_analysis_gui.scaled_{quantity}_2d.v1', role, 'float32', ('component',),
                                               f'Cursor {quantity} divided by the training RMS speed', unit),
                    layout=sdk.TensorLayout(f'data_analysis_gui.{quantity}.xy.v1', (2,), {'component': ('x', 'y')}))
    return feature, vector('action', 'velocity', 'scaled_task_unit_per_s'), vector('feedback', 'position',
                                                                                  'scaled_task_unit_s')


def fit(train, params):
    sdk, kf = _sdk()
    reg = float(params.get('refit_reg', DEFAULTS['refit_reg']))
    if abs(train.bin_seconds - BIN_SECONDS) > 1e-9:
        raise ValueError(f'{NAME} requires 20 ms bins, got {train.bin_seconds * 1000:g} ms')
    x = train.features
    keep = x.std(0) > 1e-12
    if keep.sum() < 2:
        raise ValueError(f'{NAME}: fewer than two non-constant channels in training')
    scale = float(np.sqrt(np.mean(np.sum(train.velocity ** 2, 1))))
    if scale <= 1e-12:
        raise ValueError(f'{NAME}: training movement has zero RMS speed')
    feature, action, feedback = _specs(sdk, train, keep)
    trainer = kf.RefitKFTrainer(feature_spec=feature, action_spec=action, feedback_spec=feedback, max_speed=1e9)
    raw = x[:, keep].astype(np.float32)
    ts = train.timestamps_ns()
    period = int(round(BIN_SECONDS * 1e9))
    f_parts, p_parts, t_parts, id_parts = [], [], [], []
    end = train.end_position / scale
    used = 0
    for a, z in train.sequence_bounds():
        if z - a < 2:
            continue
        used += 1
        f_parts.append(np.vstack([raw[a], raw[a:z]]))
        p_parts.append(np.vstack([train.position[a] / scale, end[a:z]]))
        t_parts.append(np.r_[ts[a] - period, ts[a:z]])
        id_parts.append(np.full(z - a + 1, int(train.sequence[a]), np.int64))
    if used < 2:
        raise ValueError(f'{NAME}: need at least two training sequences of ≥2 bins')
    position = np.concatenate(p_parts)
    source = ('data_analysis_gui.velocity_bins', 'train-rows-only')
    tracking = kf.RefitKFTrackingBlock(
        features=np.concatenate(f_parts), timestamps_ns=np.concatenate(t_parts).astype(np.int64),
        trial_ids=np.concatenate(id_parts), target_position=position.copy(), cursor_position=position,
        feature_schema_id=feature.schema_id, feature_layout_id=feature.layout_id, source_ids=source)
    calibration = kf.CalibrationBlock(raw, ts, feature.schema_id, feature.layout_id, source)
    config = kf.RefitKFFitConfig(transition_ridge_regularization=reg, observation_ridge_regularization=reg)
    result, logs = _captured(lambda: trainer.fit(kf.RefitKFFitData(calibration, tracking), config))
    metrics = dict(result.report.metrics)
    info = dict(refit_reg=reg, kinematic_scale=scale, n_features=int(keep.sum()), n_sequences=used,
                spectral_radius=metrics.get('train.velocity_spectral_radius'),
                model_version=result.artifact.model_version, sdk_log_messages=len(logs))
    return dict(artifact=result.artifact, keep=keep, scale=scale, metrics=metrics, info=info,
                feedback=params.get('refit_feedback', DEFAULTS['refit_feedback']))


def predict(model, test):
    sdk, _ = _sdk()
    artifact, keep, scale = model['artifact'], model['keep'], model['scale']
    mode = model['feedback']
    if mode not in ('decoded', 'recorded'):
        raise ValueError("refit_feedback must be 'decoded' or 'recorded'")
    runtime = sdk.create_runtime(artifact)
    fspec, bspec = artifact.feature_spec, artifact.feedback_spec
    mask = np.ones(fspec.shape, dtype=bool)
    x = test.features[:, keep].astype(np.float32)
    ts = test.timestamps_ns()
    period = int(round(BIN_SECONDS * 1e9))
    dt = BIN_SECONDS
    out = np.full((test.n, 2), np.nan)
    recorded = test.position / scale

    def run():
        for a, z in test.sequence_bounds():
            pos = recorded[a].copy()
            t_reset = int(ts[a]) - period
            runtime.reset_state(sdk.RuntimeResetRequest(timestamp_ns=t_reset, feedback=sdk.FeedbackState.from_spec(
                bspec, values=pos.astype(np.float32), timestamp_ns=t_reset)))
            for i in range(a, z):
                fb = recorded[i] if mode == 'recorded' else pos
                t = int(ts[i])
                result = runtime.infer(sdk.DecoderRequest(
                    features=sdk.FeatureTensor.from_spec(fspec, values=x[i], valid_mask=mask, timestamp_ns=t,
                                                         producer_frame_id=i),
                    feedback=sdk.FeedbackState.from_spec(bspec, values=np.asarray(fb, np.float32), timestamp_ns=t)))
                if result.status is not sdk.InferenceStatus.OK or result.output is None:
                    raise RuntimeError(f'{NAME} frame {i}: {result.status.value}: {result.diagnostics.message}')
                v = result.output.action.astype(float)
                out[i] = v * scale
                pos = pos + dt * v
    with single_thread():
        _captured(run)
    return out
