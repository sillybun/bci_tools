<!--
Vendored from github.com/sillybun/bci-decoder-sdk
commit: 8d79df0475b85ca859a02a8268f5262358a22afa
original path: src/decoders/ecog_linear_velocity_decoder/README.md
original git blob: 931e26050e0f0f9da1beb4fd637ee9993c6be3be
Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
--- end vendor header ---
-->
# ECoG Linear Velocity Decoder

This package is a baseline two-dimensional cursor-velocity decoder for
`bci.feature.ecog_iir_bandpower_20ms.v1`.
The concrete feature layout may vary in channel count, but must preserve the
schema's six-band order (excluding 0.5–4 Hz). The action layout must label its
components `(x, y)`.

The root Artifact is produced from a rest-calibration block. It stores per
`(channel, band)` mean and standard deviation, uses a positive scale floor, and
starts with the safe decoder `W=0, b=0`. Rest samples are not reused as
zero-velocity supervision.

The SDK algorithm accepts the retained rest block; protocol integration is
responsible for excluding the 2 s wash-in and passing the following 10 s
`rest start`–`rest end` interval. This package does not implement UI/LSL events.

Each supervised update:

1. applies the root Artifact's frozen normalization;
2. estimates target velocity by a causal, trial-local backward difference;
3. forms `target_velocity + (target_position-cursor_position)/0.060 s` and
   vector-norm clips it to `max_speed`;
4. fits an optionally affine Ridge map with equal total weight per trial;
5. uses SmoothBatch against the seed on the first update
   (`first_alpha=0.0` by default, preserving the complete optimum), then uses
   SmoothBatch against the previous adapted model (`alpha=0.75` by default).

Both alpha values weight the parent model. Therefore, the first update is
`(1-first_alpha) * fitted_model + first_alpha * seed`; only `W` and `b` are
blended, while rest-derived normalization stays frozen.

`fit_intercept=True` is the default. Integration CLIs may expose its inverse as
`--no-intercept`; no-intercept Artifacts contain an exactly zero `b`.

Run the synthetic contract smoke test with:

```bash
PYTHONPATH=src:. python -m decoders.ecog_linear_velocity_decoder.example
```

The fixture demonstrates Trainer/Artifact/Runtime lifecycle behavior only. It
does not establish decoding accuracy on a person, real ECoG, or closed-loop BCI.
