<!--
Vendored from github.com/sillybun/bci-decoder-sdk
commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
original path: src/decoders/madduri_coadaptive_decoder/README.md
original git blob: 6e8c328492e3c6f6993b3b524e31925bbb6f9f7c
Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
--- end vendor header ---
-->
# Madduri SmoothBatch decoder

The decoder update from Madduri et al., *Computational framework to predict and
shape human-machine interactions in closed-loop, co-adaptive neural interfaces*
(2026), integrated with the SDK Trainer / Artifact / Core / Runtime lifecycle.

```text
fit:    random D, b = 0; no training samples consumed
infer:  velocity = D @ flatten_C(features) + b
adapt:  desired = (target_position - cursor_position) / timestep
        D* = argmin ||X D.T - desired||^2 + penalty ||D||^2
        D_new = alpha D_parent + (1-alpha) D*
```

`learning_rate` names alpha, the retained **parent** weight. The default
`fit_intercept=False` leaves b exactly zero. `fit_intercept=True` fits an
unpenalized intercept and blends it with the parent. Updates take place between
Artifacts; inference does not learn or consume feedback.

## Caller-defined contracts

No catalog or preset dependency exists in the model or its examples. Supply
`FeatureSpec` and `ActionSpec` directly. Schema IDs, axis names, units, and
attributes do not select preprocessing or numerical domain restrictions.

- Feature shape: any non-scalar shape, e.g. `(32,)`, `(channel, band)`, `(2,3,2)`.
- Block features: `(sample, *feature_spec.shape)`, finite float32, signed or unsigned values.
- Flattening: C order in training and inference, recorded in runtime config.
- Action: a vector of any positive dimension N; D is `(N, prod(feature_shape))`, b is `(N,)`.
- Target/cursor: finite real `(sample, N)` arrays in the same coordinates/units.
- Timestamps: nonnegative int64-compatible integers, strictly increasing within
  each segment. Each integer trial ID occupies one contiguous run; timestamps
  may restart at a new segment. Continuous tracking uses a single segment.
- Timestep: explicit `state_update_period_ns`, positive integer, default 20 ms.
  It never comes from schema attributes. Different observed intervals produce
  training/evaluation warnings and SDK runtime telemetry; the model continues
  using its configured fixed timestep.
- `max_speed`: optional vector-norm cap in action units/second, applied to both
  desired fitting velocities and inferred outputs. Evaluation uses Artifact settings.

Mask/finiteness and schema/layout/order checks go through the SDK Runtime.
Float32 action overflow returns `NUMERICAL_ERROR` without advancing Runtime IDs.
The Core is stateless and does not implement checkpoints.

## Minimal lifecycle

```python
from bci_decoder_api import ActionSpec, FeatureSpec, SchemaDefinition, TensorLayout
from decoders.madduri_coadaptive_decoder import (
    MadduriSmoothBatchTrainer, MadduriFitConfig, MadduriAdaptConfig, make_seed_block,
)

feature = FeatureSpec(
    SchemaDefinition("user.features.v1", "feature", "float32", ("channel", "band")),
    TensorLayout("user.features.8x4", (8, 4)),
)
action = ActionSpec(
    SchemaDefinition("user.velocity.v1", "action", "float32", ("component",)),
    TensorLayout("user.velocity.3d", (3,)),
)
trainer = MadduriSmoothBatchTrainer(
    feature_spec=feature, action_spec=action, state_update_period_ns=20_000_000,
)
seed = trainer.fit(make_seed_block(feature_spec=feature), MadduriFitConfig())
# block = MadduriInteractionBlock(...) containing the logged interaction window
# child = trainer.adapt(block, seed.artifact, MadduriAdaptConfig())
# report = trainer.evaluate(child.artifact, held_out_block)
```

`validation.py` owns input, operation and Artifact checks; `fitting.py` owns
numerical operations; `config.py` owns typed settings. Validation summaries are
operation-local and never cached against mutable arrays. Model versions are
deterministic hashes of parameters, full Specs, runtime/training settings,
provenance counts/source IDs and parent version. Identical fits are reproducible.

## Examples and compatibility

Run from the repository root after installing the project:

```shell
python -m examples.standalone_madduri.example
python -m examples.standalone_madduri.example_two_process
```

The second demo models two process roles with classes and disk Artifact handoffs;
it does not spawn OS processes. Both retain the original 2-D, 50 Hz EMG-like
simulation as an example-specific task, and perform two adaptation steps.
The single-file demo also calls canonical Runtime-based evaluation.

Examples live only under `examples/standalone_madduri`; the decoder package has
no example entry points. `channel_count` seed arguments, config imports from
`data_types`, and the old EMG synthetic fixture remain available for compatibility;
new integrations use `make_seed_block(feature_spec=...)` and keep simulation/data
generation in their application or examples. The decoder package does not import
examples during training, persistence or inference.

Implementation **2.0.0** introduces explicit timestep and flattening metadata.
Version 1.x Artifacts are rejected consistently by Trainer, Runtime and persistence;
recreate them with the original seed/config and explicit timestep. There is no
automatic interpretation of their old schema attributes. Public SDK API and
Artifact format remain 2.0. No new dependencies are required.

Tests: `tests/test_madduri_coadaptive_decoder.py` and
`tests/test_standalone_madduri_example.py`.
