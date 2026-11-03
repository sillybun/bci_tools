# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: 8d79df0475b85ca859a02a8268f5262358a22afa
# original path: src/decoders/ecog_linear_velocity_decoder/example.py
# original git blob: 11f201e933fec9c6e3143610f97514e04fac1998
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Synthetic lifecycle smoke test; this is contract evidence, not BCI performance."""

from __future__ import annotations

import numpy as np

from ...bci_decoder_api import ActionSpec, FeatureSpec, TensorLayout, schema_catalog

from .data_types import (
    CURSOR_VELOCITY_SCHEMA_ID,
    ECOG_FEATURE_SCHEMA_ID,
    ECOG_BAND_LABELS,
    EcogAdaptConfig,
    EcogFitConfig,
    make_synthetic_calibration,
    make_synthetic_interaction,
)
from .trainer import EcogLinearVelocityTrainer


def main() -> None:
    feature_spec = FeatureSpec(
        schema=schema_catalog.load(ECOG_FEATURE_SCHEMA_ID),
        layout=TensorLayout(
            layout_id="bci.layout.tiantan_tsmi_ecog16_band6.v1",
            shape=(16, 6),
            axis_labels={"band": ECOG_BAND_LABELS},
        ),
    )
    action_spec = ActionSpec(
        schema=schema_catalog.load(CURSOR_VELOCITY_SCHEMA_ID),
        layout=TensorLayout(
            layout_id="bci.layout.cursor_xy_velocity.v1",
            shape=(2,),
            axis_labels={"component": ("x", "y")},
        ),
    )
    trainer = EcogLinearVelocityTrainer(
        feature_spec=feature_spec,
        action_spec=action_spec,
        max_speed=1_000.0,
    )
    rest = make_synthetic_calibration(feature_layout_id=feature_spec.layout_id)
    task = make_synthetic_interaction(rest)
    seed = trainer.fit(rest, EcogFitConfig())
    first = trainer.adapt(task, seed.artifact, EcogAdaptConfig())
    second = trainer.adapt(task, first.artifact, EcogAdaptConfig())
    report = trainer.evaluate(second.artifact, task)
    print(
        {
            "seed": seed.artifact.model_version,
            "first": first.artifact.model_version,
            "second": second.artifact.model_version,
            "normalization_frozen": bool(
                np.array_equal(
                    seed.artifact.parameters["feature_mean"],
                    second.artifact.parameters["feature_mean"],
                )
            ),
            "metrics": dict(report.metrics),
        }
    )


if __name__ == "__main__":
    main()
