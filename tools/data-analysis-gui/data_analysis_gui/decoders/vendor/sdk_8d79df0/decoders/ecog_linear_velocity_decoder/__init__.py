# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: 8d79df0475b85ca859a02a8268f5262358a22afa
# original path: src/decoders/ecog_linear_velocity_decoder/__init__.py
# original git blob: 4801f18b7d0163efb76fb43d23c9e6d7475b1e87
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Rest-calibrated adaptive linear decoder for ECoG cursor velocity."""

from .data_types import (
    CURSOR_VELOCITY_COMPONENTS,
    CURSOR_VELOCITY_SCHEMA_ID,
    ECOG_BAND_LABELS,
    ECOG_FEATURE_SCHEMA_ID,
    EcogAdaptConfig,
    EcogCalibrationBlock,
    EcogFitConfig,
    EcogInteractionBlock,
    make_synthetic_calibration,
    make_synthetic_interaction,
)
from .fitting import (
    IMPLEMENTATION_VERSION,
    MODEL_TYPE,
    desired_tracking_velocity,
    fit_trial_balanced_ridge,
    limit_speed,
    smoothbatch,
    trial_equal_sample_weights,
    trial_local_target_velocity,
    validate_ecog_linear_artifact,
)
from .runtime import EcogLinearVelocityCore
from .trainer import EcogLinearVelocityTrainer

__all__ = [
    "CURSOR_VELOCITY_COMPONENTS",
    "CURSOR_VELOCITY_SCHEMA_ID",
    "ECOG_BAND_LABELS",
    "ECOG_FEATURE_SCHEMA_ID",
    "IMPLEMENTATION_VERSION",
    "MODEL_TYPE",
    "EcogAdaptConfig",
    "EcogCalibrationBlock",
    "EcogFitConfig",
    "EcogInteractionBlock",
    "EcogLinearVelocityCore",
    "EcogLinearVelocityTrainer",
    "desired_tracking_velocity",
    "fit_trial_balanced_ridge",
    "limit_speed",
    "make_synthetic_calibration",
    "make_synthetic_interaction",
    "smoothbatch",
    "trial_equal_sample_weights",
    "trial_local_target_velocity",
    "validate_ecog_linear_artifact",
]
