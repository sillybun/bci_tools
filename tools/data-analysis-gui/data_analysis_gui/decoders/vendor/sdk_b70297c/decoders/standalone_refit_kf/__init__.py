# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/decoders/standalone_refit_kf/__init__.py
# original git blob: 76f2b18588e16c9f123e37d6ac6dff4cc0a9b3b0
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Signal-independent ReFIT Kalman filter with explicit input validation."""

from typing import TYPE_CHECKING

from .data_types import STATE_UPDATE_PERIOD_NS, CalibrationBlock, RefitKFFitData, RefitKFTrackingBlock
from .config import RefitKFAdaptConfig, RefitKFFitConfig
from .fitting import (
    IMPLEMENTATION_VERSION,
    MODEL_TYPE,
    blend_kalman_parameters,
    fit_kalman_parameters,
    limit_speed,
    moving_target_intended_velocity,
    trial_local_velocity,
    valid_transition_pairs,
)
from .runtime import RefitKFCore, infer

if TYPE_CHECKING:
    from .validation import (
        RefitKFInputValidator, validate_decoder_specs,
        validate_refit_kf_artifact, validate_refit_kf_parameters,
    )
    from .trainer import RefitKFTrainer

__all__ = [
    "IMPLEMENTATION_VERSION", "MODEL_TYPE", "STATE_UPDATE_PERIOD_NS",
    "CalibrationBlock", "RefitKFAdaptConfig", "RefitKFCore", "RefitKFFitConfig",
    "RefitKFFitData", "RefitKFInputValidator", "RefitKFTrackingBlock", "RefitKFTrainer",
    "infer",
    "blend_kalman_parameters", "fit_kalman_parameters", "limit_speed",
    "moving_target_intended_velocity", "trial_local_velocity", "valid_transition_pairs",
    "validate_decoder_specs", "validate_refit_kf_artifact", "validate_refit_kf_parameters",
]


def __getattr__(name: str):
    """Load Trainer and validators on demand while registering the Core eagerly.

    Inference-only imports do not need the offline Trainer. All model checks
    are defined by the local validation module.
    """
    from importlib import import_module

    modules = {
        "RefitKFTrainer": ".trainer",
        "RefitKFInputValidator": ".validation",
        "validate_decoder_specs": ".validation",
        "validate_refit_kf_artifact": ".validation",
        "validate_refit_kf_parameters": ".validation",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
