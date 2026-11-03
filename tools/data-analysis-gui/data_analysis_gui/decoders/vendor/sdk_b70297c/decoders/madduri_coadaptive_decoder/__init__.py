# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/decoders/madduri_coadaptive_decoder/__init__.py
# original git blob: 6cce6d65298147a195934fa2649fec39dc1eedf2
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Madduri SmoothBatch decoder over caller-defined feature and action contracts.

Online inference is the fixed affine map ``v_hat = D u + b``; the seed decoder is
random (``fit()``) and the ~20 s SmoothBatch updates are offline ``adapt()`` steps
that each produce a child Artifact. Importing this package registers
``MadduriSmoothBatchCore`` under its exact ``model_type`` registry key.
"""

from .config import STATE_UPDATE_PERIOD_NS
from .data_types import (
    EMG_FEATURE_SCHEMA_ID,
    MadduriAdaptConfig,
    MadduriFitConfig,
    MadduriInteractionBlock,
    make_seed_block,
    make_synthetic_block,
)
from .fitting import (
    IMPLEMENTATION_VERSION,
    INIT_SIGNS,
    MODEL_TYPE,
    fit_optimal_decoder,
    limit_speed,
    random_seed_decoder,
    smoothbatch,
)
from .validation import validate_madduri_artifact
from .runtime import MadduriSmoothBatchCore
from .trainer import MadduriSmoothBatchTrainer

__all__ = [
    "EMG_FEATURE_SCHEMA_ID",
    "IMPLEMENTATION_VERSION",
    "INIT_SIGNS",
    "MODEL_TYPE",
    "STATE_UPDATE_PERIOD_NS",
    "MadduriAdaptConfig",
    "MadduriFitConfig",
    "MadduriInteractionBlock",
    "MadduriSmoothBatchCore",
    "MadduriSmoothBatchTrainer",
    "fit_optimal_decoder",
    "limit_speed",
    "make_seed_block",
    "make_synthetic_block",
    "random_seed_decoder",
    "smoothbatch",
    "validate_madduri_artifact",
]
