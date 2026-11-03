# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/bci_decoder_api/__init__.py
# original git blob: 6c54631c7f169324c7fc9eb24c017144a6019622
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Public API for the transport-neutral BCI Decoder SDK."""

from .artifact import ModelArtifact, json_sha256, load_artifact, save_artifact
from .base import (
    AdaptiveDecoderTrainer,
    CheckpointableDecoderCore,
    DecoderCore,
    DecoderTrainer,
)
from .contracts import (
    CoreDecoderRequest,
    CoreInferenceDiagnostics,
    CoreInferenceResult,
    CoreRuntimeState,
    DecoderOutput,
    DecoderRequest,
    EvaluationReport,
    FeatureTensor,
    FeedbackState,
    InferenceDiagnostics,
    InferenceResult,
    InferenceStatus,
    RuntimeResetRequest,
    RuntimeState,
    TrainingProvenance,
    TrainingReport,
    TrainingResult,
)
from .registry import create_runtime, register_decoder_core, registered_model_types
from .runtime import DecoderRuntime
from .specs import ActionSpec, FeedbackSpec, FeatureSpec, SchemaDefinition, TensorLayout

__all__ = [
    "ActionSpec",
    "AdaptiveDecoderTrainer",
    "CheckpointableDecoderCore",
    "CoreDecoderRequest",
    "CoreInferenceDiagnostics",
    "CoreInferenceResult",
    "CoreRuntimeState",
    "DecoderCore",
    "DecoderOutput",
    "DecoderRequest",
    "DecoderRuntime",
    "DecoderTrainer",
    "EvaluationReport",
    "FeatureSpec",
    "FeatureTensor",
    "FeedbackSpec",
    "FeedbackState",
    "InferenceDiagnostics",
    "InferenceResult",
    "InferenceStatus",
    "ModelArtifact",
    "RuntimeResetRequest",
    "RuntimeState",
    "SchemaDefinition",
    "TensorLayout",
    "TrainingProvenance",
    "TrainingReport",
    "TrainingResult",
    "create_runtime",
    "json_sha256",
    "load_artifact",
    "register_decoder_core",
    "registered_model_types",
    "save_artifact",
]
