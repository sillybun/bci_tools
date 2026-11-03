# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: 8d79df0475b85ca859a02a8268f5262358a22afa
# original path: src/bci_decoder_api/__init__.py
# original git blob: 5e2682aa92e525e156832e221aa94072ff53a066
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Public API for the transport-neutral BCI Decoder SDK."""

from . import schema_catalog
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
    "schema_catalog",
]
