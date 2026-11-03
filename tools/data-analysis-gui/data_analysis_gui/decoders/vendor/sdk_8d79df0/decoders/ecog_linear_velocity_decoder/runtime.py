# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: 8d79df0475b85ca859a02a8268f5262358a22afa
# original path: src/decoders/ecog_linear_velocity_decoder/runtime.py
# original git blob: 5d5d5120ee4a883709015e1c9c184dc5ae51f8a4
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Stateless normalized affine ECoG velocity Core."""

from __future__ import annotations

import numpy as np

from ...bci_decoder_api import (
    CoreDecoderRequest,
    CoreInferenceDiagnostics,
    CoreInferenceResult,
    DecoderCore,
    InferenceStatus,
    ModelArtifact,
    RuntimeResetRequest,
    register_decoder_core,
)

from .fitting import (
    IMPLEMENTATION_VERSION,
    MODEL_TYPE,
    limit_speed,
    validate_ecog_linear_artifact,
)


@register_decoder_core
class EcogLinearVelocityCore(DecoderCore):
    model_type = MODEL_TYPE
    implementation_version = IMPLEMENTATION_VERSION

    def __init__(self, artifact: ModelArtifact) -> None:
        self._weights = np.array(artifact.parameters["W"], dtype=np.float64, copy=True)
        self._intercept = np.array(artifact.parameters["b"], dtype=np.float64, copy=True)
        self._mean = np.array(artifact.parameters["feature_mean"], dtype=np.float64, copy=True)
        self._scale = np.array(artifact.parameters["feature_scale"], dtype=np.float64, copy=True)
        self._max_speed = float(artifact.runtime_config["max_speed"])

    @classmethod
    def validate_artifact(cls, artifact: ModelArtifact) -> None:
        validate_ecog_linear_artifact(artifact)

    @classmethod
    def from_artifact(cls, artifact: ModelArtifact) -> "EcogLinearVelocityCore":
        return cls(artifact)

    def infer_core(self, request: CoreDecoderRequest) -> CoreInferenceResult:
        frame = request.features
        invalid = (~frame.valid_mask) | (~np.isfinite(frame.values))
        invalid_count = int(np.count_nonzero(invalid))
        if invalid_count:
            return CoreInferenceResult(
                status=InferenceStatus.INVALID_FEATURES,
                action=None,
                diagnostics=CoreInferenceDiagnostics(
                    invalid_feature_count=invalid_count,
                    reason_code="ecog_linear_velocity.invalid_feature_values",
                    message="ECoG features contain invalid or non-finite values",
                ),
            )
        z = ((frame.values.astype(np.float64) - self._mean) / self._scale).reshape(
            -1, order="C"
        )
        velocity = self._weights @ z + self._intercept
        if not np.all(np.isfinite(velocity)):
            return CoreInferenceResult(
                status=InferenceStatus.NUMERICAL_ERROR,
                action=None,
                diagnostics=CoreInferenceDiagnostics(
                    reason_code="ecog_linear_velocity.non_finite_action",
                    message="linear decoder produced a non-finite velocity",
                ),
            )
        velocity = limit_speed(velocity, self._max_speed)
        return CoreInferenceResult(
            status=InferenceStatus.OK,
            action=np.array(velocity, dtype=np.float32, copy=True),
            diagnostics=CoreInferenceDiagnostics(
                metrics={"action_speed": float(np.linalg.norm(velocity))}
            ),
        )

    def reset_core(self, request: RuntimeResetRequest | None = None) -> None:
        """No-op: the Core has no temporal state."""
