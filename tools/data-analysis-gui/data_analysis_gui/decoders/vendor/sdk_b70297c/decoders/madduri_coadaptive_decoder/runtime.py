# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/decoders/madduri_coadaptive_decoder/runtime.py
# original git blob: e6664b5a0b0cfe063c1e7ea37d7f5542032bdc47
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Co-adaptive decoder online Core: fixed affine map ``v_hat = D u + b``.

Within one Artifact the decoder is fixed, so inference is stateless and uses no
feedback. Co-adaptation happens between Artifacts via the Trainer's ``adapt()``
path, not inside this Core. Preprocessing (band-pass, rectification, decimation)
is upstream and never performed here.
"""

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
)
from .validation import validate_madduri_artifact


@register_decoder_core
class MadduriSmoothBatchCore(DecoderCore):
    model_type = MODEL_TYPE
    implementation_version = IMPLEMENTATION_VERSION

    def __init__(self, artifact: ModelArtifact) -> None:
        self._weights = np.array(artifact.parameters["D"], dtype=np.float64, copy=True)
        self._intercept = np.array(artifact.parameters["b"], dtype=np.float64, copy=True)
        raw_max_speed = artifact.runtime_config.get("max_speed")
        self._max_speed = None if raw_max_speed is None else float(raw_max_speed)

    @classmethod
    def validate_artifact(cls, artifact: ModelArtifact) -> None:
        validate_madduri_artifact(artifact)

    @classmethod
    def from_artifact(cls, artifact: ModelArtifact) -> "MadduriSmoothBatchCore":
        return cls(artifact)

    @staticmethod
    def _failure(
            status: InferenceStatus,
            *,
            reason_code: str,
            message: str,
            invalid_feature_count: int = 0,
    ) -> CoreInferenceResult:
        return CoreInferenceResult(
            status=status,
            action=None,
            diagnostics=CoreInferenceDiagnostics(
                invalid_feature_count=invalid_feature_count,
                reason_code=reason_code,
                message=message,
            ),
        )

    def infer_core(self, request: CoreDecoderRequest) -> CoreInferenceResult:
        frame = request.features
        invalid = (~frame.valid_mask) | (~np.isfinite(frame.values))
        invalid_count = int(np.count_nonzero(invalid))
        if invalid_count:
            return self._failure(
                InferenceStatus.INVALID_FEATURES,
                reason_code="madduri_smoothbatch.invalid_feature_values",
                message="features contain masked or non-finite values",
                invalid_feature_count=invalid_count,
            )

        observation = frame.values.reshape(-1, order="C").astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            velocity = self._weights @ observation + self._intercept
        if not np.all(np.isfinite(velocity)):
            return self._failure(
                InferenceStatus.NUMERICAL_ERROR,
                reason_code="madduri_smoothbatch.non_finite_action",
                message="affine decoder produced a non-finite velocity",
            )
        if self._max_speed is not None:
            velocity = limit_speed(velocity, self._max_speed)
        with np.errstate(over="ignore", invalid="ignore"):
            action = np.array(velocity, dtype=np.float32, copy=True)
        if not np.all(np.isfinite(action)):
            return self._failure(
                InferenceStatus.NUMERICAL_ERROR,
                reason_code="madduri_smoothbatch.action_float32_overflow",
                message="velocity cannot be represented as a finite float32 action",
            )
        return CoreInferenceResult(
            status=InferenceStatus.OK,
            action=action,
            diagnostics=CoreInferenceDiagnostics(
                metrics={"action_speed": float(np.linalg.norm(velocity))},
            ),
        )

    def reset_core(self, request: RuntimeResetRequest | None = None) -> None:
        """No-op: within an Artifact the decoder holds no per-frame state."""
