# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/decoders/standalone_refit_kf/runtime.py
# original git blob: 808b3ab7988c4c4db6e88dd3f94504bccd61feb4
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Checkpointable causal-position Runtime Core for production ReFIT-KF."""

from __future__ import annotations

import numpy as np

from ...bci_decoder_api import (
    CheckpointableDecoderCore,
    DecoderRequest,
    FeatureTensor,
    FeedbackState,
    create_runtime,
    CoreDecoderRequest,
    CoreInferenceDiagnostics,
    CoreInferenceResult,
    CoreRuntimeState,
    InferenceStatus,
    ModelArtifact,
    RuntimeResetRequest,
    register_decoder_core,
)
from .data_types import RefitKFTrackingBlock
from .fitting import IMPLEMENTATION_VERSION, MODEL_TYPE
from .utils import limit_speed
from .validation import (
    RefitKFRuntimeValidationError,
    validate_refit_kf_artifact,
    validate_refit_kf_checkpoint_contract,
    validate_refit_kf_checkpoint_state,
    validate_refit_kf_inference_inputs,
    validate_refit_kf_posterior,
)


STATE_SCHEMA_ID = "refit_kf_decoder_core_state_v1"


@register_decoder_core
class RefitKFCore(CheckpointableDecoderCore):
    model_type = MODEL_TYPE
    implementation_version = IMPLEMENTATION_VERSION

    def __init__(self, artifact: ModelArtifact) -> None:
        self._dimension = artifact.action_spec.shape[0]
        self._parameters = artifact.parameters
        self._mean = np.array(
            artifact.parameters["feature_mean"], dtype=np.float64, copy=True
        )
        self._scale = np.array(
            artifact.parameters["feature_scale"], dtype=np.float64, copy=True
        )
        self._max_speed = float(artifact.runtime_config["max_speed"])
        self._covariance_tolerance = float(
            artifact.runtime_config["covariance_tolerance"]
        )
        self._x = np.array(artifact.parameters["x0"], dtype=np.float64, copy=True)
        self._P = np.array(artifact.parameters["P0"], dtype=np.float64, copy=True)

    @classmethod
    def validate_artifact(cls, artifact: ModelArtifact) -> None:
        validate_refit_kf_artifact(artifact)

    @classmethod
    def from_artifact(cls, artifact: ModelArtifact) -> "RefitKFCore":
        return cls(artifact)

    def _zero_known_covariance(self, covariance: np.ndarray) -> None:
        covariance[:self._dimension, :] = 0.0
        covariance[:, :self._dimension] = 0.0
        covariance[-1, :] = 0.0
        covariance[:, -1] = 0.0

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
        feedback = request.feedback
        # Presence, shape, dtype, timestamps and finiteness belong to Runtime.
        assert feedback is not None

        try:
            A = self._parameters["A"]
            C = self._parameters["C"]
            W = self._parameters["W"]
            Q = self._parameters["Q"]
            candidate_x = A @ self._x
            candidate_P = A @ self._P @ A.T + W
            candidate_x[:self._dimension] = feedback.values.astype(np.float64)
            candidate_x[-1] = 1.0
            self._zero_known_covariance(candidate_P)

            observation = ((frame.values.astype(np.float64) - self._mean) / self._scale).reshape(
                -1, order="C"
            )
            innovation_covariance = C @ candidate_P @ C.T + Q
            kalman_gain = np.linalg.solve(
                innovation_covariance, C @ candidate_P
            ).T
            posterior_x = candidate_x + kalman_gain @ (observation - C @ candidate_x)
            identity = np.eye(candidate_P.shape[0])
            posterior_P = (
                    (identity - kalman_gain @ C)
                    @ candidate_P
                    @ (identity - kalman_gain @ C).T
                    + kalman_gain @ Q @ kalman_gain.T
            )
            posterior_P = 0.5 * (posterior_P + posterior_P.T)
            posterior_x[:self._dimension] = feedback.values.astype(np.float64)
            posterior_x[-1] = 1.0
            self._zero_known_covariance(posterior_P)
            minimum_eigenvalue = validate_refit_kf_posterior(
                posterior_x, posterior_P, covariance_tolerance=self._covariance_tolerance,
            )
            raw_velocity = posterior_x[self._dimension:2 * self._dimension]
            action = np.asarray(
                limit_speed(raw_velocity, self._max_speed), dtype=np.float32
            )
            result = CoreInferenceResult(
                status=InferenceStatus.OK,
                action=np.array(action, copy=True),
                diagnostics=CoreInferenceDiagnostics(
                    metrics={
                        "posterior_trace": float(np.trace(posterior_P)),
                        "posterior_min_eigenvalue": minimum_eigenvalue,
                        "position_gain_max_abs": float(
                            np.max(np.abs(kalman_gain[:self._dimension]))
                        ),
                        "raw_action_speed": float(np.linalg.norm(raw_velocity)),
                        "action_speed": float(np.linalg.norm(action)),
                    }
                ),
            )
        except RefitKFRuntimeValidationError as exc:
            return self._failure(
                InferenceStatus.NUMERICAL_ERROR,
                reason_code=exc.reason_code,
                message=str(exc),
            )
        except np.linalg.LinAlgError as exc:
            return self._failure(
                InferenceStatus.NUMERICAL_ERROR,
                reason_code="standalone_refit_kf.linear_algebra_failure",
                message=str(exc),
            )
        except ValueError as exc:
            return self._failure(
                InferenceStatus.NUMERICAL_ERROR,
                reason_code="standalone_refit_kf.state_update_value_error",
                message=str(exc),
            )

        self._x = posterior_x
        self._P = posterior_P
        return result

    def reset_core(self, request: RuntimeResetRequest | None = None) -> None:
        candidate_x = np.array(self._parameters["x0"], dtype=np.float64, copy=True)
        candidate_P = np.array(self._parameters["P0"], dtype=np.float64, copy=True)
        if request is not None and request.feedback is not None:
            candidate_x[:self._dimension] = request.feedback.values.astype(np.float64)
        candidate_x[-1] = 1.0
        self._x = candidate_x
        self._P = candidate_P

    def snapshot_core_state(self) -> CoreRuntimeState:
        return CoreRuntimeState(
            state_schema_id=STATE_SCHEMA_ID,
            arrays={"x": self._x, "P": self._P},
        )

    def restore_core_state(self, state: CoreRuntimeState) -> None:
        validate_refit_kf_checkpoint_contract(state, state_schema_id=STATE_SCHEMA_ID)
        candidate_x = np.array(state.arrays["x"], dtype=np.float64, copy=True)
        candidate_P = np.array(state.arrays["P"], dtype=np.float64, copy=True)
        validate_refit_kf_checkpoint_state(
            candidate_x, candidate_P, action_dimension=self._dimension,
            covariance_tolerance=self._covariance_tolerance,
        )
        self._x = candidate_x
        self._P = candidate_P


def infer(artifact: ModelArtifact, data: RefitKFTrackingBlock) -> np.ndarray:
    validate_refit_kf_inference_inputs(artifact, data)
    runtime = create_runtime(artifact)
    assert artifact.feedback_spec is not None
    actions = []
    for index, trial in enumerate(data.trial_ids):
        timestamp = int(data.timestamps_ns[index])
        feedback = FeedbackState.from_spec(
            artifact.feedback_spec, values=np.asarray(data.cursor_position[index], dtype=np.float32),
            timestamp_ns=timestamp,
        )
        if index == 0 or trial != data.trial_ids[index - 1]:
            runtime.reset_state(RuntimeResetRequest(timestamp_ns=timestamp, feedback=feedback))
        result = runtime.infer(DecoderRequest(
            features=FeatureTensor.from_spec(
                artifact.feature_spec, values=data.features[index],
                valid_mask=np.ones(artifact.feature_spec.shape, dtype=bool),
                timestamp_ns=timestamp, producer_frame_id=index,
            ), feedback=feedback,
        ))
        if result.status is not InferenceStatus.OK or result.output is None:
            raise RuntimeError(f"frame {index}: {result.status.value}: {result.diagnostics.message}")
        actions.append(result.output.action)
    return np.stack(actions)
