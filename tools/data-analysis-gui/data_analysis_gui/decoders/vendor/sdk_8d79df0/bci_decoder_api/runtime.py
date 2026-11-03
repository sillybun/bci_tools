# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: 8d79df0475b85ca859a02a8268f5262358a22afa
# original path: src/bci_decoder_api/runtime.py
# original git blob: 1690c03777fd2ea367226914e1b1bfdc9c63375b
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""SDK-owned Runtime wrapper around model-specific DecoderCore implementations."""

from __future__ import annotations

import time
from threading import RLock

import numpy as np

from .artifact import ModelArtifact
from .base import CheckpointableDecoderCore, DecoderCore
from .contracts import (
    CoreDecoderRequest,
    CoreRuntimeState,
    DecoderOutput,
    DecoderRequest,
    FeedbackState,
    InferenceDiagnostics,
    InferenceResult,
    InferenceStatus,
    RuntimeResetRequest,
    RuntimeState,
)
from .errors import RuntimeStateError, SpecValidationError


class DecoderRuntime:
    """Public stateful Runtime that owns integration protocol and update identity."""

    def __init__(self, artifact: ModelArtifact, core: DecoderCore) -> None:
        self._artifact = artifact
        self._core = core
        self._lock = RLock()
        self._last_timestamp_ns: int | None = None
        self._last_update_id: int | None = None
        self._last_producer_frame_id: int | None = None
        self._next_update_id = 0

    @property
    def artifact(self) -> ModelArtifact:
        return self._artifact

    @property
    def model_type(self) -> str:
        return self._artifact.model_type

    @property
    def model_version(self) -> str:
        return self._artifact.model_version

    @property
    def implementation_version(self) -> str:
        return self._artifact.implementation_version

    @property
    def supports_checkpoint(self) -> bool:
        return isinstance(self._core, CheckpointableDecoderCore)

    def _failure(
        self,
        status: InferenceStatus,
        started_ns: int,
        reason_code: str,
        message: str,
    ) -> InferenceResult:
        return InferenceResult(
            status=status,
            output=None,
            diagnostics=InferenceDiagnostics(
                inference_time_ns=time.perf_counter_ns() - started_ns,
                reason_code=reason_code,
                message=message,
            ),
        )

    def _validate_feedback_contract(self, feedback: FeedbackState) -> None:
        spec = self._artifact.feedback_spec
        if spec is None:
            raise SpecValidationError("the artifact does not accept feedback")
        if feedback.schema_id != spec.schema_id or feedback.layout_id != spec.layout_id:
            raise SpecValidationError("feedback schema_id/layout_id does not match the artifact")
        if feedback.values.dtype != np.dtype(spec.dtype) or feedback.values.shape != spec.shape:
            raise SpecValidationError("feedback dtype/shape does not match the artifact")
        if not np.all(np.isfinite(feedback.values)):
            raise SpecValidationError("feedback values must be finite")

    def infer(self, request: DecoderRequest) -> InferenceResult:
        """Validate one producer frame, assign an update ID, and invoke the Core."""

        started_ns = time.perf_counter_ns()
        with self._lock:
            frame = request.features
            feature_spec = self._artifact.feature_spec
            if (
                frame.schema_id != feature_spec.schema_id
                or frame.layout_id != feature_spec.layout_id
                or frame.values.dtype != np.dtype(feature_spec.dtype)
                or frame.values.shape != feature_spec.shape
            ):
                return self._failure(
                    InferenceStatus.SCHEMA_MISMATCH,
                    started_ns,
                    "sdk.feature_contract_mismatch",
                    "feature schema/layout/dtype/shape does not match the artifact",
                )
            if (
                self._last_timestamp_ns is not None
                and frame.timestamp_ns <= self._last_timestamp_ns
            ):
                return self._failure(
                    InferenceStatus.OUT_OF_ORDER,
                    started_ns,
                    "sdk.feature_timestamp_out_of_order",
                    "feature timestamp must be strictly increasing",
                )
            producer_frame_gap = 0
            if frame.producer_frame_id is not None and self._last_producer_frame_id is not None:
                if frame.producer_frame_id <= self._last_producer_frame_id:
                    return self._failure(
                        InferenceStatus.OUT_OF_ORDER,
                        started_ns,
                        "sdk.producer_frame_id_out_of_order",
                        "producer_frame_id is duplicated or out of order",
                    )
                producer_frame_gap = (
                    frame.producer_frame_id - self._last_producer_frame_id - 1
                )

            feedback_spec = self._artifact.feedback_spec
            feedback = request.feedback
            if feedback_spec is None:
                if feedback is not None:
                    return self._failure(
                        InferenceStatus.SCHEMA_MISMATCH,
                        started_ns,
                        "sdk.unexpected_feedback",
                        "the artifact does not accept feedback",
                    )
            else:
                if feedback is None:
                    return self._failure(
                        InferenceStatus.MISSING_FEEDBACK,
                        started_ns,
                        "sdk.missing_feedback",
                        "the artifact requires feedback",
                    )
                try:
                    self._validate_feedback_contract(feedback)
                except SpecValidationError as exc:
                    return self._failure(
                        InferenceStatus.SCHEMA_MISMATCH,
                        started_ns,
                        "sdk.feedback_contract_mismatch",
                        str(exc),
                    )
                maximum_age_ns = int(
                    self._artifact.runtime_config.get("maximum_feedback_age_ns", 0)
                )
                feedback_age_ns = frame.timestamp_ns - feedback.timestamp_ns
                if feedback_age_ns < 0 or feedback_age_ns > maximum_age_ns:
                    return self._failure(
                        InferenceStatus.STALE_FEEDBACK,
                        started_ns,
                        "sdk.stale_feedback",
                        "feedback is from the future or older than the configured maximum age",
                    )

            update_id = self._next_update_id
            core_started_ns = time.perf_counter_ns()
            core_result = self._core.infer_core(
                CoreDecoderRequest(
                    features=frame,
                    feedback=feedback,
                    update_id=update_id,
                )
            )
            core_inference_time_ns = time.perf_counter_ns() - core_started_ns
            metrics = dict(core_result.diagnostics.metrics)
            if producer_frame_gap:
                metrics["producer_frame_gap"] = float(producer_frame_gap)
            diagnostics = InferenceDiagnostics(
                inference_time_ns=time.perf_counter_ns() - started_ns,
                core_inference_time_ns=core_inference_time_ns,
                invalid_feature_count=core_result.diagnostics.invalid_feature_count,
                reason_code=core_result.diagnostics.reason_code,
                message=core_result.diagnostics.message,
                metrics=metrics,
            )
            if core_result.status is not InferenceStatus.OK:
                return InferenceResult(
                    status=core_result.status,
                    output=None,
                    diagnostics=diagnostics,
                )

            action = core_result.action
            if action is None:
                raise RuntimeError("DecoderCore returned OK without an action")
            action_spec = self._artifact.action_spec
            if action.dtype != np.dtype(action_spec.dtype) or action.shape != action_spec.shape:
                raise RuntimeError("DecoderCore returned an action incompatible with the Artifact")
            if not np.all(np.isfinite(action)):
                raise RuntimeError("DecoderCore returned a non-finite action")
            output = DecoderOutput(
                action=np.array(action, copy=True),
                action_schema_id=action_spec.schema_id,
                action_layout_id=action_spec.layout_id,
                input_timestamp_ns=frame.timestamp_ns,
                update_id=update_id,
                producer_frame_id=frame.producer_frame_id,
                model_version=self._artifact.model_version,
            )
            self._last_timestamp_ns = frame.timestamp_ns
            self._last_update_id = update_id
            if frame.producer_frame_id is not None:
                self._last_producer_frame_id = frame.producer_frame_id
            self._next_update_id += 1
            return InferenceResult(
                status=InferenceStatus.OK,
                output=output,
                diagnostics=diagnostics,
            )

    def reset_state(self, request: RuntimeResetRequest | None = None) -> None:
        with self._lock:
            if request is not None and request.feedback is not None:
                self._validate_feedback_contract(request.feedback)
                if (
                    request.timestamp_ns is not None
                    and request.feedback.timestamp_ns > request.timestamp_ns
                ):
                    raise SpecValidationError(
                        "reset feedback cannot be newer than the reset timestamp"
                    )
            self._core.reset_core(request)
            self._last_timestamp_ns = None
            self._last_update_id = None
            self._last_producer_frame_id = None
            self._next_update_id = 0

    def snapshot_state(self) -> RuntimeState:
        with self._lock:
            if not isinstance(self._core, CheckpointableDecoderCore):
                raise RuntimeStateError("this DecoderCore does not support checkpointing")
            core_state = self._core.snapshot_core_state()
            return RuntimeState(
                model_type=self.model_type,
                model_version=self.model_version,
                state_schema_id=core_state.state_schema_id,
                last_timestamp_ns=self._last_timestamp_ns,
                last_update_id=self._last_update_id,
                last_producer_frame_id=self._last_producer_frame_id,
                arrays=core_state.arrays,
                counters=core_state.counters,
            )

    def restore_state(self, state: RuntimeState) -> None:
        with self._lock:
            if not isinstance(self._core, CheckpointableDecoderCore):
                raise RuntimeStateError("this DecoderCore does not support checkpointing")
            if state.model_type != self.model_type or state.model_version != self.model_version:
                raise RuntimeStateError("checkpoint model identity does not match the runtime")
            self._core.restore_core_state(
                CoreRuntimeState(
                    state_schema_id=state.state_schema_id,
                    arrays=state.arrays,
                    counters=state.counters,
                )
            )
            self._last_timestamp_ns = state.last_timestamp_ns
            self._last_update_id = state.last_update_id
            self._last_producer_frame_id = state.last_producer_frame_id
            self._next_update_id = 0 if state.last_update_id is None else state.last_update_id + 1
