# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: 8d79df0475b85ca859a02a8268f5262358a22afa
# original path: src/bci_decoder_api/base.py
# original git blob: adb36938e075e2072ef5b105efd00e07294b0737
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Abstract interfaces implemented by decoder authors."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, Generic, Self, TypeVar, final

from .contracts import (
    CoreDecoderRequest,
    CoreInferenceResult,
    CoreRuntimeState,
    EvaluationReport,
    RuntimeResetRequest,
    TrainingResult,
)

if TYPE_CHECKING:
    from .artifact import ModelArtifact


TrainingDataT = TypeVar("TrainingDataT")
FitConfigT = TypeVar("FitConfigT")
AdaptationDataT = TypeVar("AdaptationDataT")
AdaptConfigT = TypeVar("AdaptConfigT")


class DecoderTrainer(ABC, Generic[TrainingDataT, FitConfigT]):
    """SDK-owned offline lifecycle with model-specific validation and fitting hooks."""

    model_type: ClassVar[str]
    implementation_version: ClassVar[str]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        overridden = {"fit", "evaluate"}.intersection(cls.__dict__)
        if overridden:
            names = ", ".join(sorted(overridden))
            raise TypeError(
                f"{cls.__name__} cannot override SDK-owned lifecycle method(s): {names}; "
                "implement the required validation and validated-operation hooks instead"
            )

    @final
    def fit(self, data: TrainingDataT, config: FitConfigT) -> TrainingResult:
        """Validate inputs, run model fitting, and validate the public result contract."""

        started_ns = time.perf_counter_ns()
        self._validate_fit_inputs(data, config)
        result = self._fit_validated(data, config)
        result = self._with_training_duration(result, time.perf_counter_ns() - started_ns)
        self._validate_training_result(result)
        if result.artifact.provenance.parent_model_version is not None:
            raise ValueError("fit() must produce a root Artifact without a parent model version")
        return result

    @abstractmethod
    def _validate_fit_inputs(self, data: TrainingDataT, config: FitConfigT) -> None:
        """Reject invalid model-specific training data or configuration."""

    @abstractmethod
    def _fit_validated(self, data: TrainingDataT, config: FitConfigT) -> TrainingResult:
        """Fit already-validated inputs and return a candidate public result."""

    @final
    def evaluate(
        self,
        artifact: "ModelArtifact",
        data: TrainingDataT,
    ) -> EvaluationReport:
        """Validate evaluation inputs, run evaluation, and validate its report."""

        started_ns = time.perf_counter_ns()
        self._validate_artifact_identity(artifact)
        self._validate_evaluation_inputs(artifact, data)
        report = self._evaluate_validated(artifact, data)
        if not isinstance(report, EvaluationReport):
            raise TypeError("_evaluate_validated() must return EvaluationReport")
        report = replace(report, duration_ns=time.perf_counter_ns() - started_ns)
        self._validate_evaluation_report(artifact, report)
        return report

    @abstractmethod
    def _validate_evaluation_inputs(
        self,
        artifact: "ModelArtifact",
        data: TrainingDataT,
    ) -> None:
        """Reject an incompatible Artifact or invalid evaluation dataset."""

    @abstractmethod
    def _evaluate_validated(
        self,
        artifact: "ModelArtifact",
        data: TrainingDataT,
    ) -> EvaluationReport:
        """Evaluate inputs that have already passed the model-specific checks."""

    def _validate_artifact_identity(self, artifact: "ModelArtifact") -> None:
        if artifact.model_type != self.model_type:
            raise ValueError(
                f"Artifact model_type {artifact.model_type!r} does not match "
                f"Trainer model_type {self.model_type!r}"
            )
        if artifact.implementation_version != self.implementation_version:
            raise ValueError(
                "Artifact implementation_version does not match this Trainer"
            )

    def _with_training_duration(
        self,
        result: TrainingResult,
        duration_ns: int,
    ) -> TrainingResult:
        if not isinstance(result, TrainingResult):
            raise TypeError("_fit_validated()/_adapt_validated() must return TrainingResult")
        return TrainingResult(
            artifact=result.artifact,
            report=replace(result.report, duration_ns=duration_ns),
        )

    def _validate_training_result(self, result: TrainingResult) -> None:
        artifact = result.artifact
        report = result.report
        self._validate_artifact_identity(artifact)
        if report.model_type != artifact.model_type:
            raise ValueError("TrainingReport model_type does not match its Artifact")
        if report.model_version != artifact.model_version:
            raise ValueError("TrainingReport model_version does not match its Artifact")

    def _validate_evaluation_report(
        self,
        artifact: "ModelArtifact",
        report: EvaluationReport,
    ) -> None:
        if report.model_type != artifact.model_type:
            raise ValueError("EvaluationReport model_type does not match its Artifact")
        if report.model_version != artifact.model_version:
            raise ValueError("EvaluationReport model_version does not match its Artifact")


class AdaptiveDecoderTrainer(
    DecoderTrainer[TrainingDataT, FitConfigT],
    Generic[TrainingDataT, FitConfigT, AdaptationDataT, AdaptConfigT],
):
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "adapt" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} cannot override SDK-owned lifecycle method adapt; "
                "implement _validate_adapt_inputs() and _adapt_validated() instead"
            )

    @final
    def adapt(
        self,
        data: AdaptationDataT,
        previous: "ModelArtifact",
        config: AdaptConfigT,
    ) -> TrainingResult:
        """Validate adaptation inputs and create a validated child Artifact."""

        started_ns = time.perf_counter_ns()
        self._validate_artifact_identity(previous)
        self._validate_adapt_inputs(data, previous, config)
        result = self._adapt_validated(data, previous, config)
        result = self._with_training_duration(result, time.perf_counter_ns() - started_ns)
        self._validate_training_result(result)
        if result.artifact.provenance.parent_model_version != previous.model_version:
            raise ValueError(
                "adapt() must set child provenance.parent_model_version to the parent model_version"
            )
        if result.artifact.model_version == previous.model_version:
            raise ValueError("adapt() must produce a new child model_version")
        return result

    @abstractmethod
    def _validate_adapt_inputs(
        self,
        data: AdaptationDataT,
        previous: "ModelArtifact",
        config: AdaptConfigT,
    ) -> None:
        """Reject invalid adaptation data, config, or parent Artifact."""

    @abstractmethod
    def _adapt_validated(
        self,
        data: AdaptationDataT,
        previous: "ModelArtifact",
        config: AdaptConfigT,
    ) -> TrainingResult:
        """Adapt inputs that have already passed the model-specific checks."""


class DecoderCore(ABC):
    """Model-specific algorithm implementation wrapped by DecoderRuntime."""

    model_type: ClassVar[str]
    implementation_version: ClassVar[str]

    @classmethod
    @abstractmethod
    def validate_artifact(cls, artifact: "ModelArtifact") -> None:
        """Validate model-specific Artifact invariants.

        SDK-owned runtime and persistence entry points call this exact hook.
        Implementations should delegate to the validator shared with the Trainer.
        """

        ...

    @classmethod
    @abstractmethod
    def from_artifact(cls, artifact: "ModelArtifact") -> Self:
        """Build a fresh Core after the SDK has validated the Artifact."""

        ...

    @abstractmethod
    def infer_core(self, request: CoreDecoderRequest) -> CoreInferenceResult: ...

    @abstractmethod
    def reset_core(self, request: RuntimeResetRequest | None = None) -> None: ...


class CheckpointableDecoderCore(DecoderCore):
    @abstractmethod
    def snapshot_core_state(self) -> CoreRuntimeState: ...

    @abstractmethod
    def restore_core_state(self, state: CoreRuntimeState) -> None: ...
