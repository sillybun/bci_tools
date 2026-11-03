# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: 8d79df0475b85ca859a02a8268f5262358a22afa
# original path: src/decoders/ecog_linear_velocity_decoder/trainer.py
# original git blob: aeed719495a5a5904af26113fbd1a6eb3da9d436
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Rest-calibrated, trial-balanced adaptive ECoG linear velocity Trainer."""

from __future__ import annotations

import numpy as np

from ...bci_decoder_api import (
    ActionSpec,
    AdaptiveDecoderTrainer,
    DecoderRequest,
    EvaluationReport,
    FeatureSpec,
    FeatureTensor,
    InferenceStatus,
    ModelArtifact,
    TrainingProvenance,
    TrainingReport,
    TrainingResult,
    create_runtime,
    json_sha256,
)
from ...bci_decoder_api.errors import (
    InsufficientDataError,
    TrainingDataError,
    TrainingNumericalError,
    TrainingSchemaError,
)

from .data_types import (
    CURSOR_VELOCITY_COMPONENTS,
    CURSOR_VELOCITY_SCHEMA_ID,
    ECOG_BAND_LABELS,
    ECOG_FEATURE_SCHEMA_ID,
    EcogAdaptConfig,
    EcogCalibrationBlock,
    EcogFitConfig,
    EcogInteractionBlock,
)
from .fitting import (
    ACTION_DIMENSION,
    IMPLEMENTATION_VERSION,
    MODEL_TYPE,
    desired_tracking_velocity,
    fit_trial_balanced_ridge,
    parameter_fingerprint,
    smoothbatch,
    trial_equal_sample_weights,
    validate_ecog_linear_artifact,
)


class EcogLinearVelocityTrainer(
    AdaptiveDecoderTrainer[
        EcogCalibrationBlock,
        EcogFitConfig,
        EcogInteractionBlock,
        EcogAdaptConfig,
    ]
):
    """Fit frozen rest normalization, then adapt a 2-D affine Ridge map."""

    model_type = MODEL_TYPE
    implementation_version = IMPLEMENTATION_VERSION

    def __init__(
        self,
        *,
        feature_spec: FeatureSpec,
        action_spec: ActionSpec,
        max_speed: float = 1_000.0,
    ) -> None:
        if feature_spec.schema_id != ECOG_FEATURE_SCHEMA_ID:
            raise ValueError("unexpected ECoG feature schema")
        if feature_spec.rank != 2 or feature_spec.axes != ("channel", "band"):
            raise ValueError("ECoG FeatureSpec must have (channel, band) axes")
        if feature_spec.shape[1] != len(ECOG_BAND_LABELS):
            raise ValueError("ECoG FeatureSpec must contain six bands")
        if feature_spec.layout.axis_labels.get("band") != ECOG_BAND_LABELS:
            raise ValueError("ECoG band labels/order do not match the schema")
        if action_spec.schema_id != CURSOR_VELOCITY_SCHEMA_ID:
            raise ValueError("unexpected cursor velocity action schema")
        if action_spec.rank != 1 or action_spec.shape != (ACTION_DIMENSION,):
            raise ValueError("the decoder emits a two-dimensional velocity action")
        if (
            action_spec.layout.axis_labels.get("component")
            != CURSOR_VELOCITY_COMPONENTS
        ):
            raise ValueError("action component order must be (x, y)")
        if (
            isinstance(max_speed, bool)
            or not isinstance(max_speed, (int, float))
            or not np.isfinite(max_speed)
            or max_speed <= 0.0
        ):
            raise ValueError("max_speed must be finite and positive")
        self.feature_spec = feature_spec
        self.action_spec = action_spec
        self.max_speed = float(max_speed)

    @property
    def feature_count(self) -> int:
        return int(np.prod(self.feature_spec.shape))

    @staticmethod
    def _validate_source_ids(source_ids: tuple[str, ...]) -> None:
        if not isinstance(source_ids, tuple) or not source_ids or any(
            not isinstance(value, str) or not value for value in source_ids
        ):
            raise TrainingDataError(
                "source_ids must be a non-empty tuple of non-empty strings"
            )

    def _validate_feature_rows(
        self,
        features: np.ndarray,
        timestamps_ns: np.ndarray,
        *,
        schema_id: str,
        layout_id: str,
        minimum_samples: int,
    ) -> int:
        if not isinstance(features, np.ndarray) or features.ndim != 3:
            raise TrainingSchemaError(
                "features must have shape (sample, channel, band)"
            )
        sample_count = int(features.shape[0])
        if sample_count < minimum_samples:
            raise InsufficientDataError(
                f"need at least {minimum_samples} samples, got {sample_count}"
            )
        if schema_id != self.feature_spec.schema_id or layout_id != self.feature_spec.layout_id:
            raise TrainingSchemaError("feature schema/layout does not match the Trainer")
        if features.shape[1:] != self.feature_spec.shape:
            raise TrainingSchemaError("feature tensor shape does not match FeatureSpec")
        if features.dtype != np.dtype("float32"):
            raise TrainingSchemaError("features must use float32")
        if not np.all(np.isfinite(features)):
            raise TrainingDataError("features must be finite")
        if not isinstance(timestamps_ns, np.ndarray) or timestamps_ns.shape != (
            sample_count,
        ):
            raise TrainingDataError("timestamps_ns must have one value per sample")
        if not np.issubdtype(timestamps_ns.dtype, np.integer):
            raise TrainingDataError("timestamps_ns must use an integer dtype")
        # Out-of-order / duplicate timestamps are tolerated: fitting substitutes
        # the nominal frame period (DEFAULT_SAMPLE_PERIOD_NS) for any dt <= 0, so
        # a rare bad timestamp no longer aborts training.
        return sample_count

    @staticmethod
    def _validate_contiguous_trials(trial_ids: np.ndarray) -> int:
        if not isinstance(trial_ids, np.ndarray) or trial_ids.ndim != 1:
            raise TrainingDataError("trial_ids must be a one-dimensional array")
        if not np.issubdtype(trial_ids.dtype, np.integer):
            raise TrainingDataError("trial_ids must use an integer dtype")
        seen: set[int] = set()
        previous: int | None = None
        counts: dict[int, int] = {}
        for raw in trial_ids:
            trial = int(raw)
            counts[trial] = counts.get(trial, 0) + 1
            if trial != previous:
                if trial in seen:
                    raise TrainingDataError("each trial must form one contiguous segment")
                seen.add(trial)
                previous = trial
        if any(count < 2 for count in counts.values()):
            raise InsufficientDataError("each trial must contain at least two samples")
        return len(counts)

    def _validate_calibration(
        self, data: EcogCalibrationBlock, *, minimum_samples: int
    ) -> int:
        sample_count = self._validate_feature_rows(
            data.features,
            data.timestamps_ns,
            schema_id=data.feature_schema_id,
            layout_id=data.feature_layout_id,
            minimum_samples=minimum_samples,
        )
        self._validate_source_ids(data.source_ids)
        return sample_count

    def _validate_interaction(
        self, data: EcogInteractionBlock, *, minimum_samples: int
    ) -> tuple[int, int]:
        sample_count = self._validate_feature_rows(
            data.features,
            data.timestamps_ns,
            schema_id=data.feature_schema_id,
            layout_id=data.feature_layout_id,
            minimum_samples=minimum_samples,
        )
        expected_vector_shape = (sample_count, ACTION_DIMENSION)
        if not isinstance(data.trial_ids, np.ndarray) or data.trial_ids.shape != (
            sample_count,
        ):
            raise TrainingDataError("trial_ids must have one value per sample")
        trial_count = self._validate_contiguous_trials(data.trial_ids)
        for name, values in (
            ("target_position", data.target_position),
            ("cursor_position", data.cursor_position),
        ):
            if not isinstance(values, np.ndarray) or values.shape != expected_vector_shape:
                raise TrainingDataError(f"{name} must have shape (sample, 2)")
            if not np.issubdtype(values.dtype, np.number) or not np.all(
                np.isfinite(values)
            ):
                raise TrainingDataError(f"{name} must be finite numeric data")
        self._validate_source_ids(data.source_ids)
        return sample_count, trial_count

    def _validate_artifact_compatibility(self, artifact: ModelArtifact) -> None:
        validate_ecog_linear_artifact(artifact)
        if (
            artifact.feature_spec != self.feature_spec
            or artifact.action_spec != self.action_spec
            or artifact.feedback_spec is not None
        ):
            raise TrainingSchemaError("Artifact tensor specs do not match this Trainer")
        if float(artifact.runtime_config["max_speed"]) != self.max_speed:
            raise TrainingSchemaError("Artifact max_speed does not match this Trainer")

    def _build_artifact(
        self,
        *,
        parameters: dict[str, np.ndarray],
        config_payload: dict[str, object],
        source_ids: tuple[str, ...],
        sample_count: int,
        sequence_count: int,
        parent_model_version: str | None,
        prefix: str,
    ) -> ModelArtifact:
        config_hash = json_sha256(config_payload)
        fingerprint = parameter_fingerprint(
            parameters,
            payload={
                "config_hash": config_hash,
                "source_ids": list(source_ids),
                "parent_model_version": parent_model_version,
            },
        )
        artifact = ModelArtifact(
            decoder_api_version="2.0",
            artifact_format_version="2.0",
            model_type=self.model_type,
            model_version=f"{prefix}-{fingerprint}",
            implementation_version=self.implementation_version,
            created_at_ns=0,
            feature_spec=self.feature_spec,
            action_spec=self.action_spec,
            feedback_spec=None,
            parameters=parameters,
            training_config=config_payload,
            runtime_config={"max_speed": self.max_speed},
            provenance=TrainingProvenance(
                source_ids=source_ids,
                sample_count=sample_count,
                sequence_count=sequence_count,
                training_code_version=self.implementation_version,
                training_config_hash=config_hash,
                parent_model_version=parent_model_version,
            ),
        )
        validate_ecog_linear_artifact(artifact)
        return artifact

    def _validate_fit_inputs(
        self, data: EcogCalibrationBlock, config: EcogFitConfig
    ) -> None:
        self._validate_calibration(data, minimum_samples=2)

    def _fit_validated(
        self, data: EcogCalibrationBlock, config: EcogFitConfig
    ) -> TrainingResult:
        features = data.features.astype(np.float64)
        feature_mean = np.mean(features, axis=0)
        raw_scale = np.std(features, axis=0, ddof=0)
        feature_scale = np.maximum(raw_scale, config.std_floor)
        parameters = {
            "W": np.zeros((ACTION_DIMENSION, self.feature_count), dtype=np.float64),
            "b": np.zeros(ACTION_DIMENSION, dtype=np.float64),
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
        }
        config_payload: dict[str, object] = {
            "mode": "fit",
            "stage": "calibration_seed",
            "normalization": "rest_mean_std_frozen",
            "std_floor": float(config.std_floor),
            "rest_as_zero_velocity_training": False,
            "supervised_update_count": 0,
            "trial_weighting": "equal_total_weight",
            "target_velocity_estimator": "trial_local_backward_difference",
            "feature_to_label_offset_ns": 0,
        }
        artifact = self._build_artifact(
            parameters=parameters,
            config_payload=config_payload,
            source_ids=data.source_ids,
            sample_count=data.features.shape[0],
            sequence_count=1,
            parent_model_version=None,
            prefix="seed-ecog-linear",
        )
        floored = int(np.count_nonzero(raw_scale < config.std_floor))
        return TrainingResult(
            artifact=artifact,
            report=TrainingReport(
                model_type=self.model_type,
                model_version=artifact.model_version,
                sample_count=data.features.shape[0],
                sequence_count=1,
                duration_ns=0,
                metrics={
                    "calibration.scale_min": float(np.min(feature_scale)),
                    "calibration.scale_floored_count": float(floored),
                },
            ),
        )

    def _validate_adapt_inputs(
        self,
        data: EcogInteractionBlock,
        previous: ModelArtifact,
        config: EcogAdaptConfig,
    ) -> None:
        self._validate_artifact_compatibility(previous)
        self._validate_interaction(data, minimum_samples=config.minimum_samples)

    def _desired_velocity(
        self, data: EcogInteractionBlock, *, tracking_time_constant_ms: float
    ) -> np.ndarray:
        return desired_tracking_velocity(
            data.target_position,
            data.cursor_position,
            data.timestamps_ns,
            data.trial_ids,
            tracking_time_constant_ms=tracking_time_constant_ms,
            max_speed=self.max_speed,
        )

    def _adapt_validated(
        self,
        data: EcogInteractionBlock,
        previous: ModelArtifact,
        config: EcogAdaptConfig,
    ) -> TrainingResult:
        update_count = int(previous.training_config["supervised_update_count"])
        feature_mean = np.array(previous.parameters["feature_mean"], copy=True)
        feature_scale = np.array(previous.parameters["feature_scale"], copy=True)
        normalized = (
            (data.features.astype(np.float64) - feature_mean) / feature_scale
        ).reshape(data.features.shape[0], -1, order="C")
        desired = self._desired_velocity(
            data, tracking_time_constant_ms=config.tracking_time_constant_ms
        )
        try:
            optimal_w, optimal_b = fit_trial_balanced_ridge(
                normalized,
                desired,
                data.trial_ids,
                penalty=config.penalty,
                fit_intercept=config.fit_intercept,
            )
        except np.linalg.LinAlgError as exc:
            raise TrainingNumericalError(str(exc)) from exc
        if update_count == 0:
            weights, intercept = smoothbatch(
                previous.parameters["W"],
                previous.parameters["b"],
                optimal_w,
                optimal_b,
                alpha=config.first_alpha,
            )
            blend_applied = float(config.first_alpha > 0.0)
        else:
            weights, intercept = smoothbatch(
                previous.parameters["W"],
                previous.parameters["b"],
                optimal_w,
                optimal_b,
                alpha=config.alpha,
            )
            blend_applied = 1.0
        if not config.fit_intercept:
            intercept = np.zeros(ACTION_DIMENSION, dtype=np.float64)
        parameters = {
            "W": np.asarray(weights, dtype=np.float64),
            "b": np.asarray(intercept, dtype=np.float64),
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
        }
        config_payload: dict[str, object] = {
            "mode": "adapt",
            "stage": "supervised_adapt",
            "normalization": "rest_mean_std_frozen",
            "rest_as_zero_velocity_training": False,
            "supervised_update_count": update_count + 1,
            "trial_weighting": "equal_total_weight",
            "target_velocity_estimator": "trial_local_backward_difference",
            "feature_to_label_offset_ns": 0,
            "ridge_penalty": float(config.penalty),
            "smoothbatch_alpha": float(config.alpha),
            "first_smoothbatch_alpha": float(config.first_alpha),
            "tracking_time_constant_ms": float(config.tracking_time_constant_ms),
            "minimum_samples": int(config.minimum_samples),
            "fit_intercept": bool(config.fit_intercept),
            "first_supervised_update_uses_full_optimum": config.first_alpha == 0.0,
        }
        artifact = self._build_artifact(
            parameters=parameters,
            config_payload=config_payload,
            source_ids=data.source_ids,
            sample_count=data.features.shape[0],
            sequence_count=int(np.unique(data.trial_ids).size),
            parent_model_version=previous.model_version,
            prefix="adapt-ecog-linear",
        )
        prediction = normalized @ weights.T + intercept
        residual = prediction - desired
        sample_weights = trial_equal_sample_weights(data.trial_ids)
        weighted_mse = float(
            np.sum(sample_weights[:, None] * residual**2)
            / (np.sum(sample_weights) * ACTION_DIMENSION)
        )
        return TrainingResult(
            artifact=artifact,
            report=TrainingReport(
                model_type=self.model_type,
                model_version=artifact.model_version,
                sample_count=data.features.shape[0],
                sequence_count=int(np.unique(data.trial_ids).size),
                duration_ns=0,
                metrics={
                    "adapt.trial_balanced_velocity_rmse": float(np.sqrt(weighted_mse)),
                    "adapt.weight_frobenius_norm": float(np.linalg.norm(weights)),
                    "adapt.smoothbatch_applied": blend_applied,
                },
            ),
        )

    def _validate_evaluation_inputs(
        self, artifact: ModelArtifact, data: EcogInteractionBlock
    ) -> None:
        self._validate_artifact_compatibility(artifact)
        self._validate_interaction(data, minimum_samples=2)

    def _evaluate_validated(
        self, artifact: ModelArtifact, data: EcogInteractionBlock
    ) -> EvaluationReport:
        runtime = create_runtime(artifact)
        desired = self._desired_velocity(
            data,
            tracking_time_constant_ms=float(
                artifact.training_config.get("tracking_time_constant_ms", 60.0)
            ),
        )
        predictions: list[np.ndarray] = []
        valid_mask = np.ones(self.feature_spec.shape, dtype=bool)
        for index in range(data.features.shape[0]):
            result = runtime.infer(
                DecoderRequest(
                    features=FeatureTensor.from_spec(
                        self.feature_spec,
                        values=data.features[index],
                        valid_mask=valid_mask,
                        timestamp_ns=int(data.timestamps_ns[index]),
                        producer_frame_id=index,
                    )
                )
            )
            if result.status is not InferenceStatus.OK or result.output is None:
                raise TrainingNumericalError(
                    f"canonical runtime evaluation failed: {result.status.value}"
                )
            predictions.append(np.array(result.output.action, dtype=np.float64))
        predicted = np.stack(predictions)
        error = predicted - desired
        axis_rmse = np.sqrt(np.mean(error**2, axis=0))
        vector_rmse = float(np.sqrt(np.mean(np.sum(error**2, axis=1))))
        trial_rmse = [
            float(np.sqrt(np.mean(np.sum(error[data.trial_ids == trial] ** 2, axis=1))))
            for trial in np.unique(data.trial_ids)
        ]
        return EvaluationReport(
            model_type=self.model_type,
            model_version=artifact.model_version,
            sample_count=data.features.shape[0],
            sequence_count=len(trial_rmse),
            duration_ns=0,
            metrics={
                "evaluation.velocity_rmse": vector_rmse,
                "evaluation.velocity_rmse_x": float(axis_rmse[0]),
                "evaluation.velocity_rmse_y": float(axis_rmse[1]),
                "evaluation.trial_macro_velocity_rmse": float(np.mean(trial_rmse)),
            },
        )
