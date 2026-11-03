# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/decoders/standalone_refit_kf/trainer.py
# original git blob: 3a14a1546172fc77b7803835d8d208654f5c8f8d
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Production Trainer for rest-normalized moving-target ReFIT-KF."""

from __future__ import annotations

import time

import numpy as np

from ...bci_decoder_api import (
    ActionSpec,
    AdaptiveDecoderTrainer,
    DecoderRequest,
    DecoderRuntime,
    EvaluationReport,
    FeatureSpec,
    FeatureTensor,
    FeedbackSpec,
    FeedbackState,
    InferenceStatus,
    ModelArtifact,
    RuntimeResetRequest,
    TrainingProvenance,
    TrainingReport,
    TrainingResult,
    create_runtime,
    json_sha256,
)
from ...bci_decoder_api.errors import (
    ArtifactValidationError,
    TrainingNumericalError,
)
from .validation import (
    RefitKFInputValidator, TrackingValidation,
    validate_refit_kf_fit_inputs, validate_refit_kf_evaluation_inputs,
    validate_refit_kf_artifact, validate_refit_kf_parameters,
    validate_refit_kf_trainer_settings, validate_refit_kf_trainer_contract,
    validate_refit_kf_frozen_normalization,
)

from .data_types import STATE_UPDATE_PERIOD_NS, RefitKFFitData, RefitKFTrackingBlock
from .utils import _trial_local_velocity_validated
from .config import (
    COVARIANCE_FLOOR_OPTION_NAMES, REGULARIZATION_NAMES, RefitKFAdaptConfig, RefitKFFitConfig,
)
from .fitting import (
    IMPLEMENTATION_VERSION,
    MODEL_TYPE,
    RIDGE_LOSS,
    blend_kalman_parameters,
    fit_kalman_parameters,
    make_trivial_parameters,
    moving_target_intended_velocity,
    parameter_fingerprint,
)


TrainingData = RefitKFFitData | RefitKFTrackingBlock


class RefitKFTrainer(
    AdaptiveDecoderTrainer[
        TrainingData,
        RefitKFFitConfig,
        RefitKFTrackingBlock,
        RefitKFAdaptConfig,
    ]
):
    """Fit and adapt a checkpointable causal-position ReFIT-KF."""

    model_type = MODEL_TYPE
    implementation_version = IMPLEMENTATION_VERSION

    def __init__(
            self,
            *,
            feature_spec: FeatureSpec,
            action_spec: ActionSpec,
            feedback_spec: FeedbackSpec,
            max_speed: float = 1_000.0,
            maximum_feedback_age_ns: int = 50_000_000,
            covariance_tolerance: float = 1e-10,
    ) -> None:
        validate_refit_kf_trainer_settings(
            feature_spec=feature_spec, action_spec=action_spec, feedback_spec=feedback_spec,
            max_speed=max_speed, maximum_feedback_age_ns=maximum_feedback_age_ns,
            covariance_tolerance=covariance_tolerance,
        )
        self._input_validator = RefitKFInputValidator(feature_spec, action_spec.shape[0])
        self.feature_spec = feature_spec
        self.action_spec = action_spec
        self.feedback_spec = feedback_spec
        self.max_speed = float(max_speed)
        self.maximum_feedback_age_ns = maximum_feedback_age_ns
        self.covariance_tolerance = float(covariance_tolerance)

    def _validate_artifact_compatibility(self, artifact: ModelArtifact) -> None:
        validate_refit_kf_artifact(artifact)
        self._validate_trainer_contract(artifact)

    def _validate_trainer_contract(self, artifact: ModelArtifact) -> None:
        validate_refit_kf_trainer_contract(
            artifact, feature_spec=self.feature_spec, action_spec=self.action_spec,
            feedback_spec=self.feedback_spec, max_speed=self.max_speed,
            maximum_feedback_age_ns=self.maximum_feedback_age_ns,
            covariance_tolerance=self.covariance_tolerance,
        )

    @staticmethod
    def _normalized(
            features: np.ndarray, feature_mean: np.ndarray, feature_scale: np.ndarray
    ) -> np.ndarray:
        return (features.astype(np.float64) - feature_mean) / feature_scale

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
            created_at_ns=time.time_ns(),
            feature_spec=self.feature_spec,
            action_spec=self.action_spec,
            feedback_spec=self.feedback_spec,
            parameters=parameters,
            training_config=config_payload,
            runtime_config={
                "state_update_period_ns": STATE_UPDATE_PERIOD_NS,
                "required_feature_period_ns": STATE_UPDATE_PERIOD_NS,
                "maximum_feedback_age_ns": self.maximum_feedback_age_ns,
                "covariance_tolerance": self.covariance_tolerance,
                "missing_feature_policy": "reject",
                "max_speed": self.max_speed,
            },
            provenance=TrainingProvenance(
                source_ids=source_ids,
                sample_count=sample_count,
                sequence_count=sequence_count,
                training_code_version=self.implementation_version,
                training_config_hash=config_hash,
                parent_model_version=parent_model_version,
            ),
        )
        try:
            validate_refit_kf_artifact(artifact)
        except ArtifactValidationError as exc:
            raise TrainingNumericalError(
                f"candidate Artifact validation failed: {exc}"
            ) from exc
        return artifact

    @staticmethod
    def _base_config(
            *,
            stage: str,
            update_count: int,
            target_radius_px: float,
            minimum_intended_speed_px_s: float,
    ) -> dict[str, object]:
        return {
            "stage": stage,
            "supervised_update_count": update_count,
            "normalization": "rest_mean_std_frozen",
            "ridge_loss": RIDGE_LOSS,
            "rest_as_zero_velocity_training": False,
            "velocity_estimator": "trial_local_fixed_dt_backward_difference",
            "state_update_period_ns": STATE_UPDATE_PERIOD_NS,
            "feature_to_label_offset_ns": 0,
            "target_radius_px": float(target_radius_px),
            "minimum_intended_speed_px_s": float(
                minimum_intended_speed_px_s
            ),
        }

    def _validate_fit_inputs(
            self, data: TrainingData, config: RefitKFFitConfig
    ) -> TrackingValidation | None:
        return validate_refit_kf_fit_inputs(
            data, self._input_validator, minimum_samples=config.minimum_samples,
        )

    def _fit_validated(
            self, data: TrainingData, config: RefitKFFitConfig,
            *, validation: TrackingValidation | None = None,
    ) -> TrainingResult:
        assert isinstance(data, RefitKFFitData)
        rest = data.calibration.features.astype(np.float64)
        feature_mean = np.mean(rest, axis=0)
        raw_scale = np.std(rest, axis=0, ddof=0)
        feature_scale = np.maximum(raw_scale, float(config.std_floor))
        tracking = data.initial_tracking
        if tracking is None:
            parameters = make_trivial_parameters(
                feature_shape=self.feature_spec.shape,
                action_dimension=self.action_spec.shape[0],
                feature_mean=feature_mean,
                feature_scale=feature_scale,
                state_update_period_ns=STATE_UPDATE_PERIOD_NS,
            )
            metrics = {
                "calibration.scale_min": float(np.min(feature_scale)),
                "calibration.scale_floored_count": float(
                    np.count_nonzero(raw_scale < config.std_floor)
                ),
            }
            payload = {
                "mode": "fit",
                **self._base_config(
                    stage="calibration_seed",
                    update_count=0,
                    target_radius_px=config.target_radius_px,
                    minimum_intended_speed_px_s=(
                        config.minimum_intended_speed_px_s
                    ),
                ),
                "std_floor": float(config.std_floor),
                **{name: float(getattr(config, name)) for name in REGULARIZATION_NAMES},
                **{name: getattr(config, name) for name in COVARIANCE_FLOOR_OPTION_NAMES},
                "p0_velocity_regularization": float(
                    config.p0_velocity_regularization
                ),
                "minimum_samples": int(config.minimum_samples),
            }
            artifact = self._build_artifact(
                parameters=parameters,
                config_payload=payload,
                source_ids=data.calibration.source_ids,
                sample_count=rest.shape[0],
                sequence_count=1,
                parent_model_version=None,
                prefix="seed-refit-kf",
            )
            sample_count, sequence_count = rest.shape[0], 1
        else:
            assert validation is not None
            _, trial_count, valid, _ = validation
            actual_velocity = _trial_local_velocity_validated(tracking.cursor_position, valid)
            normalized = self._normalized(
                tracking.features, feature_mean, feature_scale
            )
            try:
                parameters, metrics = fit_kalman_parameters(
                    normalized_features=normalized,
                    position=tracking.cursor_position,
                    velocity=actual_velocity,
                    velocity_valid=valid,
                    trial_ids=tracking.trial_ids,
                    feature_mean=feature_mean,
                    feature_scale=feature_scale,
                    state_update_period_ns=STATE_UPDATE_PERIOD_NS,
                    **{name: getattr(config, name) for name in REGULARIZATION_NAMES},
                    **{name: getattr(config, name) for name in COVARIANCE_FLOOR_OPTION_NAMES},
                    p0_velocity_regularization=(
                        config.p0_velocity_regularization
                    ),
                )
            except (np.linalg.LinAlgError, ArtifactValidationError) as exc:
                raise TrainingNumericalError(str(exc)) from exc
            payload = {
                "mode": "fit",
                **self._base_config(
                    stage="initial_supervised",
                    update_count=1,
                    target_radius_px=config.target_radius_px,
                    minimum_intended_speed_px_s=(
                        config.minimum_intended_speed_px_s
                    ),
                ),
                "std_floor": float(config.std_floor),
                **{name: float(getattr(config, name)) for name in REGULARIZATION_NAMES},
                **{name: getattr(config, name) for name in COVARIANCE_FLOOR_OPTION_NAMES},
                "p0_velocity_regularization": float(
                    config.p0_velocity_regularization
                ),
                "minimum_samples": int(config.minimum_samples),
            }
            source_ids = data.calibration.source_ids + tracking.source_ids
            valid_count = int(np.count_nonzero(valid))
            sample_count = rest.shape[0] + valid_count
            sequence_count = 1 + trial_count
            artifact = self._build_artifact(
                parameters=parameters,
                config_payload=payload,
                source_ids=source_ids,
                sample_count=sample_count,
                sequence_count=sequence_count,
                parent_model_version=None,
                prefix="initial-refit-kf",
            )
        return TrainingResult(
            artifact=artifact,
            report=TrainingReport(
                model_type=self.model_type,
                model_version=artifact.model_version,
                sample_count=sample_count,
                sequence_count=sequence_count,
                duration_ns=0,
                metrics=metrics,
            ),
        )

    def _validate_adapt_inputs(
            self,
            data: RefitKFTrackingBlock,
            previous: ModelArtifact,
            config: RefitKFAdaptConfig,
    ) -> TrackingValidation:
        self._validate_artifact_compatibility(previous)
        return self._input_validator.validate_tracking(data, minimum_samples=config.minimum_samples)

    def _adapt_validated(
            self,
            data: RefitKFTrackingBlock,
            previous: ModelArtifact,
            config: RefitKFAdaptConfig,
            *, validation: TrackingValidation | None = None,
    ) -> TrainingResult:
        assert validation is not None
        _, trial_count, velocity_valid, _ = validation
        target_velocity = _trial_local_velocity_validated(data.target_position, velocity_valid)
        actual_velocity = _trial_local_velocity_validated(data.cursor_position, velocity_valid)
        intended = moving_target_intended_velocity(
            target_position=data.target_position,
            cursor_position=data.cursor_position,
            target_velocity=target_velocity,
            actual_velocity=actual_velocity,
            velocity_valid=velocity_valid,
            target_radius_px=config.target_radius_px,
            minimum_intended_speed_px_s=config.minimum_intended_speed_px_s,
            max_speed=self.max_speed,
        )
        feature_mean = np.array(previous.parameters["feature_mean"], copy=True)
        feature_scale = np.array(previous.parameters["feature_scale"], copy=True)
        normalized = self._normalized(data.features, feature_mean, feature_scale)
        try:
            candidate, metrics = fit_kalman_parameters(
                normalized_features=normalized,
                position=data.cursor_position,
                velocity=intended,
                velocity_valid=velocity_valid,
                trial_ids=data.trial_ids,
                feature_mean=feature_mean,
                feature_scale=feature_scale,
                state_update_period_ns=STATE_UPDATE_PERIOD_NS,
                **{name: getattr(config, name) for name in REGULARIZATION_NAMES},
                **{name: getattr(config, name) for name in COVARIANCE_FLOOR_OPTION_NAMES},
                p0_velocity_regularization=(
                    config.p0_velocity_regularization
                ),
            )
            validate_refit_kf_frozen_normalization(candidate, previous)
            update_count = int(previous.training_config["supervised_update_count"])
            blend_applied = config.alpha is not None and update_count > 0
            if blend_applied:
                # The candidate and blended model are distinct numerical results.
                # The final model is checked by _build_artifact below.
                validate_refit_kf_parameters(
                    candidate,
                    feature_shape=self.feature_spec.shape,
                    action_dimension=self.action_spec.shape[0],
                    state_update_period_ns=STATE_UPDATE_PERIOD_NS,
                    covariance_tolerance=self.covariance_tolerance,
                )
                parameters = blend_kalman_parameters(
                    previous.parameters,
                    candidate,
                    alpha=float(config.alpha),
                    state_update_period_ns=STATE_UPDATE_PERIOD_NS,
                )
                validate_refit_kf_frozen_normalization(parameters, previous)
            else:
                parameters = {
                    name: np.array(value, copy=True)
                    for name, value in candidate.items()
                }
        except (np.linalg.LinAlgError, ArtifactValidationError) as exc:
            raise TrainingNumericalError(str(exc)) from exc
        payload = {
            "mode": "adapt",
            **self._base_config(
                stage="refit_adapt",
                update_count=update_count + 1,
                target_radius_px=config.target_radius_px,
                minimum_intended_speed_px_s=(
                    config.minimum_intended_speed_px_s
                ),
            ),
            **{name: float(getattr(config, name)) for name in REGULARIZATION_NAMES},
            **{name: getattr(config, name) for name in COVARIANCE_FLOOR_OPTION_NAMES},
            "p0_velocity_regularization": float(
                config.p0_velocity_regularization
            ),
            "minimum_samples": int(config.minimum_samples),
            "blend_alpha": None if config.alpha is None else float(config.alpha),
            "blend_applied": bool(blend_applied),
            "blend_semantics": "alpha_parent_plus_one_minus_alpha_candidate",
        }
        valid_count = int(np.count_nonzero(velocity_valid))
        artifact = self._build_artifact(
            parameters=parameters,
            config_payload=payload,
            source_ids=data.source_ids,
            sample_count=valid_count,
            sequence_count=trial_count,
            parent_model_version=previous.model_version,
            prefix="adapted-refit-kf",
        )
        residual_speed = np.linalg.norm(actual_velocity - target_velocity, axis=1)
        metrics.update(
            {
                "adapt.blend_applied": float(blend_applied),
                "adapt.alpha_parent": (
                    0.0 if config.alpha is None else float(config.alpha)
                ),
                "adapt.mean_relative_actual_speed": float(
                    np.mean(residual_speed[velocity_valid])
                ),
            }
        )
        return TrainingResult(
            artifact=artifact,
            report=TrainingReport(
                model_type=self.model_type,
                model_version=artifact.model_version,
                sample_count=valid_count,
                sequence_count=trial_count,
                duration_ns=0,
                metrics=metrics,
            ),
        )

    def _validate_evaluation_inputs(
            self, artifact: ModelArtifact, data: TrainingData
    ) -> tuple[TrackingValidation, DecoderRuntime]:
        # Runtime installation owns full model validation. Reuse that Runtime
        # during evaluation rather than installing the same model twice.
        runtime = create_runtime(artifact)
        self._validate_trainer_contract(artifact)
        tracking = validate_refit_kf_evaluation_inputs(data, self._input_validator)
        return tracking, runtime

    def _evaluate_validated(
            self, artifact: ModelArtifact, data: TrainingData,
            *, validation: tuple[TrackingValidation, DecoderRuntime] | None = None,
    ) -> EvaluationReport:
        assert isinstance(data, RefitKFTrackingBlock)
        assert validation is not None
        tracking, runtime = validation
        valid = tracking.velocity_valid
        actual_velocity = _trial_local_velocity_validated(data.cursor_position, valid)
        target_velocity = _trial_local_velocity_validated(data.target_position, valid)
        intended = moving_target_intended_velocity(
            target_position=data.target_position,
            cursor_position=data.cursor_position,
            target_velocity=target_velocity,
            actual_velocity=actual_velocity,
            velocity_valid=valid,
            target_radius_px=float(artifact.training_config["target_radius_px"]),
            minimum_intended_speed_px_s=float(
                artifact.training_config["minimum_intended_speed_px_s"]
            ),
            max_speed=float(artifact.runtime_config["max_speed"]),
        )
        predictions: list[np.ndarray] = []
        valid_mask = np.ones(self.feature_spec.shape, dtype=bool)
        previous_trial: int | None = None
        for index, raw_trial in enumerate(data.trial_ids):
            trial = int(raw_trial)
            feedback = FeedbackState.from_spec(
                self.feedback_spec,
                values=np.asarray(data.cursor_position[index], dtype=np.float32),
                timestamp_ns=int(data.timestamps_ns[index]),
            )
            if trial != previous_trial:
                runtime.reset_state(
                    RuntimeResetRequest(
                        timestamp_ns=int(data.timestamps_ns[index]), feedback=feedback
                    )
                )
                previous_trial = trial
            result = runtime.infer(
                DecoderRequest(
                    features=FeatureTensor.from_spec(
                        self.feature_spec,
                        values=data.features[index],
                        valid_mask=valid_mask,
                        timestamp_ns=int(data.timestamps_ns[index]),
                        producer_frame_id=index,
                    ),
                    feedback=feedback,
                )
            )
            if result.status is not InferenceStatus.OK or result.output is None:
                raise TrainingNumericalError(
                    f"canonical runtime evaluation failed: {result.status.value}"
                )
            predictions.append(np.asarray(result.output.action, dtype=np.float64))
        predicted = np.stack(predictions)
        actual_error = predicted[valid] - actual_velocity[valid]
        intended_error = predicted[valid] - intended[valid]
        trial_rmse = []
        for trial in np.unique(data.trial_ids):
            mask = valid & (data.trial_ids == trial)
            if np.any(mask):
                error = predicted[mask] - intended[mask]
                trial_rmse.append(
                    float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))
                )
        return EvaluationReport(
            model_type=self.model_type,
            model_version=artifact.model_version,
            sample_count=int(np.count_nonzero(valid)),
            sequence_count=len(trial_rmse),
            duration_ns=0,
            metrics={
                "evaluation.actual_velocity_rmse": float(
                    np.sqrt(np.mean(np.sum(actual_error ** 2, axis=1)))
                ),
                "evaluation.intended_velocity_rmse": float(
                    np.sqrt(np.mean(np.sum(intended_error ** 2, axis=1)))
                ),
                "evaluation.trial_macro_intended_velocity_rmse": float(
                    np.mean(trial_rmse)
                ),
            },
        )
