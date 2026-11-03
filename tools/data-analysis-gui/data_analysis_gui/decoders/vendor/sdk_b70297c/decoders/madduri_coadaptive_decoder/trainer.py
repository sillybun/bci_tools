# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/decoders/madduri_coadaptive_decoder/trainer.py
# original git blob: f79c206328d069041ee501ee82efd46313ed440e
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Random-seed and SmoothBatch lifecycle over caller-defined tensor contracts."""

from dataclasses import asdict
from typing import cast

import numpy as np

from ...bci_decoder_api import (
    ActionSpec, AdaptiveDecoderTrainer, DecoderRequest, EvaluationReport,
    FeatureSpec, FeatureTensor, InferenceStatus, ModelArtifact, TrainingProvenance,
    TrainingReport, TrainingResult, create_runtime, json_sha256,
)
from ...bci_decoder_api.errors import ParameterValidationError, TrainingDataError, TrainingNumericalError

from .config import STATE_UPDATE_PERIOD_NS, MadduriAdaptConfig, MadduriFitConfig
from .data_types import MadduriInteractionBlock
from .fitting import (
    IMPLEMENTATION_VERSION, MODEL_TYPE, fit_optimal_decoder, limit_speed,
    parameter_fingerprint, random_seed_decoder, smoothbatch,
)
from .validation import (
    BlockValidation, MadduriInputValidator, spec_payload, validate_adapt_config,
    validate_fit_config, validate_madduri_artifact, validate_madduri_parameters,
    validate_trainer_contract, validate_trainer_settings,
)


class MadduriSmoothBatchTrainer(
    AdaptiveDecoderTrainer[MadduriInteractionBlock, MadduriFitConfig, MadduriInteractionBlock, MadduriAdaptConfig]
):
    model_type = MODEL_TYPE
    implementation_version = IMPLEMENTATION_VERSION

    def __init__(self, *, feature_spec: FeatureSpec, action_spec: ActionSpec,
                 max_speed: float | None = None,
                 state_update_period_ns: int = STATE_UPDATE_PERIOD_NS) -> None:
        validate_trainer_settings(feature_spec, action_spec, state_update_period_ns, max_speed)
        self.feature_spec = feature_spec
        self.action_spec = action_spec
        self.state_update_period_ns = state_update_period_ns
        self.max_speed = None if max_speed is None else float(max_speed)

    @property
    def feature_count(self) -> int:
        return int(np.prod(self.feature_spec.shape))

    @property
    def channel_count(self) -> int:
        """Compatibility alias for the flattened feature count."""
        return self.feature_count

    @property
    def action_dimension(self) -> int:
        return self.action_spec.shape[0]

    def _validate_block(self, data, *, minimum_samples, require_task_signals) -> BlockValidation:
        validate_trainer_settings(self.feature_spec, self.action_spec, self.state_update_period_ns, self.max_speed)
        return MadduriInputValidator(self.feature_spec, self.action_dimension).validate_block(
            data, minimum_samples=minimum_samples, require_task_signals=require_task_signals,
            state_update_period_ns=self.state_update_period_ns,
        )

    def _validate_fit_inputs(self, data: MadduriInteractionBlock, config: MadduriFitConfig) -> BlockValidation:
        validate_fit_config(config)
        summary = self._validate_block(data, minimum_samples=0, require_task_signals=False)
        if summary.sample_count:
            raise TrainingDataError("random seed fit consumes no data; pass an empty seed block")
        return summary

    def _validate_adapt_inputs(
        self, data: MadduriInteractionBlock, previous: ModelArtifact, config: MadduriAdaptConfig,
    ) -> BlockValidation:
        validate_adapt_config(config)
        validate_madduri_artifact(previous)
        validate_trainer_contract(previous, self.feature_spec, self.action_spec, self.state_update_period_ns)
        return self._validate_block(data, minimum_samples=config.minimum_samples, require_task_signals=True)

    def _validate_evaluation_inputs(self, artifact: ModelArtifact, data: MadduriInteractionBlock) -> BlockValidation:
        validate_madduri_artifact(artifact)
        validate_trainer_contract(artifact, self.feature_spec, self.action_spec, self.state_update_period_ns)
        return self._validate_block(data, minimum_samples=1, require_task_signals=True)

    @staticmethod
    def _desired_velocity(data, *, period_ns, max_speed) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            velocity = (np.asarray(data.target_position, dtype=np.float64)
                        - np.asarray(data.cursor_position, dtype=np.float64)) / (period_ns / 1e9)
        if not np.all(np.isfinite(velocity)):
            raise TrainingNumericalError("desired velocity contains non-finite values")
        return velocity if max_speed is None else limit_speed(velocity, max_speed)

    def _build_artifact(self, *, parameters, config_payload, data, summary,
                        parent_model_version, prefix) -> ModelArtifact:
        runtime_config = {
            "max_speed": self.max_speed, "state_update_period_ns": self.state_update_period_ns,
            "required_feature_period_ns": self.state_update_period_ns,
            "flatten_order": "C", "missing_feature_policy": "reject",
        }
        config_hash = json_sha256(config_payload)
        identity = {
            "model_type": self.model_type, "implementation_version": self.implementation_version,
            "feature_spec": spec_payload(self.feature_spec), "action_spec": spec_payload(self.action_spec),
            "runtime_config": runtime_config, "training_config_hash": config_hash,
            "source_ids": data.source_ids, "sample_count": summary.sample_count,
            "sequence_count": summary.sequence_count, "parent_model_version": parent_model_version,
        }
        fingerprint = parameter_fingerprint(parameters, payload={"contract_hash": json_sha256(identity)})
        artifact = ModelArtifact(
            decoder_api_version="2.0", artifact_format_version="2.0",
            model_type=self.model_type, model_version=f"{prefix}-{fingerprint}",
            implementation_version=self.implementation_version,
            created_at_ns=0,  # Deterministic sentinel; persistence preserves it.
            feature_spec=self.feature_spec, action_spec=self.action_spec, feedback_spec=None,
            parameters=parameters, training_config=config_payload, runtime_config=runtime_config,
            provenance=TrainingProvenance(
                source_ids=data.source_ids, sample_count=summary.sample_count,
                sequence_count=summary.sequence_count, training_code_version=self.implementation_version,
                training_config_hash=config_hash, parent_model_version=parent_model_version,
            ),
        )
        validate_madduri_artifact(artifact)
        return artifact

    def _fit_validated(self, data, config, *, validation: object = None) -> TrainingResult:
        assert isinstance(validation, BlockValidation)
        weights, intercept = random_seed_decoder(
            channel_count=self.feature_count, action_dimension=self.action_dimension,
            init_sign=config.init_sign, scale=config.scale, random_seed=config.random_seed,
        )
        artifact = self._build_artifact(
            parameters={"D": weights, "b": intercept}, config_payload={"mode": "fit", **asdict(config)},
            data=data, summary=validation, parent_model_version=None, prefix="seed-madduri",
        )
        return TrainingResult(artifact, TrainingReport(
            model_type=self.model_type, model_version=artifact.model_version,
            sample_count=0, sequence_count=0, duration_ns=0,
            metrics={"seed.weight_abs_max": float(np.max(np.abs(weights)))},
        ))

    def _adapt_validated(self, data, previous, config, *, validation: object = None) -> TrainingResult:
        assert isinstance(validation, BlockValidation)
        desired = self._desired_velocity(data, period_ns=self.state_update_period_ns, max_speed=self.max_speed)
        design = data.features.reshape(validation.sample_count, self.feature_count, order="C").astype(np.float64)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                optimal_weights, optimal_intercept = fit_optimal_decoder(
                    design, desired, penalty=config.penalty, fit_intercept=config.fit_intercept,
                )
                weights, intercept = smoothbatch(
                    previous.parameters["D"], previous.parameters["b"], optimal_weights, optimal_intercept,
                    learning_rate=config.learning_rate,
                )
                if not config.fit_intercept:
                    intercept = np.zeros(self.action_dimension, dtype=np.float64)
                parameters = {"D": weights, "b": intercept}
                validate_madduri_parameters(parameters, feature_count=self.feature_count,
                                            action_dimension=self.action_dimension)
                residual = desired - (design @ weights.T + intercept)
                rmse = float(np.sqrt(np.mean(residual ** 2)))
                norm = float(np.linalg.norm(weights))
                if not np.isfinite(rmse) or not np.isfinite(norm):
                    raise FloatingPointError("non-finite adaptation metrics")
        except (np.linalg.LinAlgError, FloatingPointError, ValueError, ParameterValidationError) as exc:
            raise TrainingNumericalError(str(exc)) from exc
        artifact = self._build_artifact(
            parameters=parameters, config_payload={"mode": "adapt", **asdict(config)}, data=data,
            summary=validation, parent_model_version=previous.model_version, prefix="adapt-madduri",
        )
        return TrainingResult(artifact, TrainingReport(
            model_type=self.model_type, model_version=artifact.model_version,
            sample_count=validation.sample_count, sequence_count=validation.sequence_count, duration_ns=0,
            metrics={"adapt.tracking_velocity_rmse": rmse, "adapt.weight_frobenius_norm": norm},
            warnings=validation.warnings,
        ))

    def _evaluate_validated(self, artifact, data, *, validation: object = None) -> EvaluationReport:
        assert isinstance(validation, BlockValidation)
        runtime = create_runtime(artifact)
        desired = self._desired_velocity(
            data, period_ns=cast(int, artifact.runtime_config["state_update_period_ns"]),
            max_speed=artifact.runtime_config["max_speed"],
        )
        predicted = []
        valid_mask = np.ones(artifact.feature_spec.shape, dtype=bool)
        previous_group = None
        for index, group in enumerate(data.trial_ids):
            if group != previous_group:
                runtime.reset_state()
                previous_group = group
            result = runtime.infer(DecoderRequest(features=FeatureTensor.from_spec(
                artifact.feature_spec, values=data.features[index], valid_mask=valid_mask,
                timestamp_ns=int(data.timestamps_ns[index]), producer_frame_id=index,
            )))
            if result.status is not InferenceStatus.OK or result.output is None:
                raise TrainingNumericalError(f"canonical runtime evaluation failed: {result.diagnostics.reason_code}")
            predicted.append(result.output.action.astype(np.float64))
        with np.errstate(over="ignore", invalid="ignore"):
            rmse = float(np.sqrt(np.mean((np.stack(predicted) - desired) ** 2)))
        if not np.isfinite(rmse):
            raise TrainingNumericalError("non-finite evaluation metric")
        return EvaluationReport(
            model_type=self.model_type, model_version=artifact.model_version,
            sample_count=validation.sample_count, sequence_count=validation.sequence_count, duration_ns=0,
            metrics={"evaluation.tracking_velocity_rmse": rmse}, warnings=validation.warnings,
        )
