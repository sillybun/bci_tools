# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/decoders/madduri_coadaptive_decoder/validation.py
# original git blob: b555de186e1b994960711ffbe61628497cc2a56c
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Model checks grouped by lifecycle boundary; never transform or retain inputs."""

from typing import NamedTuple

import numpy as np

from ...bci_decoder_api import ActionSpec, FeatureSpec, ModelArtifact, json_sha256
from ...bci_decoder_api.errors import (
    ArtifactValidationError, InsufficientDataError, ParameterValidationError,
    TrainingDataError, TrainingSchemaError,
)

from .config import MadduriAdaptConfig, MadduriFitConfig, finite_real, positive_integer
from .data_types import MadduriInteractionBlock
from .fitting import IMPLEMENTATION_VERSION, MODEL_TYPE


# Input specs and interaction blocks

def validate_decoder_specs(feature_spec: FeatureSpec, action_spec: ActionSpec) -> None:
    if not isinstance(feature_spec, FeatureSpec) or feature_spec.rank < 1:
        raise ValueError("features must have at least one axis")
    if not isinstance(action_spec, ActionSpec) or action_spec.rank != 1:
        raise ValueError("velocity action must be a vector")


class BlockValidation(NamedTuple):
    """Operation-local summary, not a certificate for mutable input arrays."""

    sample_count: int
    sequence_count: int
    warnings: tuple[str, ...] = ()


class MadduriInputValidator:
    def __init__(self, feature_spec: FeatureSpec, action_dimension: int) -> None:
        self.feature_spec = feature_spec
        self.action_dimension = positive_integer(action_dimension, name="action_dimension")

    def validate_block(
            self, data: MadduriInteractionBlock, *, minimum_samples: int,
            require_task_signals: bool, state_update_period_ns: int,
    ) -> BlockValidation:
        if not isinstance(data, MadduriInteractionBlock):
            raise TrainingDataError("data must be MadduriInteractionBlock")
        features = data.features
        if not isinstance(features, np.ndarray) or features.ndim != self.feature_spec.rank + 1:
            raise TrainingSchemaError("features must have shape (sample, *feature_spec.shape)")
        count = features.shape[0]
        if count < minimum_samples:
            raise InsufficientDataError(f"need at least {minimum_samples} samples, got {count}")
        if (
                data.feature_schema_id != self.feature_spec.schema_id
                or data.feature_layout_id != self.feature_spec.layout_id
                or features.shape[1:] != self.feature_spec.shape
                or features.dtype != np.dtype("float32")
        ):
            raise TrainingSchemaError("feature schema/layout/shape/dtype does not match Trainer")
        if not np.all(np.isfinite(features)):
            raise TrainingDataError("features must be finite")
        for name, array in (("timestamps_ns", data.timestamps_ns), ("trial_ids", data.trial_ids)):
            if (
                    not isinstance(array, np.ndarray) or array.shape != (count,)
                    or not np.issubdtype(array.dtype, np.integer)
            ):
                raise TrainingDataError(f"{name} must be an integer array with one value per sample")
        if np.any(data.timestamps_ns < 0) or np.any(data.timestamps_ns > np.iinfo(np.int64).max):
            raise TrainingDataError("timestamps_ns must be nonnegative and int64-compatible")
        seen: set[int] = set()
        last_group = None
        last_time = None
        mismatch_count = 0
        for group_value, time_value in zip(data.trial_ids, data.timestamps_ns):
            group, timestamp = int(group_value), int(time_value)
            if group != last_group:
                if group in seen:
                    raise TrainingDataError("each trial ID must occupy one contiguous segment")
                seen.add(group)
                last_group = group
            else:
                interval = timestamp - last_time
                if interval <= 0:
                    raise TrainingDataError("timestamps_ns must be strictly increasing within each trial")
                mismatch_count += interval != state_update_period_ns
            last_time = timestamp
        if not isinstance(data.source_ids, tuple) or not data.source_ids or any(
                not isinstance(value, str) or not value.strip() for value in data.source_ids
        ):
            raise TrainingDataError("source_ids must be a non-empty tuple of non-empty strings")
        for name, values in (("target_position", data.target_position), ("cursor_position", data.cursor_position)):
            if values is None and not require_task_signals:
                continue
            if (
                    not isinstance(values, np.ndarray) or values.shape != (count, self.action_dimension)
                    or not (np.issubdtype(values.dtype, np.floating) or np.issubdtype(values.dtype, np.integer))
            ):
                raise TrainingDataError(f"{name} must be a real array of shape (sample, action_dimension)")
            if not np.all(np.isfinite(values)):
                raise TrainingDataError(f"{name} must be finite")
        warnings = () if not mismatch_count else (
            f"{mismatch_count} timestamp intervals differ from state_update_period_ns; using fixed model timestep",
        )
        return BlockValidation(count, len(seen), warnings)


# Trainer settings and operation inputs

def validate_trainer_settings(feature_spec, action_spec, period_ns, max_speed) -> None:
    validate_decoder_specs(feature_spec, action_spec)
    positive_integer(period_ns, name="state_update_period_ns")
    if max_speed is not None:
        finite_real(max_speed, name="max_speed", positive=True)


def validate_fit_config(config: MadduriFitConfig) -> None:
    if not isinstance(config, MadduriFitConfig):
        raise TrainingDataError("config must be MadduriFitConfig")
    config.__post_init__()


def validate_adapt_config(config: MadduriAdaptConfig) -> None:
    if not isinstance(config, MadduriAdaptConfig):
        raise TrainingDataError("config must be MadduriAdaptConfig")
    config.__post_init__()


def validate_trainer_contract(artifact, feature_spec, action_spec, period_ns) -> None:
    if artifact.feature_spec != feature_spec or artifact.action_spec != action_spec:
        raise TrainingSchemaError("Artifact tensor specs do not match this Trainer")
    if artifact.runtime_config["state_update_period_ns"] != period_ns:
        raise TrainingSchemaError("Artifact timestep does not match this Trainer")


# Parameters and Artifact metadata

def validate_madduri_parameters(parameters, *, feature_count: int, action_dimension: int) -> None:
    expected = {"D": (action_dimension, feature_count), "b": (action_dimension,)}
    if set(parameters) != set(expected):
        raise ParameterValidationError("decoder parameter names are incomplete or unexpected")
    for name, shape in expected.items():
        array = parameters[name]
        if (
                not isinstance(array, np.ndarray) or array.shape != shape
                or not np.issubdtype(array.dtype, np.floating)
        ):
            raise ParameterValidationError(f"parameter {name} has the wrong shape or dtype")
        if not np.all(np.isfinite(array)):
            raise ParameterValidationError(f"parameter {name} contains non-finite values")


def spec_payload(spec) -> dict:
    """Canonical identity payload including caller-defined layout ordering."""
    payload: dict[str, object] = {
        "schema": {
            "schema_id": spec.schema_id, "tensor_role": spec.schema.tensor_role,
            "dtype": spec.dtype, "axes": spec.axes, "description": spec.schema.description,
            "value_unit": spec.schema.value_unit, "attributes": dict(spec.schema.attributes),
        },
        "layout": {
            "layout_id": spec.layout_id, "shape": spec.shape,
            "axis_labels": dict(spec.layout.axis_labels),
        },
    }
    if isinstance(spec, FeatureSpec):
        payload.update(time_axis=spec.time_axis, sample_period_ns=spec.sample_period_ns,
                       window_alignment=spec.window_alignment)
    return payload


def validate_madduri_artifact(artifact: ModelArtifact) -> None:
    if artifact.model_type != MODEL_TYPE:
        raise ArtifactValidationError(f"expected model_type {MODEL_TYPE!r}")
    if artifact.implementation_version != IMPLEMENTATION_VERSION:
        raise ArtifactValidationError("Artifact implementation_version does not match this implementation")
    if artifact.feedback_spec is not None:
        raise ArtifactValidationError("Madduri inference uses no feedback")
    fit_intercept = False
    try:
        validate_decoder_specs(artifact.feature_spec, artifact.action_spec)
        runtime = artifact.runtime_config
        if not {"state_update_period_ns", "required_feature_period_ns", "max_speed", "flatten_order",
                "missing_feature_policy"} <= runtime.keys():
            raise ValueError("runtime config is incomplete")
        period = positive_integer(runtime["state_update_period_ns"], name="state_update_period_ns")
        positive_integer(runtime["required_feature_period_ns"], name="required_feature_period_ns")
        if runtime["required_feature_period_ns"] != period:
            raise ValueError("required_feature_period_ns must equal state_update_period_ns")
        if runtime["max_speed"] is not None:
            finite_real(runtime["max_speed"], name="max_speed", positive=True)
        if runtime["flatten_order"] != "C" or runtime["missing_feature_policy"] != "reject":
            raise ValueError("runtime requires C flattening and reject missing-feature policy")
        training = dict(artifact.training_config)
        mode = training.pop("mode", None)
        if mode == "fit":
            MadduriFitConfig(**training)
            if artifact.provenance.parent_model_version is not None:
                raise ValueError("seed Artifact must have no parent")
            if artifact.provenance.sample_count != 0 or artifact.provenance.sequence_count != 0:
                raise ValueError("random seed must record zero consumed samples/sequences")
        elif mode == "adapt":
            config = MadduriAdaptConfig(**training)
            fit_intercept = config.fit_intercept
            if not artifact.provenance.parent_model_version or artifact.provenance.parent_model_version == artifact.model_version:
                raise ValueError("adapted Artifact requires a distinct parent")
            if artifact.provenance.sample_count < config.minimum_samples:
                raise ValueError("adapted sample_count is below minimum_samples")
            if not 1 <= artifact.provenance.sequence_count <= artifact.provenance.sample_count:
                raise ValueError("adapted sequence_count is invalid")
        else:
            raise ValueError("training mode must be fit or adapt")
        if artifact.provenance.training_config_hash != json_sha256(artifact.training_config):
            raise ValueError("training_config_hash does not match training_config")
        if artifact.provenance.training_code_version != artifact.implementation_version:
            raise ValueError("training_code_version does not match implementation_version")
        if not artifact.provenance.source_ids or any(
                not isinstance(value, str) or not value.strip() for value in artifact.provenance.source_ids
        ):
            raise ValueError("Artifact source_ids must be non-empty")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtifactValidationError(str(exc)) from exc
    validate_madduri_parameters(
        artifact.parameters, feature_count=int(np.prod(artifact.feature_spec.shape)),
        action_dimension=artifact.action_spec.shape[0],
    )
    if not fit_intercept and np.any(artifact.parameters["b"] != 0):
        raise ParameterValidationError("no-intercept model requires exactly zero b")
