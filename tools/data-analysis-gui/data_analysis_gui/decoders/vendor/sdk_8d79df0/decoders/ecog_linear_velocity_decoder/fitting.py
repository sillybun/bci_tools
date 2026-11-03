# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: 8d79df0475b85ca859a02a8268f5262358a22afa
# original path: src/decoders/ecog_linear_velocity_decoder/fitting.py
# original git blob: a08e4ebd9e44cd99d605ba3ca8cc68a480d90be5
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Numerical helpers and complete Artifact validation."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np

from ...bci_decoder_api import ModelArtifact
from ...bci_decoder_api.errors import ArtifactValidationError, ParameterValidationError

from .data_types import (
    CURSOR_VELOCITY_COMPONENTS,
    CURSOR_VELOCITY_SCHEMA_ID,
    ECOG_BAND_LABELS,
    ECOG_FEATURE_SCHEMA_ID,
)


MODEL_TYPE = "ecog_linear_velocity"
IMPLEMENTATION_VERSION = "1.0.0"
ACTION_DIMENSION = 2
DEFAULT_TRACKING_TIME_CONSTANT_MS = 60.0
# Nominal ECoG feature frame period (20 ms). Used as the fixed within-trial dt
# for target-velocity differencing, so out-of-order or duplicate timestamps
# cannot affect fitting.
DEFAULT_SAMPLE_PERIOD_NS = 20_000_000


def limit_speed(velocity: np.ndarray, maximum_speed: float) -> np.ndarray:
    values = np.asarray(velocity, dtype=np.float64)
    if values.shape[-1] != ACTION_DIMENSION:
        raise ValueError("velocity must end in two components")
    if (
        isinstance(maximum_speed, bool)
        or not isinstance(maximum_speed, (int, float))
        or not np.isfinite(maximum_speed)
        or maximum_speed <= 0.0
    ):
        raise ValueError("maximum_speed must be finite and positive")
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values * np.minimum(1.0, maximum_speed / np.maximum(norm, 1e-12))


def trial_equal_sample_weights(trial_ids: np.ndarray) -> np.ndarray:
    trials = np.asarray(trial_ids)
    if trials.ndim != 1 or trials.size == 0:
        raise ValueError("trial_ids must be a non-empty vector")
    unique, inverse, counts = np.unique(trials, return_inverse=True, return_counts=True)
    return trials.size / (unique.size * counts[inverse].astype(np.float64))


def trial_local_target_velocity(
    target_position: np.ndarray,
    timestamps_ns: np.ndarray,
    trial_ids: np.ndarray,
) -> np.ndarray:
    """Causal backward target-position difference, reset at every trial.

    The per-sample dt is fixed at the nominal frame period
    (``DEFAULT_SAMPLE_PERIOD_NS``) rather than derived from ``timestamps_ns``,
    so out-of-order or duplicate timestamps never affect the result.
    """

    target = np.asarray(target_position, dtype=np.float64)
    timestamps = np.asarray(timestamps_ns)
    trials = np.asarray(trial_ids)
    if target.ndim != 2 or target.shape[1] != ACTION_DIMENSION:
        raise ValueError("target_position must have shape (sample, 2)")
    if timestamps.shape != (target.shape[0],) or trials.shape != (target.shape[0],):
        raise ValueError("timestamps_ns/trial_ids must align with target_position")
    dt_seconds = DEFAULT_SAMPLE_PERIOD_NS / 1e9
    velocity = np.zeros_like(target, dtype=np.float64)
    for index in range(1, target.shape[0]):
        if trials[index] != trials[index - 1]:
            continue
        velocity[index] = (target[index] - target[index - 1]) / dt_seconds
    return velocity


def desired_tracking_velocity(
    target_position: np.ndarray,
    cursor_position: np.ndarray,
    timestamps_ns: np.ndarray,
    trial_ids: np.ndarray,
    *,
    tracking_time_constant_ms: float = DEFAULT_TRACKING_TIME_CONSTANT_MS,
    max_speed: float,
) -> np.ndarray:
    """Target feedforward plus proportional position-error correction."""

    target = np.asarray(target_position, dtype=np.float64)
    cursor = np.asarray(cursor_position, dtype=np.float64)
    if cursor.shape != target.shape:
        raise ValueError("cursor_position must match target_position")
    time_constant_seconds = float(tracking_time_constant_ms) / 1_000.0
    if not np.isfinite(time_constant_seconds) or time_constant_seconds <= 0.0:
        raise ValueError("tracking_time_constant_ms must be finite and positive")
    target_velocity = trial_local_target_velocity(target, timestamps_ns, trial_ids)
    desired = target_velocity + (target - cursor) / time_constant_seconds
    return limit_speed(desired, max_speed)


def fit_trial_balanced_ridge(
    features: np.ndarray,
    target: np.ndarray,
    trial_ids: np.ndarray,
    *,
    penalty: float,
    fit_intercept: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Ridge with total sample weight N and equal total weight per trial."""

    design = np.asarray(features, dtype=np.float64)
    desired = np.asarray(target, dtype=np.float64)
    if design.ndim != 2 or desired.shape != (design.shape[0], ACTION_DIMENSION):
        raise ValueError("features/target have incompatible shapes")
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(desired)):
        raise ValueError("features/target must contain only finite values")
    if (
        isinstance(penalty, bool)
        or not isinstance(penalty, (int, float))
        or not np.isfinite(penalty)
        or penalty < 0.0
    ):
        raise ValueError("penalty must be finite and non-negative")
    if not isinstance(fit_intercept, bool):
        raise ValueError("fit_intercept must be a bool")
    weights = trial_equal_sample_weights(trial_ids)
    if weights.shape != (design.shape[0],):
        raise ValueError("trial_ids must have one value per sample")
    if fit_intercept:
        feature_mean = np.average(design, axis=0, weights=weights)
        target_mean = np.average(desired, axis=0, weights=weights)
        x = design - feature_mean
        y = desired - target_mean
    else:
        feature_mean = np.zeros(design.shape[1])
        target_mean = np.zeros(ACTION_DIMENSION)
        x, y = design, desired
    sqrt_w = np.sqrt(weights)[:, None]
    xw, yw = x * sqrt_w, y * sqrt_w
    gram = xw.T @ xw + float(penalty) * np.eye(x.shape[1])
    cross = xw.T @ yw
    try:
        coefficients = np.linalg.solve(gram, cross)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram) @ cross
    decoder = coefficients.T
    intercept = target_mean - decoder @ feature_mean
    if not fit_intercept:
        intercept = np.zeros(ACTION_DIMENSION, dtype=np.float64)
    return decoder, intercept


def smoothbatch(
    prior_weights: np.ndarray,
    prior_intercept: np.ndarray,
    optimal_weights: np.ndarray,
    optimal_intercept: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend a supervised optimum onto its parent; alpha weights the parent."""

    value = float(alpha)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("alpha must lie in [0, 1)")
    return (
        value * np.asarray(prior_weights) + (1.0 - value) * np.asarray(optimal_weights),
        value * np.asarray(prior_intercept)
        + (1.0 - value) * np.asarray(optimal_intercept),
    )


def parameter_fingerprint(
    parameters: Mapping[str, np.ndarray], *, payload: Mapping[str, object]
) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )
    for name, array in sorted(parameters.items()):
        value = np.ascontiguousarray(array)
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()[:12]


def validate_ecog_linear_artifact(artifact: ModelArtifact) -> None:
    if artifact.model_type != MODEL_TYPE:
        raise ArtifactValidationError(f"expected model_type {MODEL_TYPE!r}")
    if (
        artifact.implementation_version.split(".", 1)[0]
        != IMPLEMENTATION_VERSION.split(".", 1)[0]
    ):
        raise ArtifactValidationError("incompatible implementation major version")
    if artifact.feedback_spec is not None:
        raise ArtifactValidationError("ECoG linear velocity inference uses no feedback")
    if artifact.feature_spec.schema_id != ECOG_FEATURE_SCHEMA_ID:
        raise ArtifactValidationError("unexpected ECoG feature schema")
    if artifact.feature_spec.rank != 2 or artifact.feature_spec.axes != ("channel", "band"):
        raise ArtifactValidationError("ECoG features must have (channel, band) rank-2 layout")
    if artifact.feature_spec.shape[1] != len(ECOG_BAND_LABELS):
        raise ArtifactValidationError("ECoG feature layout must contain six bands")
    if artifact.feature_spec.layout.axis_labels.get("band") != ECOG_BAND_LABELS:
        raise ArtifactValidationError("ECoG band labels/order do not match the schema")
    if artifact.action_spec.schema_id != CURSOR_VELOCITY_SCHEMA_ID:
        raise ArtifactValidationError("unexpected cursor velocity action schema")
    if artifact.action_spec.rank != 1 or artifact.action_spec.shape != (2,):
        raise ArtifactValidationError("action must be a two-dimensional velocity")
    if (
        artifact.action_spec.layout.axis_labels.get("component")
        != CURSOR_VELOCITY_COMPONENTS
    ):
        raise ArtifactValidationError("action component order must be (x, y)")
    feature_shape = artifact.feature_spec.shape
    feature_count = int(np.prod(feature_shape))
    expected = {
        "W": (2, feature_count),
        "b": (2,),
        "feature_mean": feature_shape,
        "feature_scale": feature_shape,
    }
    if set(artifact.parameters) != set(expected):
        raise ParameterValidationError("parameter names are incomplete or unexpected")
    for name, shape in expected.items():
        value = artifact.parameters[name]
        if value.shape != shape or not np.issubdtype(value.dtype, np.floating):
            raise ParameterValidationError(f"parameter {name} has wrong shape or dtype")
        if not np.all(np.isfinite(value)):
            raise ParameterValidationError(f"parameter {name} contains non-finite values")
    if np.any(artifact.parameters["feature_scale"] <= 0.0):
        raise ParameterValidationError("feature_scale must be strictly positive")
    max_speed = artifact.runtime_config.get("max_speed")
    if isinstance(max_speed, bool) or not isinstance(max_speed, (int, float)):
        raise ArtifactValidationError("runtime max_speed must be a real number")
    if not np.isfinite(float(max_speed)) or float(max_speed) <= 0.0:
        raise ArtifactValidationError("runtime max_speed must be finite and positive")
    count = artifact.training_config.get("supervised_update_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ArtifactValidationError("supervised_update_count must be a non-negative integer")
    if artifact.training_config.get("normalization") != "rest_mean_std_frozen":
        raise ArtifactValidationError("normalization contract is missing")
    if artifact.training_config.get("rest_as_zero_velocity_training") is not False:
        raise ArtifactValidationError("rest must not be used as zero-velocity training data")
    if artifact.training_config.get("trial_weighting") != "equal_total_weight":
        raise ArtifactValidationError("trial weighting contract is missing")
    if (
        artifact.training_config.get("target_velocity_estimator")
        != "trial_local_backward_difference"
    ):
        raise ArtifactValidationError("target-velocity estimator contract is missing")
    if artifact.training_config.get("feature_to_label_offset_ns") != 0:
        raise ArtifactValidationError("feature_to_label_offset_ns must be zero")
    stage = artifact.training_config.get("stage")
    if stage not in ("calibration_seed", "supervised_adapt"):
        raise ArtifactValidationError("artifact stage is invalid")
    if (count == 0) != (stage == "calibration_seed"):
        raise ArtifactValidationError("artifact stage/update count are inconsistent")
    if count == 0:
        if np.any(artifact.parameters["W"] != 0.0) or np.any(
            artifact.parameters["b"] != 0.0
        ):
            raise ParameterValidationError("calibration seed must have W=b=0")
    else:
        for name in (
            "ridge_penalty",
            "smoothbatch_alpha",
            "tracking_time_constant_ms",
            "fit_intercept",
        ):
            if name not in artifact.training_config:
                raise ArtifactValidationError(f"training config is missing {name}")
        penalty = artifact.training_config["ridge_penalty"]
        alpha = artifact.training_config["smoothbatch_alpha"]
        first_alpha = artifact.training_config.get("first_smoothbatch_alpha", 0.0)
        tc_ms = artifact.training_config["tracking_time_constant_ms"]
        fit_intercept = artifact.training_config["fit_intercept"]
        if isinstance(penalty, bool) or not isinstance(penalty, (int, float)):
            raise ArtifactValidationError("ridge_penalty must be a real number")
        if not np.isfinite(float(penalty)) or float(penalty) < 0.0:
            raise ArtifactValidationError("ridge_penalty must be finite and non-negative")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ArtifactValidationError("smoothbatch_alpha must be a real number")
        if not np.isfinite(float(alpha)) or not 0.0 <= float(alpha) < 1.0:
            raise ArtifactValidationError("smoothbatch_alpha must lie in [0, 1)")
        if isinstance(first_alpha, bool) or not isinstance(first_alpha, (int, float)):
            raise ArtifactValidationError(
                "first_smoothbatch_alpha must be a real number"
            )
        if not np.isfinite(float(first_alpha)) or not 0.0 <= float(first_alpha) < 1.0:
            raise ArtifactValidationError(
                "first_smoothbatch_alpha must lie in [0, 1)"
            )
        if isinstance(tc_ms, bool) or not isinstance(tc_ms, (int, float)):
            raise ArtifactValidationError("tracking_time_constant_ms must be a real number")
        if not np.isfinite(float(tc_ms)) or float(tc_ms) <= 0.0:
            raise ArtifactValidationError(
                "tracking_time_constant_ms must be finite and positive"
            )
        if not isinstance(fit_intercept, bool):
            raise ArtifactValidationError("fit_intercept must be a bool")
        if not fit_intercept and np.any(artifact.parameters["b"] != 0.0):
            raise ParameterValidationError("no-intercept artifacts must have b=0")
