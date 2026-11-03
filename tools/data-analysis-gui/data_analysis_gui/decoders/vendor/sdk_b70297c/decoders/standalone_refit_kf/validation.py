# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/decoders/standalone_refit_kf/validation.py
# original git blob: c68af0df826412f01c5840f73e1efb9f34e0e420
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""ReFIT-KF validation, grouped by lifecycle boundary.

Sections cover input specs and blocks, operation inputs, shared numerical rules,
Trainer settings and adaptation invariants, model parameters and Artifact
metadata, computed posterior state, and checkpoint restoration.

Rules only inspect data; they never transform inputs or commit state. Public
per-frame feature/feedback contracts remain the SDK Runtime's responsibility.
Example dataset preparation is validated under examples/standalone_refit_kf.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

import numpy as np

from ...bci_decoder_api import (
    ActionSpec, CoreRuntimeState, FeatureSpec, FeedbackSpec, ModelArtifact, json_sha256,
)
from ...bci_decoder_api.errors import (
    ArtifactValidationError, InsufficientDataError, ParameterValidationError,
    RuntimeStateError, TrainingDataError, TrainingNumericalError, TrainingSchemaError,
)

from .data_types import (
    STATE_UPDATE_PERIOD_NS, CalibrationBlock, RefitKFFitData, RefitKFTrackingBlock,
)
from .config import COVARIANCE_FLOOR_OPTION_NAMES, REGULARIZATION_NAMES
from .fitting import IMPLEMENTATION_VERSION, MODEL_TYPE, RIDGE_LOSS
from .utils import (
    valid_transition_pairs,
    report_frame_period_mismatch, timestamps_as_int64, validate_trial_timestamps,
)

STRUCTURE_ATOL = 1e-10


# ---------------------------------------------------------------------------
# Input specs and calibration/tracking blocks
# ---------------------------------------------------------------------------


def validate_decoder_specs(
        feature_spec: FeatureSpec,
        action_spec: ActionSpec,
        feedback_spec: FeedbackSpec | None,
) -> None:
    """Require matching velocity/position vectors; no fixed coordinate labels."""
    if feature_spec.rank < 1:
        raise ValueError("features must have at least one axis")
    if action_spec.rank != 1:
        raise ValueError("velocity action must be a vector")
    if feedback_spec is None or feedback_spec.shape != action_spec.shape:
        raise ValueError("position feedback and velocity action dimensions must match")


class TrackingValidation(NamedTuple):
    """Per-operation summary, never a persistent validation certificate."""

    sample_count: int
    trial_count: int
    velocity_valid: np.ndarray
    transition_count: int


class RefitKFInputValidator:
    """Validate calibration and tracking blocks against a deployment contract."""

    def __init__(self, feature_spec: FeatureSpec, action_dimension: int = 2) -> None:
        if isinstance(action_dimension, bool) or not isinstance(action_dimension, int) or action_dimension < 1:
            raise ValueError("action_dimension must be a positive integer")
        self.feature_spec = feature_spec
        self.action_dimension = action_dimension

    @staticmethod
    def _validate_source_ids(source_ids: tuple[str, ...]) -> None:
        if not isinstance(source_ids, tuple) or not source_ids or any(
                not isinstance(value, str) or not value for value in source_ids
        ):
            raise TrainingDataError(
                "source_ids must be a non-empty tuple of non-empty strings"
            )

    @staticmethod
    def _timestamps_as_int64(timestamps_ns: np.ndarray) -> np.ndarray:
        if not isinstance(timestamps_ns, np.ndarray):
            raise TrainingDataError("timestamps_ns must be a one-dimensional array")
        try:
            return timestamps_as_int64(timestamps_ns)
        except ValueError as exc:
            raise TrainingDataError(str(exc)) from exc

    def _validate_feature_rows(
            self,
            features: np.ndarray,
            timestamps_ns: np.ndarray,
            *,
            schema_id: str,
            layout_id: str,
            minimum_samples: int,
    ) -> int:
        if not isinstance(features, np.ndarray) or features.ndim != self.feature_spec.rank + 1:
            raise TrainingSchemaError(
                "features must have shape (sample, *feature_spec.shape)"
            )
        sample_count = int(features.shape[0])
        if sample_count < minimum_samples:
            raise InsufficientDataError(
                f"need at least {minimum_samples} rows, got {sample_count}"
            )
        if (
                schema_id != self.feature_spec.schema_id
                or layout_id != self.feature_spec.layout_id
        ):
            raise TrainingSchemaError("feature schema/layout does not match Trainer")
        if features.shape[1:] != self.feature_spec.shape:
            raise TrainingSchemaError("feature tensor shape does not match FeatureSpec")
        if features.dtype != np.dtype("float32"):
            raise TrainingSchemaError("features must use float32")
        if not np.all(np.isfinite(features)):
            raise TrainingDataError("features must contain only finite values")
        timestamps = self._timestamps_as_int64(timestamps_ns)
        if timestamps.shape != (sample_count,):
            raise TrainingDataError("timestamps_ns must have one value per sample")
        return sample_count

    @staticmethod
    def _validate_trials(
            trial_ids: np.ndarray, sample_count: int
    ) -> int:
        if not isinstance(trial_ids, np.ndarray) or trial_ids.shape != (sample_count,):
            raise TrainingDataError("trial_ids must have one value per sample")
        if not np.issubdtype(trial_ids.dtype, np.integer):
            raise TrainingDataError("trial_ids must use an integer dtype")
        previous: int | None = None
        counts: dict[int, int] = {}
        for raw in trial_ids:
            trial = int(raw)
            if trial != previous:
                if trial in counts:
                    raise TrainingDataError("each trial must be one contiguous segment")
                previous = trial
            counts[trial] = counts.get(trial, 0) + 1
        if len(counts) < 2:
            raise InsufficientDataError("at least two trials are required")
        if any(count < 3 for count in counts.values()):
            raise InsufficientDataError("each trial requires at least three rows")
        return len(counts)

    def validate_calibration(
            self, data: CalibrationBlock, *, minimum_samples: int = 2
    ) -> int:
        if not isinstance(data, CalibrationBlock):
            raise TrainingDataError("calibration must be CalibrationBlock")
        count = self._validate_feature_rows(
            data.features,
            data.timestamps_ns,
            schema_id=data.feature_schema_id,
            layout_id=data.feature_layout_id,
            minimum_samples=minimum_samples,
        )
        self._validate_source_ids(data.source_ids)
        report_frame_period_mismatch(
            np.diff(data.timestamps_ns.astype(np.int64)), context="calibration timestamps_ns",
        )
        return count

    def validate_tracking(
            self, data: RefitKFTrackingBlock, *, minimum_samples: int
    ) -> TrackingValidation:
        if not isinstance(data, RefitKFTrackingBlock):
            raise TrainingDataError("tracking data must be RefitKFTrackingBlock")
        sample_count = self._validate_feature_rows(
            data.features,
            data.timestamps_ns,
            schema_id=data.feature_schema_id,
            layout_id=data.feature_layout_id,
            minimum_samples=minimum_samples,
        )
        trial_count = self._validate_trials(data.trial_ids, sample_count)
        expected = (sample_count, self.action_dimension)
        for name, values in (
                ("target_position", data.target_position),
                ("cursor_position", data.cursor_position),
        ):
            if not isinstance(values, np.ndarray) or values.shape != expected:
                raise TrainingDataError(f"{name} must have shape (sample, {self.action_dimension})")
            if (
                    not np.issubdtype(values.dtype, np.number)
                    or not np.isrealobj(values)
                    or not np.all(np.isfinite(values))
            ):
                raise TrainingDataError(
                    f"{name} must contain finite real numeric values"
                )
        self._validate_source_ids(data.source_ids)
        valid = np.r_[False, data.trial_ids[1:] == data.trial_ids[:-1]]
        report_frame_period_mismatch(
            np.diff(data.timestamps_ns.astype(np.int64))[valid[1:]],
            context="within-trial timestamps_ns",
        )
        valid_count = int(np.count_nonzero(valid))
        if valid_count < minimum_samples:
            raise InsufficientDataError(
                f"need at least {minimum_samples} velocity-valid rows, got {valid_count}"
            )
        previous, _ = valid_transition_pairs(data.trial_ids, valid)
        if previous.size < 2:
            raise InsufficientDataError("need at least two valid velocity transitions")
        return TrackingValidation(sample_count, trial_count, valid, int(previous.size))


# ---------------------------------------------------------------------------
# Trainer operation inputs
# ---------------------------------------------------------------------------


def validate_refit_kf_fit_inputs(
        data: RefitKFFitData, validator: RefitKFInputValidator, *, minimum_samples: int,
) -> TrackingValidation | None:
    if not isinstance(data, RefitKFFitData):
        raise TrainingDataError("fit() requires RefitKFFitData")
    validator.validate_calibration(data.calibration)
    if data.initial_tracking is not None:
        return validator.validate_tracking(data.initial_tracking, minimum_samples=minimum_samples)
    return None


def validate_refit_kf_evaluation_inputs(
        data: RefitKFTrackingBlock, validator: RefitKFInputValidator,
) -> TrackingValidation:
    if not isinstance(data, RefitKFTrackingBlock):
        raise TrainingDataError("evaluate() requires RefitKFTrackingBlock")
    return validator.validate_tracking(data, minimum_samples=4)


# ---------------------------------------------------------------------------
# Batch-inference inputs
# ---------------------------------------------------------------------------


def validate_refit_kf_inference_inputs(artifact: ModelArtifact, data: RefitKFTrackingBlock) -> None:
    """Check batch identity and global timestamp order across trial resets.

    Individual feature and feedback frames are checked by SDK Runtime.
    """
    if (data.feature_schema_id, data.feature_layout_id) != (
        artifact.feature_spec.schema_id, artifact.feature_spec.layout_id,
    ):
        raise ValueError("inference data schema/layout does not match the artifact")
    validate_trial_timestamps(data.timestamps_ns, data.trial_ids)


# ---------------------------------------------------------------------------
# Shared numerical and configuration rules
# ---------------------------------------------------------------------------


class _CovarianceValidationError(ValueError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _validate_covariance(
        covariance: np.ndarray, tolerance: float, *, positive_definite: bool = False,
) -> float | None:
    """Shared finite/symmetric/PSD rule; return the minimum eigenvalue for PSD.

    Callers own shape/dtype checks and translate failures to their boundary's
    error type. Q uses Cholesky instead of a redundant eigenvalue calculation.
    """
    if not np.all(np.isfinite(covariance)):
        raise _CovarianceValidationError("non_finite", "contains non-finite values")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=tolerance):
        raise _CovarianceValidationError("not_symmetric", "must be symmetric")
    if positive_definite:
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise _CovarianceValidationError("not_pd", "must be positive definite") from exc
        return None
    eigenvalues = np.linalg.eigvalsh(covariance)
    minimum = float(np.min(eigenvalues))
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if minimum < -tolerance * scale:
        raise _CovarianceValidationError("not_psd", "must be positive semidefinite")
    return minimum


def _real_config_value(
        config: Mapping, name: str, *, context: str, nonnegative: bool = False,
) -> float:
    raw = config.get(name)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ArtifactValidationError(f"{context} {name} must be a real number")
    value = float(raw)
    if not np.isfinite(value) or (value < 0.0 if nonnegative else value <= 0.0):
        bound = "non-negative" if nonnegative else "positive"
        raise ArtifactValidationError(f"{context} {name} must be finite and {bound}")
    return value


def _positive_integer(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArtifactValidationError(f"{name} must be positive")


# ---------------------------------------------------------------------------
# Trainer configuration
# ---------------------------------------------------------------------------


def validate_refit_kf_trainer_settings(
        *, feature_spec: FeatureSpec, action_spec: ActionSpec, feedback_spec: FeedbackSpec,
        max_speed: float, maximum_feedback_age_ns: int, covariance_tolerance: float,
) -> None:
    validate_decoder_specs(feature_spec, action_spec, feedback_spec)
    settings = {"max_speed": max_speed, "covariance_tolerance": covariance_tolerance}
    for name in settings:
        try:
            _real_config_value(settings, name, context="runtime")
        except ArtifactValidationError as exc:
            raise ValueError(f"{name} must be finite and positive") from exc
    try:
        _positive_integer(maximum_feedback_age_ns, name="maximum_feedback_age_ns")
    except ArtifactValidationError as exc:
        raise ValueError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Model compatibility and adaptation invariants
# ---------------------------------------------------------------------------


def validate_refit_kf_trainer_contract(
        artifact: ModelArtifact, *, feature_spec: FeatureSpec, action_spec: ActionSpec,
        feedback_spec: FeedbackSpec, max_speed: float, maximum_feedback_age_ns: int,
        covariance_tolerance: float,
) -> None:
    """Compare settings after full model validation; do not rescan parameters."""
    if (
            artifact.feature_spec != feature_spec or artifact.action_spec != action_spec
            or artifact.feedback_spec != feedback_spec
    ):
        raise TrainingSchemaError("Artifact tensor specs do not match this Trainer")
    runtime = artifact.runtime_config
    if (
            float(runtime["max_speed"]) != max_speed
            or int(runtime["maximum_feedback_age_ns"]) != maximum_feedback_age_ns
            or float(runtime["covariance_tolerance"]) != covariance_tolerance
    ):
        raise TrainingSchemaError("Artifact Runtime configuration does not match Trainer")


def validate_refit_kf_frozen_normalization(
        parameters: Mapping[str, np.ndarray], parent: ModelArtifact,
) -> None:
    for name in ("feature_mean", "feature_scale"):
        if name not in parameters:
            raise TrainingNumericalError(f"candidate is missing frozen normalization parameter {name}")
        if not np.array_equal(parameters[name], parent.parameters[name]):
            raise TrainingNumericalError(f"candidate {name} differs from frozen parent normalization")


# ---------------------------------------------------------------------------
# Produced model parameters
# ---------------------------------------------------------------------------


def validate_refit_kf_parameters(
        parameters: Mapping[str, np.ndarray],
        *,
        feature_shape: tuple[int, ...],
        action_dimension: int = 2,
        state_update_period_ns: int,
        covariance_tolerance: float,
) -> None:
    state_dimension = 2 * action_dimension + 1
    velocity_slice = slice(action_dimension, 2 * action_dimension)
    feature_count = int(np.prod(feature_shape))
    expected = {
        "A": (state_dimension, state_dimension),
        "C": (feature_count, state_dimension),
        "W": (state_dimension, state_dimension),
        "Q": (feature_count, feature_count),
        "x0": (state_dimension,),
        "P0": (state_dimension, state_dimension),
        "feature_mean": feature_shape,
        "feature_scale": feature_shape,
    }
    if set(parameters) != set(expected):
        raise ParameterValidationError("parameter names are incomplete or unexpected")
    for name, shape in expected.items():
        value = parameters[name]
        if not isinstance(value, np.ndarray) or value.shape != shape or not np.issubdtype(value.dtype, np.floating):
            raise ParameterValidationError(f"parameter {name} has wrong shape or dtype")
        if name not in ("W", "P0", "Q") and not np.all(np.isfinite(value)):
            raise ParameterValidationError(f"parameter {name} contains non-finite values")
    if np.any(parameters["feature_scale"] <= 0.0):
        raise ParameterValidationError("feature_scale must be strictly positive")

    A, W, Q, x0, P0 = (
        parameters[name] for name in ("A", "W", "Q", "x0", "P0")
    )
    dt = state_update_period_ns / 1e9
    expected_A = np.zeros_like(A)
    expected_A[:action_dimension, :action_dimension] = np.eye(action_dimension)
    expected_A[:action_dimension, velocity_slice] = dt * np.eye(action_dimension)
    expected_A[-1, -1] = 1.0
    fixed_mask = np.ones_like(A, dtype=bool)
    fixed_mask[velocity_slice, velocity_slice] = False
    if not np.allclose(
            A[fixed_mask], expected_A[fixed_mask], rtol=0.0, atol=STRUCTURE_ATOL
    ):
        raise ParameterValidationError("A violates causal-position block structure")
    if float(np.max(np.abs(np.linalg.eigvals(A[velocity_slice, velocity_slice])))) >= 1.0:
        raise ParameterValidationError("A velocity dynamics are unstable: spectral radius must be < 1")
    structural = np.r_[np.arange(action_dimension), 2 * action_dimension]
    if not np.allclose(W[structural, :], 0.0, atol=STRUCTURE_ATOL) or not np.allclose(
            W[:, structural], 0.0, atol=STRUCTURE_ATOL
    ):
        raise ParameterValidationError("W may contain noise only in velocity block")
    if not np.allclose(
            P0[structural, :], 0.0, atol=STRUCTURE_ATOL
    ) or not np.allclose(P0[:, structural], 0.0, atol=STRUCTURE_ATOL):
        raise ParameterValidationError(
            "P0 may contain covariance only in velocity block"
        )
    if not np.allclose(x0[:-1], 0.0, atol=STRUCTURE_ATOL) or not np.isclose(
            x0[-1], 1.0, atol=STRUCTURE_ATOL
    ):
        raise ParameterValidationError("x0 must use zero position/velocity and bias one")
    for name, covariance in (("W", W), ("P0", P0), ("Q", Q)):
        try:
            _validate_covariance(covariance, covariance_tolerance, positive_definite=name == "Q")
        except _CovarianceValidationError as exc:
            raise ParameterValidationError(f"{name} {exc}") from exc


# ---------------------------------------------------------------------------
# Complete Artifact: model contract, runtime configuration and provenance
# ---------------------------------------------------------------------------


def validate_refit_kf_artifact(artifact: ModelArtifact) -> None:
    if artifact.model_type != MODEL_TYPE:
        raise ArtifactValidationError(f"expected model_type {MODEL_TYPE!r}")
    if artifact.implementation_version.split(".", 1)[0] != IMPLEMENTATION_VERSION.split(
            ".", 1
    )[0]:
        raise ArtifactValidationError("incompatible implementation major version")
    try:
        validate_decoder_specs(
            artifact.feature_spec, artifact.action_spec, artifact.feedback_spec
        )
    except ValueError as exc:
        raise ArtifactValidationError(str(exc)) from exc

    raw_period = artifact.runtime_config.get("state_update_period_ns")
    if (
            isinstance(raw_period, bool)
            or not isinstance(raw_period, int)
            or raw_period != STATE_UPDATE_PERIOD_NS
    ):
        raise ArtifactValidationError("state_update_period_ns must equal 20_000_000")
    required_period = artifact.runtime_config.get("required_feature_period_ns", STATE_UPDATE_PERIOD_NS)
    if (
            isinstance(required_period, bool)
            or not isinstance(required_period, int)
            or required_period != STATE_UPDATE_PERIOD_NS
    ):
        raise ArtifactValidationError("required_feature_period_ns must equal 20_000_000")
    raw_age = artifact.runtime_config.get("maximum_feedback_age_ns")
    _positive_integer(raw_age, name="maximum_feedback_age_ns")
    tolerance = _real_config_value(artifact.runtime_config, "covariance_tolerance", context="runtime")
    _real_config_value(artifact.runtime_config, "max_speed", context="runtime")
    if artifact.runtime_config.get("missing_feature_policy") != "reject":
        raise ArtifactValidationError("missing_feature_policy must be 'reject'")
    validate_refit_kf_parameters(
        artifact.parameters,
        feature_shape=artifact.feature_spec.shape,
        action_dimension=artifact.action_spec.shape[0],
        state_update_period_ns=raw_period,
        covariance_tolerance=tolerance,
    )

    config = artifact.training_config
    provenance = artifact.provenance
    if (
            not provenance.source_ids
            or any(
        not isinstance(source_id, str) or not source_id
        for source_id in provenance.source_ids
    )
            or provenance.sample_count <= 0
            or provenance.sequence_count <= 0
    ):
        raise ArtifactValidationError("training provenance data identity/counts are invalid")
    if provenance.training_code_version != artifact.implementation_version:
        raise ArtifactValidationError("training provenance code version is inconsistent")
    if provenance.training_config_hash != json_sha256(config):
        raise ArtifactValidationError("training provenance config hash is inconsistent")
    if config.get("state_update_period_ns") != STATE_UPDATE_PERIOD_NS:
        raise ArtifactValidationError(
            "training state_update_period_ns must equal 20_000_000"
        )
    if config.get("normalization") != "rest_mean_std_frozen":
        raise ArtifactValidationError("normalization contract is missing")
    if config.get("rest_as_zero_velocity_training") is not False:
        raise ArtifactValidationError("rest cannot be zero-velocity supervision")
    if config.get("velocity_estimator") not in (
            "trial_local_fixed_dt_backward_difference",  # Existing Artifacts.
            "trial_local_timestamp_backward_difference",
    ):
        raise ArtifactValidationError("velocity estimator contract is missing")
    if config.get("feature_to_label_offset_ns") != 0:
        raise ArtifactValidationError("feature_to_label_offset_ns must be zero")
    if "ridge_loss" in config and config["ridge_loss"] != RIDGE_LOSS:
        raise ArtifactValidationError(f"training ridge_loss must equal {RIDGE_LOSS!r}")
    if any(name in config for name in COVARIANCE_FLOOR_OPTION_NAMES):
        if not isinstance(config.get("auto_covariance_floor"), bool):
            raise ArtifactValidationError("training auto_covariance_floor must be a bool")
        for name in COVARIANCE_FLOOR_OPTION_NAMES[1:]:
            _real_config_value(config, name, context="training", nonnegative=True)
    for name in ("target_radius_px", "p0_velocity_regularization"):
        _real_config_value(config, name, context="training")
    if any(name in config for name in REGULARIZATION_NAMES) or "regularization" not in config:
        for name in REGULARIZATION_NAMES:
            _real_config_value(config, name, context="training")
    else:
        # Existing Artifacts recorded one shared value for all four operations.
        _real_config_value(config, "regularization", context="training")
    _real_config_value(config, "minimum_intended_speed_px_s", context="training", nonnegative=True)
    minimum_samples = config.get("minimum_samples")
    if isinstance(minimum_samples, bool) or not isinstance(minimum_samples, int) or minimum_samples < 4:
        raise ArtifactValidationError("training minimum_samples must be an integer of at least 4")
    if config.get("mode") == "fit":
        _real_config_value(config, "std_floor", context="training")
    count = config.get("supervised_update_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ArtifactValidationError("supervised_update_count must be non-negative")
    stage = config.get("stage")
    if stage not in ("calibration_seed", "initial_supervised", "refit_adapt"):
        raise ArtifactValidationError("training stage is invalid")
    if stage == "calibration_seed":
        if (
                config.get("mode") != "fit"
                or count != 0
                or artifact.provenance.parent_model_version is not None
        ):
            raise ArtifactValidationError("calibration seed lineage/count is invalid")
        params = artifact.parameters
        dimension = artifact.action_spec.shape[0]
        velocity_slice = slice(dimension, 2 * dimension)
        if (
                np.any(params["A"][velocity_slice, velocity_slice] != 0.0)
                or np.any(params["W"] != 0.0)
                or np.any(params["P0"] != 0.0)
                or np.any(params["C"] != 0.0)
                or not np.array_equal(params["Q"], np.eye(params["Q"].shape[0]))
        ):
            raise ParameterValidationError("calibration seed must be the trivial KF")
    elif stage == "initial_supervised":
        if (
                config.get("mode") != "fit"
                or count != 1
                or artifact.provenance.parent_model_version is not None
        ):
            raise ArtifactValidationError("initial-supervised lineage/count is invalid")
    else:
        alpha = config.get("blend_alpha")
        blend_applied = config.get("blend_applied")
        valid_alpha = alpha is None or (
                not isinstance(alpha, bool)
                and isinstance(alpha, (int, float))
                and np.isfinite(float(alpha))
                and 0.0 <= float(alpha) < 1.0
        )
        if (
                config.get("mode") != "adapt"
                or count < 1
                or artifact.provenance.parent_model_version is None
                or not valid_alpha
                or not isinstance(blend_applied, bool)
                or blend_applied != (alpha is not None and count > 1)
                or config.get("blend_semantics")
                != "alpha_parent_plus_one_minus_alpha_candidate"
        ):
            raise ArtifactValidationError("adapted lineage/count is invalid")

# ---------------------------------------------------------------------------
# Core posterior state
# ---------------------------------------------------------------------------


class RefitKFRuntimeValidationError(FloatingPointError):
    """A rejected computed state, with a stable inference reason code."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def validate_refit_kf_posterior(
        state: np.ndarray, covariance: np.ndarray, *, covariance_tolerance: float,
) -> float:
    """Check computed x/P before commit; return the covariance diagnostic."""
    if not np.all(np.isfinite(state)):
        raise RefitKFRuntimeValidationError(
            "standalone_refit_kf.posterior_state_non_finite",
            "posterior state contains non-finite values",
        )
    try:
        minimum = _validate_covariance(covariance, covariance_tolerance)
        assert minimum is not None
        return minimum
    except _CovarianceValidationError as exc:
        raise RefitKFRuntimeValidationError(
            f"standalone_refit_kf.posterior_covariance_{exc.reason}",
            f"posterior covariance {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Checkpoint identity and state
# ---------------------------------------------------------------------------


def validate_refit_kf_checkpoint_contract(
        state: CoreRuntimeState, *, state_schema_id: str,
) -> None:
    """Check checkpoint identity and fields before the Core reads its arrays."""
    if state.state_schema_id != state_schema_id:
        raise RuntimeStateError("checkpoint core-state schema is not supported")
    if state.counters:
        raise RuntimeStateError("checkpoint counters are unexpected")
    if set(state.arrays) != {"x", "P"}:
        raise RuntimeStateError("checkpoint arrays are incomplete or unexpected")


def validate_refit_kf_checkpoint_state(
        state: np.ndarray, covariance: np.ndarray, *,
        action_dimension: int, covariance_tolerance: float,
) -> None:
    """Check copied checkpoint arrays without converting or changing them."""
    state_dimension = 2 * action_dimension + 1
    if state.shape != (state_dimension,) or covariance.shape != (state_dimension, state_dimension):
        raise RuntimeStateError("checkpoint array shapes do not match Core")
    # Reuse posterior finite/PSD checks; checkpoint errors use the public
    # restoration error type rather than an online inference status.
    try:
        validate_refit_kf_posterior(state, covariance, covariance_tolerance=covariance_tolerance)
    except RefitKFRuntimeValidationError as exc:
        message = (
            "checkpoint contains non-finite state"
            if exc.reason_code.endswith("posterior_state_non_finite") else str(exc)
        )
        raise RuntimeStateError(message) from exc
    if not np.isclose(state[-1], 1.0, rtol=0.0, atol=covariance_tolerance):
        raise RuntimeStateError("checkpoint affine bias must equal one")
    if not np.allclose(
            covariance[:action_dimension, :], 0.0, atol=covariance_tolerance,
    ) or not np.allclose(covariance[:, :action_dimension], 0.0, atol=covariance_tolerance):
        raise RuntimeStateError("checkpoint position covariance must be zero")
    if not np.allclose(
            covariance[-1, :], 0.0, atol=covariance_tolerance,
    ) or not np.allclose(covariance[:, -1], 0.0, atol=covariance_tolerance):
        raise RuntimeStateError("checkpoint affine-bias covariance must be zero")
