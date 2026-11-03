# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/decoders/standalone_refit_kf/fitting.py
# original git blob: ae952294003f5d07c2f1e6d545ac32bd25dc3ed5
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Numerical fitting and parameter blending helpers for ReFIT-KF."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np

from .data_types import STATE_UPDATE_PERIOD_NS
from .utils import (
    limit_speed, moving_target_intended_velocity, trial_local_velocity, valid_transition_pairs,
)
MODEL_TYPE = "standalone_refit_kf"
IMPLEMENTATION_VERSION = "3.3.0"
RIDGE_LOSS = "mean_squared_error_per_sample"


def ridge(
        design: np.ndarray,
        target: np.ndarray,
        regularization: float,
        *,
        penalize_last_column: bool = True,
) -> np.ndarray:
    """Minimize squared error summed over outputs and averaged over samples, plus L2."""
    sample_count = design.shape[0]
    if sample_count == 0:
        raise ValueError("ridge requires at least one sample")
    penalty = np.eye(design.shape[1])
    if not penalize_last_column:
        penalty[-1, -1] = 0.0
    # Multiply the mean-loss normal equations by N to avoid rescaling the
    # data matrices. This is equivalent to X.T @ X / N + regularization * D.
    gram = design.T @ design + (sample_count * float(regularization)) * penalty
    return np.linalg.solve(gram, design.T @ target)


def velocity_transition(
    velocity: np.ndarray, trial_ids: np.ndarray, valid: np.ndarray,
    transition_ridge_regularization: float,
) -> np.ndarray:
    """Estimate the velocity block without changing or checking its eigenvalues."""
    previous, following = valid_transition_pairs(trial_ids, valid)
    return ridge(velocity[previous], velocity[following], transition_ridge_regularization).T


# Preserve the former private ridge alias.
_ridge = ridge


def _covariance(
        residual: np.ndarray, regularization: float, *,
        auto_floor: bool = False, relative_ratio: float = 1e-4,
) -> tuple[np.ndarray, dict[str, float]]:
    """Add absolute or residual-scaled diagonal loading and report its trace fraction."""
    covariance = residual.T @ residual / max(residual.shape[0], 1)
    covariance = 0.5 * (covariance + covariance.T)
    variance_scale = float(np.trace(covariance) / covariance.shape[0])
    floor = float(regularization)
    if auto_floor:
        floor = max(floor, float(relative_ratio) * variance_scale)
    covariance += floor * np.eye(covariance.shape[0])
    return covariance, {
        "residual_variance_mean": variance_scale,
        "floor_effective": floor,
        "floor_fraction": floor / (variance_scale + floor),
    }


def _trial_start_velocity_second_moment(
        velocity: np.ndarray,
        velocity_valid: np.ndarray,
        trial_ids: np.ndarray,
        p0_velocity_regularization: float,
) -> np.ndarray:
    """Estimate a zero-mean velocity prior from each trial's first valid row."""

    velocities = np.asarray(velocity, dtype=np.float64)
    valid = np.asarray(velocity_valid, dtype=bool)
    trials = np.asarray(trial_ids)
    first_valid: list[int] = []
    start = 0
    while start < trials.size:
        end = start + 1
        while end < trials.size and trials[end] == trials[start]:
            end += 1
        local_valid = np.flatnonzero(valid[start:end])
        if local_valid.size == 0:
            raise np.linalg.LinAlgError(
                "each trial requires at least one valid velocity for P0"
            )
        first_valid.append(start + int(local_valid[0]))
        start = end
    if not first_valid:
        raise np.linalg.LinAlgError("at least one trial is required to estimate P0")
    initial_velocity = velocities[np.asarray(first_valid, dtype=np.int64)]
    second_moment = initial_velocity.T @ initial_velocity / initial_velocity.shape[0]
    second_moment = 0.5 * (second_moment + second_moment.T)
    second_moment += float(p0_velocity_regularization) * np.eye(velocities.shape[1])
    return second_moment


def _structured_parameters(
        *,
        velocity_transition: np.ndarray,
        velocity_covariance: np.ndarray,
        initial_velocity_covariance: np.ndarray,
        observation: np.ndarray,
        observation_covariance: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        state_update_period_ns: int,
) -> dict[str, np.ndarray]:
    dt = state_update_period_ns / 1e9
    action_dimension = velocity_transition.shape[0]
    state_dimension = 2 * action_dimension + 1
    velocity_slice = slice(action_dimension, 2 * action_dimension)
    transition = np.zeros((state_dimension, state_dimension), dtype=np.float64)
    transition[:action_dimension, :action_dimension] = np.eye(action_dimension)
    transition[:action_dimension, velocity_slice] = dt * np.eye(action_dimension)
    transition[velocity_slice, velocity_slice] = velocity_transition
    transition[-1, -1] = 1.0
    process_covariance = np.zeros_like(transition)
    process_covariance[velocity_slice, velocity_slice] = velocity_covariance
    initial_mean = np.zeros(state_dimension, dtype=np.float64)
    initial_mean[-1] = 1.0
    initial_covariance = np.zeros_like(transition)
    initial_covariance[velocity_slice, velocity_slice] = initial_velocity_covariance
    return {
        "A": transition,
        "C": np.asarray(observation, dtype=np.float64),
        "W": process_covariance,
        "Q": np.asarray(observation_covariance, dtype=np.float64),
        "x0": initial_mean,
        "P0": initial_covariance,
        "feature_mean": np.asarray(feature_mean, dtype=np.float64),
        "feature_scale": np.asarray(feature_scale, dtype=np.float64),
    }


def fit_kalman_parameters(
        *,
        normalized_features: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        velocity_valid: np.ndarray,
        trial_ids: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        state_update_period_ns: int,
        transition_ridge_regularization: float,
        observation_ridge_regularization: float,
        process_covariance_floor_regularization: float,
        observation_covariance_floor_regularization: float,
        p0_velocity_regularization: float,
        auto_covariance_floor: bool = False,
        process_covariance_floor_relative_ratio: float = 1e-4,
        observation_covariance_floor_relative_ratio: float = 1e-4,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Fit structured causal-position KF parameters on valid velocity rows."""

    features = np.asarray(normalized_features, dtype=np.float64)
    positions = np.asarray(position, dtype=np.float64)
    velocities = np.asarray(velocity, dtype=np.float64)
    valid = np.asarray(velocity_valid, dtype=bool)
    trials = np.asarray(trial_ids)
    flattened = features.reshape(features.shape[0], -1, order="C")
    previous, following = valid_transition_pairs(trials, valid)

    fitted_transition = velocity_transition(
        velocities, trials, valid, transition_ridge_regularization
    )
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(fitted_transition))))
    # Report the fitted radius here; acceptance belongs to validation, which
    # checks the complete candidate before blending or Artifact publication.
    velocity_residual = (
            velocities[following] - velocities[previous] @ fitted_transition.T
    )
    velocity_covariance, process_metrics = _covariance(
        velocity_residual, process_covariance_floor_regularization,
        auto_floor=auto_covariance_floor,
        relative_ratio=process_covariance_floor_relative_ratio,
    )
    initial_velocity_covariance = _trial_start_velocity_second_moment(
        velocities, valid, trials, p0_velocity_regularization
    )

    states = np.concatenate(
        (positions, velocities, np.ones((positions.shape[0], 1))), axis=1
    )
    observation_coefficients = ridge(
        states[valid],
        flattened[valid],
        observation_ridge_regularization,
        penalize_last_column=False,
    )
    observation = observation_coefficients.T
    observation_residual = flattened[valid] - states[valid] @ observation.T
    observation_covariance, observation_metrics = _covariance(
        observation_residual, observation_covariance_floor_regularization,
        auto_floor=auto_covariance_floor,
        relative_ratio=observation_covariance_floor_relative_ratio,
    )
    parameters = _structured_parameters(
        velocity_transition=fitted_transition,
        velocity_covariance=velocity_covariance,
        initial_velocity_covariance=initial_velocity_covariance,
        observation=observation,
        observation_covariance=observation_covariance,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        state_update_period_ns=state_update_period_ns,
    )
    metrics = {
        "train.velocity_transition_rmse": float(
            np.sqrt(np.mean(velocity_residual ** 2))
        ),
        "train.velocity_spectral_radius": spectral_radius,
        "train.observation_rmse": float(np.sqrt(np.mean(observation_residual ** 2))),
        "train.valid_observation_count": float(np.count_nonzero(valid)),
        "train.valid_transition_count": float(previous.size),
        **{f"train.process_covariance.{name}": value for name, value in process_metrics.items()},
        **{f"train.observation_covariance.{name}": value for name, value in observation_metrics.items()},
        "train.initial_velocity_prior_trace": float(
            np.trace(initial_velocity_covariance)
        ),
        "train.initial_velocity_prior_max_eigenvalue": float(
            np.max(np.linalg.eigvalsh(initial_velocity_covariance))
        ),
    }
    return parameters, metrics


def make_trivial_parameters(
        *,
        feature_shape: tuple[int, ...],
        action_dimension: int = 2,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        state_update_period_ns: int,
) -> dict[str, np.ndarray]:
    feature_count = int(np.prod(feature_shape))
    return _structured_parameters(
        velocity_transition=np.zeros((action_dimension, action_dimension), dtype=np.float64),
        velocity_covariance=np.zeros((action_dimension, action_dimension), dtype=np.float64),
        initial_velocity_covariance=np.zeros((action_dimension, action_dimension), dtype=np.float64),
        observation=np.zeros((feature_count, 2 * action_dimension + 1), dtype=np.float64),
        observation_covariance=np.eye(feature_count, dtype=np.float64),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        state_update_period_ns=state_update_period_ns,
    )


def blend_kalman_parameters(
        parent: Mapping[str, np.ndarray],
        candidate: Mapping[str, np.ndarray],
        *,
        alpha: float,
        state_update_period_ns: int,
) -> dict[str, np.ndarray]:
    """Blend learned blocks with alpha weighting the parent."""

    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must be finite in [0, 1)")
    value = float(alpha)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("alpha must be finite in [0, 1)")
    if state_update_period_ns != STATE_UPDATE_PERIOD_NS:
        raise ValueError("state_update_period_ns must equal 20_000_000")
    action_dimension = (parent["A"].shape[0] - 1) // 2
    velocity_slice = slice(action_dimension, 2 * action_dimension)
    parent_av = parent["A"][velocity_slice, velocity_slice]
    candidate_av = candidate["A"][velocity_slice, velocity_slice]
    parent_wv = parent["W"][velocity_slice, velocity_slice]
    candidate_wv = candidate["W"][velocity_slice, velocity_slice]
    parent_pv0 = parent["P0"][velocity_slice, velocity_slice]
    candidate_pv0 = candidate["P0"][velocity_slice, velocity_slice]
    return _structured_parameters(
        velocity_transition=value * parent_av + (1.0 - value) * candidate_av,
        velocity_covariance=value * parent_wv + (1.0 - value) * candidate_wv,
        initial_velocity_covariance=(
                value * parent_pv0 + (1.0 - value) * candidate_pv0
        ),
        observation=value * parent["C"] + (1.0 - value) * candidate["C"],
        observation_covariance=(
                value * parent["Q"] + (1.0 - value) * candidate["Q"]
        ),
        feature_mean=np.array(parent["feature_mean"], copy=True),
        feature_scale=np.array(parent["feature_scale"], copy=True),
        state_update_period_ns=state_update_period_ns,
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
