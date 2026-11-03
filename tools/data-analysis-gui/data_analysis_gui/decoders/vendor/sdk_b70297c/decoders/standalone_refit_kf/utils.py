# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/decoders/standalone_refit_kf/utils.py
# original git blob: 907ad4cda8cd4aec20acb1e11dd58c4d0ce4c1b1
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Shared timestamp, motion and ReFIT intention helpers.

Keep fixed-step timing, trial-local velocity/transition calculations and
motion helpers here. Model fitting and lifecycle validation remain separate.
"""

from __future__ import annotations

import logging

import numpy as np

from .data_types import STATE_UPDATE_PERIOD_NS

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timestamps and fixed-step cadence
# ---------------------------------------------------------------------------


def report_frame_period_mismatch(intervals_ns: np.ndarray, *, context: str) -> None:
    """Log unexpected positive frame intervals without changing fixed-step processing."""
    mismatches = np.flatnonzero(intervals_ns != STATE_UPDATE_PERIOD_NS)
    if mismatches.size:
        _logger.error(
            "%s: %d frame interval(s) differ from 20 ms; first actual interval=%d ns, "
            "expected=20_000_000 ns. Continuing with one fixed 20 ms step per input frame.",
            context, mismatches.size, int(intervals_ns[mismatches[0]]),
        )


def timestamps_as_int64(timestamps_ns: np.ndarray) -> np.ndarray:
    """Validate a non-negative, strictly increasing timestamp vector in int64 range."""
    timestamps = np.asarray(timestamps_ns)
    if timestamps.ndim != 1:
        raise ValueError("timestamps_ns must be a one-dimensional array")
    if not np.issubdtype(timestamps.dtype, np.integer):
        raise ValueError("timestamps_ns must use an integer dtype")
    if timestamps.size and (
            np.any(timestamps < 0)
            or int(np.max(timestamps)) > np.iinfo(np.int64).max
    ):
        raise ValueError("timestamps_ns must be non-negative int64 values")
    timestamps = timestamps.astype(np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps_ns must be strictly increasing")
    return timestamps


def validate_trial_timestamps(timestamps_ns: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
    """Validate trial alignment/order and log within-trial cadence mismatches."""
    trials = np.asarray(trial_ids)
    timestamps = np.asarray(timestamps_ns)
    if trials.ndim != 1 or timestamps.shape != trials.shape:
        raise ValueError("timestamps_ns must align with trial_ids")
    timestamps = timestamps_as_int64(timestamps)
    intervals = np.diff(timestamps)
    same_trial = trials[1:] == trials[:-1]
    report_frame_period_mismatch(intervals[same_trial], context="within-trial timestamps_ns")
    return timestamps

# ---------------------------------------------------------------------------
# Trial-local velocity and state transitions
# ---------------------------------------------------------------------------


def trial_local_velocity(
        position: np.ndarray,
        trial_ids: np.ndarray,
        timestamps_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-20-ms backward velocity; report cadence mismatches and exclude trial-first rows."""

    raw_values = np.asarray(position)
    if not np.issubdtype(raw_values.dtype, np.number) or not np.isrealobj(raw_values):
        raise ValueError("position must contain real numeric values")
    values = raw_values.astype(np.float64)
    trials = np.asarray(trial_ids)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("position must have shape (sample, D), D >= 1")
    if trials.shape != (values.shape[0],):
        raise ValueError("trial_ids must align with position")
    if not np.all(np.isfinite(values)):
        raise ValueError("position must contain finite values")
    timestamps = np.asarray(timestamps_ns)
    if timestamps.shape != (values.shape[0],):
        raise ValueError("timestamps_ns must align with position")
    validate_trial_timestamps(timestamps, trials)
    valid = np.r_[False, trials[1:] == trials[:-1]] if trials.size else np.zeros(0, dtype=bool)
    return _trial_local_velocity_validated(values, valid), valid


def _trial_local_velocity_validated(position: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Compute velocities after input validation; ``valid`` excludes trial starts.

    Internal training callers reuse the mask from validation. The public
    trial_local_velocity entry point continues to validate untrusted arrays.
    """
    values = np.asarray(position, dtype=np.float64)
    velocity = np.zeros_like(values)
    indices = np.flatnonzero(valid)
    velocity[indices] = (values[indices] - values[indices - 1]) / (STATE_UPDATE_PERIOD_NS / 1e9)
    return velocity


def valid_transition_pairs(
        trial_ids: np.ndarray, velocity_valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    trials = np.asarray(trial_ids)
    valid = np.asarray(velocity_valid, dtype=bool)
    if trials.ndim != 1 or valid.shape != trials.shape:
        raise ValueError("trial_ids and velocity_valid must be aligned vectors")
    previous = np.arange(max(trials.size - 1, 0), dtype=np.int64)
    keep = (
            (trials[:-1] == trials[1:])
            & valid[:-1]
            & valid[1:]
    )
    return previous[keep], previous[keep] + 1

# ---------------------------------------------------------------------------
# ReFIT motion helpers
# ---------------------------------------------------------------------------


def limit_speed(velocity: np.ndarray, maximum_speed: float) -> np.ndarray:
    values = np.asarray(velocity, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("velocity must end in a nonempty component axis")
    if (
            isinstance(maximum_speed, bool)
            or not isinstance(maximum_speed, (int, float))
            or not np.isfinite(maximum_speed)
            or maximum_speed <= 0.0
    ):
        raise ValueError("maximum_speed must be finite and positive")
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values * np.minimum(1.0, float(maximum_speed) / np.maximum(norm, 1e-12))


def moving_target_intended_velocity(
        *,
        target_position: np.ndarray,
        cursor_position: np.ndarray,
        target_velocity: np.ndarray,
        actual_velocity: np.ndarray,
        velocity_valid: np.ndarray,
        target_radius_px: float,
        max_speed: float,
        minimum_intended_speed_px_s: float = 50.0,
) -> np.ndarray:
    """Rotate target-relative velocity with an outside-target startup floor."""

    target = np.asarray(target_position, dtype=np.float64)
    cursor = np.asarray(cursor_position, dtype=np.float64)
    target_v = np.asarray(target_velocity, dtype=np.float64)
    actual_v = np.asarray(actual_velocity, dtype=np.float64)
    valid = np.asarray(velocity_valid, dtype=bool)
    expected = target.shape
    if (
            target.ndim != 2
            or target.shape[1] < 1
            or cursor.shape != expected
            or target_v.shape != expected
            or actual_v.shape != expected
            or valid.shape != (target.shape[0],)
    ):
        raise ValueError("moving-target arrays have incompatible shapes")
    if (
            isinstance(target_radius_px, bool)
            or not isinstance(target_radius_px, (int, float))
            or not np.isfinite(target_radius_px)
            or target_radius_px <= 0.0
    ):
        raise ValueError("target_radius_px must be finite and positive")
    if (
            isinstance(minimum_intended_speed_px_s, bool)
            or not isinstance(minimum_intended_speed_px_s, (int, float))
            or not np.isfinite(minimum_intended_speed_px_s)
            or minimum_intended_speed_px_s < 0.0
    ):
        raise ValueError(
            "minimum_intended_speed_px_s must be finite and non-negative"
        )
    result = np.array(target_v, dtype=np.float64, copy=True)
    displacement = target - cursor
    distance = np.linalg.norm(displacement, axis=1)
    relative_speed = np.linalg.norm(actual_v - target_v, axis=1)
    # The agreed one-second distance scale makes min(v_min, distance / 1 s)
    # dimensionally a speed while preserving the requested numeric taper.
    startup_floor = np.minimum(float(minimum_intended_speed_px_s), distance)
    intended_relative_speed = np.maximum(relative_speed, startup_floor)
    outside = valid & (distance > float(target_radius_px))
    result[outside] += (
            intended_relative_speed[outside, None]
            * displacement[outside]
            / distance[outside, None]
    )
    return limit_speed(result, max_speed)
