# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/decoders/standalone_refit_kf/config.py
# original git blob: 4f768910559f41d190fffd22dfb6f2e19c5c572e
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Fit and adaptation configuration for ReFIT-KF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_types import STATE_UPDATE_PERIOD_NS


REGULARIZATION_NAMES = (
    "transition_ridge_regularization",
    "observation_ridge_regularization",
    "process_covariance_floor_regularization",
    "observation_covariance_floor_regularization",
)
COVARIANCE_FLOOR_OPTION_NAMES = (
    "auto_covariance_floor",
    "process_covariance_floor_relative_ratio",
    "observation_covariance_floor_relative_ratio",
)


def _finite_positive(value: object, *, name: str) -> float:
    if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _finite_nonnegative(value: object, *, name: str) -> float:
    if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _minimum_samples(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 4:
        raise ValueError("minimum_samples must be an integer of at least 4")
    return value


def _validate_covariance_floor_options(config: RefitKFFitConfig | RefitKFAdaptConfig) -> None:
    if not isinstance(config.auto_covariance_floor, bool):
        raise ValueError("auto_covariance_floor must be a bool")
    for name in COVARIANCE_FLOOR_OPTION_NAMES[1:]:
        _finite_nonnegative(getattr(config, name), name=name)


@dataclass(frozen=True)
class RefitKFFitConfig:
    """Frozen normalization and Stage-1 fitting; ridge weights use mean sample loss."""

    minimum_samples: int = 20
    transition_ridge_regularization: float = 1e-4
    observation_ridge_regularization: float = 1e-4
    process_covariance_floor_regularization: float = 1e-4
    observation_covariance_floor_regularization: float = 1e-4
    target_radius_px: float = 30.0
    minimum_intended_speed_px_s: float = 50.0
    p0_velocity_regularization: float = 2_500.0
    std_floor: float = 1e-6
    state_update_period_ns: int = STATE_UPDATE_PERIOD_NS
    auto_covariance_floor: bool = False
    process_covariance_floor_relative_ratio: float = 1e-4
    observation_covariance_floor_relative_ratio: float = 1e-4

    def __post_init__(self) -> None:
        _finite_positive(self.std_floor, name="std_floor")
        for name in REGULARIZATION_NAMES:
            _finite_positive(getattr(self, name), name=name)
        _validate_covariance_floor_options(self)
        _finite_positive(
            self.p0_velocity_regularization,
            name="p0_velocity_regularization",
        )
        _minimum_samples(self.minimum_samples)
        if (
                isinstance(self.state_update_period_ns, bool)
                or not isinstance(self.state_update_period_ns, int)
                or self.state_update_period_ns != STATE_UPDATE_PERIOD_NS
        ):
            raise ValueError("state_update_period_ns must equal 20_000_000")
        _finite_positive(self.target_radius_px, name="target_radius_px")
        _finite_nonnegative(
            self.minimum_intended_speed_px_s,
            name="minimum_intended_speed_px_s",
        )


@dataclass(frozen=True)
class RefitKFAdaptConfig:
    """Moving-target ReFIT and blending; ridge weights use mean sample loss."""

    minimum_samples: int = 20
    transition_ridge_regularization: float = 1e-4
    observation_ridge_regularization: float = 1e-4
    process_covariance_floor_regularization: float = 1e-4
    observation_covariance_floor_regularization: float = 1e-4
    target_radius_px: float = 30.0
    minimum_intended_speed_px_s: float = 50.0
    p0_velocity_regularization: float = 2_500.0
    alpha: float | None = None  # old model param weighting
    auto_covariance_floor: bool = False
    process_covariance_floor_relative_ratio: float = 1e-4
    observation_covariance_floor_relative_ratio: float = 1e-4

    def __post_init__(self) -> None:
        for name in REGULARIZATION_NAMES:
            _finite_positive(getattr(self, name), name=name)
        _validate_covariance_floor_options(self)
        _finite_positive(
            self.p0_velocity_regularization,
            name="p0_velocity_regularization",
        )
        _minimum_samples(self.minimum_samples)
        _finite_positive(self.target_radius_px, name="target_radius_px")
        _finite_nonnegative(
            self.minimum_intended_speed_px_s,
            name="minimum_intended_speed_px_s",
        )
        if self.alpha is not None and (
                isinstance(self.alpha, bool)
                or not isinstance(self.alpha, (int, float))
                or not np.isfinite(self.alpha)
                or not 0.0 <= float(self.alpha) < 1.0
        ):
            raise ValueError("alpha must be None or finite in [0, 1)")
