# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: 8d79df0475b85ca859a02a8268f5262358a22afa
# original path: src/decoders/ecog_linear_velocity_decoder/data_types.py
# original git blob: ebe265542add43f453862b57b36b413921e07b16
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Training data and configuration contracts for the ECoG velocity decoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ECOG_FEATURE_SCHEMA_ID = "bci.feature.ecog_iir_bandpower_20ms.v1"
CURSOR_VELOCITY_SCHEMA_ID = "bci.action.cursor_velocity_px_per_s.v1"
ECOG_BAND_LABELS = (
    "theta",
    "mu_alpha",
    "beta",
    "low_gamma",
    "high_gamma",
    "very_high_gamma",
)
CURSOR_VELOCITY_COMPONENTS = ("x", "y")


@dataclass(frozen=True)
class EcogCalibrationBlock:
    """Rest calibration features used only to estimate frozen mean and scale."""

    features: np.ndarray
    timestamps_ns: np.ndarray
    feature_schema_id: str
    feature_layout_id: str
    source_ids: tuple[str, ...] = ("synthetic-ecog-rest-v1",)


@dataclass(frozen=True)
class EcogInteractionBlock:
    """Supervised task block aligned to ECoG feature right-edge timestamps."""

    features: np.ndarray
    timestamps_ns: np.ndarray
    trial_ids: np.ndarray
    target_position: np.ndarray
    cursor_position: np.ndarray
    feature_schema_id: str
    feature_layout_id: str
    source_ids: tuple[str, ...] = ("synthetic-ecog-task-v1",)


@dataclass(frozen=True)
class EcogFitConfig:
    """Rest-normalization seed configuration."""

    std_floor: float = 1e-6

    def __post_init__(self) -> None:
        if (
            isinstance(self.std_floor, bool)
            or not isinstance(self.std_floor, (int, float))
            or not np.isfinite(self.std_floor)
            or self.std_floor <= 0.0
        ):
            raise ValueError("std_floor must be finite and positive")


@dataclass(frozen=True)
class EcogAdaptConfig:
    """Trial-balanced ridge and SmoothBatch configuration.

    ``first_alpha`` weights the calibration seed on the first supervised
    update; ``alpha`` weights the previous adapted model thereafter.
    """

    penalty: float = 1.0
    alpha: float = 0.75
    tracking_time_constant_ms: float = 60.0
    minimum_samples: int = 2
    fit_intercept: bool = True
    first_alpha: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.penalty, bool)
            or not isinstance(self.penalty, (int, float))
            or not np.isfinite(self.penalty)
            or self.penalty < 0.0
        ):
            raise ValueError("penalty must be finite and non-negative")
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not np.isfinite(self.alpha)
            or not 0.0 <= self.alpha < 1.0
        ):
            raise ValueError("alpha must lie in [0, 1)")
        if (
            isinstance(self.first_alpha, bool)
            or not isinstance(self.first_alpha, (int, float))
            or not np.isfinite(self.first_alpha)
            or not 0.0 <= self.first_alpha < 1.0
        ):
            raise ValueError("first_alpha must lie in [0, 1)")
        if (
            isinstance(self.tracking_time_constant_ms, bool)
            or not isinstance(self.tracking_time_constant_ms, (int, float))
            or not np.isfinite(self.tracking_time_constant_ms)
            or self.tracking_time_constant_ms <= 0.0
        ):
            raise ValueError("tracking_time_constant_ms must be finite and positive")
        if (
            isinstance(self.minimum_samples, bool)
            or not isinstance(self.minimum_samples, int)
            or self.minimum_samples < 2
        ):
            raise ValueError("minimum_samples must be at least 2")
        if not isinstance(self.fit_intercept, bool):
            raise ValueError("fit_intercept must be a bool")


def make_synthetic_calibration(
    *,
    channel_count: int = 16,
    band_count: int = 6,
    sample_count: int = 80,
    random_seed: int = 4,
    feature_layout_id: str | None = None,
) -> EcogCalibrationBlock:
    """Deterministic synthetic rest fixture; it is not physiological evidence."""

    rng = np.random.default_rng(random_seed)
    offset = rng.normal(size=(channel_count, band_count))
    scale = rng.uniform(0.5, 2.0, size=(channel_count, band_count))
    values = offset + rng.normal(size=(sample_count, channel_count, band_count)) * scale
    return EcogCalibrationBlock(
        features=values.astype(np.float32),
        timestamps_ns=np.arange(1, sample_count + 1, dtype=np.int64) * 20_000_000,
        feature_schema_id=ECOG_FEATURE_SCHEMA_ID,
        feature_layout_id=feature_layout_id
        or f"bci.layout.tiantan_tsmi_ecog{channel_count}_band{band_count}.v1",
    )


def make_synthetic_interaction(
    calibration: EcogCalibrationBlock,
    *,
    trial_count: int = 4,
    samples_per_trial: int = 20,
    random_seed: int = 9,
) -> EcogInteractionBlock:
    """Synthetic contract fixture with task-correlated ECoG features."""

    rng = np.random.default_rng(random_seed)
    shape = calibration.features.shape[1:]
    mean = calibration.features.astype(np.float64).mean(axis=0)
    scale = np.maximum(calibration.features.astype(np.float64).std(axis=0), 1e-6)
    mixing = rng.normal(scale=0.3, size=(2, int(np.prod(shape))))
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    cursors: list[np.ndarray] = []
    trials: list[int] = []
    timestamps: list[int] = []
    timestamp = 0
    dt = 0.02
    for trial in range(trial_count):
        angle = 2.0 * np.pi * trial / trial_count
        velocity = 120.0 * np.array([np.cos(angle), np.sin(angle)])
        target = np.zeros(2, dtype=np.float64)
        for _ in range(samples_per_trial):
            z = velocity @ mixing + rng.normal(scale=0.02, size=mixing.shape[1])
            features.append((mean + scale * z.reshape(shape)).astype(np.float32))
            target = target + velocity * dt
            targets.append(target.copy())
            cursors.append(target - velocity * 0.060)
            trials.append(trial)
            timestamp += 20_000_000
            timestamps.append(timestamp)
    return EcogInteractionBlock(
        features=np.stack(features),
        timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        trial_ids=np.asarray(trials, dtype=np.int64),
        target_position=np.stack(targets),
        cursor_position=np.stack(cursors),
        feature_schema_id=calibration.feature_schema_id,
        feature_layout_id=calibration.feature_layout_id,
    )
