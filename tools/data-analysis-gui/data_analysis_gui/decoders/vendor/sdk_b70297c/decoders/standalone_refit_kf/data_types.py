# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/decoders/standalone_refit_kf/data_types.py
# original git blob: 0b59c3ec38447441c91484da68ffee7f0b16756d
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Calibration and tracking input contracts for ReFIT-KF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATE_UPDATE_PERIOD_NS = 20_000_000


@dataclass(frozen=True)
class CalibrationBlock:
    """Reference features used only to estimate frozen normalization."""

    features: np.ndarray
    timestamps_ns: np.ndarray
    feature_schema_id: str
    feature_layout_id: str
    source_ids: tuple[str, ...] = ("calibration",)


@dataclass(frozen=True)
class RefitKFTrackingBlock:
    """Closed-loop tracking rows aligned to feature right edges."""

    features: np.ndarray
    timestamps_ns: np.ndarray
    trial_ids: np.ndarray
    target_position: np.ndarray
    cursor_position: np.ndarray
    feature_schema_id: str
    feature_layout_id: str
    source_ids: tuple[str, ...] = ("tracking",)


@dataclass(frozen=True)
class RefitKFFitData:
    """Rest calibration plus an optional Stage-1 supervised tracking block."""

    calibration: CalibrationBlock
    initial_tracking: RefitKFTrackingBlock | None = None
