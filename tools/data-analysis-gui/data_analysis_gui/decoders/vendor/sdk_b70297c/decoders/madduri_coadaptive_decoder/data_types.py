# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/decoders/madduri_coadaptive_decoder/data_types.py
# original git blob: eed3fdb0bb2bad33512cc1d1377fe456c568bd69
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Generic interaction data and backward-compatible seed/fixture helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...bci_decoder_api import FeatureSpec

from .config import (
    MadduriAdaptConfig as MadduriAdaptConfig,
    MadduriFitConfig as MadduriFitConfig,
)

EMG_FEATURE_SCHEMA_ID = "bci.feature.emg_causal_envelope_50hz.v1"

__all__ = [
    "EMG_FEATURE_SCHEMA_ID", "MadduriInteractionBlock", "MadduriFitConfig",
    "MadduriAdaptConfig", "make_seed_block", "make_synthetic_block",
]


@dataclass(frozen=True)
class MadduriInteractionBlock:
    """Closed-loop data with caller-defined feature and coordinate contracts.

    Fields:
        features: finite float32, shape ``(sample, *feature_spec.shape)``.
            All feature elements are flattened in C order inside the algorithm.
        timestamps_ns: nonnegative int64-compatible observation times.
        trial_ids: integer segment IDs; each ID occupies one contiguous run.
        feature_schema_id / feature_layout_id: must equal the Trainer FeatureSpec.
        target_position: task target, shape ``(sample, action_dimension)``. Required for
            adapt/evaluate; may be ``None`` for a random seed fit.
        cursor_position: observed position, same shape as target_position.
        source_ids: provenance identifiers for the data sources.
    """

    features: np.ndarray
    timestamps_ns: np.ndarray
    trial_ids: np.ndarray
    feature_schema_id: str
    feature_layout_id: str
    target_position: np.ndarray | None = None
    cursor_position: np.ndarray | None = None
    source_ids: tuple[str, ...] = ("interaction-block",)


def make_seed_block(
        *,
        feature_spec: FeatureSpec | None = None,
        channel_count: int | None = None,
        feature_layout_id: str | None = None,
        source_ids: tuple[str, ...] = ("random-initialization",),
) -> MadduriInteractionBlock:
    """An empty block for ``fit()``: the random seed decoder consumes no data.

    It carries only the feature schema/layout identity (so the produced Artifact
    matches the Trainer) and zero samples — making explicit that no data is used.
    """

    if feature_spec is not None:
        if channel_count is not None or feature_layout_id is not None:
            raise ValueError("pass feature_spec without legacy channel/layout arguments")
        shape = feature_spec.shape
        schema_id = feature_spec.schema_id
        layout_id = feature_spec.layout_id
    else:
        from .config import positive_integer

        channel_count = positive_integer(channel_count, name="channel_count")
        shape = (channel_count,)
        schema_id = EMG_FEATURE_SCHEMA_ID
        layout_id = feature_layout_id or f"emg_envelope_{channel_count}ch_v1"
    return MadduriInteractionBlock(
        features=np.zeros((0, *shape), dtype=np.float32),
        timestamps_ns=np.zeros((0,), dtype=np.int64),
        trial_ids=np.zeros((0,), dtype=np.int64),
        feature_schema_id=schema_id,
        feature_layout_id=layout_id,
        source_ids=source_ids,
    )


def make_synthetic_block(
        *,
        random_seed: int = 7,
        trial_count: int = 8,
        samples_per_trial: int = 40,
        channel_count: int = 64,
        feature_layout_id: str | None = None,
) -> MadduriInteractionBlock:
    """Create a small nonnegative EMG block with matching target/cursor signals.

    A random nonnegative channel mixing maps a per-trial direction to the EMG
    envelope, and the target/cursor are set so ``(tau - y) / Δt`` equals that
    direction's velocity. This is a contract fixture, not real EMG.
    """

    rng = np.random.default_rng(random_seed)
    # Agonist/antagonist muscle drive: nonnegative envelope that still encodes
    # signed direction, so a linear decoder can recover velocity (unlike a
    # sign-blind |velocity| mapping, where 0 deg and 180 deg are identical).
    mixing = np.abs(rng.normal(scale=1.0, size=(channel_count, 4)))
    baseline = rng.uniform(0.5, 1.5, size=channel_count)
    step_ns = 20_000_000  # 50 Hz cadence == Δt = 20 ms.
    delta_t_s = step_ns / 1_000_000_000.0
    speed = 100.0  # px/s stand-in for the task speed.

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    cursors: list[np.ndarray] = []
    trial_ids: list[int] = []
    timestamps: list[int] = []
    timestamp = 0

    for trial_id in range(trial_count):
        angle = 2.0 * np.pi * trial_id / trial_count
        velocity = speed * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        drive = np.array(
            [
                max(velocity[0], 0.0),
                max(-velocity[0], 0.0),
                max(velocity[1], 0.0),
                max(-velocity[1], 0.0),
            ],
            dtype=np.float64,
        ) / speed
        for _ in range(samples_per_trial):
            envelope = baseline + mixing @ drive
            envelope = np.abs(envelope + 0.02 * rng.normal(size=channel_count))
            features.append(envelope.astype(np.float32))
            # cursor at origin, target chosen so (target - cursor) / Δt == velocity.
            cursors.append(np.zeros(2, dtype=np.float64))
            targets.append(velocity * delta_t_s)
            trial_ids.append(trial_id)
            timestamp += step_ns
            timestamps.append(timestamp)

    channels = channel_count
    return MadduriInteractionBlock(
        features=np.stack(features),
        timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        trial_ids=np.asarray(trial_ids, dtype=np.int64),
        feature_schema_id=EMG_FEATURE_SCHEMA_ID,
        feature_layout_id=feature_layout_id or f"emg_envelope_{channels}ch_v1",
        target_position=np.stack(targets),
        cursor_position=np.stack(cursors),
    )
