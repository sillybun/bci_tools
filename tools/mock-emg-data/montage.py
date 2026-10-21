"""Frozen SAGA channel order and Bipolar-32 pairing used by the mock source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


N_ELECTRODES = 64


SAGA_64_CHANNEL_LABELS = (
    "R1C8", "R2C8", "R3C8", "R4C8", "R1C7", "R2C7", "R3C7", "R4C7",
    "R1C6", "R2C6", "R3C6", "R4C6", "R1C5", "R1C4", "R1C3", "R1C2",
    "R1C1", "R2C5", "R2C4", "R2C3", "R2C2", "R2C1", "R3C5", "R3C4",
    "R3C3", "R3C2", "R3C1", "R4C5", "R4C4", "R4C3", "R4C2", "R4C1",
    "R5C1", "R5C2", "R5C3", "R5C4", "R5C5", "R6C1", "R6C2", "R6C3",
    "R6C4", "R6C5", "R7C1", "R7C2", "R7C3", "R7C4", "R7C5", "R8C1",
    "R8C2", "R8C3", "R8C4", "R8C5", "R5C6", "R6C6", "R7C6", "R8C6",
    "R5C7", "R6C7", "R7C7", "R8C7", "R5C8", "R6C8", "R7C8", "R8C8",
)


def fixed_pair_labels() -> tuple[tuple[str, str, str], ...]:
    """Return ``(pair, positive, negative)`` in the frozen 4x8 map order."""

    return tuple(
        (f"R{row}C{column}-R{row + 1}C{column}",
         f"R{row}C{column}",
         f"R{row + 1}C{column}")
        for column in range(1, 9)
        for row in (1, 3, 5, 7)
    )


@dataclass(frozen=True)
class Bipolar32Montage:
    """Label-derived indices for the frozen SAGA Bipolar-32 montage."""

    channel_labels: tuple[str, ...]
    pair_labels: tuple[str, ...]
    positive_indices: tuple[int, ...]
    negative_indices: tuple[int, ...]

    @classmethod
    def from_channel_labels(cls, channel_labels: Sequence[str]) -> "Bipolar32Montage":
        labels = tuple(str(label).strip() for label in channel_labels)
        if len(labels) != N_ELECTRODES:
            raise ValueError(
                f"Bipolar-32 requires exactly {N_ELECTRODES} channel labels, "
                f"got {len(labels)}"
            )
        if any(not label for label in labels):
            raise ValueError("channel labels must be non-empty")
        if len(set(labels)) != len(labels):
            raise ValueError("channel labels must be unique")

        by_label = {label: index for index, label in enumerate(labels)}
        pairs = fixed_pair_labels()
        required = {label for _, positive, negative in pairs for label in (positive, negative)}
        missing = sorted(required - by_label.keys())
        if missing:
            raise ValueError(f"channel labels are missing labels required by Bipolar-32: {missing}")

        return cls(
            channel_labels=labels,
            pair_labels=tuple(pair for pair, _, _ in pairs),
            positive_indices=tuple(by_label[positive] for _, positive, _ in pairs),
            negative_indices=tuple(by_label[negative] for _, _, negative in pairs),
        )

    def difference(self, raw_ct: np.ndarray) -> np.ndarray:
        """Apply positive-minus-negative pairing to one ``(64, T)`` array."""

        raw = np.asarray(raw_ct)
        if raw.ndim != 2 or raw.shape[0] != N_ELECTRODES:
            raise ValueError(
                f"raw EMG must have shape ({N_ELECTRODES}, T), got {raw.shape}"
            )
        positive = np.asarray(self.positive_indices, dtype=np.intp)
        negative = np.asarray(self.negative_indices, dtype=np.intp)
        return np.ascontiguousarray(raw[positive] - raw[negative])
