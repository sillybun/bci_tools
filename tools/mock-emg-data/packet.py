"""Decode DataLink's fixed-size packetized raw EMG LSL samples.

The wire format deliberately carries no shape metadata.  ``channel_count`` is
therefore a local deployment parameter and every LSL sample must contain
exactly ``channel_count * 40`` finite numeric values in NumPy C order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

PACKET_SAMPLES = 40
PHYSICAL_SAMPLE_RATE_HZ = 2_000.0
PACKET_DURATION_SECONDS = PACKET_SAMPLES / PHYSICAL_SAMPLE_RATE_HZ


class PacketContractError(ValueError):
    """One LSL sample or timestamp violates the raw EMG wire contract."""


@dataclass(frozen=True)
class DecodedPacket:
    """One decoded packet and its reconstructed physical-sample timestamps."""

    raw_ct: np.ndarray
    packet_timestamp: float
    sample_timestamps: np.ndarray


class PacketDecoder:
    """Stateful validator/decoder for packetized raw EMG.

    LSL timestamps are interpreted as the time of the final physical sample in
    each packet.  With 40 samples at 2 kHz, the first reconstructed timestamp
    is therefore 19.5 ms before the packet timestamp.
    """

    def __init__(self, channel_count: int, *, require_monotonic: bool = True) -> None:
        if isinstance(channel_count, bool) or not isinstance(channel_count, int):
            raise TypeError("channel_count must be an integer")
        if channel_count <= 0:
            raise ValueError("channel_count must be positive")
        self.channel_count = channel_count
        self.require_monotonic = require_monotonic
        self._last_timestamp: float | None = None

    @property
    def expected_length(self) -> int:
        return self.channel_count * PACKET_SAMPLES

    @property
    def last_timestamp(self) -> float | None:
        return self._last_timestamp

    def reset_timestamp(self) -> None:
        """Forget timestamp history, for example when reconnecting a stream."""

        self._last_timestamp = None

    def decode(self, sample: Sequence[float] | np.ndarray, timestamp: float) -> DecodedPacket:
        """Validate and reshape one flat LSL sample into ``[channel, time]``."""

        try:
            source = np.asarray(sample)
        except (TypeError, ValueError) as exc:
            raise PacketContractError("packet values must be numeric") from exc
        if source.ndim != 1:
            raise PacketContractError(
                f"packet must be one-dimensional, got shape {source.shape}"
            )
        if source.size != self.expected_length:
            raise PacketContractError(
                f"packet length {source.size} does not match C*40={self.expected_length} "
                f"for C={self.channel_count}"
            )
        if source.dtype.kind not in "fiu":
            raise PacketContractError(
                f"packet values must have a real numeric dtype, got {source.dtype}"
            )
        packet = np.asarray(source, dtype=np.float32)
        if not np.isfinite(packet).all():
            raise PacketContractError("packet contains NaN or infinity")

        try:
            packet_timestamp = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise PacketContractError("packet timestamp must be numeric") from exc
        if not np.isfinite(packet_timestamp):
            raise PacketContractError("packet timestamp must be finite")
        if packet_timestamp < 0.0:
            raise PacketContractError("packet timestamp must be non-negative")
        if (
            self.require_monotonic
            and self._last_timestamp is not None
            and packet_timestamp <= self._last_timestamp
        ):
            raise PacketContractError(
                f"packet timestamp {packet_timestamp} is not later than "
                f"previous timestamp {self._last_timestamp}"
            )

        raw_ct = packet.reshape(self.channel_count, PACKET_SAMPLES, order="C")
        offsets = np.arange(PACKET_SAMPLES - 1, -1, -1, dtype=np.float64)
        sample_timestamps = packet_timestamp - offsets / PHYSICAL_SAMPLE_RATE_HZ
        self._last_timestamp = packet_timestamp
        return DecodedPacket(raw_ct, packet_timestamp, sample_timestamps)

    def decode_chunk(
        self,
        samples: Iterable[Sequence[float] | np.ndarray],
        timestamps: Iterable[float],
    ) -> list[DecodedPacket]:
        """Decode every row returned by ``StreamInlet.pull_chunk`` in order."""

        sample_rows = list(samples)
        packet_timestamps = list(timestamps)
        if len(sample_rows) != len(packet_timestamps):
            raise PacketContractError(
                "pull_chunk returned different sample and timestamp counts: "
                f"{len(sample_rows)} != {len(packet_timestamps)}"
            )
        return [
            self.decode(sample, timestamp)
            for sample, timestamp in zip(sample_rows, packet_timestamps)
        ]


def decode_pull_chunk(
    samples: Iterable[Sequence[float] | np.ndarray],
    timestamps: Iterable[float],
    channel_count: int,
    *,
    require_monotonic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience API returning ``raw[N,C,40]`` and packet timestamps."""

    decoded = PacketDecoder(
        channel_count, require_monotonic=require_monotonic
    ).decode_chunk(samples, timestamps)
    if not decoded:
        return (
            np.empty((0, channel_count, PACKET_SAMPLES), dtype=np.float32),
            np.empty(0, dtype=np.float64),
        )
    return (
        np.stack([packet.raw_ct for packet in decoded]),
        np.asarray([packet.packet_timestamp for packet in decoded], dtype=np.float64),
    )


def flatten_packet(raw_ct: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Validate and flatten one raw packet with the wire contract's C order."""

    packet = np.asarray(raw_ct, dtype=np.float32)
    if packet.ndim != 2 or packet.shape[0] <= 0 or packet.shape[1] != PACKET_SAMPLES:
        raise ValueError(
            f"raw packet must have shape [C,{PACKET_SAMPLES}], got {packet.shape}"
        )
    if not np.isfinite(packet).all():
        raise ValueError("raw packet contains NaN or infinity")
    return packet.reshape(-1, order="C")
