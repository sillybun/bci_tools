"""Read-only backend for monitoring packetized signed raw EMG.

History and acquisition deliberately retain the raw transport contract. A
separate display-only branch may apply a causal band-pass and/or signed
Bipolar-32 subtraction, but it does not perform CAR, notch filtering, envelope
extraction, QC, training, recording, or inference.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import threading
import time
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal

from lsl_discovery import ExternalStreamContract, resolve_external_stream, stream_identity
from stream_metrics import StreamMetrics
from montage import (
    N_ELECTRODES,
    Bipolar32Montage,
    SAGA_64_CHANNEL_LABELS,
)
from packet import (
    PACKET_DURATION_SECONDS,
    PACKET_SAMPLES,
    PHYSICAL_SAMPLE_RATE_HZ,
    DecodedPacket,
    PacketContractError,
    PacketDecoder,
)

DEFAULT_STREAM_NAME = "emg_raw_signed_2khz"
DEFAULT_STREAM_TYPE = "EMG"
DEFAULT_PACKET_RATE_HZ = 1.0 / PACKET_DURATION_SECONDS
DEFAULT_HISTORY_SECONDS = 10.0
MAX_CHANNEL_COUNT = 512
MAX_HISTORY_BYTES = 128 * 1024 * 1024
DISPLAY_MODE_RAW = "Raw-64"
DISPLAY_MODE_BIPOLAR32 = "signed Bipolar-32"
DISPLAY_MODES = (DISPLAY_MODE_RAW, DISPLAY_MODE_BIPOLAR32)
DEFAULT_BANDPASS_LOW_HZ = 1.0
DEFAULT_BANDPASS_HIGH_HZ = 200.0
DISPLAY_BANDPASS_ORDER = 4


def default_channel_labels(channel_count: int) -> tuple[str, ...]:
    """Return the explicit SAGA preset for 64 channels, generic labels otherwise."""

    if channel_count == N_ELECTRODES:
        return SAGA_64_CHANNEL_LABELS
    return tuple(f"ch{index + 1:03d}" for index in range(channel_count))


def estimated_history_bytes(channel_count: int, history_seconds: float) -> int:
    """Estimate raw float32 plus one float64 timestamp per 2 kHz point."""

    sample_count = math.ceil(float(history_seconds) * PHYSICAL_SAMPLE_RATE_HZ)
    return sample_count * (int(channel_count) * np.dtype(np.float32).itemsize + 8)


@dataclass(frozen=True)
class MonitorConfig:
    stream_name: str = DEFAULT_STREAM_NAME
    stream_type: str = DEFAULT_STREAM_TYPE
    source_id: str | None = None
    channel_count: int = 64
    channel_labels: tuple[str, ...] = ()
    history_seconds: float = DEFAULT_HISTORY_SECONDS
    resolve_timeout_s: float = 0.5
    info_timeout_s: float = 2.0
    pull_timeout_s: float = 0.2

    def validated(self) -> "MonitorConfig":
        name = self.stream_name.strip()
        stream_type = self.stream_type.strip()
        source_id = self.source_id.strip() if self.source_id else None
        if not name:
            raise ValueError("stream name must not be empty")
        if not stream_type:
            raise ValueError("stream type must not be empty")
        if isinstance(self.channel_count, bool) or not isinstance(self.channel_count, int):
            raise TypeError("channel_count must be an integer")
        if not 1 <= self.channel_count <= MAX_CHANNEL_COUNT:
            raise ValueError(f"channel_count must be in [1, {MAX_CHANNEL_COUNT}]")
        labels = tuple(label.strip() for label in self.channel_labels)
        if not labels:
            labels = default_channel_labels(self.channel_count)
        if len(labels) != self.channel_count:
            raise ValueError(
                f"channel label count {len(labels)} does not match C={self.channel_count}"
            )
        if any(not label for label in labels):
            raise ValueError("channel labels must not be empty")
        if len(set(labels)) != len(labels):
            raise ValueError("channel labels must be unique")
        if not math.isfinite(self.history_seconds) or self.history_seconds <= 0:
            raise ValueError("history_seconds must be finite and > 0")
        history_bytes = estimated_history_bytes(
            self.channel_count, self.history_seconds
        )
        if history_bytes > MAX_HISTORY_BYTES:
            raise ValueError(
                "raw history estimate "
                f"{history_bytes / 1024**2:.1f} MiB exceeds 128 MiB limit; "
                "reduce C or history_seconds"
            )
        for field_name in ("resolve_timeout_s", "info_timeout_s", "pull_timeout_s"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and > 0")
        return MonitorConfig(
            stream_name=name,
            stream_type=stream_type,
            source_id=source_id,
            channel_count=self.channel_count,
            channel_labels=labels,
            history_seconds=float(self.history_seconds),
            resolve_timeout_s=float(self.resolve_timeout_s),
            info_timeout_s=float(self.info_timeout_s),
            pull_timeout_s=float(self.pull_timeout_s),
        )

    @property
    def expected_lsl_width(self) -> int:
        return self.channel_count * PACKET_SAMPLES

    @property
    def contract(self) -> ExternalStreamContract:
        return ExternalStreamContract(
            name=self.stream_name,
            stream_type=self.stream_type,
            channel_count=self.expected_lsl_width,
            channel_format="float32",
            nominal_srate=DEFAULT_PACKET_RATE_HZ,
            require_source_id=False,
            source_option="--source-id",
        )


def parse_channel_labels(text: str, channel_count: int) -> tuple[str, ...]:
    stripped = str(text).strip()
    labels = tuple(part.strip() for part in stripped.split(",")) if stripped else ()
    config = MonitorConfig(channel_count=channel_count, channel_labels=labels)
    return config.validated().channel_labels


def validate_stream_info(info: Any, config: MonitorConfig) -> dict[str, Any]:
    """Validate an already selected header, useful for workers and tests."""

    config = config.validated()
    identity = stream_identity(info)
    expected = {
        "name": config.stream_name,
        "type": config.stream_type,
        "channel_count": config.expected_lsl_width,
        "channel_format": "float32",
    }
    problems: list[str] = []
    for key in ("name", "type", "channel_count", "channel_format"):
        if identity[key] != expected[key]:
            problems.append(f"{key}={identity[key]!r}, expected {expected[key]!r}")
    nominal_hz = float(identity["nominal_hz"])
    if (
        not math.isfinite(nominal_hz)
        or abs(nominal_hz - DEFAULT_PACKET_RATE_HZ) > 0.5
    ):
        problems.append(
            f"nominal_hz={identity['nominal_hz']!r}, expected {DEFAULT_PACKET_RATE_HZ:g}"
        )
    if config.source_id is not None and identity["source_id"] != config.source_id:
        problems.append(
            f"source_id={identity['source_id']!r}, expected {config.source_id!r}"
        )
    if problems:
        raise RuntimeError("LSL header mismatch: " + "; ".join(problems))
    return identity


class RawEmgHistory:
    """Bounded full-resolution raw sample history indexed by physical time."""

    def __init__(self, channel_count: int, history_seconds: float) -> None:
        if channel_count <= 0:
            raise ValueError("channel_count must be positive")
        if history_seconds <= 0:
            raise ValueError("history_seconds must be > 0")
        self.channel_count = int(channel_count)
        self.history_seconds = float(history_seconds)
        self._timestamps: deque[np.ndarray] = deque()
        self._values: deque[np.ndarray] = deque()
        self._sample_count = 0

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def latest_timestamp(self) -> float | None:
        return float(self._timestamps[-1][-1]) if self._timestamps else None

    def clear(self) -> None:
        """Discard all stored samples without changing the history contract."""

        self._timestamps.clear()
        self._values.clear()
        self._sample_count = 0

    def append(self, packet: DecodedPacket) -> None:
        raw = np.asarray(packet.raw_ct, dtype=np.float32)
        stamps = np.asarray(packet.sample_timestamps, dtype=np.float64)
        if raw.shape != (self.channel_count, PACKET_SAMPLES):
            raise ValueError(
                f"raw packet shape {raw.shape} does not match "
                f"({self.channel_count}, {PACKET_SAMPLES})"
            )
        if stamps.shape != (PACKET_SAMPLES,):
            raise ValueError("physical timestamp vector has the wrong shape")
        if self._timestamps and stamps[0] <= self._timestamps[-1][-1]:
            raise ValueError("physical sample timestamps are not strictly ordered")
        self._timestamps.append(stamps.copy())
        self._values.append(raw.copy())
        self._sample_count += PACKET_SAMPLES
        cutoff = stamps[-1] - self.history_seconds
        while self._timestamps and self._timestamps[0][-1] < cutoff:
            self._sample_count -= self._timestamps[0].size
            self._timestamps.popleft()
            self._values.popleft()

    def append_many(self, packets: Sequence[DecodedPacket]) -> None:
        for packet in packets:
            self.append(packet)

    def window(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        if not self._timestamps:
            return (
                np.empty(0, dtype=np.float64),
                np.empty((self.channel_count, 0), dtype=np.float32),
            )
        cutoff = self._timestamps[-1][-1] - float(seconds)
        selected_timestamps: list[np.ndarray] = []
        selected_values: list[np.ndarray] = []
        for timestamps_chunk, values_chunk in zip(
            reversed(self._timestamps), reversed(self._values)
        ):
            selected_timestamps.append(timestamps_chunk)
            selected_values.append(values_chunk)
            if timestamps_chunk[0] <= cutoff:
                break
        selected_timestamps.reverse()
        selected_values.reverse()
        timestamps = np.concatenate(selected_timestamps)
        values = np.concatenate(selected_values, axis=1)
        first = int(np.searchsorted(timestamps, cutoff, side="left"))
        return timestamps[first:], values[:, first:]

    def channel_statistics(
        self, seconds: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _timestamps, values = self.window(seconds)
        return raw_channel_statistics(values)


@dataclass(frozen=True)
class DisplayBandpassConfig:
    """Validated display-only Butterworth band-pass parameters."""

    low_hz: float = DEFAULT_BANDPASS_LOW_HZ
    high_hz: float = DEFAULT_BANDPASS_HIGH_HZ
    sample_rate_hz: float = PHYSICAL_SAMPLE_RATE_HZ
    order: int = DISPLAY_BANDPASS_ORDER

    def validated(self) -> "DisplayBandpassConfig":
        low = float(self.low_hz)
        high = float(self.high_hz)
        sample_rate = float(self.sample_rate_hz)
        if not all(math.isfinite(value) for value in (low, high, sample_rate)):
            raise ValueError("band-pass frequencies and sample rate must be finite")
        if sample_rate <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if self.order != DISPLAY_BANDPASS_ORDER:
            raise ValueError(
                f"display band-pass order must be {DISPLAY_BANDPASS_ORDER}"
            )
        nyquist = sample_rate / 2.0
        if not 0 < low < high < nyquist:
            raise ValueError(
                f"band-pass must satisfy 0 < low < high < Nyquist ({nyquist:g} Hz)"
            )
        return DisplayBandpassConfig(
            low_hz=low,
            high_hz=high,
            sample_rate_hz=sample_rate,
            order=DISPLAY_BANDPASS_ORDER,
        )


class OnlineDisplayBandpass:
    """Stateful causal SOS filter for the monitor's display-only branch.

    State is preserved across contiguous packets. It is initialized from each
    channel's first sample and reset before a packet following a stream gap or
    non-increasing packet timestamp, so filter state never crosses a time
    discontinuity.
    """

    def __init__(
        self,
        channel_count: int,
        config: DisplayBandpassConfig | None = None,
    ) -> None:
        if isinstance(channel_count, bool) or not isinstance(channel_count, int):
            raise TypeError("channel_count must be an integer")
        if channel_count <= 0:
            raise ValueError("channel_count must be positive")
        self.channel_count = channel_count
        self.config = (config or DisplayBandpassConfig()).validated()
        from scipy.signal import butter

        self.sos = np.ascontiguousarray(
            butter(
                self.config.order,
                [self.config.low_hz, self.config.high_hz],
                btype="bandpass",
                fs=self.config.sample_rate_hz,
                output="sos",
            ),
            dtype=np.float64,
        )
        self._zi: np.ndarray | None = None
        self._last_packet_timestamp: float | None = None
        self._last_sample_timestamp: float | None = None

    @property
    def has_state(self) -> bool:
        return self._zi is not None

    def reset(self) -> None:
        """Forget filter and timestamp state after a stream discontinuity."""

        self._zi = None
        self._last_packet_timestamp = None
        self._last_sample_timestamp = None

    def _validate_packet(self, packet: DecodedPacket) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(packet.raw_ct)
        stamps = np.asarray(packet.sample_timestamps, dtype=np.float64)
        expected = (self.channel_count, PACKET_SAMPLES)
        if values.shape != expected:
            raise ValueError(f"raw EMG packet must have shape {expected}, got {values.shape}")
        if values.dtype != np.float32:
            raise TypeError(f"raw EMG packet must have dtype float32, got {values.dtype}")
        if stamps.shape != (PACKET_SAMPLES,):
            raise ValueError("physical timestamp vector has the wrong shape")
        if not np.isfinite(values).all() or not np.isfinite(stamps).all():
            raise ValueError("raw EMG packet contains non-finite values or timestamps")
        if not np.all(np.diff(stamps) > 0):
            raise ValueError("physical sample timestamps are not strictly ordered")
        packet_timestamp = float(packet.packet_timestamp)
        if not math.isfinite(packet_timestamp):
            raise ValueError("packet timestamp must be finite")
        return values, stamps

    def process_packet(self, packet: DecodedPacket) -> DecodedPacket:
        """Filter one packet, preserving state only across contiguous time."""

        from scipy.signal import sosfilt, sosfilt_zi

        raw, stamps = self._validate_packet(packet)
        timestamp = float(packet.packet_timestamp)
        if self._last_packet_timestamp is not None:
            delta = timestamp - self._last_packet_timestamp
            overlaps_previous = (
                self._last_sample_timestamp is not None
                and stamps[0] <= self._last_sample_timestamp
            )
            if delta <= 0 or delta > 1.5 * PACKET_DURATION_SECONDS or overlaps_previous:
                self._zi = None
        values = np.ascontiguousarray(raw, dtype=np.float64)
        if self._zi is None:
            self._zi = (
                sosfilt_zi(self.sos)[:, None, :]
                * values[:, 0][None, :, None]
            )
        filtered, self._zi = sosfilt(self.sos, values, axis=1, zi=self._zi)
        self._last_packet_timestamp = timestamp
        self._last_sample_timestamp = float(stamps[-1])
        return DecodedPacket(
            raw_ct=np.ascontiguousarray(filtered, dtype=np.float32),
            packet_timestamp=timestamp,
            sample_timestamps=stamps.copy(),
        )


def raw_channel_statistics(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute signed-raw RMS, peak-to-peak, and mean from one snapshot."""

    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError("raw values must have shape [channel, time]")
    if values.shape[1] == 0:
        empty = np.full(values.shape[0], np.nan, dtype=np.float64)
        return empty.copy(), empty.copy(), empty.copy()
    values64 = values.astype(np.float64, copy=False)
    rms = np.sqrt(np.mean(values64 * values64, axis=1))
    p2p = np.ptp(values64, axis=1)
    mean = np.mean(values64, axis=1)
    return rms, p2p, mean


def transform_display_values(
    raw_ct: np.ndarray,
    channel_labels: Sequence[str],
    mode: str,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return labels and traces for a stateless monitor display transform.

    The Bipolar-32 branch is signed positive-minus-negative only. It deliberately
    does not perform CAR, filtering, feature extraction, QC, or inference.
    """

    raw = np.asarray(raw_ct)
    if raw.ndim != 2:
        raise ValueError(f"raw display values must have shape [channel, time], got {raw.shape}")
    labels = tuple(str(label).strip() for label in channel_labels)
    if len(labels) != raw.shape[0]:
        raise ValueError(
            f"channel label count {len(labels)} does not match raw C={raw.shape[0]}"
        )
    if mode == DISPLAY_MODE_RAW:
        return labels, raw
    if mode == DISPLAY_MODE_BIPOLAR32:
        montage = Bipolar32Montage.from_channel_labels(labels)
        return montage.pair_labels, montage.difference(raw)
    raise ValueError(f"unknown display mode {mode!r}; expected one of {DISPLAY_MODES}")


def bipolar32_grid_position(index: int) -> tuple[int, int]:
    """Map frozen column-major pair order onto the QC 4-row by 8-column grid."""

    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 32:
        raise ValueError("Bipolar-32 pair index must be an integer in [0, 31]")
    return index % 4, index // 4


def _split_on_time_gaps(timestamps: np.ndarray) -> list[slice]:
    if timestamps.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(timestamps) > 3.0 / PHYSICAL_SAMPLE_RATE_HZ) + 1
    bounds = np.concatenate(([0], breaks, [timestamps.size]))
    return [
        slice(int(left), int(right))
        for left, right in zip(bounds[:-1], bounds[1:])
    ]


def decimate_minmax(
    timestamps: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    max_points: int,
) -> list[np.ndarray]:
    """Return gap-separated min/max decimated ``[N,2]`` display segments."""

    ts = np.asarray(timestamps, dtype=np.float64)
    ys = np.asarray(values, dtype=np.float64)
    if ts.ndim != 1 or ys.ndim != 1 or ts.size != ys.size:
        raise ValueError("timestamps and values must be equal-length vectors")
    if max_points < 2:
        raise ValueError("max_points must be >= 2")
    output: list[np.ndarray] = []
    for part in _split_on_time_gaps(ts):
        part_ts, part_y = ts[part], ys[part]
        if part_ts.size <= max_points:
            output.append(np.column_stack((part_ts, part_y)))
            continue
        bucket_count = max(1, max_points // 2)
        edges = np.linspace(0, part_ts.size, bucket_count + 1, dtype=int)
        points: list[tuple[float, float]] = []
        for left, right in zip(edges[:-1], edges[1:]):
            if right <= left:
                continue
            local = part_y[left:right]
            indices = [left + int(np.argmin(local)), left + int(np.argmax(local))]
            for index in sorted(set(indices)):
                points.append((float(part_ts[index]), float(part_y[index])))
        output.append(np.asarray(points, dtype=np.float64))
    return output


class _StatusBoundLog:
    """Tiny structlog-compatible adapter surfacing discovery rejections."""

    def __init__(
        self,
        status: Callable[[str], None],
        context: dict[str, Any] | None = None,
    ) -> None:
        self._status = status
        self._context = dict(context or {})

    def bind(self, **kwargs: Any) -> "_StatusBoundLog":
        return _StatusBoundLog(self._status, {**self._context, **kwargs})

    def info(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def warning(self, *_args: Any, **_kwargs: Any) -> None:
        if self._context.get("event") != "stream_candidates_rejected":
            return
        details: list[str] = []
        for candidate in self._context.get("candidates", []):
            reasons = candidate.get("reasons", [])
            details.append(", ".join(str(reason) for reason in reasons))
        rendered = "; ".join(detail for detail in details if detail)
        self._status("header rejected: " + (rendered or "contract mismatch"))

    def error(self, *_args: Any, **_kwargs: Any) -> None:
        pass


class RawStreamWorker(QThread):
    """Resolve, validate, pull, and decode packets outside the GUI thread."""

    connected = Signal(object)
    batch_received = Signal(object)
    health_updated = Signal(object)
    status_changed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        config: MonitorConfig,
        *,
        resolver: Callable[..., Any] = resolve_external_stream,
        inlet_factory: Callable[..., Any] | None = None,
        lsl_clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__()
        self.config = config.validated()
        self._resolver = resolver
        self._inlet_factory = inlet_factory
        self._lsl_clock = lsl_clock
        self._stop_event = threading.Event()
        self.metrics = StreamMetrics(expected_period_s=PACKET_DURATION_SECONDS)
        self.invalid_total = 0
        self.out_of_order_total = 0
        self._out_of_order_window = 0
        self.gap_total = 0
        self._gap_window = 0

    def request_stop(self) -> None:
        self._stop_event.set()
        self.requestInterruption()

    def _stopping(self) -> bool:
        return self._stop_event.is_set() or self.isInterruptionRequested()

    def _emit_health(self, now: float, elapsed: float) -> None:
        snapshot = self.metrics.snapshot(interval_s=elapsed, now_lsl_time_s=now)
        snapshot["invalid_total"] = self.invalid_total
        snapshot["out_of_order_total"] = self.out_of_order_total
        snapshot["out_of_order_window"] = self._out_of_order_window
        snapshot["gap_total"] = self.gap_total
        snapshot["gap_window"] = self._gap_window
        self._out_of_order_window = 0
        self._gap_window = 0
        self.health_updated.emit(snapshot)

    def run(self) -> None:
        inlet = None
        pending: list[DecodedPacket] = []
        last_emit = time.monotonic()
        last_health = last_emit
        try:
            stream = self._resolver(
                stop_event=self._stop_event,
                log=_StatusBoundLog(self.status_changed.emit),
                contract=self.config.contract,
                description="signed raw EMG monitor",
                source_id=self.config.source_id,
                resolve_timeout_s=self.config.resolve_timeout_s,
            )
            if stream is None or self._stopping():
                return
            if self._inlet_factory is None:
                from pylsl import StreamInlet

                inlet_factory = StreamInlet
            else:
                inlet_factory = self._inlet_factory
            inlet = inlet_factory(
                stream,
                max_buflen=math.ceil(self.config.history_seconds + 1),
                recover=True,
            )
            info = inlet.info(timeout=self.config.info_timeout_s)
            identity = validate_stream_info(info, self.config)
            correction_s = float(
                inlet.time_correction(timeout=self.config.info_timeout_s)
            )
            if not math.isfinite(correction_s):
                raise RuntimeError("LSL time_correction is not finite")
            identity["time_correction_s"] = correction_s
            self.connected.emit(identity)
            self.status_changed.emit("connected")
            decoder = PacketDecoder(self.config.channel_count, require_monotonic=True)
            while not self._stopping():
                samples, timestamps = inlet.pull_chunk(
                    timeout=self.config.pull_timeout_s,
                    max_samples=max(1, round(DEFAULT_PACKET_RATE_HZ)),
                )
                now_monotonic = time.monotonic()
                if len(samples) != len(timestamps):
                    self.invalid_total += max(len(samples), len(timestamps), 1)
                    self.status_changed.emit("contract error: pull_chunk count mismatch")
                else:
                    for sample, timestamp in zip(samples, timestamps):
                        try:
                            timestamp_value = float(timestamp) + correction_s
                        except (TypeError, ValueError):
                            timestamp_value = float("nan")
                        previous_timestamp = decoder.last_timestamp
                        try:
                            packet = decoder.decode(sample, timestamp_value)
                        except (PacketContractError, TypeError, ValueError) as error:
                            self.invalid_total += 1
                            if (
                                math.isfinite(timestamp_value)
                                and timestamp_value >= 0
                                and previous_timestamp is not None
                                and timestamp_value <= previous_timestamp
                            ):
                                self.out_of_order_total += 1
                                self._out_of_order_window += 1
                            self.status_changed.emit(f"contract error: {error}")
                        else:
                            # Only contract-valid decoded packets may advance
                            # the last-sample clock used for age/gap metrics.
                            self.metrics.observe([timestamp_value])
                            if (
                                previous_timestamp is not None
                                and timestamp_value - previous_timestamp
                                > 1.5 * PACKET_DURATION_SECONDS
                            ):
                                self.gap_total += 1
                                self._gap_window += 1
                            pending.append(packet)
                if pending and now_monotonic - last_emit >= 0.05:
                    self.batch_received.emit(tuple(pending))
                    pending.clear()
                    last_emit = now_monotonic
                if now_monotonic - last_health >= 0.5:
                    now_lsl = (
                        self._lsl_clock() if self._lsl_clock else _lsl_local_clock()
                    )
                    self._emit_health(now_lsl, now_monotonic - last_health)
                    last_health = now_monotonic
            if pending:
                self.batch_received.emit(tuple(pending))
        except Exception as error:
            if not self._stopping():
                self.metrics.mark_lost()
                self.failed.emit(str(error))
        finally:
            if inlet is not None:
                try:
                    inlet.close_stream()
                except Exception:
                    pass


def _lsl_local_clock() -> float:
    from pylsl import local_clock

    return float(local_clock())
