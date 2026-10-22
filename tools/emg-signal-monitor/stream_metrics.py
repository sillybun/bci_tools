#!/usr/bin/env python3
"""Thread-safe health metrics for LSL stream consumers."""

from __future__ import annotations

import threading
from typing import Any, Iterable


_LSL_CHANNEL_FORMAT_NAMES = {
    0: "undefined",
    1: "float32",
    2: "double64",
    3: "string",
    4: "int32",
    5: "int16",
    6: "int8",
    7: "int64",
}


def lsl_channel_format_name(value: Any) -> str:
    """Return one readable channel-format name for real and test LSL streams."""
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return str(value)
    return _LSL_CHANNEL_FORMAT_NAMES.get(numeric, f"unknown:{numeric}")


class StreamMetrics:
    """Thread-safe receive-window metrics based on LSL timestamps."""

    def __init__(self, *, expected_period_s: float | None = None) -> None:
        self.expected_period_s = expected_period_s
        self._lock = threading.Lock()
        self.total = 0
        self.nonfinite_total = 0
        self.negative_total = 0
        self.out_of_order_total = 0
        self.gap_total = 0
        self.lost_total = 0
        self.reconnect_total = 0
        self.last_lsl_time_s: float | None = None
        self._window_count = 0
        self._window_seq_first: int | None = None
        self._window_gap_count = 0
        self._window_gap_max_s = 0.0
        self._window_out_of_order = 0
        self._window_nonfinite = 0
        self._window_negative = 0

    def observe(
        self,
        timestamps: Iterable[float],
        *,
        nonfinite_count: int = 0,
        negative_count: int = 0,
    ) -> tuple[int | None, int | None]:
        timestamps = [float(ts) for ts in timestamps]
        if not timestamps:
            return None, None
        with self._lock:
            first_seq = self.total + 1
            if self._window_seq_first is None:
                self._window_seq_first = first_seq
            for timestamp in timestamps:
                if self.last_lsl_time_s is not None:
                    delta = timestamp - self.last_lsl_time_s
                    if delta <= 0:
                        self.out_of_order_total += 1
                        self._window_out_of_order += 1
                    elif (
                        self.expected_period_s is not None
                        and delta > 3.0 * self.expected_period_s
                    ):
                        self.gap_total += 1
                        self._window_gap_count += 1
                        self._window_gap_max_s = max(self._window_gap_max_s, delta)
                self.last_lsl_time_s = timestamp
            self.total += len(timestamps)
            self._window_count += len(timestamps)
            self.nonfinite_total += int(nonfinite_count)
            self.negative_total += int(negative_count)
            self._window_nonfinite += int(nonfinite_count)
            self._window_negative += int(negative_count)
            return first_seq, self.total

    def mark_lost(self) -> None:
        with self._lock:
            self.lost_total += 1

    def mark_reconnected(self) -> None:
        with self._lock:
            self.reconnect_total += 1

    def snapshot(self, *, interval_s: float, now_lsl_time_s: float) -> dict[str, Any]:
        interval_s = max(float(interval_s), 1e-9)
        with self._lock:
            age_ms = (
                max(0.0, now_lsl_time_s - self.last_lsl_time_s) * 1e3
                if self.last_lsl_time_s is not None
                else None
            )
            result = {
                "seq_first": self._window_seq_first,
                "seq_last": self.total if self._window_count else None,
                "rx_window": self._window_count,
                "rx_total": self.total,
                "actual_rate_hz": round(self._window_count / interval_s, 3),
                "gap_window": self._window_gap_count,
                "gap_total": self.gap_total,
                "gap_max_ms": round(self._window_gap_max_s * 1e3, 3),
                "out_of_order_window": self._window_out_of_order,
                "out_of_order_total": self.out_of_order_total,
                "nonfinite_window": self._window_nonfinite,
                "nonfinite_total": self.nonfinite_total,
                "negative_window": self._window_negative,
                "negative_total": self.negative_total,
                "last_sample_age_ms": round(age_ms, 3) if age_ms is not None else None,
                "lost_total": self.lost_total,
                "reconnect_total": self.reconnect_total,
            }
            self._window_count = 0
            self._window_seq_first = None
            self._window_gap_count = 0
            self._window_gap_max_s = 0.0
            self._window_out_of_order = 0
            self._window_nonfinite = 0
            self._window_negative = 0
            return result

    def totals(self) -> dict[str, int]:
        with self._lock:
            return {
                "rx_total": self.total,
                "gap_total": self.gap_total,
                "out_of_order_total": self.out_of_order_total,
                "nonfinite_total": self.nonfinite_total,
                "negative_total": self.negative_total,
                "lost_total": self.lost_total,
                "reconnect_total": self.reconnect_total,
            }
