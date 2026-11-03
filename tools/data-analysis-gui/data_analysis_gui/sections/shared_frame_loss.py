"""Frame-clock calculations shared by the loss and summary sections."""

import numpy as np

FPGA_UNIT_S = 1e-3          # labelled 100 us; device span matches the LSL span only at 1 ms/unit
BASE_PERIOD_S = 0.0192     # measured acquisition frame period


def transmission_totals(times, threshold, fpga):
    """Whole-session fractions with the same denominator rules as S03."""
    span = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    intervals_ms = np.diff(times) * 1000
    gap_s = float(intervals_ms[intervals_ms > threshold * 1000].sum() / 1000)
    return dict(recording_span_s=span, long_receive_gap_threshold_s=threshold,
                long_receive_gap_seconds=gap_s, long_receive_gap_time_fraction=gap_s / span if span > 0 else None,
                fpga_matched_received_frames=fpga['n_frames'] if fpga is not None else None,
                fpga_frames_lost=fpga['frames_lost'] if fpga is not None else None,
                fpga_frame_lost_rate=fpga['drop_rate'] if fpga is not None else None)


def meta_column(session, label, fallback):
    labels = list(session.metadata_labels or [])
    if label in labels:
        return labels.index(label)
    return fallback if session.metadata.shape[1] > fallback and not labels else None



def metadata_frame_indices(session):
    """Locate metadata rows in the neural stream without truncating either stream."""
    times = session.neural_times
    if len(session.metadata) == len(times):
        return np.arange(len(times))
    if not len(session.metadata) or not len(times):
        return None
    meta_times = session.metadata_times
    if meta_times is None:
        return None
    if np.any(np.diff(times) < 0):
        return None
    indices = np.searchsorted(times, meta_times)
    if (np.any(indices >= len(times)) or np.any(np.diff(indices) <= 0)
            or not np.array_equal(times[indices], meta_times)):
        return None
    return indices


def fpga_analysis(session, frame_indices=None):
    """Per-step FPGA deltas, frame multiples and lost frames (None when no clock)."""
    col = meta_column(session, 'device_timestamp_100us', 4)
    if col is None or session.metadata.shape[1] <= col:
        return None
    if frame_indices is None:
        frame_indices = metadata_frame_indices(session)
    if frame_indices is None or len(frame_indices) < 2:
        return None
    clock = session.metadata[:, col].astype(np.int64)
    d = np.diff(clock)
    adjacent = np.diff(frame_indices) == 1
    valid = (d > 0) & adjacent
    k = np.where(valid, np.rint(d * FPGA_UNIT_S / BASE_PERIOD_S), 1).astype(int)
    lost = np.where(valid, np.maximum(k - 1, 0), 0)
    n = len(clock)
    residual = np.abs(d[valid] * FPGA_UNIT_S - k[valid] * BASE_PERIOD_S) * 1000
    span_fpga = float(d[valid].sum() * FPGA_UNIT_S)
    matched_times = session.neural_times[frame_indices]
    span_lsl = float(matched_times[-1] - matched_times[0])
    # Consumers index these arrays on the complete neural receive timeline.
    starts = frame_indices[:-1]
    shape = len(session.neural_times) - 1
    delta_full, k_full, lost_full = np.zeros(shape, d.dtype), np.ones(shape, int), np.zeros(shape, int)
    valid_full = np.zeros(shape, bool)
    delta_full[starts], k_full[starts], lost_full[starts], valid_full[starts] = d, k, lost, valid
    return dict(delta=delta_full, k=k_full, lost=lost_full, valid=valid_full, n_frames=n,
                frame_indices=frame_indices,
                n_steps_gt_one_frame=int((k > 1).sum()), frames_lost=int(lost.sum()),
                lost_seconds=float(lost.sum() * BASE_PERIOD_S),
                drop_rate=float(lost.sum() / (n + lost.sum())) if n else float('nan'),
                n_nonpositive_steps=int((d <= 0).sum()),
                n_steps_crossing_missing_metadata=int((~adjacent).sum()),
                max_residual_ms=float(residual.max()) if len(residual) else float('nan'),
                unit_ratio_at_1ms=span_fpga / span_lsl if span_lsl > 0 else float('nan'),
                max_lost_at_one_step=int(lost.max()) if len(lost) else 0)



def overlap(event_lo, event_hi, lefts, rights):
    """(n_windows, n_events) bool: event interval overlaps window [left, right)."""
    if not len(event_lo) or not len(lefts):
        return np.zeros((len(lefts), len(event_lo)), bool)
    return (event_lo[None, :] < rights[:, None]) & (event_hi[None, :] > lefts[:, None])
