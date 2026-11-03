"""One loaded recording: raw arrays plus the full trial table (before filtering).

Trial reconstruction follows the event state machine used by both source projects:
T_TRIAL_START ... (T_TARGET_ACQUIRED | MISS) ... T_TRIAL_END, cross-checked with the
cursor stream's trial/block identity while `active_trial == 1`. Every trial that
cannot be reconstructed keeps a row with an explicit `incomplete_reason`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
import threading

import numpy as np

from .. import config as C
from .catalog import Recording, describe, run_folder
from .xdf_io import desc_value, electrode_ids, load_streams

TRIAL_FIELDS = ('trial_index', 'start', 'end', 'complete', 'incomplete_reason', 'inferred_end',
                'trial_id', 'block', 'order', 'direction', 'angle', 'start_x', 'start_y',
                'target_x', 'target_y', 'acquired', 'miss', 'outcome', 'acquire_time', 'miss_time', 'duration',
                'cursor_samples')


def recording_window_policy(recording: Recording) -> str | None:
    """Exclude the post-task transmission outage on 11-06."""
    if 'analysis_window_policy' in recording.header_info:
        return recording.header_info['analysis_window_policy']
    if run_folder(recording.path) is not None:
        return 'through_task_end'
    return 'through_task_end_or_first_25_min' if recording.date == '2026-11-06' else None


def _limit_stream(stream, start, stop):
    times = stream['time_stamps']
    keep = (times >= start) & (times <= stop)
    values = stream['time_series']
    return dict(stream, time_stamps=times[keep],
                time_series=[values[i] for i in np.flatnonzero(keep)] if isinstance(values, list) else values[keep])


def _direction(segment: np.ndarray, task_mode: str):
    """(index 0..7, angle degrees, start xy, target xy) or a reason string.

    Center-out: target angle from the origin, nearest 45°. Grid: angle from the
    first active cursor sample to the fixed target, bins [-22.5°, 22.5°) = 0°.
    """
    target = segment[0, C.C_TX:C.C_TY + 1].astype(float)
    start = np.zeros(2) if task_mode == 'centerout' else segment[0, C.C_X:C.C_Y + 1].astype(float)
    if not np.isfinite(np.r_[start, target]).all():
        return 'nonfinite_direction_coordinates'
    if task_mode == 'grid' and not np.all(segment[:, C.C_TX:C.C_TY + 1] == target):
        return 'target_changes_within_trial'
    delta = target - start
    if np.linalg.norm(delta) <= 1e-9:
        return 'zero_direction_displacement'
    angle = float(np.degrees(np.arctan2(delta[1], delta[0])))
    if task_mode == 'centerout':
        # Majority target within the trial (robust to a single stale sample).
        values, counts = np.unique(np.round(segment[:, C.C_TX:C.C_TY + 1], 6), axis=0, return_counts=True)
        tx, ty = values[counts.argmax()]
        angle = float(np.degrees(np.arctan2(ty, tx)))
        index = int(np.rint(angle / 45)) % 8
    else:
        scaled = (angle + 22.5) / 45
        if abs(scaled - np.rint(scaled)) <= 1e-12:
            scaled = float(np.rint(scaled))
        index = int(np.floor(scaled)) % 8
    return index, angle % 360, start, target


@dataclass
class Session:
    recording: Recording
    counts: np.ndarray            # frames x 512, small integers
    neural_times: np.ndarray      # receive timestamps (s)
    metadata: np.ndarray          # frames x 4|5
    metadata_labels: list
    cursor_times: np.ndarray
    cursor: np.ndarray            # samples x 12
    event_times: np.ndarray
    events: list                  # [event_type, code, system_ns, iso]
    electrode_ids: np.ndarray     # one-based electrode id per column
    decoder_times: np.ndarray | None = None
    decoder_velocity: np.ndarray | None = None
    stream_audit: list = field(default_factory=list)
    header: dict = field(default_factory=dict)
    metadata_times: np.ndarray | None = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, recording: Recording | str | Path) -> 'Session':
        if not isinstance(recording, Recording):
            path = Path(recording)
            recording = describe(path, path.parent)
        streams, audit = load_streams(recording.path)
        missing = [n for n in C.REQUIRED_STREAMS if n not in streams]
        if missing:
            raise ValueError(f'XDF 缺少数据流：{missing}')
        window_policy = recording_window_policy(recording)
        window_note = ''
        if window_policy is not None:
            start = streams['spike_counts']['time_stamps'][0]
            stop = (np.inf if window_policy == 'through_task_end' else start + 25 * 60)
            events = streams['cursortasks_events']
            ends = [t for t, row in zip(events['time_stamps'], events['time_series'])
                    if start < t <= stop and row[0] == 'T_SESSION_END']
            if ends:
                stop = max(ends)
                window_note = f'仅分析首个神经帧至 T_SESSION_END（{(stop - start) / 60:.2f} 分钟）。'
            elif window_policy != 'through_task_end':
                window_note = '前 25 分钟内没有 T_SESSION_END，分析范围使用首个神经帧起的前 25 分钟。'
            else:
                window_note = '未记录 T_SESSION_END，保留 XDF 的全部录制范围；不按完整行为 CSV 补齐。'
            analysis_streams = (*C.REQUIRED_STREAMS, C.DECODER_STREAM)
            for name in analysis_streams:
                if name in streams:
                    streams[name] = _limit_stream(streams[name], start, stop)
        spikes = streams['spike_counts']
        counts = np.asarray(spikes['time_series'])
        if counts.ndim != 2 or counts.shape[1] != C.N_COLUMNS or not np.isfinite(counts).all():
            raise ValueError('spike_counts 必须为 512 列有限计数')
        if counts.min() < 0 or not np.array_equal(counts, np.rint(counts)):
            raise ValueError('spike_counts 不是非负整数计数')
        meta = streams['spike_counts_metadata']
        from .xdf_io import channel_labels
        decoder = streams.get(C.DECODER_STREAM)
        header = {k: desc_value(spikes, k) for k in
                  ('window_ms', 'num_channels_total', 'protocol', 'window_ms_basis', 'device_timestamp_basis',
                   'channel_mapping_generation', 'profile_id')}
        header['task_mode'] = desc_value(streams['cursortasks_events'], 'task_mode')
        header['session_name'] = desc_value(streams['cursortasks_events'], 'session_name')
        header['subject_id'] = desc_value(streams['cursortasks_events'], 'subject_id')
        header['spike_stream_name'] = spikes['info']['name'][0]
        if window_note:
            header['analysis_window_note'] = window_note
        session = cls(
            recording=recording,
            counts=counts.astype(np.uint8 if counts.max() < 256 else np.int32),
            neural_times=np.asarray(spikes['time_stamps'], float),
            metadata=np.asarray(meta['time_series'], float),
            metadata_labels=channel_labels(meta),
            cursor_times=np.asarray(streams['cursortasks_cursor_state']['time_stamps'], float),
            cursor=np.asarray(streams['cursortasks_cursor_state']['time_series'], float),
            event_times=np.asarray(streams['cursortasks_events']['time_stamps'], float),
            events=[list(e) for e in streams['cursortasks_events']['time_series']],
            electrode_ids=electrode_ids(spikes),
            decoder_times=np.asarray(decoder['time_stamps'], float) if decoder else None,
            decoder_velocity=np.asarray(decoder['time_series'], float) if decoder else None,
            stream_audit=audit, header=header,
            metadata_times=np.asarray(meta['time_stamps'], float))
        if len(session.electrode_ids) != C.N_COLUMNS:
            raise ValueError('电极 ID 数量与 512 列不一致')
        return session

    # --------------------------------------------------------------- properties
    @property
    def task_mode(self) -> str:
        return self.recording.task_mode

    @property
    def label(self) -> str:
        return self.recording.label

    @property
    def frame_seconds(self) -> float:
        try:
            return float(self.header.get('window_ms') or 20) / 1000
        except (TypeError, ValueError):
            return C.FRAME_SECONDS

    @cached_property
    def effective_frame_period(self) -> float:
        """Frame period for coverage checks, in seconds."""
        if (self.header.get('spike_stream_name') == 'spike_counts_pipe'
                and 'device_timestamp_100us' in self.metadata_labels
                and np.array_equal(self.metadata_times, self.neural_times)
                and len(self.neural_times) > 1):
            clock = self.metadata[:, self.metadata_labels.index('device_timestamp_100us')]
            steps = np.diff(clock)
            host_span = self.neural_times[-1] - self.neural_times[0]
            # Confirm the effective 1 ms clock unit against the whole recording.
            if (steps.min() > 0 and host_span > 0
                    and np.isclose((clock[-1] - clock[0]) / 1000, host_span, rtol=0.02)):
                regular = np.abs(steps - np.median(steps)) <= 1
                if regular.mean() >= 0.9:
                    return float(steps[regular].mean() / 1000)
        # Legacy estimate from receive intervals, excluding long outages.
        dt = np.diff(self.neural_times)
        span = dt[dt <= 0.5].sum()
        n = int((dt <= 0.5).sum())
        return float(span / n) if n else self.frame_seconds

    @cached_property
    def rates(self) -> np.ndarray:
        """Frames x 512 in nominal Hz (count / configured window)."""
        return self.counts.astype(np.float32) / np.float32(self.frame_seconds)

    @cached_property
    def neural_cumsum(self) -> np.ndarray:
        """Prefix sums of counts for O(1) window means; row 0 is zero."""
        out = np.zeros((len(self.counts) + 1, self.counts.shape[1]), np.float64)
        np.cumsum(self.counts, axis=0, out=out[1:])
        return out

    @cached_property
    def trials(self) -> list[dict]:
        return reconstruct_trials(self)

    def trial_array(self, key, rows=None, dtype=float) -> np.ndarray:
        rows = self.trials if rows is None else [self.trials[i] for i in rows]
        return np.array([r[key] for r in rows], dtype=dtype)

    # ---------------------------------------------------------- neural windows
    def window_stats(self, lefts, rights):
        """Frame count and max boundary-inclusive gap for many [left, right) windows."""
        nt = self.neural_times
        lefts, rights = np.asarray(lefts, float), np.asarray(rights, float)
        lo = np.searchsorted(nt, lefts, side='left')
        hi = np.searchsorted(nt, rights, side='left')
        n = hi - lo
        gap = np.empty(len(lefts))
        for i, (a, b, l, r) in enumerate(zip(lo, hi, lefts, rights)):
            gap[i] = np.max(np.diff(np.r_[l, nt[a:b], r])) if b > a else r - l
        return lo, hi, n, gap

    def window_mean_rates(self, lefts, rights) -> tuple[np.ndarray, np.ndarray]:
        """Mean Hz per window (NaN rows where no frame) and frame counts."""
        lo, hi, n, _ = self.window_stats(lefts, rights)
        cs = self.neural_cumsum
        total = cs[hi] - cs[lo]
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = total / n[:, None] / self.frame_seconds
        mean[n == 0] = np.nan
        return mean, n

    def binned_rates(self, anchors, tmin, tmax, bin_s) -> tuple[np.ndarray, np.ndarray]:
        """Event-aligned rates: (n_anchor, n_bin, 512) Hz, NaN where a bin has no frame."""
        edges = np.arange(tmin, tmax + 1e-9, bin_s)
        anchors = np.asarray(anchors, float)
        lefts = (anchors[:, None] + edges[None, :-1]).ravel()
        rights = (anchors[:, None] + edges[None, 1:]).ravel()
        mean, _ = self.window_mean_rates(lefts, rights)
        return mean.reshape(len(anchors), len(edges) - 1, -1), edges

    def active_cursor(self, row: dict) -> np.ndarray:
        """Indices of active cursor samples inside a trial interval."""
        ct, cc = self.cursor_times, self.cursor
        if not np.isfinite(row['start']) or not np.isfinite(row['end']):
            return np.zeros(0, int)
        lo, hi = np.searchsorted(ct, [row['start'], row['end']], side='left')
        idx = np.arange(lo, min(hi + 1, len(ct)))
        idx = idx[(ct[idx] <= row['end']) & (cc[idx, C.C_ACTIVE] == 1)]
        return idx


def reconstruct_trials(session: Session) -> list[dict]:
    et, events = session.event_times, session.events
    types = [e[0] for e in events]
    rows: list[dict] = []

    def new_row(start, start_event):
        return dict(start=float(start), end=np.nan, complete=False, incomplete_reason='', inferred_end=False,
                    trial_id=-1, block=-1, order=-1, direction=-1, angle=np.nan, start_x=np.nan, start_y=np.nan,
                    target_x=np.nan, target_y=np.nan, acquired=0, miss=0, outcome='unknown',
                    acquire_time=np.nan, miss_time=np.nan, duration=np.nan, cursor_samples=0, cursor_start_x=np.nan,
                    cursor_start_y=np.nan, n_targets=0, target_radius=np.nan, direction_offset_deg=np.nan,
                    _start_event=start_event)

    active = None
    for i, (t, kind) in enumerate(zip(et, types)):
        if kind == 'T_TRIAL_START':
            if active is not None:
                # A new START while a trial is open: the open trial never ended (nested).
                active['incomplete_reason'] = 'nested_start'
                active['_next_start'] = float(t)
                rows.append(active)
            active = new_row(t, i)
        elif kind in ('T_TARGET_ACQUIRED', 'MISS') and active is not None:
            if kind == 'T_TARGET_ACQUIRED':
                active['acquired'] += 1
                if not np.isfinite(active['acquire_time']):
                    active['acquire_time'] = float(t) - active['start']
            else:
                active['miss'] += 1
                if not np.isfinite(active['miss_time']):
                    active['miss_time'] = float(t) - active['start']
        elif kind == 'T_TRIAL_END':
            if active is None:
                row = new_row(np.nan, -1)
                row.update(end=float(t), incomplete_reason='end_without_start')
                rows.append(row)
                continue
            active['end'] = float(t)
            active['complete'] = True
            rows.append(active)
            active = None
    if active is not None:
        active['incomplete_reason'] = 'start_without_end'
        rows.append(active)

    # Candidate end for start-without-end trials: last event before the next start.
    for row in rows:
        if row['incomplete_reason'] in ('start_without_end', 'nested_start'):
            limit = row.pop('_next_start', np.inf)
            later = [t for t, k in zip(et, types) if row['start'] < t < limit and k != 'T_TRIAL_START']
            row['_candidate_end'] = float(max(later)) if later else np.nan

    for index, row in enumerate(rows):
        row['trial_index'] = index
        row.pop('_start_event', None)
        # Exactly one acquisition and no MISS = success; repeated acquisitions or
        # acquisition + MISS are ambiguous and never silently counted as success.
        if row['acquired'] == 1 and row['miss'] == 0:
            row['outcome'] = 'success'
        elif row['miss'] >= 1 and row['acquired'] == 0:
            row['outcome'] = 'miss'
        elif row['acquired'] or row['miss']:
            row['outcome'] = 'ambiguous'
        if not np.isfinite(row['start']):
            continue
        end = row['end'] if row['complete'] else row.get('_candidate_end', np.nan)
        probe = dict(row, end=end)
        idx = session.active_cursor(probe)
        row['cursor_samples'] = int(len(idx))
        if not len(idx):
            row['incomplete_reason'] = row['incomplete_reason'] or 'no_active_cursor_samples'
            continue
        segment = session.cursor[idx]
        identity = segment[:, [C.C_TRIAL, C.C_BLOCK, C.C_ORDER]]
        if np.any(identity != identity[0]):
            # Keep the majority identity but flag it.
            values, counts = np.unique(identity, axis=0, return_counts=True)
            identity0 = values[counts.argmax()]
            row['incomplete_reason'] = row['incomplete_reason'] or 'cursor_identity_changes_within_trial'
        else:
            identity0 = identity[0]
        row['trial_id'], row['block'], row['order'] = (int(v) for v in identity0)
        row['cursor_start_x'], row['cursor_start_y'] = map(float, segment[0, C.C_X:C.C_Y + 1])
        row['n_targets'] = int(len(np.unique(np.round(segment[:, C.C_TX:C.C_TY + 1], 6), axis=0)))
        label = _direction(segment, session.task_mode)
        if isinstance(label, str):
            row['incomplete_reason'] = row['incomplete_reason'] or label
        else:
            row['direction'], row['angle'] = label[0], label[1]
            row['start_x'], row['start_y'] = map(float, label[2])
            row['target_x'], row['target_y'] = map(float, label[3])
            row['target_radius'] = float(np.hypot(row['target_x'] - row['start_x'], row['target_y'] - row['start_y']))
            row['direction_offset_deg'] = float((row['angle'] - row['direction'] * 45 + 180) % 360 - 180)                 if session.task_mode == 'centerout' else 0.0
        if row['complete']:
            row['duration'] = row['end'] - row['start']
        elif np.isfinite(end):
            row['_candidate_duration'] = end - row['start']
    # The same trial number on two event-defined trials is a reconstruction conflict.
    ids = [r['trial_id'] for r in rows if r['trial_id'] >= 0]
    duplicated = {i for i in ids if ids.count(i) > 1}
    for row in rows:
        if row['trial_id'] in duplicated:
            row['incomplete_reason'] = row['incomplete_reason'] or 'trial_id_conflict'
    return rows


# ------------------------------------------------------------------ caching
_CACHE: dict[str, Session] = {}
_LOCK = threading.Lock()


def get_session(recording: Recording, max_cached: int = 6) -> Session:
    """Process-wide session cache keyed by file path (loading is ~1 s per file)."""
    key = str(recording.path.resolve())
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
    session = Session.load(recording)
    with _LOCK:
        _CACHE[key] = session
        while len(_CACHE) > max_cached:
            _CACHE.pop(next(iter(_CACHE)))
    return session


def clear_cache():
    with _LOCK:
        _CACHE.clear()
