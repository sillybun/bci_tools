"""Synthetic center-out/grid sessions with the real stream layout (tests and demo)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config as C
from .catalog import Recording
from .session import Session


def make_session(seed=0, n_blocks=12, task_mode='centerout', online=False, date='2026-01-01',
                 electrode_ids=None, tuned_fraction=0.3, fpga=True, drop_every=4000, gap_at=None,
                 miss_every=17, saturate_channels=8, run='001') -> Session:
    rng = np.random.default_rng(seed)
    frame = C.FRAME_SECONDS
    ids = np.asarray(electrode_ids if electrode_ids is not None else np.arange(1, 513))
    n_ch = len(ids)
    preferred = rng.uniform(0, 2 * np.pi, n_ch)
    depth = np.where(rng.random(n_ch) < tuned_fraction, rng.uniform(5, 30, n_ch), 0)
    base = rng.gamma(2.0, 6.0, n_ch)
    base[rng.random(n_ch) < 0.15] = 0                     # silent channels
    base[:saturate_channels] = 140                        # near the 2-bit ceiling

    events, event_times, cursor_rows, cursor_times = [], [], [], []
    t = 1.0
    trial_overall = 0
    schedule = []
    for block in range(1, n_blocks + 1):
        events.append(['T_BLOCK_START', '3', '0', '']); event_times.append(t); t += 0.05
        for order, direction in enumerate(rng.permutation(8), start=1):
            trial_overall += 1
            angle = np.radians(direction * 45)
            target = np.array([np.cos(angle), np.sin(angle)]) * 0.4
            start_xy = np.zeros(2) if task_mode == 'centerout' else rng.uniform(-0.3, 0.3, 2)
            if task_mode == 'grid':
                target = start_xy + np.array([np.cos(angle), np.sin(angle)]) * 0.3
            miss = miss_every and trial_overall % miss_every == 0
            duration = 11.0 if miss else rng.uniform(1.6, 3.0)
            start = t
            events.append(['T_TRIAL_START', '5', '0', '']); event_times.append(start)
            # cursor at 60 Hz while active
            ts = np.arange(start + 0.005, start + duration, 1 / 60)
            reach = np.clip((ts - start - 0.25) / 0.8, 0, 1)[:, None]
            pos = start_xy + (target - start_xy) * (reach if not miss else 0.3 * reach)
            pos = pos + rng.normal(0, 0.004, pos.shape)
            for ti, p in zip(ts, pos):
                cursor_times.append(ti)
                cursor_rows.append([0, 0, p[0], p[1], 0, 0, target[0], target[1], block, order, trial_overall, 1])
            schedule.append((start, start + duration, angle, miss))
            if miss:
                events.append(['MISS', '9', '0', '']); event_times.append(start + duration - 0.01)
            else:
                events.append(['T_TARGET_ACQUIRED', '13', '0', '']); event_times.append(start + duration - 0.22)
            events.append(['T_TRIAL_END', '6', '0', '']); event_times.append(start + duration)
            events.append(['T_ITI_START', '11', '0', '']); event_times.append(start + duration + 0.005)
            iti_ts = np.arange(start + duration + 0.02, start + duration + 1.0, 1 / 60)
            for ti in iti_ts:
                cursor_times.append(ti)
                cursor_rows.append([0, 0, 0, 0, 0, 0, target[0], target[1], block, order, trial_overall, 0])
            events.append(['T_ITI_END', '12', '0', '']); event_times.append(start + duration + 1.0)
            t = start + duration + 1.01
        events.append(['T_BLOCK_END', '4', '0', '']); event_times.append(t); t += 0.05
    # one orphan end, as observed in real files
    events.insert(0, ['T_TRIAL_END', '6', '0', '']); event_times.insert(0, 0.5)

    neural_times = np.arange(0.4, t + 0.5, frame)
    if gap_at is not None:
        keep = ~((neural_times > gap_at) & (neural_times < gap_at + 0.5))
        neural_times = neural_times[keep]
    rate = np.tile(base, (len(neural_times), 1))
    for start, end, angle, miss in schedule:
        a, b = np.searchsorted(neural_times, [start + 0.1, start + 1.4])
        rate[a:b] += np.maximum(depth * np.cos(angle - preferred), -base)
    counts = np.minimum(rng.poisson(np.clip(rate, 0, None) * frame), C.MAX_CODE).astype(np.uint8)
    n = len(neural_times)
    window = np.arange(n, dtype=float)
    start_us = np.round((neural_times - frame) * 1e6)
    meta_cols = [window, start_us, start_us + 20000, counts.sum(1)]
    labels = ['window_index', 'start_timestamp', 'end_timestamp', 'raw_total_count']
    if fpga:
        steps = np.full(n, 19.2)
        if drop_every:
            steps[drop_every::drop_every] = 19.2 * 3
        meta_cols.append(np.round(np.cumsum(steps)))
        labels.append('device_timestamp_100us')
    order = np.argsort(event_times, kind='stable')
    recording = Recording(Path(f'synthetic_{seed}.xdf'), f'synthetic/{date}/{seed}.xdf', date, task_mode,
                          online, run, '', fpga)
    session = Session(
        recording=recording, counts=counts, neural_times=neural_times, metadata=np.column_stack(meta_cols),
        metadata_labels=labels, cursor_times=np.asarray(cursor_times), cursor=np.asarray(cursor_rows, float),
        event_times=np.asarray(event_times)[order], events=[events[i] for i in order], electrode_ids=ids,
        header={'window_ms': '20', 'task_mode': task_mode})
    if online:
        # Online decoder velocity: noisy copy of the cursor velocity at the neural rate.
        ct, cc = session.cursor_times, session.cursor
        vx = np.gradient(cc[:, C.C_X], ct)
        vy = np.gradient(cc[:, C.C_Y], ct)
        dv = np.column_stack([np.interp(neural_times, ct, vx), np.interp(neural_times, ct, vy)])
        session.decoder_times = neural_times.copy()
        session.decoder_velocity = dv + rng.normal(0, 0.05, dv.shape)
    return session
