"""Online CursorTasks outcomes and task-specific throughput."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime
import math
from pathlib import Path
import re

import numpy as np


BPS_LABELS = {'centerout': 'Fitts throughput (bits/s)', 'grid': 'Achieved bitrate (bits/s)'}
TASK_LABELS = {'centerout': 'CO8', 'grid': 'Grid'}
METHOD = ('Online sessions only. Task outcomes are independent of the neural trial filter. Success rate = '
          'acquisitions / attempts with a known outcome, including timeouts and misses. RT = first target '
          'acquisition minus trial start, for successful attempts; markers are used when a matching trials.csv '
          'is unavailable. RT includes movement and any acquisition dwell. Report median and interquartile range. '
          'CO8 Fitts throughput = sum of nominal difficulty bits over successful trials / sum of their outbound '
          'movement times, using per-block difficulty from block_metrics.csv. Grid achieved bitrate = successful '
          'selections × log₂(number of grid cells) / total active attempt duration, including timeouts and excluding '
          'inter-trial intervals. Grid dimensions come from the session log. BPS is missing when these inputs '
          'cannot be verified.')


from .archive_logs import archived_logs


def source_files(recording):
    """Find a behavior bundle assigned to exactly one nearby XDF."""
    folder = recording.path.parent
    root = folder.parent if folder.name.lower() == 'xdf' else folder
    trials = root / 'behavior' / 'trials.csv'
    if not trials.is_file():
        trials = root / 'trials.csv'
    if not trials.is_file():
        return {}
    recordings = [*root.glob('*.xdf'), *(root / 'xdf').glob('*.xdf')]
    if len(recordings) != 1 or recordings[0].resolve() != recording.path.resolve():
        # A normalized session can retain both early and main recording parts.
        # Only its explicitly annotated main recording represents the complete
        # CSV; assigning that CSV to the early fragment would double-count it.
        from .catalog import session_metadata
        metadata = session_metadata(recording.path)
        main = [r for r in metadata.get('recordings', []) if r.get('segment') == 'main']
        if len(main) != 1 or recording.path.name != Path(main[0].get('path', '')).name:
            return {}
    files = {'trials': trials}
    if recording.task_mode == 'centerout':
        blocks = trials.with_name('block_metrics.csv')
        if blocks.is_file():
            files['blocks'] = blocks
    else:
        logs = archived_logs(root / 'logs', 'grid-task*.log')
        if logs:
            files.update({f'log_{i}': path for i, path in enumerate(logs)})
        else:
            fallback = (archived_logs(root / 'logs', 'cursor_tasks_app.log') or
                        archived_logs(root / 'logs', 'task.log'))
            if fallback:
                files['log_0'] = fallback[0]
    return files


def _read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _positive(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return np.nan
    return number if math.isfinite(number) and number > 0 else np.nan


def _rt(row):
    try:
        delta = datetime.fromisoformat(row['acquisition_iso']) - datetime.fromisoformat(row['trial_start_iso'])
    except (KeyError, ValueError, TypeError):
        return np.nan
    return _positive(delta.total_seconds())


def _from_csv(files, mode):
    raw = _read(files['trials'])
    if not raw or any(r.get('task_mode', '').lower() != mode for r in raw):
        raise ValueError('trials.csv 为空或任务类型与 XDF 不一致')
    if len({r['session_timestamp_iso'] for r in raw}) != 1 or len({r['trial_num_overall'] for r in raw}) != len(raw):
        raise ValueError('trials.csv 含多个会话或重复 trial')
    difficulty = {}
    if 'blocks' in files:
        for row in _read(files['blocks']):
            if row.get('task_mode') == mode:
                difficulty[int(row['block_num'])] = _positive(row['fitts_index_of_difficulty_bits'])
    grid_sizes = set()
    for key, path in files.items():
        if key.startswith('log_'):
            log = path.read_text(encoding='utf-8', errors='replace')
            grid_sizes.update(int(a) * int(b) for a, b in re.findall(r'Grid [^\n]*\brows=(\d+),\s*cols=(\d+)', log))
    targets = next(iter(grid_sizes)) if len(grid_sizes) == 1 else None
    trials = []
    for row in raw:
        block, outcome = int(row['block_num']), row['outcome'].strip().lower()
        success = outcome == 'acquired'
        bits = difficulty.get(block, np.nan) if mode == 'centerout' else (
            math.log2(targets) if targets and targets > 1 else np.nan)
        trials.append(dict(block=block, trial=int(row['trial_num_overall']), outcome=outcome,
                           rt_s=_rt(row) if success else np.nan, bits=bits,
                           duration_s=_rt(row) if success else _positive(row.get('trial_duration_sec')),
                           timing_source='start_to_acquisition' if success else 'active_trial_duration',
                           outbound_s=_positive(row.get('outbound_movement_time_sec'))))
    return trials


def collect(session):
    files = source_files(session.recording)
    note = ''
    if files:
        try:
            return _from_csv(files, session.task_mode), str(files['trials']), note
        except (KeyError, ValueError, OSError, TypeError, csv.Error) as exc:
            note = f'行为 CSV 无法使用（{exc}），回退到 XDF 事件。'
    else:
        note = '缺少可与这份 XDF 唯一匹配的行为 CSV，使用 XDF 事件。'
    trials = [dict(block=r['block'] if r['block'] >= 0 else None, trial=r['trial_index'] + 1,
                   outcome={'success': 'acquired', 'miss': 'miss'}.get(r['outcome'], 'unknown'),
                   rt_s=_positive(r['acquire_time']) if r['outcome'] == 'success' else np.nan,
                   duration_s=_positive(r['acquire_time'] if r['outcome'] == 'success' else r.get('miss_time')),
                   timing_source='start_to_acquisition' if r['outcome'] == 'success' else 'start_to_miss',
                   outbound_s=np.nan, bits=np.nan)
              for r in session.trials if r['complete']]
    return trials, 'XDF event markers', note


def metrics(trials, mode):
    outcomes = Counter(r['outcome'] for r in trials)
    attempts = [r for r in trials if r['outcome'] in ('acquired', 'timeout', 'miss', 'incorrect', 'wrong_target')]
    successes = [r for r in attempts if r['outcome'] == 'acquired']
    rt = [r['rt_s'] for r in successes if np.isfinite(r['rt_s'])]
    q1, median, q3 = np.quantile(rt, [.25, .5, .75]) if rt else [np.nan] * 3
    if mode == 'centerout':
        duration = [r['outbound_s'] for r in successes]
        bits = [r['bits'] for r in successes]
    else:
        duration = [r['duration_s'] for r in attempts]
        bits = [r['bits'] if r['outcome'] == 'acquired' else 0 for r in attempts]
        if not all(np.isfinite(r['bits']) for r in attempts):
            bits = [np.nan]
    unknown = sum(outcomes.values()) - len(attempts)
    bps = (sum(bits) / sum(duration) if duration and not unknown
           and np.isfinite(duration).all() and np.isfinite(bits).all()
           and sum(duration) > 0 else np.nan)
    return dict(attempts=len(attempts), successes=len(successes), unknown_outcomes=unknown,
                success_pct=100 * len(successes) / len(attempts) if attempts else np.nan, bps=bps,
                rt_n=len(rt), rt_median_s=float(median), rt_q1_s=float(q1), rt_q3_s=float(q3))


def block_success_summary(trials):
    """Distribution of original task-block success rates, independent of neural filters."""
    blocks = defaultdict(list)
    for trial in trials:
        if trial.get('block') is not None:
            blocks[trial['block']].append(trial)
    known = {'acquired', 'timeout', 'miss', 'incorrect', 'wrong_target'}
    rates = []
    for group in blocks.values():
        attempts = [r for r in group if r['outcome'] in known]
        if attempts:
            rates.append(100 * sum(r['outcome'] == 'acquired' for r in attempts) / len(attempts))
    q1, q3 = np.quantile(rates, [.25, .75]) if rates else [np.nan] * 2
    attempts = [r for r in trials if r['outcome'] in known]
    mean = 100 * sum(r['outcome'] == 'acquired' for r in attempts) / len(attempts) if attempts else np.nan
    return dict(success_pct_q1=float(q1), success_pct_mean=float(mean),
                success_pct_q3=float(q3), success_blocks=len(rates))


def run_success_summaries(sessions):
    """Prefer a run's complete CSV over its selected early XDF fragments."""
    from .catalog import run_folder
    groups = defaultdict(list)
    identities = []
    for session in sessions:
        rec = session.recording
        folder = run_folder(rec.path)
        identity = str(folder.resolve()) if folder else (rec.date, rec.task_mode, rec.online, rec.run or rec.key)
        identities.append(identity)
        if session.task_mode in TASK_LABELS:
            trials, source, note = collect(session)
            groups[identity].append((trials, source, note))
    summaries = {}
    for identity in identities:
        entries = groups[identity]
        csv_entries = [entry for entry in entries if entry[1] != 'XDF event markers']
        selected = csv_entries[:1] if csv_entries else entries
        trials = [row for rows, _, _ in selected for row in rows]
        summaries[identity] = dict(block_success_summary(trials),
                                   success_source='; '.join(dict.fromkeys(source for _, source, _ in selected)))
    return [summaries[identity] for identity in identities]
