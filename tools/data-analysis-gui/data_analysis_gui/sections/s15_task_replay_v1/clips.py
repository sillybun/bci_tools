"""Locate replayable 60-second BPS peaks on the original recording clock."""
import csv
import math

import numpy as np


CLIP_CHOICES = ('全程', 'BPS最高60s', '最高60s前后各60s')
CLIP_IDS = dict(zip(CLIP_CHOICES, ('full', 'best60', 'context180')))


def _saved_peak(files, cursor, times, mode):
    path = files.get('rolling_metrics')
    ns = cursor[:, 0]
    if path is None or not np.isfinite(ns).all() or ns[0] < 1e15 or np.any(np.diff(ns) <= 0):
        return None
    system = (ns-ns[0])/1e9
    elapsed = times-times[0]
    if not np.isclose(system[-1], elapsed[-1], rtol=.02, atol=.05):
        return None
    candidates, all_bps = [], []
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            try:
                seconds = float(row['window_sec'])
                end_elapsed = float(row['window_end_elapsed_sec'])
                count = int(row['acquired'])
                cells = int(row['target_count'])
                bps = float(row['achieved_bitrate_bps'])
                stop_system = (int(row['system_timestamp_ns'])-int(ns[0]))/1e9
            except (KeyError, ValueError, TypeError, OverflowError):
                continue
            if (row.get('task_mode') != mode or not math.isclose(seconds, 60., abs_tol=1e-6)
                    or end_elapsed < 60.-1e-6 or count < 0 or cells < 2
                    or not math.isfinite(bps) or bps<0 or not math.isfinite(stop_system)):
                continue
            all_bps.append(bps)
            # Select only windows fully captured by this recording, not a full-session CSV tail.
            if stop_system-60 < -1e-6 or stop_system > system[-1]+1e-6:
                continue
            right = float(np.interp(stop_system, system, elapsed))
            left = right-60.
            if left < -1e-6:
                continue
            candidates.append(dict(start=max(0.,left), end=right, bps=bps,
                                   successes=count, target_count=cells,
                                   source='saved rolling_metrics.csv / achieved_bitrate_bps',
                                   source_timestamp_ns=row['system_timestamp_ns']))
    if not candidates:
        return None
    peak = min(candidates, key=lambda r:(-r['bps'], r['end']))
    peak['whole_task_saved_peak_bps'] = max(all_bps)
    return peak


def replay_clips(files, cursor, times, events, geo, mode):
    """Use saved task BPS only. Never manufacture a metric for missing records."""
    duration = float(times[-1]-times[0])
    full = dict(id='full', label='全程', start=0., end=duration, available=True)
    unavailable = '录制不足完整60秒，或缺少可验证的BPS数据'
    peak = _saved_peak(files, cursor, times, mode) if duration >= 60.-1e-6 else None
    if peak is None:
        return [full, dict(id='best60',label='BPS最高60s',available=False,reason=unavailable),
                dict(id='context180',label='最高60s前后各60s',available=False,reason=unavailable)], None
    best = dict(id='best60',label='BPS最高60s',available=True,**peak)
    left, right = max(0.,peak['start']-60), min(duration,peak['end']+60)
    context = dict(id='context180',label='最高60s前后各60s',available=True,
                   start=left,end=right,peak_start=peak['start'],peak_end=peak['end'],bps=peak['bps'],
                   requested_start=peak['start']-60,requested_end=peak['end']+60,
                   clipped=left>peak['start']-60+1e-6 or right<peak['end']+60-1e-6)
    return [full,best,context], peak
