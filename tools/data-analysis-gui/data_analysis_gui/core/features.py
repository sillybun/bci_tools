"""Behaviour-aligned bins: cursor-displacement velocity and trailing neural windows.

Same protocol as both source projects: bin k covers [t0+k·Δ, t0+(k+1)·Δ) after
T_TRIAL_START + offset; velocity = (p(right) − p(left)) / Δ from positions linearly
interpolated between bracketing active samples of the same trial (gap ≤ cursor_gap);
neural feature = mean Hz over the trailing `neural_window` ending at the right edge,
with ≥ min_frames frames and a boundary-inclusive gap ≤ max_gap.

Velocity-bin quality (criterion 8, velocity cohort only) and its thresholds come from
`cohort.criteria` unless overridden: `velocity_bin_policy = drop_trial` removes every
bin of a trial with any bad bin in the evaluated range; `drop_bin` keeps the good bins.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config as C
from .filters import Cohort

BIN_REASON = {'beyond_trial_end': 'velocity_beyond_trial_end', 'cursor_gap': 'velocity_cursor_gap',
              'cursor_foreign': 'velocity_cursor_foreign', 'neural_frames_or_gap': 'velocity_neural_short_window'}


@dataclass
class VelocityBins:
    features: np.ndarray       # bins x 512 (Hz)
    velocity: np.ndarray       # bins x 2 (task units / s)
    position: np.ndarray       # bins x 2 at the left edge
    trial: np.ndarray          # cohort row index per bin
    bin: np.ndarray            # bin index within trial
    sequence: np.ndarray       # contiguous run id (never spans gaps or trials)
    block: np.ndarray
    direction: np.ndarray
    time: np.ndarray           # right-edge receive time
    exclusions: dict           # reason -> bin count
    trial_complete: np.ndarray  # per cohort row: all bins of the evaluated range valid
    audit: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.velocity)

    def subset(self, mask):
        keys = ('features', 'velocity', 'position', 'trial', 'bin', 'sequence', 'block', 'direction', 'time')
        values = {k: getattr(self, k)[mask] for k in keys}
        return VelocityBins(**values, exclusions=self.exclusions, trial_complete=self.trial_complete, audit=self.audit)


def _interp(times, positions, t, max_gap):
    j = int(np.searchsorted(times, t, side='left'))
    if j < len(times) and times[j] == t:
        return positions[j]
    if j == 0 or j == len(times):
        return None
    if times[j] - times[j - 1] > max_gap + 1e-9:
        return None
    frac = (t - times[j - 1]) / (times[j] - times[j - 1])
    return positions[j - 1] + frac * (positions[j] - positions[j - 1])


def velocity_bins(cohort: Cohort, offset=None, bin_s=0.1, n_bins: int | None = None, until='window',
                  neural_window=None, min_frames=None, max_gap=None, cursor_gap=None, policy=None,
                  audit=False) -> VelocityBins:
    """Bins per direction-cohort trial.

    offset defaults to the analysis-window start; with until='window' n_bins defaults
    to the number of whole bins in the analysis window. until='end' uses every full
    bin until T_TRIAL_END, until='acquire' stops at the first T_TARGET_ACQUIRED.
    """
    crit = cohort.criteria
    offset = crit.window_start if offset is None else offset
    if n_bins is None and until == 'window':
        n_bins = max(1, int(np.floor((crit.window_end - offset) / bin_s + 1e-8)))
    neural_window = crit.velocity_neural_window if neural_window is None else neural_window
    min_frames = crit.velocity_min_frames if min_frames is None else min_frames
    max_gap = crit.velocity_max_gap if max_gap is None else max_gap
    cursor_gap = crit.velocity_cursor_gap if cursor_gap is None else cursor_gap
    policy = crit.velocity_bin_policy if policy is None else policy
    s = cohort.session
    ct, cc = s.cursor_times, s.cursor
    reasons: dict[str, int] = {}
    trial_reasons = [[] for _ in range(cohort.n)]
    complete = np.zeros(cohort.n, bool)
    seq = 0
    lefts_all, rights_all, meta = [], [], []
    for i, row in enumerate(cohort.rows):
        idx = s.active_cursor(row)
        foreign = cc[idx, C.C_TRIAL] != row['trial_id']
        times, positions = ct[idx][~foreign], cc[idx][~foreign][:, C.C_X:C.C_Y + 1]
        if until == 'acquire' and np.isfinite(row['acquire_time']):
            limit = row['acquire_time']
        else:
            limit = row['end'] - row['start']
        count = int(np.floor((limit - offset) / bin_s + 1e-8))
        if n_bins is not None:
            count = min(count, n_bins) if until != 'window' else n_bins
        valid_all = count > 0 and (n_bins is None or count == n_bins)
        run_open = False
        for k in range(max(count, 0)):
            left = row['start'] + offset + k * bin_s
            right = left + bin_s
            why = ''
            if right > row['end'] + 1e-9:
                why = 'beyond_trial_end'
            else:
                p0 = _interp(times, positions, left, cursor_gap)
                p1 = _interp(times, positions, right, cursor_gap)
                if p0 is None or p1 is None:
                    # Distinguish a sampling gap from samples that belong to another trial.
                    near = (ct[idx] >= left - cursor_gap) & (ct[idx] <= right + cursor_gap)
                    why = 'cursor_foreign' if np.any(foreign & near) else 'cursor_gap'
            if why:
                reasons[why] = reasons.get(why, 0) + 1
                trial_reasons[i].append(BIN_REASON[why])
                valid_all = False
                run_open = False
                continue
            if not run_open:
                seq += 1
                run_open = True
            lefts_all.append(right - neural_window)
            rights_all.append(right)
            meta.append((i, k, seq, p0, p1, right))
        complete[i] = valid_all
    out = {k: [] for k in ('features', 'velocity', 'position', 'trial', 'bin', 'sequence', 'block', 'direction', 'time')}
    if meta:
        _, _, n, gap = s.window_stats(lefts_all, rights_all)
        feats, _ = s.window_mean_rates(lefts_all, rights_all)
        ok = (n >= min_frames) & (gap <= max_gap + 1e-9)
        reasons['neural_frames_or_gap'] = int((~ok).sum())
        for m in np.flatnonzero(~ok):
            i = meta[m][0]
            complete[i] = False
            trial_reasons[i].append(BIN_REASON['neural_frames_or_gap'])
        keep_trial = complete if policy == 'drop_trial' else np.ones(cohort.n, bool)
        blocks, directions = cohort.blocks, cohort.directions
        last_seq, seq_new = None, 0
        for m, (i, k, sq, p0, p1, right) in enumerate(meta):
            if not ok[m] or not keep_trial[i]:
                last_seq = None
                continue
            if sq != last_seq or (out['bin'] and out['bin'][-1] != k - 1) or (out['trial'] and out['trial'][-1] != i):
                seq_new += 1
            last_seq = sq
            out['features'].append(feats[m])
            out['velocity'].append((p1 - p0) / bin_s)
            out['position'].append(p0)
            out['trial'].append(i)
            out['bin'].append(k)
            out['sequence'].append(seq_new)
            out['block'].append(blocks[i])
            out['direction'].append(directions[i])
            out['time'].append(right)
    included = complete.copy() if policy == 'drop_trial' else np.isin(np.arange(cohort.n), out['trial'])
    audit_info = dict(included=included, reasons=[sorted(set(r)) for r in trial_reasons],
                      bins_bad=np.array([len(r) for r in trial_reasons]), policy=policy,
                      settings=dict(offset=offset, bin_s=bin_s, n_bins=n_bins, until=until, neural_window=neural_window,
                                    min_frames=min_frames, max_gap=max_gap, cursor_gap=cursor_gap))
    if not out['trial']:
        empty = np.zeros((0, 2))
        return VelocityBins(np.zeros((0, C.N_COLUMNS)), empty, empty, *(np.zeros(0, int) for _ in range(5)),
                            np.zeros(0), reasons, complete, audit_info)
    arrays = {k: np.asarray(v) for k, v in out.items()}
    return VelocityBins(**arrays, exclusions=reasons, trial_complete=complete, audit=audit_info)
