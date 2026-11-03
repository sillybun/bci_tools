"""Trial-selection criteria (versioned) and the resulting direction / velocity cohorts.

Design rules (see README · Trial 筛选规范):
  * Every trial gets, for every criterion, a status pass / fail / flag / n/a with a
    reason code. A final status and one primary reason are derived from a fixed
    priority order (PRIORITY), so the same trial always reports the same cause.
  * Two cohorts: the direction cohort uses criteria 1 (incomplete), 2 (outcome),
    4 (direction label), timing, window coverage; the velocity cohort additionally
    applies velocity-bin quality (criterion 3) — only velocity/KF analyses use it.
  * Flags never exclude unless the corresponding "exclude flagged" switch is on
    (sensitivity analysis).
  * Trial-level problems are recorded, never raised; only session-level data errors
    (inconsistent duplicate streams, non-monotonic timestamps) stop loading.
  * Thresholds live in TrialFilter, are saved/loaded as a versioned JSON config and
    are part of every result signature — fix them before looking at results.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, fields, replace
from functools import cached_property
import json
from pathlib import Path

import numpy as np

from .session import Session

CRITERIA_VERSION = 'trial-filter-v2'

# Criterion groups in priority order; the first failing group gives the primary reason.
PRIORITY = ('incomplete', 'outcome', 'direction', 'timing', 'window', 'neural', 'block', 'velocity')
GROUP_LABELS = {
    'incomplete': '1 · Incomplete trial', 'outcome': '2 · Success only（含结局不明确）',
    'direction': '3 · 方向标签有效性', 'timing': '4 · Overtime / 过快', 'window': '5 · 窗口须在 trial 内',
    'neural': '6 · 窗口内神经覆盖率', 'block': '7 · block 范围', 'velocity': '8 · 速度 bin 质量（仅速度队列）',
    'flags': 'Flag（不排除）',
}
REASON_LABELS = {
    # incomplete
    'end_without_start': '缺 START：有 T_TRIAL_END 无对应 T_TRIAL_START',
    'start_without_end': '缺 END：记录结束时 trial 仍未结束',
    'nested_start': '嵌套：trial 未结束即出现新的 T_TRIAL_START',
    'trial_id_conflict': 'trial id 冲突：两个事件 trial 使用同一 trial 编号',
    'no_active_cursor_samples': '无 active cursor 样本',
    'cursor_identity_changes_within_trial': 'trial 内 cursor 的 trial/block 编号变化',
    # outcome
    'not_success': '非成功（未命中或 MISS）',
    'ambiguous_outcome': '结局不明确（ACQUIRED ≥2 次，或 ACQUIRED 与 MISS 同时出现）',
    # direction
    'nonfinite_direction_coordinates': '方向坐标非有限值',
    'target_changes_within_trial': 'trial 内 target 变化（不止一个 target）',
    'zero_direction_displacement': '起点与目标重合',
    'direction_angle_off_grid': 'target 角度偏离最近的 45° 方向超过容差',
    'target_radius_out_of_range': 'target 到中心距离超出预期范围',
    'start_not_at_centre': '起始 cursor 位置不在中心附近',
    # timing
    'overtime': 'Overtime：计时超过阈值',
    'too_fast': '过快：start → 命中短于阈值',
    # window / neural / block
    'shorter_than_window': '窗口长于 trial：窗口终点超过 T_TRIAL_END（会混入 ITI 数据）',
    'outside_neural_recording': '分析窗口落在神经记录范围之外',
    'no_neural_in_window': '分析窗口内无神经数据',
    'neural_coverage_low': '窗口内神经帧数比例低于阈值',
    'neural_gap_in_window': '窗口内神经接收空档超过阈值',
    'block_out_of_range': 'block 不在所选范围内',
    # velocity
    'velocity_cursor_gap': '速度 bin：cursor 采样空档超过阈值',
    'velocity_cursor_foreign': '速度 bin：cursor 不属于本 trial 或不在 active 状态',
    'velocity_neural_short_window': '速度 bin：短窗口内神经帧不足或空档过大',
    'velocity_beyond_trial_end': '速度 bin：超出 trial 结束',
    # flags
    'acquired_before_window_end': 'Flag：命中早于窗口结束',
}
GROUP_OF = {
    **{k: 'incomplete' for k in ('end_without_start', 'start_without_end', 'nested_start', 'trial_id_conflict',
                                 'no_active_cursor_samples', 'cursor_identity_changes_within_trial')},
    'not_success': 'outcome', 'ambiguous_outcome': 'outcome',
    **{k: 'direction' for k in ('nonfinite_direction_coordinates', 'target_changes_within_trial',
                                'zero_direction_displacement', 'direction_angle_off_grid',
                                'target_radius_out_of_range', 'start_not_at_centre')},
    'overtime': 'timing', 'too_fast': 'timing', 'shorter_than_window': 'window',
    **{k: 'neural' for k in ('outside_neural_recording', 'no_neural_in_window', 'neural_coverage_low',
                             'neural_gap_in_window')},
    'block_out_of_range': 'block',
    **{k: 'velocity' for k in ('velocity_cursor_gap', 'velocity_cursor_foreign', 'velocity_neural_short_window',
                               'velocity_beyond_trial_end')},
    'acquired_before_window_end': 'flags',
}
INCOMPLETE_REASONS = {k for k, g in GROUP_OF.items() if g == 'incomplete'}
DIRECTION_REASONS = {k for k, g in GROUP_OF.items() if g == 'direction'}

# GUI metadata: (field, label, kind, group[basic|advanced|session], help)
FIELD_SPEC = (
    ('exclude_incomplete', '排除 Incomplete trial（缺 START / 缺 END / 嵌套 / trial id 冲突）', 'bool', 'basic', ''),
    ('success_only', '仅 Success trial（结局不明确单列并排除）', 'bool', 'basic', ''),
    ('window_start', '分析窗口起点 / s（相对 T_TRIAL_START）', 'float', 'basic', ''),
    ('window_end', '分析窗口终点 / s', 'float', 'basic', ''),
    ('require_window_within_trial', '窗口长于 trial 则排除（trial_shorter_than_window）', 'bool', 'basic', ''),
    ('overtime_enabled', '排除 Overtime', 'bool', 'basic', ''),
    ('overtime_basis', 'Overtime 计时口径', 'choice:start_to_acquire,start_to_end', 'basic',
     'start→命中（推荐；END 含约 0.2–0.3 s 停留）或 start→END；未命中的 trial 一律用 start→END'),
    ('overtime_mode', 'Overtime 阈值形式', 'choice:fixed,median_mad', 'basic',
     'fixed = 固定秒数；median_mad = 本 session 成功 trial 的 median + k·MAD（MAD×1.4826）'),
    ('overtime_seconds', 'Overtime 固定阈值 / s', 'float', 'basic', ''),
    ('overtime_mad_k', 'Overtime median + k·MAD 中的 k', 'float', 'basic', ''),
    ('neural_check', '检查窗口内神经覆盖率', 'bool', 'basic', ''),
    ('min_frame_fraction', '最少帧数比例（相对窗口应有帧数）', 'float', 'basic',
     '应有帧数 = 窗口长度 / 本 session 实测帧周期（约 19.2 ms）'),
    ('max_gap_seconds', '窗口内最大接收空档 / s（含窗口边界）', 'float', 'basic', ''),
    ('direction_check', '检查方向标签有效性', 'bool', 'advanced', ''),
    ('angle_tolerance_deg', 'CO8：target 角度与最近 45° 的容差 / °', 'float', 'advanced', ''),
    ('target_radius_tolerance', 'CO8：target 到中心距离相对 session 中位数的容差（比例）', 'float', 'advanced', ''),
    ('start_radius_fraction', 'CO8：起始 cursor 到中心距离 ≤ target 距离 × 此比例', 'float', 'advanced', ''),
    ('too_fast_enabled', '过快 trial 检查（start → 命中）', 'bool', 'advanced', ''),
    ('too_fast_seconds', '过快阈值 / s', 'float', 'advanced', '目前最快约 0.95 s'),
    ('too_fast_action', '过快 trial 处理', 'choice:flag,exclude', 'advanced', ''),
    ('exclude_flagged_acquired', '敏感性分析：排除「命中早于窗口结束」的 flag trial', 'bool', 'advanced',
     '主分析保留；flag 同时记录命中后数据占窗口的比例'),
    ('velocity_bin_policy', '速度 bin 坏 bin 处理', 'choice:drop_trial,drop_bin', 'advanced',
     'drop_trial = 固定窗口内任一 bin 不合格则整个 trial 退出速度队列；drop_bin = 只去掉该 bin'),
    ('velocity_cursor_gap', '速度 bin：cursor 最大采样空档 / s', 'float', 'advanced', ''),
    ('velocity_neural_window', '速度 bin：神经短窗口长度 / s（截至 bin 右端）', 'float', 'advanced', ''),
    ('velocity_min_frames', '速度 bin：短窗口最少帧数', 'int', 'advanced', ''),
    ('velocity_max_gap', '速度 bin：短窗口最大空档 / s', 'float', 'advanced', ''),
    ('block_min', 'block 下限（空 = 不限）', 'optint', 'advanced', ''),
    ('block_max', 'block 上限（空 = 不限）', 'optint', 'advanced', ''),
    ('min_velocity_trials', 'Session 合格：速度 trial ≥', 'int', 'session', ''),
    ('min_trials_per_direction', 'Session 合格：每方向 trial ≥', 'int', 'session', ''),
    ('min_blocks', 'Session 合格：时间 block ≥', 'int', 'session', ''),
    ('outage_seconds', 'Session：标记神经长时间断流 ≥ / s', 'float', 'session', ''),
)


@dataclass(frozen=True)
class TrialFilter:
    criteria_version: str = CRITERIA_VERSION
    # basic
    exclude_incomplete: bool = True
    success_only: bool = True
    window_start: float = 0.2
    window_end: float = 1.2
    require_window_within_trial: bool = True
    overtime_enabled: bool = True
    overtime_basis: str = 'start_to_acquire'
    overtime_mode: str = 'fixed'
    overtime_seconds: float = 10.0
    overtime_mad_k: float = 5.0
    neural_check: bool = True
    min_frame_fraction: float = 0.8
    max_gap_seconds: float = 0.15
    # advanced
    direction_check: bool = True
    angle_tolerance_deg: float = 5.0
    target_radius_tolerance: float = 0.2
    start_radius_fraction: float = 0.25
    too_fast_enabled: bool = True
    too_fast_seconds: float = 0.8
    too_fast_action: str = 'flag'
    exclude_flagged_acquired: bool = False
    velocity_bin_policy: str = 'drop_trial'
    velocity_cursor_gap: float = 0.10
    velocity_neural_window: float = 0.20
    velocity_min_frames: int = 4
    velocity_max_gap: float = 0.15
    block_min: int | None = None
    block_max: int | None = None
    # session level
    min_velocity_trials: int = 80
    min_trials_per_direction: int = 8
    min_blocks: int = 10
    outage_seconds: float = 5.0

    def __post_init__(self):
        if not self.window_end > self.window_start:
            raise ValueError('分析窗口终点必须大于起点')
        if not 0 < self.min_frame_fraction <= 1:
            raise ValueError('最少帧数比例须在 (0, 1]')
        positive = ('overtime_seconds', 'overtime_mad_k', 'max_gap_seconds', 'angle_tolerance_deg',
                    'target_radius_tolerance', 'start_radius_fraction', 'too_fast_seconds', 'velocity_cursor_gap',
                    'velocity_neural_window', 'velocity_max_gap', 'outage_seconds')
        for name in positive:
            if not getattr(self, name) > 0:
                raise ValueError(f'{name} 必须为正数')
        if self.velocity_min_frames < 1:
            raise ValueError('velocity_min_frames 必须 ≥ 1')
        choices = dict(overtime_basis=('start_to_acquire', 'start_to_end'), overtime_mode=('fixed', 'median_mad'),
                       too_fast_action=('flag', 'exclude'), velocity_bin_policy=('drop_trial', 'drop_bin'))
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f'{name} 必须为 {allowed}')

    # Compatibility with earlier callers.
    @property
    def min_frames(self) -> int:
        return self.velocity_min_frames

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict | None):
        values = dict(values or {})
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in values.items() if k in known}
        values['criteria_version'] = CRITERIA_VERSION
        try:
            return cls(**values)
        except (TypeError, ValueError):
            return cls()

    def with_(self, **changes):
        return replace(self, **changes)

    # ------------------------------------------------------------ config file
    def save(self, path):
        Path(path).write_text(json.dumps(dict(self.to_dict(), _comment='Trial-selection thresholds; fix before '
                                              'looking at results.'), ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding='utf-8'))
        if values.get('criteria_version') not in (None, CRITERIA_VERSION):
            raise ValueError(f'配置版本 {values.get("criteria_version")} 与当前 {CRITERIA_VERSION} 不一致')
        values.pop('_comment', None)
        return cls(**{k: v for k, v in values.items() if k in {f.name for f in fields(cls)}})


@dataclass
class Cohort:
    """Direction cohort (`rows`) plus the lazily computed velocity cohort."""
    session: Session
    criteria: TrialFilter
    table: list[dict] = field(repr=False)        # one row per reconstructed trial
    include: np.ndarray = field(repr=False)      # bool per table row: direction cohort
    artifacts: object = field(default=None, repr=False, compare=False)

    # ----------------------------------------------------------------- arrays
    @cached_property
    def rows(self) -> list[dict]:
        return [r for r, keep in zip(self.table, self.include) if keep]

    def array(self, key, dtype=float) -> np.ndarray:
        return np.array([r[key] for r in self.rows], dtype=dtype)

    @property
    def n(self) -> int:
        return int(self.include.sum())

    @cached_property
    def starts(self):
        return self.array('start')

    @cached_property
    def ends(self):
        return self.array('end')

    @cached_property
    def directions(self):
        return self.array('direction', int)

    @cached_property
    def blocks(self):
        return self.array('block', int)

    @cached_property
    def orders(self):
        return self.array('order', int)

    @cached_property
    def trial_ids(self):
        return self.array('trial_id', int)

    @cached_property
    def window_rates(self) -> np.ndarray:
        """Trials x 512 mean Hz in the analysis window."""
        def compute():
            rates, _ = self.session.window_mean_rates(self.starts + self.criteria.window_start,
                                                       self.starts + self.criteria.window_end)
            return rates
        if self.artifacts is not None:
            return self.artifacts.remember(('trial_window_rates', id(self)), compute)
        return compute()

    def rates_in(self, start: float, end: float, anchor: str = 'start') -> np.ndarray:
        anchors = self.starts if anchor == 'start' else self.starts + self.array('acquire_time')
        rates, _ = self.session.window_mean_rates(anchors + start, anchors + end)
        return rates

    # ------------------------------------------------------------ velocity
    @cached_property
    def velocity_audit(self) -> dict:
        """Velocity-cohort status per direction-cohort row (fixed analysis window, 100 ms bins).

        Returns dict(included=bool[n], reasons=list[list[str]], bins_bad=int[n]).
        """
        from .features import velocity_bins

        def compute():
            return velocity_bins(self, audit=True).audit

        if self.artifacts is not None:
            return self.artifacts.remember(('cohort_velocity_audit', id(self)), compute)
        return compute()

    @property
    def velocity_include(self) -> np.ndarray:
        return self.velocity_audit['included']

    # ----------------------------------------------------------------- audit
    def reason_counts(self) -> Counter:
        return Counter(reason for r in self.table for reason in r['exclusion_reasons'])

    def flag_counts(self) -> Counter:
        return Counter(f for r in self.table for f in r['flags'])

    def primary_counts(self) -> Counter:
        return Counter(r['primary_reason'] for r in self.table if r['primary_reason'])

    def summary(self) -> dict:
        complete = sum(1 for r in self.table if r['complete'] and not r['incomplete_reason'])
        status = self.session_status()
        return dict(
            recording=self.session.label, criteria_version=self.criteria.criteria_version,
            total_trial_rows=len(self.table), complete_trials=complete,
            success_trials=sum(r['outcome'] == 'success' for r in self.table),
            included_trials=self.n, velocity_trials=int(self.velocity_include.sum()) if self.n else 0,
            blocks=len(np.unique(self.blocks)) if self.n else 0,
            direction_counts=np.bincount(self.directions, minlength=8).tolist() if self.n else [0] * 8,
            exclusion_counts=dict(self.reason_counts()), primary_reason_counts=dict(self.primary_counts()),
            flag_counts=dict(self.flag_counts()), session_qualified=status['qualified'],
            session_issues=status['issues'], session_type=status['session_type'])

    def session_status(self) -> dict:
        """Session-level qualification (minimum samples), type label and neural outages."""
        c, s = self.criteria, self.session
        vel = self.velocity_include if self.n else np.zeros(0, bool)
        per_direction = np.bincount(self.directions[vel], minlength=8) if self.n else np.zeros(8, int)
        blocks = len(np.unique(self.blocks)) if self.n else 0
        issues = []
        if int(vel.sum()) < c.min_velocity_trials:
            issues.append(f'速度 trial {int(vel.sum())} < {c.min_velocity_trials}')
        if per_direction.min() < c.min_trials_per_direction:
            issues.append(f'最少方向 trial {int(per_direction.min())} < {c.min_trials_per_direction}')
        if blocks < c.min_blocks:
            issues.append(f'时间 block {blocks} < {c.min_blocks}')
        nt = s.neural_times
        gaps = np.diff(nt)
        outages = []
        for i in np.flatnonzero(gaps >= c.outage_seconds):
            a, b = float(nt[i]), float(nt[i + 1])
            affected = sum(1 for r in self.table if np.isfinite(r['start']) and np.isfinite(r['end'])
                           and r['start'] < b and r['end'] > a)
            outages.append(dict(start_s=a - float(nt[0]), end_s=b - float(nt[0]), duration_s=b - a,
                                trials_overlapping=affected))
        events_end = float(s.event_times[-1]) if len(s.event_times) else float(nt[-1])
        tail = max(0.0, events_end - float(nt[-1]))
        if tail >= c.outage_seconds:
            outages.append(dict(start_s=float(nt[-1] - nt[0]), end_s=events_end - float(nt[0]), duration_s=tail,
                                trials_overlapping=sum(1 for r in self.table if np.isfinite(r['start'])
                                                       and r['start'] > nt[-1]), note='neural ended before task'))
        online = s.recording.online or s.decoder_velocity is not None
        return dict(qualified=not issues, issues=issues, velocity_trials=int(vel.sum()),
                    velocity_per_direction=per_direction.tolist(), blocks=blocks,
                    session_type='online（闭环：cursor 由解码器驱动，速度标签含义不同）' if online else 'offline',
                    online=online, outages=outages)

    def exclusion_table(self) -> list[dict]:
        vel = dict(zip((r['trial_index'] for r in self.rows), self.velocity_include)) if self.n else {}
        vreason = dict(zip((r['trial_index'] for r in self.rows), self.velocity_audit['reasons'])) if self.n else {}
        out = []
        for r in self.table:
            row = dict(trial_index=r['trial_index'], trial_id=r['trial_id'], block=r['block'], direction=r['direction'],
                       outcome=r['outcome'], start=round(r['start'], 3) if np.isfinite(r['start']) else '',
                       duration=round(r['duration'], 3) if np.isfinite(r['duration']) else '',
                       acquire_time=round(r['acquire_time'], 3) if np.isfinite(r['acquire_time']) else '',
                       status=r['status'], primary_reason=r['primary_reason'],
                       primary_reason_text=REASON_LABELS.get(r['primary_reason'], r['primary_reason']),
                       direction_cohort=bool(r['included']),
                       velocity_cohort=bool(vel.get(r['trial_index'], False)),
                       velocity_reasons=';'.join(vreason.get(r['trial_index'], [])),
                       flags=';'.join(r['flags']),
                       post_acquire_window_fraction=round(r['post_acquire_fraction'], 3)
                       if np.isfinite(r['post_acquire_fraction']) else '')
            for group in PRIORITY[:-1] + ('flags',):
                row[f'check_{group}'] = r['checks'].get(group, 'pass')
            row['all_reasons'] = '; '.join(REASON_LABELS.get(x, x) for x in r['exclusion_reasons'])
            out.append(row)
        return out


def _overtime_threshold(session: Session, criteria: TrialFilter) -> float:
    if criteria.overtime_mode == 'fixed':
        return criteria.overtime_seconds
    times = []
    for r in session.trials:
        if r['outcome'] == 'success' and r['complete']:
            times.append(r['acquire_time'] if criteria.overtime_basis == 'start_to_acquire' else r['duration'])
    times = np.asarray([t for t in times if np.isfinite(t)])
    if not len(times):
        return criteria.overtime_seconds
    median = float(np.median(times))
    mad = float(np.median(np.abs(times - median))) * 1.4826
    return median + criteria.overtime_mad_k * mad


def apply_filter(session: Session, criteria: TrialFilter) -> Cohort:
    c = criteria
    nt = session.neural_times
    period = session.effective_frame_period
    expected_frames = (c.window_end - c.window_start) / period
    overtime = _overtime_threshold(session, c)
    radii = np.array([r['target_radius'] for r in session.trials if np.isfinite(r.get('target_radius', np.nan))])
    radius_median = float(np.median(radii)) if len(radii) else np.nan
    table = []
    for base in session.trials:
        row = dict(base)
        fails: list[str] = []
        flags: list[str] = []
        # ---- 1 incomplete
        if row['incomplete_reason']:
            fails.append(row['incomplete_reason'])
        if not c.exclude_incomplete:
            # Optional relaxation: close a trial that never ended at its last event
            # before the next start; keep the majority id of a changing cursor id.
            if row['incomplete_reason'] in ('start_without_end', 'nested_start'):
                end = base.get('_candidate_end', np.nan)
                if np.isfinite(end) and row['direction'] >= 0:
                    fails.remove(row['incomplete_reason'])
                    row.update(end=end, inferred_end=True, duration=end - row['start'])
            if 'cursor_identity_changes_within_trial' in fails:
                fails.remove('cursor_identity_changes_within_trial')
        # ---- 2 outcome
        if c.success_only and np.isfinite(row['start']):
            if row['outcome'] == 'ambiguous':
                fails.append('ambiguous_outcome')
            elif row['outcome'] != 'success':
                fails.append('not_success')
        # ---- 3 direction label validity
        if c.direction_check and row['direction'] >= 0:
            if row['n_targets'] > 1:
                fails.append('target_changes_within_trial')
            if session.task_mode == 'centerout':
                if abs(row['direction_offset_deg']) > c.angle_tolerance_deg:
                    fails.append('direction_angle_off_grid')
                start_r = np.hypot(row['cursor_start_x'], row['cursor_start_y'])
                if np.isfinite(start_r) and np.isfinite(row['target_radius']) and \
                        start_r > c.start_radius_fraction * row['target_radius']:
                    fails.append('start_not_at_centre')
                # Grid targets lie at varying distances by design; only CO8 has one radius.
                if np.isfinite(radius_median) and np.isfinite(row['target_radius']) and \
                        abs(row['target_radius'] - radius_median) > c.target_radius_tolerance * radius_median:
                    fails.append('target_radius_out_of_range')
        # ---- 4 timing
        timing = row['acquire_time'] if (c.overtime_basis == 'start_to_acquire' and np.isfinite(row['acquire_time'])) \
            else row['duration']
        row['timing_seconds'] = timing
        if c.overtime_enabled and np.isfinite(timing) and timing > overtime:
            fails.append('overtime')
        if c.too_fast_enabled and np.isfinite(row['acquire_time']) and row['acquire_time'] < c.too_fast_seconds:
            (fails if c.too_fast_action == 'exclude' else flags).append('too_fast')
        # ---- 5 window inside trial
        if c.require_window_within_trial and np.isfinite(row['duration']) and row['duration'] < c.window_end - 1e-9:
            fails.append('shorter_than_window')
        # ---- 6 neural coverage
        if c.neural_check and np.isfinite(row['start']):
            left, right = row['start'] + c.window_start, row['start'] + c.window_end
            if left < nt[0] or right > nt[-1]:
                fails.append('outside_neural_recording')
            else:
                _, _, n, gap = session.window_stats([left], [right])
                row['window_frames'] = int(n[0])
                row['window_max_gap'] = float(gap[0])
                if n[0] == 0:
                    fails.append('no_neural_in_window')
                else:
                    if n[0] < c.min_frame_fraction * expected_frames - 1e-9:
                        fails.append('neural_coverage_low')
                    if gap[0] > c.max_gap_seconds + 1e-9:
                        fails.append('neural_gap_in_window')
        # ---- 7 block
        if (c.block_min is not None and row['block'] < c.block_min) or \
                (c.block_max is not None and row['block'] > c.block_max):
            fails.append('block_out_of_range')
        # ---- flags
        row['post_acquire_fraction'] = np.nan
        if np.isfinite(row['acquire_time']) and row['acquire_time'] < c.window_end:
            frac = (c.window_end - max(row['acquire_time'], c.window_start)) / (c.window_end - c.window_start)
            row['post_acquire_fraction'] = float(min(max(frac, 0.0), 1.0))
            if c.exclude_flagged_acquired:
                fails.append('acquired_before_window_end')
            else:
                flags.append('acquired_before_window_end')
        hard_invalid = not np.isfinite(row['start']) or not np.isfinite(row['end']) or row['direction'] < 0
        if hard_invalid and not fails:
            fails.append('start_without_end' if not np.isfinite(row['end']) else 'no_active_cursor_samples')
        fails = list(dict.fromkeys(fails))
        checks = {}
        for reason in fails:
            checks.setdefault(GROUP_OF.get(reason, 'incomplete'), 'fail')
        for reason in flags:
            checks.setdefault(GROUP_OF.get(reason, 'flags'), 'flag')
        if not c.neural_check:
            checks.setdefault('neural', 'n/a')
        row['checks'] = checks
        row['exclusion_reasons'] = fails
        row['flags'] = list(dict.fromkeys(flags))
        ordered = sorted(fails, key=lambda x: PRIORITY.index(GROUP_OF.get(x, 'incomplete'))
                         if GROUP_OF.get(x, 'incomplete') in PRIORITY else len(PRIORITY))
        row['primary_reason'] = ordered[0] if ordered else ''
        row['included'] = not fails
        row['status'] = 'excluded' if fails else ('included_flagged' if row['flags'] else 'included')
        table.append(row)
    include = np.array([r['included'] for r in table], bool)
    return Cohort(session, criteria, table, include)
