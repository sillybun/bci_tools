"""Timestamp-driven playback of recorded cursor states; no task or hardware runtime."""
from __future__ import annotations

import json
import re
import time

import numpy as np

from ... import config as C


def source_files(recording):
    """Only use a behavior bundle uniquely assigned to this XDF, including main parts."""
    from ...core.behavior import source_files as behavior_files
    files = behavior_files(recording)
    if not files:
        return {}
    root = files['trials'].parent
    root = root.parent if root.name == 'behavior' else root
    for key, candidates in [('task_config', ['task_config.json','config/task_config.json']),
                            ('task_log', ['logs/task.log','logs/cursor_tasks_app.log'])]:
        for relative in candidates:
            path = root / relative
            if path.is_file():
                files[key] = path
                break
    if 'task_log' not in files:
        candidates = sorted((root/'logs').glob('*-task*.log'))
        if len(candidates)==1:
            files['task_log'] = candidates[0]
    return files


def _number(value, default):
    try:
        value = float(value)
        return value if np.isfinite(value) and value > 0 else default
    except (ValueError, TypeError):
        return default


def geometry(session, cursor, files):
    """Use actual runtime window from log; CALCULATED sizes precede nominal factors."""
    notes = []
    config = {}
    if 'task_config' in files:
        try:
            config = json.loads(files['task_config'].read_text(encoding='utf-8-sig'))
            if not isinstance(config, dict):
                raise ValueError('配置必须为对象')
        except (OSError, ValueError) as exc:
            config = {}
            notes.append(f'任务配置无法读取：{exc}；使用记录坐标估计显示范围。')
    def group(key):
        value = config.get(key, {})
        return value if isinstance(value, dict) else {}
    system, calc, visual = group('SYSTEM'), group('CALCULATED'), group('VISUALS')
    grid, co = group('GRID_TASK'), group('CENTER_OUT_TASK')
    width, height = _number(system.get('WIN_WIDTH'), 0), _number(system.get('WIN_HEIGHT'), 0)
    log = files.get('task_log')
    text = log.read_text(encoding='utf-8-sig', errors='replace') if log else ''
    match = re.search(r'Actual window size:\s*\[\s*(\d+)[,\s]+(\d+)\s*\]', text)
    if match:
        width, height = map(float, match.groups())
    coords = cursor[:, [C.C_X, C.C_Y, C.C_TX, C.C_TY]]
    if not width or not height:
        finite = np.where(np.isfinite(coords), np.abs(coords), 0)
        width = max(800., float(np.max(finite[:, [0, 2]])) * 2.3)
        height = max(600., float(np.max(finite[:, [1, 3]])) * 2.3)
        notes.append('缺少实际窗口尺寸，显示范围由已记录的坐标估计。')
    short = min(width, height)
    target = _number(calc.get('TARGET_SIZE_PX'), short * _number(
        co.get('TARGET_SIZE_FACTOR') if session.task_mode == 'centerout' else visual.get('TARGET_SIZE_FACTOR'), .1))
    radius = _number(calc.get('CURSOR_SIZE_PX'), short * _number(visual.get('CURSOR_SIZE_FACTOR'), .015))
    rows = min(100, int(_number(grid.get('GRID_ROWS'), 0)))
    cols = min(100, int(_number(grid.get('GRID_COLS'), 0)))
    cell = target
    if session.task_mode == 'grid':
        margin = _number(calc.get('GRID_MARGIN_PX'), short * _number(visual.get('GRID_MARGIN_FACTOR'), .05))
        if rows and cols and grid.get('AUTO_SCALE_TO_WINDOW', True):
            fitted = min(max(1,width-2*margin)/cols,max(1,height-2*margin)/rows)
            cell = max(_number(grid.get('MIN_CELL_SIZE_PX'), target),
                       fitted * _number(grid.get('WINDOW_FILL_RATIO'), .9))
        # Recorded centre spacing is more precise than the rounded task.log cell size.
        steps = []
        for axis in [C.C_TX, C.C_TY]:
            values = np.unique(np.round(cursor[np.isfinite(cursor[:, axis]), axis], 5))
            diffs = np.diff(values)
            steps.extend(diffs[diffs > 1e-3])
        if steps and not (rows and cols):
            cell = float(np.min(steps))
        elif not (rows and cols):
            m = re.search(r'final_cell=([\d.]+)px', text)
            if m:
                cell = float(m[1])
        if not rows or not cols:
            # Do not invent a full grid from a subset of visited targets.
            rows = cols = 0
            notes.append('缺少网格行列配置，仅显示已记录的目标。')
        else:
            rows = min(rows, max(1,int((height-2*margin)//cell)))
            cols = min(cols, max(1,int((width-2*margin)//cell)))
    elif not config:
        notes.append('缺少目标尺寸配置，目标及 cursor 大小为示意。')
    return dict(width=width, height=height, cursor_radius=radius, target_size=target,
                target_shape='circle' if session.task_mode == 'centerout' and co.get('TARGET_SHAPE', 'circle') == 'circle' else 'rect',
                grid_rows=rows, grid_cols=cols, cell_size=cell), notes


def build_payload(session, params, check_cancel=lambda: None):
    raw = np.asarray(session.cursor, float)
    times = np.asarray(session.cursor_times, float)
    if raw.ndim != 2 or raw.shape[1] <= C.C_ACTIVE or len(raw) != len(times):
        raise ValueError('需要包含位置、速度、目标和 trial 状态的 cursor_state 数据流。')
    keep = np.isfinite(times) & np.isfinite(raw[:, C.C_X:C.C_Y+1]).all(axis=1)
    order = np.argsort(times[keep], kind='stable')
    times, cursor = times[keep][order], raw[keep][order]
    # Equal timestamps: last recorded state wins; never divide by zero in playback.
    unique = np.r_[np.diff(times) > 0, True] if len(times) else np.array([], bool)
    times, cursor = times[unique], cursor[unique]
    if len(times) < 2:
        raise ValueError('有效 cursor 位置和时间戳少于 2 帧，无法回放。')
    check_cancel()
    files = source_files(session.recording)
    geo, notes = geometry(session, cursor, files)
    origin = float(times[0])
    relative = times - origin
    dt = np.diff(relative)
    max_gap = float(params.get('max_gap', .25))
    # Never derive motion through gaps or trial resets. Recorded velocity is retained.
    active = cursor[:, C.C_ACTIVE] == 1
    block = np.where(np.isfinite(cursor[:, C.C_BLOCK]), cursor[:, C.C_BLOCK], -1).astype(int)
    trial = np.where(np.isfinite(cursor[:, C.C_TRIAL]), cursor[:, C.C_TRIAL], -1).astype(int)
    missing_v = ~np.isfinite(cursor[:, C.C_VX:C.C_VY+1]).all(axis=1)
    saved_velocity = cursor[:, C.C_VX:C.C_VY+1].copy()
    derived = np.full_like(saved_velocity, np.nan)
    continuous = ((dt <= max_gap) & active[1:] & active[:-1]
                  & (block[1:]==block[:-1]) & (trial[1:]==trial[:-1]))
    differences = np.diff(cursor[:,C.C_X:C.C_Y+1],axis=0)/dt[:,None]
    derived[1:][continuous] = differences[continuous]
    moving = np.any(np.linalg.norm(differences[continuous],axis=1) > 1e-6)
    choice = params.get('velocity_source','自动')
    valid = saved_velocity[~missing_v]
    use_derived = (choice=='位置差分' or (choice=='自动' and moving and
                   (not len(valid) or np.all(np.abs(valid)<1e-9))))
    if use_derived:
        velocity = derived
        velocity_source = '位置差分速度'
        notes.append('速度使用同一连续活动 trial 内相邻位置的 Δposition/Δtime；trial 复位和采样缺口处为 N/A。原始保存速度仍保留在 JSON 的 saved_vx/saved_vy 列。')
        if choice=='自动':notes.append('检测到位置有运动而保存速度全部为 0 或缺失，自动改用位置差分速度。')
    else:
        velocity = saved_velocity.copy()
        velocity[missing_v] = np.nan
        velocity_source = '保存速度'
    starts = np.flatnonzero(active & np.r_[True, (~active[:-1]) | (trial[1:] != trial[:-1]) | (block[1:] != block[:-1])])
    trials = [dict(time=float(relative[i]), block=int(block[i]), trial=int(trial[i])) for i in starts]
    events = []
    for t, row in zip(session.event_times, session.events):
        if times[0] <= t <= times[-1] and row:
            events.append([float(t-origin), str(row[0])])
    events.sort(key=lambda r:r[0])
    # Compact data, keeping EVERY observed sample; only display rendering is frame-limited.
    values = np.column_stack([relative, cursor[:,C.C_X:C.C_Y+1], velocity,
                              cursor[:,C.C_TX:C.C_TY+1], block, trial, active.astype(int),saved_velocity])
    frames = [[None if not np.isfinite(v) else float(v) for v in row] for row in values]
    columns = ['time_s','x','y','vx','vy','target_x','target_y','block','trial','active','saved_vx','saved_vy']
    notes.append('使用 XDF cursor_state 原始时间戳和位置复现运动；速度用于数值和箭头显示，不重新积分。')
    notes.append('回放保留本记录已加载范围内的全部 cursor 帧，不使用成功 trial 或神经分析筛选。')
    notes.append('时间 0 是本记录首个有效 cursor 帧，可能位于任务或 trial 中途；不补齐录制范围以外的轨迹。')
    if np.any(missing_v):
        notes.append(f'{int(missing_v.sum())} 帧速度不可用，显示为 N/A，不补零。')
    if np.any(dt > max_gap):
        notes.append(f'{int((dt > max_gap).sum())} 处 cursor 采样间隔超过 {max_gap:g} 秒；间隔内标记缺失，不插值跨越。')
    notes.extend(['目标外观按保存配置重建；未记录的视觉特效、辅助提示和奖励动画不作精确复现。'])
    return dict(format='cursortasks-replay-v1', label=session.label, task_mode=session.task_mode,
                velocity_source=velocity_source,
                columns=columns, frames=frames, trials=trials, events=events, geometry=geo,
                duration=float(relative[-1]), max_gap=max_gap, tail_seconds=float(params.get('tail_seconds', 1)),
                sources={key:str(p) for key,p in files.items()}, notes=notes,
                stats=dict(samples=len(frames), trials=len(trials), gaps=int(np.sum(dt>max_gap)),
                           duration_s=float(relative[-1]), recorded_velocity_samples=int((~missing_v).sum())))


def sample_at(payload, elapsed):
    """Last observed discrete state; interpolate xy only inside a continuous trial."""
    frames = payload['frames']
    # Numpy search avoids creating a second large Python list on every frame.
    times = payload.get('_times')
    if times is None:
        times = np.array([r[0] for r in frames])
        payload['_times'] = times
    elapsed = max(0., min(float(elapsed), payload['duration']))
    i = max(0, int(np.searchsorted(times, elapsed, side='right'))-1)
    state = list(frames[i])
    gap = False
    if i+1 < len(frames):
        following = frames[i+1]
        delta = following[0]-state[0]
        gap = delta > payload['max_gap'] and elapsed > state[0] + 1e-6
        continuous = (state[7:10] == following[7:10] and state[9] == 1 and not gap
                      and delta <= payload['max_gap'])
        if continuous:
            fraction = (elapsed-state[0])/delta
            state[1] += (following[1]-state[1])*fraction
            state[2] += (following[2]-state[2])*fraction
    return i, state, gap


class PlaybackClock:
    """Elapsed time follows monotonic wall time, independent of UI frame count."""
    def __init__(self, duration, now=time.perf_counter):
        self.duration, self.now = duration, now
        self.offset, self.rate, self.playing, self.anchor = 0., 1., False, now()

    def position(self):
        value = self.offset + ((self.now()-self.anchor)*self.rate if self.playing else 0)
        return min(self.duration, max(0., value))

    def pause(self):
        self.offset = self.position()
        self.playing = False

    def play(self):
        if self.position() >= self.duration:
            self.offset = 0.
        self.anchor = self.now()
        self.playing = True

    def seek(self, value):
        self.offset = min(self.duration, max(0., float(value)))
        self.anchor = self.now()

    def speed(self, rate):
        rate = float(rate)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError('回放速度必须为正数')
        self.seek(self.position())
        self.rate = rate
