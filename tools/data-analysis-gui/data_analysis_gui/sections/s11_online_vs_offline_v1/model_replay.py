"""Replay saved position/velocity KF matrices in the logged application order.

This is a matrix-level audit, never labelled an exact runtime reproduction.
Application timestamps are wall time; saved cursor clock pairs supply the LSL map.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ...core.catalog import day_folder, session_metadata
from ...core import style
from ...decoders._blas import single_thread
from ..base import new_figure


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def resolve_run(recording, selected=''):
    if selected:
        root = Path(selected).expanduser()
        if not root.is_dir():
            raise ValueError(f'在线模型目录不存在：{root}')
        found = sorted(root.rglob('online_update_swap_history.jsonl'))
    else:
        # A split XDF belongs to one explicit task-session bundle. Never borrow
        # another session's model history when this one has no applied updates.
        xdf_dir = Path(recording.path).parent
        if xdf_dir.name.lower() == 'xdf' and (
                xdf_dir.parent.parent.name == 'sessions' or session_metadata(recording.path)):
            found = sorted((xdf_dir.parent / 'refit').rglob('online_update_swap_history.jsonl'))
            if len(found) > 1:
                raise ValueError('该任务 session 存在多个在线更新目录，请在 S11 指定在线模型目录')
            return found[0].parent if found else None
        # Resolve the owning run even for a nested early recording segment.
        from ...core.catalog import run_folder
        owner = run_folder(recording.path)
        if owner is not None:
            packaged_refit = owner / 'refit'
            return packaged_refit if (packaged_refit / 'online_update_swap_history.jsonl').is_file() else None
        # Legacy packages keep the XDF in <run>/xdf beside that run's refit.
        xdf_dir = Path(recording.path).parent
        if xdf_dir.name.lower() == 'xdf':
            packaged_refit = xdf_dir.parent / 'refit'
            if (packaged_refit / 'online_update_swap_history.jsonl').is_file():
                return packaged_refit
        root = day_folder(recording.path)
        found = sorted(root.rglob('online_update_swap_history.jsonl')) if root else []
    if len(found) > 1:
        raise ValueError('当天存在多个在线更新 session，请在 S11 指定在线模型目录')
    return found[0].parent if found else None


def relocate(historical, root, day):
    original = Path(str(historical).replace('\\', '/'))
    # Retain the complete suffix under either session/day; basename fallback is
    # accepted only when unique, never simply take the first matching weight.
    parts = original.parts[1:] if original.anchor else original.parts
    for base in (root, day):
        if base is None:
            continue
        for index in range(len(parts)):
            candidate = base.joinpath(*parts[index:])
            if candidate.is_file():
                return candidate
    if original.is_file():
        return original
    candidates = list(day.rglob(original.name)) if day else []
    if len(candidates) != 1:
        raise ValueError(f'无法唯一定位实际权重：{historical}')
    return candidates[0]


def manifest(recording, params):
    root = resolve_run(recording, params.get('online_model_dir', ''))
    if root is None:
        return None
    swaps = [json.loads(line) for line in (root/'online_update_swap_history.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    swaps = [s for s in swaps if s.get('model_name') == 'position_velocity_kf' and
             s.get('auto_swap', {}).get('swaps', {}).get('position_velocity_kf', {}).get('status') == 'applied']
    swaps.sort(key=lambda s: float(s['applied_time']))
    if not swaps:
        raise ValueError('模型目录没有 position_velocity_kf 实际 applied 切换记录')
    if len({float(s['applied_time']) for s in swaps}) != len(swaps):
        raise ValueError('切换日志有重复 applied_time，不能确定唯一切换顺序')
    day = day_folder(recording.path)
    first = swaps[0]['auto_swap']['swaps']['position_velocity_kf']
    initial = params.get('initial_model_path')
    if initial:
        initial = Path(initial).expanduser()
        if initial.is_dir():
            initial = initial / 'INITIAL_position_velocity_kf.pkl'
        if not initial.is_file():
            raise ValueError(f'初始权重文件不存在：{initial}')
    else:
        packaged_initial = root.parent / 'initial_weights' / 'INITIAL_position_velocity_kf.pkl'
        initial = (packaged_initial if packaged_initial.is_file()
                   else relocate(first['old_weights_path'], root, day))
    paths = [initial]
    paths += [relocate(s['auto_swap']['swaps']['position_velocity_kf']['active_weights_path_after_swap'], root, day) for s in swaps]
    blocks = sorted(root.glob('block_*_data.npz'))
    if not blocks:
        raise ValueError('模型目录缺少 block_*_data.npz，无法核对实际输入/输出')
    return dict(root=root, swaps=swaps, weights=paths, blocks=blocks)


def signature_dependencies(recording, params):
    from ...storage import recording_sha
    files = manifest(recording, params)
    if files is None:
        return {'status': 'no_model_directory'}
    paths = set(files['weights'] + files['blocks'] + list(files['root'].glob('*.json')) +
                [files['root']/'online_update_swap_history.jsonl'])
    return dict(directory=str(files['root']), files=[dict(path=str(p), sha256=recording_sha(p)) for p in sorted(paths)])


def load_model(path, channels):
    # Saved files are user-selected local model artifacts, not downloaded code.
    with Path(path).open('rb') as stream:
        model = pickle.load(stream)
    if model.get('model_type') != 'position_velocity':
        raise ValueError(f'不支持的在线模型类型：{path}')
    shapes = {'A': (4,4), 'C': (channels,4), 'Q': (4,4), 'R': (channels,channels),
              'P0': (4,4), 'state_offset': (4,), 'observation_bias': (channels,)}
    for name, shape in shapes.items():
        array = np.asarray(model[name], float)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'{path.name}: {name} 维度或数值不符合 position_velocity 合约')
        model[name] = array
    if model.get('metadata', {}).get('neural_preprocessing', {}).get('method', 'raw') != 'raw':
        raise ValueError('此重放目前仅支持 raw 神经计数输入')
    return model


def filter_saved(data, models, model_index, progress=lambda _:None, feedback='decoded'):
    """No regression, lag search, output seeding or fitted post-hoc transforms."""
    pred = np.full((len(model_index),2), np.nan)
    continuous = np.zeros(len(pred), bool)
    state, covariance = None, None
    with single_thread():
        for index, model in enumerate(models):
            progress(f'saved KF weights {index+1}/{len(models)}')
            rows = np.flatnonzero(model_index == index)
            if not len(rows):
                continue
            A,C,Q,R = [model[k] for k in ('A','C','Q','R')]
            riC = cho_solve(cho_factor(R, lower=True), C)
            precision = C.T @ riC
            natural = (data['neural'][rows].astype(float)-model['observation_bias']) @ riC
            velocity_precision = precision[2:,2:]
            for local, i in enumerate(rows):
                trial_start = i == 0 or data['trial_num_overall'][i] != data['trial_num_overall'][i-1]
                if trial_start:
                    state = np.r_[data.get('feedback_position',data['position'])[i], 0., 0.]
                    covariance = model['P0'].copy()
                    intact = True
                elif data['window_index'][i] != data['window_index'][i-1]+1:
                    intact = False  # unrecorded neural inputs: state cannot be reconstructed
                predicted = A@state + model['state_offset']
                pcov = A@covariance@A.T + Q
                if feedback == 'recorded':
                    position = data.get('feedback_position',data['position'])[i]
                    predicted[:2] = position
                    pv = np.linalg.inv(np.linalg.inv(pcov[2:,2:]) + velocity_precision)
                    drive = natural[local,2:] - precision[2:,:2]@position
                    velocity = predicted[2:] + pv@(drive - velocity_precision@predicted[2:])
                    state = np.r_[position, velocity]
                    covariance = np.zeros((4,4));covariance[2:,2:]=pv
                else:
                    covariance = np.linalg.inv(np.linalg.inv(pcov) + precision)
                    state = predicted + covariance@(natural[local] - precision@predicted)
                pred[i] = state[2:]
                continuous[i] = intact
    return pred, continuous


def replay(session, params, progress=lambda _:None):
    files = manifest(session.recording, params)
    if files is None:
        return None
    keys = ('neural','lsl_time','window_index','decoded_velocity','position','trial_num_overall','block_num')
    data = {key:[] for key in keys}
    clock_differences = []
    for path in files['blocks']:
        with np.load(path, allow_pickle=False) as block:
            for key in keys:
                data[key].append(block[key])
            pairs = block['cursor_state_samples_raw']
            ct = block['cursor_state_lsl_time_raw']
            clock_differences.extend(pairs[:,0]*1e-9-ct)
    data = {k:np.concatenate(v) for k,v in data.items()}
    if np.any(np.diff(data['window_index']) <= 0) or np.any(np.diff(data['lsl_time']) < 0):
        raise ValueError('block 日志中的帧序号/时间不单调，不能唯一重放')
    # Verify identity on exact frame indices; early log rows can precede XDF start.
    indices = np.searchsorted(session.metadata[:,0], data['window_index'])
    clipped = np.minimum(indices, len(session.counts)-1)
    matched = (indices<len(session.counts)) & (session.metadata[clipped,0] == data['window_index'])
    if matched.sum() < 8:
        raise ValueError('模型日志与当前 XDF 没有足够匹配帧，请检查在线记录和模型 session')
    if not np.array_equal(session.counts[clipped[matched]], data['neural'][matched]):
        raise ValueError('模型日志与当前 XDF 同帧序号的神经计数不一致，拒绝关联')
    if np.max(np.abs(session.neural_times[clipped[matched]]-data['lsl_time'][matched])) > 1e-5:
        raise ValueError('模型日志与当前 XDF 的 LSL 时间不匹配，拒绝关联')
    if len(session.decoder_times) != len(session.counts) or not np.allclose(
            session.decoder_velocity[clipped[matched],:2], data['decoded_velocity'][matched], atol=1e-5, rtol=1e-6):
        raise ValueError('模型日志的在线输出与当前 XDF 的逐帧 decoder 输出不一致')
    clock_differences = np.asarray(clock_differences)
    clock_differences = clock_differences[np.isfinite(clock_differences)]
    if not len(clock_differences):
        raise ValueError('缺少墙钟与 LSL 配对，无法映射 applied_time')
    offset = float(np.median(clock_differences))
    boundaries = np.array([float(s['applied_time'])-offset for s in files['swaps']])
    model_index = np.searchsorted(boundaries, data['lsl_time'], side='right')
    models = [load_model(path, data['neural'].shape[1]) for path in files['weights']]
    original_ids = models[0].get('metadata',{}).get('physical_channel_ids')
    if original_ids is not None and not np.array_equal(original_ids,session.electrode_ids):
        raise ValueError('初始权重物理电极顺序与 XDF 不一致，不能按相同列序重放')
    feedback = params.get('saved_position_feedback','recorded')
    position = data['position'].astype(float).copy()
    cursor_index = np.searchsorted(session.cursor_times,data['lsl_time'],side='right')-1
    have_position = cursor_index>=0
    chosen = np.clip(cursor_index,0,len(session.cursor_times)-1)
    age = data['lsl_time']-session.cursor_times[chosen]
    have_position &= np.isfinite(session.cursor[chosen,2:4]).all(1) & (age<=params.get('feedback_max_age',.25))
    position[have_position] = session.cursor[chosen[have_position],2:4]
    data['feedback_position'] = position
    data['feedback_from_xdf'] = have_position
    data['feedback_age_s'] = age
    pred, continuous = filter_saved(data, models, model_index, progress, feedback=feedback)
    if feedback=='recorded':
        complete_feedback = True
        for i in range(len(matched)):
            if i==0 or data['trial_num_overall'][i]!=data['trial_num_overall'][i-1]:
                complete_feedback = True
            if matched[i] and not have_position[i]:
                complete_feedback = False
            continuous[i] &= complete_feedback
    score_mask = matched & continuous
    return dict(data=data, pred=pred, matched=matched, score_mask=score_mask,
                model_index=model_index, boundaries=boundaries, files=files, models=models,
                clock_offset=offset, clock_p95_ms=float(np.percentile(abs(clock_differences-offset),95)*1000))


def append_replay(ctx, result):
    info = replay(ctx.session, ctx.params, ctx.progress)
    if info is None:
        result.summary['saved_model_replay_status'] = 'no_model_directory'
        return
    from .section import scores
    data, pred, mask = info['data'], info['pred'], info['score_mask']
    measured = data['decoded_velocity']
    metrics = scores(measured[mask], pred[mask])
    per_model = []
    for index,path in enumerate(info['files']['weights']):
        selected = mask & (info['model_index']==index)
        per_model.append(dict(model_index=index, weights_path=str(path), **scores(measured[selected],pred[selected])))
    events = [dict(update_index=s['update_index'], applied_unix_s=float(s['applied_time']),
                   mapped_lsl_s=float(t), weights_path=str(path)) for s,t,path in
              zip(info['files']['swaps'],info['boundaries'],info['files']['weights'][1:])]
    fig = new_figure(13,8)
    axes = fig.subplots(2,2)
    t0 = ctx.cohort.starts.min()
    start = t0+ctx.params['zoom_start']
    shown = info['matched'] & (data['lsl_time']>=start) & (data['lsl_time']<=start+ctx.params['zoom_len'])
    if not shown.any():
        start = data['lsl_time'][info['matched']][0]
        shown = info['matched'] & (data['lsl_time']>=start) & (data['lsl_time']<=start+ctx.params['zoom_len'])
    for axis,ax in enumerate(axes[0]):
        for tr in np.unique(data['trial_num_overall'][shown]):
            rows = np.flatnonzero(shown & (data['trial_num_overall']==tr))
            # Break lines at missing neural inputs as well as trial boundaries.
            for segment in np.split(rows,np.flatnonzero(np.diff(data['window_index'][rows])!=1)+1):
                ax.plot(data['lsl_time'][segment]-t0,measured[segment,axis],color='#2176ae',lw=.9)
                y = np.where(mask[segment],pred[segment,axis],np.nan)
                ax.plot(data['lsl_time'][segment]-t0,y,color='#8e5ea2',lw=.8,ls='--')
        ax.plot([],[],color='#2176ae',label='Recorded online output')
        ax.plot([],[],color='#8e5ea2',ls='--',label='Saved-weight matrix replay')
        for boundary in info['boundaries']:
            if start<=boundary<=start+ctx.params['zoom_len']:
                ax.axvline(boundary-t0,color='#c0392b',lw=.7)
        ax.set(title=f'{"AB"[axis]}  Velocity {"xy"[axis]}',xlabel='Time from first included trial / s',ylabel='Velocity / units s⁻¹')
        ax.legend(fontsize=6,frameon=False)
    ax = axes[1,0]
    ax.bar(range(len(per_model)),[r['rmse'] for r in per_model],color='#8e5ea2')
    ax.set(title='C  Replay error by applied weight segment',xlabel='Weight segment (0 = initial)',ylabel='Vector RMSE vs online')
    ax.set_xticks(range(len(per_model)))
    ax = axes[1,1]
    error = np.linalg.norm(pred-measured,axis=1)
    good = mask & np.isfinite(error)
    ax.hist(error[good],bins=60,color='#2176ae')
    ax.set_yscale('log')
    ax.set(title='D  Frame error distribution (matched, complete input runs)',xlabel='Vector error / units s⁻¹',ylabel='Frames (log)')
    feedback_label = ('XDF position feedback' if ctx.params.get('saved_position_feedback','recorded')=='recorded'
                      else 'KF position estimate after trial initialization')
    fig.suptitle(f'{ctx.session.label} · saved KF replay ({feedback_label}) · R²={metrics["r2_xy"]:.3f}; '
                 f'RMSE={metrics["rmse"]:.2f}\nRecorded-data replay; original runtime reset/swap behavior is unverified',fontsize=9)
    fig.supxlabel(f'{int(mask.sum()):,} scored / {int(info["matched"].sum()):,} matched frames; '
                  f'{len(events)} applied swaps; wall↔LSL mapping residual P95={info["clock_p95_ms"]:.3f} ms. '
                  'No model fitting or output-based correction.',fontsize=7)
    rows = [dict(window_index=int(data['window_index'][i]),lsl_s=float(data['lsl_time'][i]),
                 trial=int(data['trial_num_overall'][i]),model_index=int(info['model_index'][i]),
                 scored=bool(mask[i]),recorded_vx=float(measured[i,0]),recorded_vy=float(measured[i,1]),
                 feedback_x=float(data['feedback_position'][i,0]),feedback_y=float(data['feedback_position'][i,1]),
                 feedback_from_xdf=bool(data['feedback_from_xdf'][i]),feedback_age_s=float(data['feedback_age_s'][i]),
                 replay_vx=float(pred[i,0]),replay_vy=float(pred[i,1])) for i in np.flatnonzero(info['matched'])]
    method = ('Saved position_velocity KF matrix replay in applied_time order, NOT ridge/ARX reproduction. '
              'Directory identity verified by exact frame IDs, neural counts, LSL timestamps and recorded '
              'decoder output. Raw counts, no six-frame concatenation: lag_bins is a training label shift. '
              'Initial and updated A/C/Q/R/bias/P0 used as saved; updated files already include training blend, '
              f'no second blending. Position feedback={ctx.params.get("saved_position_feedback","recorded")}. '
              'Recorded: causal last XDF cursor position at/before neural frame, maximum age configured; '
              'clamp predicted/posterior position and zero position covariance, condition the velocity '
              'substate using saved C/R. No future interpolation or recorded velocity input. Before XDF '
              'start, saved block positions warm up the state. Missing/stale feedback within an XDF '
              'trial excludes scoring until next reset. Decoded: position only initializes the trial, '
              'full KF thereafter. Trial-start state=[position,0 velocity], P=P0. State/covariance '
              'preserved across weight swaps. Standard linear KF in information-form algebra; no output '
              'seeding or fit. Applied wall times mapped to LSL by median paired cursor timestamp offset. '
              'This mapping is not an exact frame-level switch acknowledgement. Missing within-trial neural '
              'frames invalidate scoring until next trial. Scores restricted to exact matching XDF frames, '
              'all logged active trials, independent of Topbar filtering. Runtime source/reset/swap/lag/output '
              'postprocessing semantics remain UNVERIFIED; errors displayed, not presented as exact reproduction. '
              f'Directory={info["files"]["root"]}; clock offset={info["clock_offset"]:.9f} s; '
              f'clock residual P95={info["clock_p95_ms"]:.6f} ms.')
    result.add_figure('Saved online model replay audit',fig,method=method,
                      data={'matched_frames':rows,'applied_weights':per_model,'weight_switches':events})
    # Promote the real-artifact comparison above the existing statistical surrogate.
    for attr in ('figures','methods','figure_data'):
        values = getattr(result,attr)
        values.insert(0,values.pop())
    result.tables.update(saved_model_replay_frames=rows,saved_model_replay_segments=per_model,
                         saved_model_weight_switches=events)
    result.summary.update(saved_model_replay_status='matrix_replay_runtime_unverified',
                          saved_model_directory=str(info['files']['root']), saved_model_replay_metrics=metrics,
                          saved_model_matched_frames=int(info['matched'].sum()),saved_model_scored_frames=int(mask.sum()),
                          saved_model_weight_switches=len(events),saved_model_clock_mapping_p95_ms=info['clock_p95_ms'])
    result.summary['saved_model_position_feedback'] = ctx.params.get('saved_position_feedback','recorded')
    result.notes.append('已按实际 applied 日志重放保存的 KF 矩阵；原在线 runtime 的重置、切换及后处理语义尚未核对，'
                        '不能称为精确复现。切换后误差需检查 runtime，而非通过拟合输出消除。')
