"""Identify target assistance from saved KF replay, online velocity and actual cursor motion."""
from pathlib import Path
import sys
import json
import csv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from matplotlib.figure import Figure
from data_analysis_gui.core.catalog import describe
from data_analysis_gui.core.session import Session
from data_analysis_gui.sections.s11_online_vs_offline_v1.model_replay import replay
from data_analysis_gui.sections.s11_online_vs_offline_v1.section import scores

ROOT = Path('E:/AitBCI/data/keaiduo/20261106')
OUT = Path('E:/AitBCI/data_analysis_gui_output/keaiduo/comparisons/20261106_assist_mechanism')
RUNS = ['run-002-CO8-online', 'run-003-CO8-online-restrained-assist-200',
        'run-004-CO8-online-restrained-assist-100']


def smooth(raw, same, alpha=.8):
    values = raw.copy()
    for i in range(1, len(raw)):
        values[i] = alpha * raw[i] + (1-alpha) * (values[i-1] if same[i] else 0)
    return values


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def hit_time(position, times, target, radius, dwell):
    entry = None
    for p, t in zip(position, times):
        if np.linalg.norm(p-target) <= radius:
            if entry is None:
                entry = t
            if t-entry >= dwell:
                return float(t-times[0])
        else:
            entry = None
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, blocks, trial_rows = [], [], []
    example = None
    for name in RUNS:
        folder = ROOT/name
        config = json.loads((folder/'config/task_config.json').read_text(encoding='utf-8'))
        ctrl = config['CONTROL']
        base = float(ctrl.get('BCI_TARGET_ASSIST_SPEED', 0)) if ctrl.get('BCI_TARGET_ASSIST_ENABLED') else 0.
        decay = float(ctrl.get('BCI_TARGET_ASSIST_DECAY_PER_BLOCK', 0))
        radius = float(config['CALCULATED']['TARGET_SIZE_PX'])/2
        dwell = float(config['GENERAL_TIMING']['DWELL_TIME_SEC'])
        session = Session.load(describe(sorted((folder/'xdf').glob('*.xdf'))[-1], ROOT))
        replay_file = OUT/(name+'_decoded_position_replay.npz')
        if not replay_file.exists():
            info = replay(session, {'saved_position_feedback': 'decoded'})
            np.savez_compressed(replay_file, pred=info['pred'], mask=info['score_mask'],
                                lsl_time=info['data']['lsl_time'])
        with np.load(replay_file) as saved:
            replay_time, replay_velocity, replay_mask = saved['lsl_time'], saved['pred'], saved['mask']
        with np.load(OUT/(name+'_replay.npz')) as original_replay:
            assert np.array_equal(original_replay['lsl_time'], replay_time)
            replay_reference = original_replay['online']
        replay_score = scores(replay_reference[replay_mask], replay_velocity[replay_mask])
        c, times = session.cursor, session.cursor_times
        same = np.r_[False, (np.diff(c[:,10])==0) & (np.diff(c[:,8])==0) & (c[1:,11]>0) & (c[:-1,11]>0)]
        previous = np.vstack([c[:1,2:4], c[:-1,2:4]])
        direction = unit(c[:,6:8]-previous)
        speed = np.maximum(0, base * (1-decay*(c[:,8]-1)))
        assist = direction * speed[:,None]
        # One neural-frame consumption delay, chosen in the unassisted control before assisted comparisons.
        ix = np.searchsorted(session.decoder_times, times-.02, side='right')-1
        raw = session.decoder_velocity[np.clip(ix,0,len(session.decoder_times)-1),:2]
        brain = smooth(raw, same)
        ri = np.searchsorted(replay_time, times-.02, side='right')-1
        ri = np.clip(ri, 0, len(replay_time)-1)
        replay_brain = smooth(replay_velocity[ri], same)
        dt = np.r_[np.nan, np.diff(c[:,0])*1e-9]
        valid = same & (ix>=0) & (dt>.005) & (dt<.04)
        # Exclude post-acquisition feedback from inference of active control.
        active_motion = np.zeros(len(c), bool)
        for tr in session.trials:
            stop = tr['start'] + (tr['acquire_time'] if tr['outcome']=='success' else tr.get('miss_time', np.nan))
            if not tr['complete'] or not np.isfinite(stop):
                continue
            active_motion |= (times>=tr['start']) & (times<=stop) & (c[:,10]==tr['trial_id'])
        valid &= active_motion
        candidates = dict(no_assist=brain, radial_add=brain+assist,
                          fixed_direction_add=brain+unit(c[:,6:8])*speed[:,None],
                          half_blend=.5*brain+.5*assist, assist_only=assist)
        errors = {key: float(np.sqrt(np.mean(np.sum((c[valid,4:6]-v[valid])**2,axis=1))))
                  for key,v in candidates.items()}
        model_ok = valid & replay_mask[ri] & (times-replay_time[ri] < .15)
        model_errors = {key: float(np.sqrt(np.mean(np.sum((c[model_ok,4:6]-v[model_ok])**2,axis=1))))
                        for key,v in dict(no_assist=replay_brain, radial_add=replay_brain+assist).items()}
        remainder = c[:,4:6]-brain
        projection = np.sum(remainder*direction, axis=1)
        coeff = float(np.sum(projection[valid]*speed[valid])/np.sum(speed[valid]**2)) if base else None
        low = valid & (np.linalg.norm(brain,axis=1)<10) & (speed>=50)
        low_result = dict(frames=int(low.sum()),
                          median_targetward_cursor_speed=float(np.median(np.sum(c[low,4:6]*direction[low],axis=1))) if low.any() else None,
                          median_assist_speed=float(np.median(speed[low])) if low.any() else None)
        for b in np.unique(c[valid,8]):
            keep = valid & (c[:,8]==b)
            blocks.append(dict(run=name, block=int(b), frames=int(keep.sum()),
                               configured_speed=float(np.median(speed[keep])),
                               measured_residual_radial_median=float(np.median(projection[keep])),
                               no_assist_rmse=float(np.sqrt(np.mean(np.sum((c[keep,4:6]-brain[keep])**2,axis=1)))),
                               radial_add_rmse=float(np.sqrt(np.mean(np.sum((c[keep,4:6]-brain[keep]-assist[keep])**2,axis=1))))))
        this_trials = []
        for tr in session.trials:
            if not tr['complete'] or tr['outcome'] not in ('success','miss'):
                continue
            duration = tr['acquire_time'] if tr['outcome']=='success' else tr.get('miss_time',np.nan)
            if not np.isfinite(duration):
                continue
            indices = np.flatnonzero((c[:,10]==tr['trial_id']) & (times>=tr['start']) &
                                     (times<=tr['start']+duration+.04) & (c[:,11]>0))
            if len(indices)<5 or times[indices[0]]-tr['start']>.12 or np.max(np.diff(times[indices]))>.06:
                continue
            wall = c[indices,0]*1e-9
            steps = np.minimum(np.diff(wall), .05)
            trajectories = {}
            target = c[indices[0],6:8]
            for label, velocities in [('online_no_assist',brain),('online_add',brain),
                                      ('saved_model_no_assist',replay_brain),('saved_model_add',replay_brain)]:
                pos = np.zeros((len(indices),2));pos[0]=c[indices[0],2:4]
                for j, (i, step) in enumerate(zip(indices[1:], steps),1):
                    velocity = velocities[i]
                    if label.endswith('_add'):
                        velocity = velocity+speed[i]*unit(target-pos[j-1])
                    pos[j] = pos[j-1]+velocity*step
                trajectories[label]=pos
            row = dict(run=name, trial=int(tr['trial_id']), block=int(tr['block']),
                       actual_outcome=tr['outcome'], actual_active_seconds=float(duration),
                       covered_seconds=float(wall[-1]-wall[0]), assist_speed=float(speed[indices[0]]))
            for label,pos in trajectories.items():
                row[label+'_position_rmse']=float(np.sqrt(np.mean(np.sum((pos-c[indices,2:4])**2,axis=1))))
                row[label+'_hit_seconds']=hit_time(pos,wall,target,radius,dwell)
            this_trials.append(row)
            if base==200 and row['assist_speed']>=160 and tr['outcome']=='success' and row['online_add_position_rmse']<20:
                if example is None or row['saved_model_add_position_rmse']<example['row']['saved_model_add_position_rmse']:
                    example=dict(row=row, trajectories=trajectories, actual=c[indices,2:4],
                                 target=target, radius=radius, time=wall-wall[0])
        trial_rows.extend(this_trials)
        summary=dict(run=name, valid_cursor_frames=int(valid.sum()), saved_model_replay=replay_score,
                     cursor_velocity_rmse=errors, saved_model_cursor_velocity_rmse=model_errors,
                     estimated_assist_gain=coeff, near_zero_decoder=low_result,
                     trajectory_trials=len(this_trials), target_radius=radius, dwell_seconds=dwell,
                     median_position_rmse={key:float(np.median([row[key+'_position_rmse'] for row in this_trials]))
                                           for key in ['online_no_assist','online_add','saved_model_no_assist','saved_model_add']})
        summaries.append(summary)
        print(json.dumps(summary),flush=True)
    result=dict(summaries=summaries,
                method='Saved matrices replay uses full KF position estimate after per-trial position initialization, actual applied model order, original neural counts; no output seeding or fitted output correction. Decoder-consumption lag=20ms and smoothing alpha=.8 were checked against the unassisted run. The assistance gain is a separate descriptive residual diagnostic, not a correction to replay.',
                example_selection='Low-replay-error illustrative successful trial among early assist200 blocks; population medians are reported separately.',
                qualifications=['Weight/reset/timestamp semantics are not exactly reproduced; replay error is reported.',
                                'Fixed recorded neural inputs and recorded initial positions are used in position simulations; these are not causal estimates of how the monkey would behave without assistance.',
                                'Simulated acquisition is only checked within the recorded trial horizon; missing hit is not a claim of permanent failure.'])
    (OUT/'assist_audit_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    write_csv(OUT/'assist_speed_by_block.csv',blocks)
    write_csv(OUT/'trial_position_and_acquisition.csv',trial_rows)
    fig=Figure(figsize=(12,8),layout='constrained');axes=fig.subplots(2,2)
    for name,color in [(RUNS[1],'#7b3294'),(RUNS[2],'#007f88')]:
        group=[r for r in blocks if r['run']==name]
        label='Initial assist 200' if '200' in name else 'Initial assist 100'
        axes[0,0].plot([r['block'] for r in group],[r['configured_speed'] for r in group],color=color,ls='--')
        axes[0,0].plot([r['block'] for r in group],[r['measured_residual_radial_median'] for r in group],color=color,label=label)
    axes[0,0].set(xlabel='Block',ylabel='Targetward residual (px/s)',title='Configured assistance vs observed residual')
    axes[0,0].legend(frameon=False)
    x=np.arange(3);axes[0,1].bar(x-.16,[r['cursor_velocity_rmse']['no_assist'] for r in summaries],width=.32,label='Decoder only')
    axes[0,1].bar(x+.16,[r['cursor_velocity_rmse']['radial_add'] for r in summaries],width=.32,label='Decoder + target assist')
    axes[0,1].set(xticks=x,xticklabels=['No assist','Assist200','Assist100'],ylabel='Vector RMSE (px/s)',title='Cursor velocity reconstruction')
    axes[0,1].legend(frameon=False)
    if example:
        ax=axes[1,0];ax.plot(*example['actual'].T,color='black',label='Actual cursor')
        for key,color in [('saved_model_no_assist','#2176ae'),('saved_model_add','#bd5021')]:
            ax.plot(*example['trajectories'][key].T,color=color,label='Saved KF only' if 'no_assist' in key else 'Saved KF + assist')
        from matplotlib.patches import Circle
        ax.add_patch(Circle(example['target'],example['radius'],fill=False,color='#15803d'))
        ax.set(aspect='equal',xlabel='Cursor x (px)',ylabel='Cursor y (px)',title=f"Assist200 example, trial {example['row']['trial']}")
        ax.legend(frameon=False,fontsize=7)
        ax=axes[1,1]
        for key,color in [('actual','black'),('saved_model_no_assist','#2176ae'),('saved_model_add','#bd5021')]:
            pos=example['actual'] if key=='actual' else example['trajectories'][key]
            ax.plot(example['time'],np.linalg.norm(pos-example['target'],axis=1),color=color,label=key.replace('_',' '))
        ax.axhline(example['radius'],color='#15803d',ls='--',label='Target radius')
        ax.set(xlabel='Time in recorded trial (s)',ylabel='Distance to target center (px)',title='Position and acquisition timing')
        ax.legend(frameon=False,fontsize=7)
        (OUT/'example_trial.json').write_text(json.dumps(example['row'],indent=2),encoding='utf-8')
    for ax in axes.ravel():ax.grid(alpha=.2)
    fig.suptitle('2026-11-06: additive target assistance audit')
    fig.savefig(OUT/'assist_mechanism_audit.png',dpi=180);fig.clear()


if __name__=='__main__':main()
