"""Saved KF replay: causal equations, actual swap order, identity and cache inputs."""
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data_analysis_gui.sections.s11_online_vs_offline_v1.model_replay import (
    filter_saved, manifest, replay, signature_dependencies, resolve_run,
)


def model(scale=1):
    return dict(model_type='position_velocity', A=np.eye(4), C=np.array([[1.,0,.2,0],[0,1,0,.2]]),
                Q=np.eye(4)*.1, R=np.eye(2)*scale, P0=np.eye(4),
                state_offset=np.zeros(4),observation_bias=np.array([.1,.2]), metadata={'lag_bins':6})


def test_packaged_model_resolution_is_scoped_to_recording(tmp_path):
    day = tmp_path / '20261104'
    run = day / 'CO8_run-002_online'
    refit = run / 'refit'
    refit.mkdir(parents=True)
    other_refit = day / 'grid_run-001_online' / 'refit'
    other_refit.mkdir(parents=True)
    (other_refit / 'online_update_swap_history.jsonl').touch()
    initial_dir = run / 'initial_weights'
    initial_dir.mkdir()
    initial = initial_dir / 'INITIAL_position_velocity_kf.pkl'
    initial.touch()
    updated = refit / 'refit_models' / 'position_velocity_kf' / 'updated.pkl'
    updated.parent.mkdir(parents=True)
    updated.touch()
    (refit / 'block_0001_data.npz').touch()
    event = dict(model_name='position_velocity_kf', applied_time=1002,
                 auto_swap={'swaps': {'position_velocity_kf': dict(
                     status='applied', old_weights_path='C:/old/training_name.pkl',
                     active_weights_path_after_swap='C:/old/refit_models/position_velocity_kf/updated.pkl')}})
    (refit / 'online_update_swap_history.jsonl').write_text(json.dumps(event))
    recording = SimpleNamespace(path=run / 'xdf' / 'recording.xdf')
    for params in ({}, dict(online_model_dir=str(refit), initial_model_path=str(initial_dir))):
        assert manifest(recording, params)['weights'] == [initial, updated]


def inputs():
    return dict(neural=np.array([[2.,3],[4,1],[5,2],[2,5],[6,1],[3,8]]),
                position=np.tile([1.,2.],(6,1)),window_index=np.arange(100,106),
                trial_num_overall=np.array([1,1,1,2,2,2]))


def test_filter_matches_direct_kf_and_does_not_use_future_or_recorded_output():
    data=inputs()
    models=[model(),model(2)]
    schedule=np.array([0,0,0,1,1,1])
    actual,good=filter_saved(data,models,schedule)
    expected=[]
    for i,index in enumerate(schedule):
        m=models[index]
        if i==0 or data['trial_num_overall'][i]!=data['trial_num_overall'][i-1]:
            state=np.r_[data['position'][i],0.,0.];P=m['P0'].copy()
        xp=m['A']@state+m['state_offset'];pp=m['A']@P@m['A'].T+m['Q']
        C=m['C'];K=np.linalg.solve(C@pp@C.T+m['R'],C@pp).T
        state=xp+K@(data['neural'][i]-m['observation_bias']-C@xp)
        P=(np.eye(4)-K@C)@pp
        expected.append(state[2:])
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    assert good.all()
    modified={**data,'neural':data['neural'].copy(),'decoded_velocity':np.ones((6,2))*1e9}
    modified['neural'][-1]*=5
    changed,_=filter_saved(modified,models,schedule)
    np.testing.assert_array_equal(actual[:-1],changed[:-1])


def test_missing_neural_frames_exclude_until_trial_reset():
    data=inputs()
    data['window_index']=np.array([100,102,103,104,105,106])
    _,good=filter_saved(data,[model()],np.zeros(6,int))
    assert good.tolist()==[True,False,False,True,True,True]


def test_recorded_position_feedback_matches_conditioned_velocity_kf():
    data=inputs();m=model();expected=[]
    for i in range(len(data['neural'])):
        if i==0 or data['trial_num_overall'][i]!=data['trial_num_overall'][i-1]:
            state=np.r_[data['position'][i],0.,0.];P=m['P0'].copy()
        xp=m['A']@state;pp=m['A']@P@m['A'].T+m['Q']
        Cv=m['C'][:,2:];pos=data['position'][i]
        residual=data['neural'][i]-m['observation_bias']-m['C'][:,:2]@pos-Cv@xp[2:]
        K=np.linalg.solve(Cv@pp[2:,2:]@Cv.T+m['R'],Cv@pp[2:,2:]).T
        velocity=xp[2:]+K@residual
        P=np.zeros((4,4));P[2:,2:]=(np.eye(2)-K@Cv)@pp[2:,2:]
        state=np.r_[pos,velocity];expected.append(velocity)
    pred,good=filter_saved(data,[m],np.zeros(6,int),feedback='recorded')
    np.testing.assert_allclose(pred,expected,atol=1e-12)
    assert good.all()


def test_artifact_association_chronology_and_cache_dependency(tmp_path):
    day=tmp_path/'20261102';run=day/'training_run003'/'session'
    run.mkdir(parents=True)
    initial=day/'initial.pkl';updated=run/'updated.pkl'
    initial.write_bytes(pickle.dumps(model()));updated.write_bytes(pickle.dumps(model(2)))
    history=run/'online_update_swap_history.jsonl'
    event=dict(model_name='position_velocity_kf',update_index=5,applied_time=1002.5,
               auto_swap={'swaps':{'position_velocity_kf':dict(status='applied',
                  old_weights_path='C:/old/initial.pkl',active_weights_path_after_swap='C:/old/updated.pkl')}})
    history.write_text(json.dumps(event)+'\n')
    data=inputs();schedule=np.array([0,0,0,1,1,1])
    pred,_=filter_saved(data,[model(),model(2)],schedule)
    times=np.arange(6,dtype=float)
    raw_cursor=np.zeros((6,12));raw_cursor[:,0]=(times+1000)*1e9
    np.savez(run/'block_0001_data.npz',**data,lsl_time=times,decoded_velocity=pred,
             block_num=np.ones(6),cursor_state_samples_raw=raw_cursor,cursor_state_lsl_time_raw=times)
    # At least eight exact matching frames are required by the production association.
    more={k:np.concatenate([v,v]) for k,v in data.items()}
    more['window_index']=np.arange(100,112);more['trial_num_overall'][6:]+=2
    times=np.arange(12,dtype=float);raw_cursor=np.zeros((12,12));raw_cursor[:,0]=(times+1000)*1e9
    schedule=np.searchsorted([2.5],times,side='right')
    pred,_=filter_saved(more,[model(),model(2)],schedule)
    np.savez(run/'block_0001_data.npz',**more,lsl_time=times,decoded_velocity=pred,
             block_num=np.ones(12),cursor_state_samples_raw=raw_cursor,cursor_state_lsl_time_raw=times)
    recording=SimpleNamespace(path=day/'run-004_spike_online.xdf')
    cursor=np.zeros((12,12));cursor[:,2:4]=more['position']
    session=SimpleNamespace(recording=recording,metadata=more['window_index'][:,None],counts=more['neural'],
                            neural_times=times,decoder_times=times,decoder_velocity=pred,cursor_times=times,cursor=cursor)
    info=replay(session,{'saved_position_feedback':'decoded'})
    np.testing.assert_array_equal(info['model_index'],schedule)
    np.testing.assert_allclose(info['pred'],pred,atol=1e-12)
    assert info['score_mask'].all() and info['clock_offset']==pytest.approx(1000)
    recorded=replay(session,{'saved_position_feedback':'recorded'})
    assert recorded['data']['feedback_from_xdf'].all()
    np.testing.assert_array_equal(recorded['data']['feedback_position'],more['position'])
    # A future cursor position cannot change earlier predictions.
    session.cursor[-1,2:4]=1e6
    changed=replay(session,{'saved_position_feedback':'recorded'})
    np.testing.assert_array_equal(recorded['pred'][:-1],changed['pred'][:-1])
    session.cursor[-1,2:4]=more['position'][-1]
    before=signature_dependencies(recording,{})
    updated.write_bytes(pickle.dumps(model(3)))
    after=signature_dependencies(recording,{})
    assert before!=after
    session.counts=session.counts.copy();session.counts[0,0]+=1
    with pytest.raises(ValueError,match='神经计数不一致'):
        replay(session,{})
    other=day/'other';other.mkdir();(other/history.name).write_text(history.read_text())
    with pytest.raises(ValueError,match='多个'):
        resolve_run(recording)
    assert resolve_run(recording,str(run))==run
