import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data_analysis_gui.core.catalog import Recording
from data_analysis_gui.sections.s15_task_replay_v1.replay import PlaybackClock, build_payload, sample_at, source_files
from data_analysis_gui.sections.s15_task_replay_v1.section import SECTION


def session(tmp_path, mode='grid'):
    root=tmp_path/'run-005-Grid-offline'
    (root/'xdf').mkdir(parents=True)
    path=root/'xdf/main.xdf';path.write_bytes(b'xdf')
    (root/'trials.csv').write_text('trial_num_overall\n1\n',encoding='utf-8')
    (root/'task_config.json').write_text(json.dumps(dict(
        TASK_INFO=dict(TASK_MODE=mode),SYSTEM=dict(WIN_WIDTH=1024,WIN_HEIGHT=768),
        GRID_TASK=dict(GRID_ROWS=8,GRID_COLS=8,AUTO_SCALE_TO_WINDOW=True,WINDOW_FILL_RATIO=.9,MIN_CELL_SIZE_PX=30),
        CALCULATED=dict(CURSOR_SIZE_PX=16.2,TARGET_SIZE_PX=108,GRID_MARGIN_PX=54))),encoding='utf-8')
    (root/'logs').mkdir();(root/'logs/task.log').write_text('Actual window size: [1920 1080]',encoding='utf-8')
    cursor=np.zeros((5,12));cursor[:,2]=[0,10,900,950,960];cursor[:,3]=[0,5,0,5,10]
    cursor[:,4:6]=[70,80];cursor[:,6:8]=[382.725,54.675];cursor[:,8]=1
    cursor[:,10]=[1,1,2,2,2];cursor[:,11]=1
    rec=Recording(path,'main','2026-11-18',mode,False,'005')
    return SimpleNamespace(cursor=cursor,cursor_times=np.array([100,100.1,100.2,101,101.1]),
                           event_times=np.array([100.1]),events=[['T_TARGET_ACQUIRED']],
                           recording=rec,task_mode=mode,label='replay test')


def test_original_times_velocity_geometry_and_discontinuities(tmp_path):
    s=session(tmp_path);p=build_payload(s,SECTION.defaults())
    assert len(p['frames'])==5
    assert p['geometry']['width']==1920 and p['geometry']['height']==1080
    assert p['geometry']['cell_size']==pytest.approx(109.35)
    assert p['geometry']['cursor_radius']==16.2
    _,state,gap=sample_at(p,.05)
    assert state[1:3]==pytest.approx([5,2.5])
    assert state[3:5]==[70,80]  # Saved control velocity is not derived/integrated.
    assert not gap
    assert sample_at(p,.15)[1][1]==10  # No interpolation into a different trial.
    assert sample_at(p,.5)[2]  # No fabricated path across a recording gap.
    assert sample_at(p,1)[1][1]==950
    assert len(p['trials'])==2


def test_duplicates_invalid_positions_missing_speed(tmp_path):
    s=session(tmp_path)
    s.cursor_times[1]=100
    s.cursor[2,2]=np.nan
    s.cursor[3,4]=np.nan
    p=build_payload(s,SECTION.defaults())
    assert len(p['frames'])==3
    assert p['frames'][0][1]==10  # Last state at duplicate timestamp wins.
    assert p['frames'][1][3] is None
    json.dumps(p,allow_nan=False)


def test_clock_is_wall_time_and_speed_changes_are_continuous():
    now=[100.]
    c=PlaybackClock(10,now=lambda:now[0]);c.play()
    now[0]+=2;assert c.position()==2
    c.speed(2);assert c.position()==2
    now[0]+=1;assert c.position()==4
    c.pause();now[0]+=9;assert c.position()==4
    c.seek(9);c.play();now[0]+=3;assert c.position()==10
    c.play();assert c.position()==0
    with pytest.raises(ValueError):c.speed(0)


def test_manual_zero_velocity_falls_back_to_position_difference(tmp_path):
    s=session(tmp_path);s.cursor[:,4:6]=0
    p=build_payload(s,SECTION.defaults())
    assert p['velocity_source']=='位置差分速度'
    assert p['frames'][1][3:5]==pytest.approx([100,50])
    assert p['frames'][2][3:5]==[None,None]  # Reset between trials.
    assert p['frames'][3][3:5]==[None,None]  # Long gap.
    assert p['frames'][1][10:12]==[0,0]  # Preserve raw saved values.
    assert p['frames'][0][3:5]==[None,None]
    saved=build_payload(s,{**SECTION.defaults(),'velocity_source':'保存速度'})
    assert saved['velocity_source']=='保存速度' and saved['frames'][1][3:5]==[0,0]


def test_ambiguous_behavior_bundle_not_assigned(tmp_path):
    s=session(tmp_path)
    s.recording.path.with_name('early.xdf').write_bytes(b'early')
    assert source_files(s.recording)=={}


def test_normalized_legacy_config_and_actual_window(tmp_path):
    s=session(tmp_path,mode='centerout');root=s.recording.path.parent.parent
    (root/'config').mkdir()
    (root/'task_config.json').rename(root/'config/task_config.json')
    (root/'logs/task.log').rename(root/'logs/cursor_tasks_app.log')
    files=source_files(s.recording)
    assert files['task_config']==root/'config/task_config.json'
    p=build_payload(s,SECTION.defaults())
    assert p['geometry']['width']==1920 and p['geometry']['target_shape']=='circle'


def test_storage_hashes_and_batch_exports_player(tmp_path):
    from data_analysis_gui import storage
    from data_analysis_gui.exporting import export_batch,export_result
    from data_analysis_gui.core.filters import TrialFilter
    from data_analysis_gui.sections.base import AnalysisContext
    s=session(tmp_path)
    ctx=AnalysisContext([s],[],TrialFilter(),SECTION.defaults())
    result=SECTION.run(ctx)
    assert '__REPLAY_PAYLOAD__' not in result.attachments['task_replay.html']
    folder=tmp_path/'result'
    inputs=dict(recordings=[dict(sha256='test')],criteria={},params=ctx.params,code='code')
    record=storage.save_result(result,folder,'signature',inputs,SECTION,[s.recording])
    assert storage.load_cached(folder,'signature')
    batch=export_batch(tmp_path,'figure')
    export_result(folder,record,[s.recording],batch)
    assert len(list(Path(batch['root']).rglob('task_replay.html')))==1
    (folder/record['attachments']['task_replay.json']).write_text('{}',encoding='utf-8')
    assert storage.load_cached(folder,'signature') is None
    for _,fig in result.figures:fig.clear()


def test_replay_sources_invalidate_signature(tmp_path,monkeypatch):
    from data_analysis_gui import storage
    from data_analysis_gui.core.filters import TrialFilter
    s=session(tmp_path)
    monkeypatch.setattr(storage,'recording_sha',lambda path:'xdf-hash')
    def sig():return storage.fingerprint(storage.signature_inputs(SECTION,[s.recording],TrialFilter(),SECTION.defaults()))
    before=sig()
    config=s.recording.path.parent.parent/'task_config.json'
    config.write_text('{}',encoding='utf-8')
    assert sig()!=before


def test_embedded_player_controls_and_cleanup(tmp_path):
    import tkinter as tk
    from data_analysis_gui.gui.task_replay import ReplayView
    try:
        root=tk.Tk()
    except tk.TclError:
        pytest.skip('Tk display unavailable')
    root.withdraw()
    try:
        payload=build_payload(session(tmp_path),SECTION.defaults())
        view=ReplayView(root,payload)
        view.pack(fill='both',expand=True)
        root.geometry('900x600');root.update()
        view.seek(.05)
        assert view.clock.position()==pytest.approx(.05)
        view.jump(1)
        assert view.clock.position()==pytest.approx(.2)
        view.toggle();assert view.clock.playing
        view.clock.speed(2);assert view.clock.rate==2
        view.pause();assert not view.clock.playing
        view.reset();assert view.clock.position()==0
        view.destroy();root.update()
        assert view._timer is None
    finally:
        root.destroy()
