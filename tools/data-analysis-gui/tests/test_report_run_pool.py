from pathlib import Path
import csv,json,os
from datetime import datetime,timedelta
import numpy as np
from data_analysis_gui.report_run_pool import RunPool,file_inputs


def test_section_layout_preserves_original_and_indexes_plot_data(tmp_path):
    from data_analysis_gui import storage
    source=tmp_path/'source';(source/'tables').mkdir(parents=True);(source/'figures').mkdir()
    (source/'tables/channels.csv').write_text('channel,value\n0,3\n')
    (source/'figures/tuning.png').write_bytes(b'plot')
    record=dict(section='s05',signature='source-signature',tables={'channels':'tables/channels.csv'},
                figures=['figures/tuning.png'],digests={p:storage.file_digest(source/p) for p in ('tables/channels.csv','figures/tuning.png')})
    original=json.dumps(record);(source/'result.json').write_text(original,encoding='utf-8')
    pool=RunPool(tmp_path/'individual_runs')
    entry=pool.section(dict(folder=str(source),record=record,job=dict(section='s05',recordings=[str(tmp_path/'20261120/run-001-Grid-offline/input.xdf')])))
    assert entry.relative_to(pool.root).parts[:3]==('S05','2026-11-20_run-001_Grid-offline','analysis')
    updated=json.loads((entry/'result.json').read_text(encoding='utf-8'))
    assert updated['tables']['channels']=='data/tables/channels.csv'
    assert (source/'result.json').read_text(encoding='utf-8')==original
    assert pool._valid(entry)
    index=json.loads((entry.parent.parent/'index.json').read_text(encoding='utf-8'))
    assert index['variants'][0]['figures']==['figures/tuning.png']
    assert 'data/tables/channels.csv' in index['variants'][0]['data']
    assert (entry.parent.parent/'index.html').is_file()


def test_pool_reuses_valid_artifacts_and_reports_share_storage(tmp_path):
    pool=RunPool(tmp_path/'individual_runs');calls=[]
    def build(folder):
        calls.append(1);(folder/'plot.png').write_bytes(b'original')
        return {'values':[0,float('nan'),3]}
    data,entry,hit=pool.get('BGA','run-001',{'source':'a'},build)
    assert not hit and np.isnan(data['values'][1])
    first=pool.publish(entry,tmp_path/'report1')[0]
    other=RunPool(pool.root)
    _,again,hit=other.get('BGA','run-001',{'source':'a'},lambda p:(_ for _ in ()).throw(AssertionError('must not recompute')))
    second=other.publish(again,tmp_path/'report2')[0]
    assert hit and len(calls)==1 and os.path.samefile(first,second)
    assert first.stat().st_nlink>=3
    _,changed,hit=pool.get('BGA','run-001',{'source':'b'},build)
    assert not hit and changed!=entry
    assert first.read_bytes()==b'original'
    forced=RunPool(pool.root,refresh=True)
    def rebuild(folder):(folder/'plot.png').write_bytes(b'rebuilt');return {}
    forced.get('BGA','run-001',{'source':'a'},rebuild)
    assert first.read_bytes()==b'original'
    assert (entry/'figures/plot.png').read_bytes()==b'rebuilt'


def test_pool_rebuilds_corruption_and_source_edits_change_key(tmp_path):
    source=tmp_path/'input.csv';source.write_text('first')
    pool=RunPool(tmp_path/'pool');calls=[]
    def build(folder):calls.append(1);(folder/'result.csv').write_text('verified');return {}
    key=file_inputs({'input':source});_,entry,_=pool.get('data','run',key,build)
    (entry/'data/result.csv').write_text('corrupted')
    _,_,hit=pool.get('data','run',key,build)
    assert not hit and len(calls)==2
    source.write_text('second')
    _,new,hit=pool.get('data','run',file_inputs({'input':source}),build)
    assert not hit and new!=entry


def test_behavior_pool_reuses_data_in_summary_only_mode(tmp_path,monkeypatch):
    from matplotlib.figure import Figure
    from data_analysis_gui import report_behavior as module
    from data_analysis_gui.core import behavior
    monkeypatch.setattr(Figure,'savefig',lambda *a,**k:None)
    run=tmp_path/'20261120/run-001-Grid-offline';run.mkdir(parents=True)
    (run/'task_config.json').write_text(json.dumps({'GRID_TASK':{'GRID_ROWS':8,'GRID_COLS':8}}))
    start=datetime(2026,11,20)
    with (run/'trials.csv').open('w',newline='') as f:
        fields=['task_mode','session_timestamp_iso','trial_num_overall','block_num','outcome','trial_start_iso','acquisition_iso','trial_duration_sec']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i in range(7):
            t=start+timedelta(seconds=10*i)
            w.writerow(dict(task_mode='grid',session_timestamp_iso=str(start),trial_num_overall=i+1,block_num=1,outcome='acquired',trial_start_iso=t.isoformat(),acquisition_iso=(t+timedelta(seconds=10)).isoformat(),trial_duration_sec=10))
    for i in range(2):
        if i:monkeypatch.setattr(behavior,'_from_csv',lambda *a,**k:(_ for _ in ()).throw(AssertionError('must reuse cached run data')))
        work=tmp_path/f'report{i}';work.mkdir();pool=RunPool(tmp_path/'pool')
        result=module.render_offline_behavior(work,[str(run)],100,lambda *a,**k:None,include_runs=False,pool=pool)
        assert result['runs'][0]['rt_median_s']==10
        assert pool.events[0]['cached']==bool(i)
        assert not list(work.rglob('runs'))


def test_bga_pool_skips_both_xdf_loading_and_drawing_on_second_report(tmp_path,monkeypatch):
    from matplotlib.figure import Figure
    from data_analysis_gui import report_bga as module
    run=tmp_path/'20261120/run-001-Grid-offline';run.mkdir(parents=True)
    xdf=run/'first.xdf';xdf.write_bytes(b'test')
    mapping=tmp_path/'layout.xlsx';mapping.write_bytes(b'layout')
    layout=dict(source=str(mapping),bga_position=np.column_stack([np.arange(1024)%30+1,np.arange(1024)//30+1]),position=np.column_stack([np.arange(1024)//8,np.arange(1024)%8]))
    monkeypatch.setattr(module,'read_layout',lambda *a:layout)
    stream={'info':{'desc':[{'channel_ids':['1,4,8']}]},'time_series':np.array([[0,1,3]]),'time_stamps':[1.]}
    monkeypatch.setattr(module,'load_streams',lambda p:({'spike_counts':stream},[]))
    report=dict(recordings=[{'path':str(xdf)}]);outputs=[]
    for i in range(2):
        if i:
            monkeypatch.setattr(module,'load_streams',lambda *a:(_ for _ in ()).throw(AssertionError('must not reread XDF')))
            monkeypatch.setattr(Figure,'savefig',lambda *a,**k:(_ for _ in ()).throw(AssertionError('must not redraw')))
        work=tmp_path/f'report{i}';work.mkdir()
        result=module.render_daily_bga(work,report,pool=RunPool(tmp_path/'pool'))
        assert result['days'][0]['transmitted_channels']==3
        outputs.append(work/result['days'][0]['files']['png'])
    assert os.path.samefile(*outputs)


def test_spatial_pool_reuses_run_plots_and_invalidates_only_changed_run(tmp_path,monkeypatch):
    from matplotlib.figure import Figure
    from data_analysis_gui import report_spatial as module
    def write_figure(fig,path,**kwargs):Path(path).write_bytes(b'figure')
    monkeypatch.setattr(Figure,'savefig',write_figure)
    monkeypatch.setattr(module,'sites_figure',lambda *a,**k:Figure())
    monkeypatch.setattr(module,'profile_figure',lambda *a,**k:Figure())
    entries=[]
    for sid in ('s05','s14'):
        folder=tmp_path/sid;folder.mkdir()
        name='anova_channels' if sid=='s05' else 'channel_weights'
        text=('electrode,active,tested,significant\n1,True,True,True\n' if sid=='s05' else
              'channel_0based,transmitted,vx,vy,norm,event_hz\n0,True,1,2,3,4\n')
        (folder/'table.csv').write_text(text)
        entries.append(dict(job={'section':sid,'recordings':[sid+'.xdf']},folder=str(folder),record={'summary':{'recording':sid,'decoder':'ridge','weight_basis':'per_SD'},'tables':{name:'table.csv'}}))
    for i in range(3):
        if i==2:
            source=tmp_path/'s05/table.csv';source.write_text(source.read_text().replace('True,True,True','True,True,False'))
        work=tmp_path/f'report{i}';work.mkdir();pool=RunPool(tmp_path/'pool')
        module.render_report_spatial(work,entries,dict(recordings=[{},{}],include_run_exports=True),100,lambda *a,**k:None,pool=pool)
        assert [e['cached'] for e in pool.events]==([False,False] if i==0 else [True,True] if i==1 else [False,True])
