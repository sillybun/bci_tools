import csv,json
from datetime import datetime,timedelta
import pytest
from data_analysis_gui.control_comparison import discover,collect,source_signature
from data_analysis_gui.report_run_pool import RunPool

def make_run(root,day,number,bci=False,seconds=10):
    folder=root/day/f'run-{number:03d}-Grid-test';folder.mkdir(parents=True)
    cfg={'GRID_TASK':{'GRID_ROWS':8,'GRID_COLS':8,'PREVIEW_NEXT_TARGET':True},'CONTROL':{'BCI_ENABLED':bci,'JOYSTICK_ENABLED':True},'GENERAL_TIMING':{'DWELL_TIME_SEC':.2}}
    (folder/'task_config.json').write_text(json.dumps(cfg))
    start=datetime.strptime(day,'%Y%m%d')
    fields=['task_mode','session_timestamp_iso','trial_num_overall','block_num','outcome','trial_start_iso','acquisition_iso','trial_duration_sec']
    with (folder/'trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i in range(7):
            t=start+timedelta(seconds=i*seconds)
            w.writerow(dict(task_mode='grid',session_timestamp_iso=str(start),trial_num_overall=i+1,block_num=1,outcome='acquired',trial_start_iso=t.isoformat(),acquisition_iso=(t+timedelta(seconds=seconds)).isoformat(),trial_duration_sec=seconds))
    return folder

def test_discover_csv_only_days_and_exclude_1116(tmp_path):
    a=make_run(tmp_path,'20261110',1);b=make_run(tmp_path,'20261117',1)
    make_run(tmp_path,'20261116',1);c=make_run(tmp_path,'20261123',1)
    assert discover([a/'xdf/test.xdf',c/'xdf/test.xdf'])==[a,b,c]

def test_control_join_short_run_and_source_invalidation(tmp_path):
    a=make_run(tmp_path,'20261123',1,False,5);b=make_run(tmp_path,'20261123',2,True,10)
    pool=RunPool(tmp_path/'pool');data=collect([a,b],pool)
    x,y=data['rows'];assert x['control']=='Joystick' and y['control']=='BCI'
    assert x['windows']==1 and y['windows']==11
    assert x['bps_median']==pytest.approx(1.2) and y['bps_median']==pytest.approx(.6)
    assert x['rt_median_s']==5 and y['rt_median_s']==10
    assert x['preview_next_target'] is True
    assert len({r['config_key'] for r in data['rows']})==2
    old=source_signature([a,b]);cfg=json.loads((a/'task_config.json').read_text());cfg['GRID_TASK']['PREVIEW_NEXT_TARGET']=False
    (a/'task_config.json').write_text(json.dumps(cfg))
    assert old!=source_signature([a,b])
    data=collect([a,b],pool);assert data['rows'][0]['preview_next_target'] is False

def test_binary_workbook_attachment_survives_export(tmp_path):
    from data_analysis_gui import storage
    from data_analysis_gui.sections.base import SectionResult
    from data_analysis_gui.sections.x10_control_comparison_v1.section import SECTION
    result=SectionResult(attachments={'stats.xlsx':b'PK\x03\x04\x00binary\xff','note.txt':'中文'})
    out=tmp_path/'result'
    saved=storage.save_result(result,out,'test',dict(recordings=[],criteria={},params={},code='test'),SECTION,[])
    assert (out/saved['attachments']['stats.xlsx']).read_bytes()==b'PK\x03\x04\x00binary\xff'
    assert (out/saved['attachments']['note.txt']).read_text(encoding='utf-8')=='中文'
