import csv,json
from datetime import datetime,timedelta
import pytest
from matplotlib.figure import Figure
from data_analysis_gui.report_behavior import render_offline_behavior,CATEGORIES


@pytest.mark.parametrize('online',[False,True])
@pytest.mark.parametrize('include_runs',[False,True])
def test_csv_only_grid_report_and_empty_session(tmp_path,online,include_runs):
    day=tmp_path/'20261119';run=day/'run-001-Grid-offline';run.mkdir(parents=True)
    (run/'task_config.json').write_text(json.dumps({'GRID_TASK':{'GRID_ROWS':8,'GRID_COLS':8}}))
    fields=['task_mode','session_timestamp_iso','trial_num_overall','block_num','outcome','trial_start_iso','acquisition_iso','trial_duration_sec']
    start=datetime(2026,11,19,10)
    with (run/'trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i in range(7):
            t=start+timedelta(seconds=i*10)
            w.writerow(dict(task_mode='grid',session_timestamp_iso=str(start),trial_num_overall=i+1,block_num=1,outcome='acquired',trial_start_iso=t.isoformat(),acquisition_iso=(t+timedelta(seconds=10)).isoformat(),trial_duration_sec=10))
    empty=day/'run-002-Grid-offline';empty.mkdir();(empty/'trials.csv').write_text(','.join(fields)+'\n')
    out=tmp_path/'report';out.mkdir()
    result=render_offline_behavior(out,[str(run),str(empty)],100,lambda *a,**k:None,online=online,include_runs=include_runs)
    assert len(result['pngs'])==(4 if include_runs else 1)
    assert result['runs'][1]['status']=='empty session'
    category='06_在线脑控行为_Grid/CSV行为分析' if online else CATEGORIES['grid']
    path=out/category/'summary_all_runs'/'run_summary.csv'
    with path.open(encoding='utf-8-sig') as f:row=next(csv.DictReader(f))
    assert float(row['bps_median'])==pytest.approx(.6)
    assert float(row['rt_median_s'])==10
    assert all((out/name).is_file() for name in result['pngs'])
    manifest='online_behavior_manifest.json' if online else 'offline_behavior_manifest.json'
    assert (out/manifest).exists()
    assert bool(list(out.rglob('runs')))==include_runs


def test_offline_summary_keeps_both_1116_scopes(tmp_path,monkeypatch):
    monkeypatch.setattr(Figure,'savefig',lambda *a,**k:None)
    folders=[]
    for day in ('20261116','20261120'):
        run=tmp_path/day/'run-001-Grid-offline';run.mkdir(parents=True);folders.append(str(run))
        (run/'task_config.json').write_text(json.dumps({'GRID_TASK':{'GRID_ROWS':8,'GRID_COLS':8}}))
        start=datetime.strptime(day,'%Y%m%d')
        with (run/'trials.csv').open('w',newline='') as stream:
            fields=['task_mode','session_timestamp_iso','trial_num_overall','block_num','outcome','trial_start_iso','acquisition_iso','trial_duration_sec']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for i in range(7):
                t=start+timedelta(seconds=i*10)
                writer.writerow(dict(task_mode='grid',session_timestamp_iso=str(start),trial_num_overall=i+1,block_num=1,outcome='acquired',trial_start_iso=t.isoformat(),acquisition_iso=(t+timedelta(seconds=10)).isoformat(),trial_duration_sec=10))
    out=tmp_path/'report';out.mkdir()
    render_offline_behavior(out,folders,100,lambda *a,**k:None)
    summary=out/CATEGORIES['grid']/'summary_all_runs'
    def dates(name):
        with (summary/name).open(encoding='utf-8-sig') as stream:return {r['date'] for r in csv.DictReader(stream)}
    assert dates('run_summary.csv')=={'20261116','20261120'}
    assert dates('Run_BPS_TTT_summary_exclude1116.csv')=={'20261120'}
