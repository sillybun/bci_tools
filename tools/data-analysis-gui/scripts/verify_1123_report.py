"""Validate the requested date coverage, cached outputs, and fresh CSV behavior."""
from pathlib import Path
import csv,json,math,sys
from datetime import datetime
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_analysis_gui.sections.s13_online_behavior_v1 import rolling_bps

AUDIT=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/import_20261123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    request=read(AUDIT/'report_request.json');spec=request['report']
    root=sorted(Path(spec['root']).glob(spec['name']+'_v*'),key=lambda p:int(p.name.rsplit('_v',1)[1]))[-1]
    report=read(root/'report.json')
    assert report['complete'] and not report['failures'],report['failures']
    assert report['run_count']==spec['run_count']==60
    assert report['recording_count']==39
    assert not report['include_run_exports']
    assert report['analysis_excluded_dates']==['2026-11-16']
    assert report['spatial']['electrode_position_range']==[0,127]
    assert Path(report['spatial']['layout_path']).name=='天枢3.0_ChannelMap20261116.xlsx'
    assert set(r['path'] for r in report['selection'])==set(r['path'] for r in spec['recordings'])
    assert any(r['date']=='2026-11-10' for r in report['selection'])
    assert any(r['date']=='2026-11-23' for r in report['selection'])
    assert not any(r['date']=='2026-11-16' for r in report['selection'])
    behaviors=[read(root/name) for name in ('offline_behavior_manifest.json','online_behavior_manifest.json')]
    assert len(behaviors[0]['runs'])==38 and len(behaviors[1]['runs'])==21
    windows_checked=0;ttt_checked=0;new_runs=[]
    for b in behaviors:
        assert b['window_seconds']==60 and b['step_seconds']==1 and b['method']==rolling_bps.METHOD
        for r in b['runs']:
            assert '20261116' not in r['folder']
            if '20261123' not in r['folder']:continue
            new_runs.append(dict(folder=r['folder'],status=r['status'],trials=r.get('trials')))
            if r['status']!='ok':continue
            folder=Path(r['folder']);trial=rows(folder/'trials.csv')
            rawrt=[(datetime.fromisoformat(t['acquisition_iso'])-datetime.fromisoformat(t['trial_start_iso'])).total_seconds()
                   for t in trial if t['outcome']=='acquired' and t.get('acquisition_iso') and t.get('trial_start_iso')]
            if rawrt:
                assert math.isclose(r['rt_median_s'],float(np.median(rawrt)),abs_tol=1e-6),(folder,r['rt_median_s'],np.median(rawrt))
                ttt_checked+=1
    # Verify every pooled 1123 behavior window without requiring per-run report exports.
    pool=Path(spec['root'])/'individual_runs/BPS_TTT'
    for p in pool.glob('2026-11-23*/statistics/*/data/calculation.json'):
        data=read(p);windows=data['windows']
        for w in windows:
            assert w['window_seconds']==60 and w['step_seconds']==1
            seconds=w['effective_window_seconds']
            assert 0<seconds<=60+1e-7
            assert math.isclose(w['window_ratio'],seconds/60,abs_tol=1e-9)
            if w.get('grid_cells') and math.isfinite(w['bps']):
                assert math.isclose(w['bps'],w['successes']*math.log2(w['grid_cells'])/seconds,abs_tol=1e-9)
            if seconds<60-1e-7:assert len(windows)==1
            else:assert w['window_end_active_s']>=60-1e-7
            windows_checked+=1
        for a,b in zip(windows,windows[1:]):
            if a['segment_id']==b['segment_id']:assert math.isclose(b['window_end_active_s']-a['window_end_active_s'],1,abs_tol=1e-7)
    csv_count=0
    for p in root.rglob('*.csv'):
        if 'summary_all_runs' not in p.parts:continue
        for r in rows(p):
            if 'date' in r:assert r['date'] not in ('20261116','2026-11-16'),p
        csv_count+=1
    result=dict(report=str(root),complete=True,neural_recordings=39,archive_runs=60,
        offline_csv_runs=38,online_csv_runs=21,new_runs=new_runs,independent_ttt_runs=ttt_checked,
        pooled_new_windows_checked=windows_checked,summary_tables_checked=csv_count,
        pngs=len(list(root.rglob('*.png'))),excluded_dates=report['analysis_excluded_dates'],cache=report['run_pool'])
    (AUDIT/'report_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='cache'},ensure_ascii=False))
if __name__=='__main__':main()
