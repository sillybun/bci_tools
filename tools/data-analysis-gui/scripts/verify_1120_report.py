"""Review report coverage and independently check a new day's behavior metrics."""
from pathlib import Path
from datetime import datetime
import csv,json,math,shutil
import numpy as np
from PIL import Image,ImageOps,ImageDraw

A=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/import_20261120')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    roots=sorted((A.parent/'report').glob('2026-11-11_2026-11-20_52runs_v*'))
    assert roots,'Report not finished'
    root=roots[-1];report=read(root/'report.json');assert report['complete'] and not report['failures']
    assert report['run_count']==52 and len(report['selection'])==26
    assert not any(r['date']=='2026-11-16' for r in report['selection'])
    offline=read(root/'offline_behavior_manifest.json');online=read(root/'online_behavior_manifest.json')
    assert len(offline['runs'])==37 and len(online['runs'])==14
    assert any('20261120' in r['folder'] and 'run-015-' in r['folder'] for r in online['runs'])
    assert any('20261120' in r['folder'] and 'run-013-' in r['folder'] for r in offline['runs'])
    count=0;partial=0
    for path in root.rglob('rolling_bps_windows.csv'):
        data=rows(path)
        for r in data:
            assert float(r['window_seconds'])==60 and float(r['step_seconds'])==1
            t=float(r['effective_window_seconds']);assert 0<t<=60+1e-8
            assert math.isclose(float(r['window_ratio']),t/60,abs_tol=1e-8)
            if t<60-1e-8:assert len(data)==1;partial+=1
            else:assert float(r['window_end_active_s'])>=60-1e-8
            count+=1
        for a,b in zip(data,data[1:]):
            if a['segment_id']==b['segment_id']:assert math.isclose(float(b['window_end_active_s'])-float(a['window_end_active_s']),1,abs_tol=1e-7)
    for p in root.rglob('Run_BPS_TTT_summary_exclude1116.csv'):
        assert all(r['date']!='20261116' for r in rows(p))
    scope_files=list(root.rglob('Run_BPS_TTT_summary_exclude1116.csv'));assert len(scope_files)==2
    # Raw timestamp arithmetic for an online run that has no XDF.
    run=next(Path(r['folder']) for r in online['runs'] if 'run-015-' in r['folder'])
    raw=rows(run/'trials.csv');valid=[r for r in raw if r['outcome']=='acquired']
    rt=[(datetime.fromisoformat(r['acquisition_iso'])-datetime.fromisoformat(r['trial_start_iso'])).total_seconds() for r in valid]
    result=next(r for r in online['runs'] if r['folder']==str(run));assert math.isclose(float(result['rt_median_s']),float(np.median(rt)),abs_tol=1e-6)
    windows_path=next(p for p in root.rglob('rolling_bps_windows.csv') if '20261120_run-015-' in str(p))
    win=rows(windows_path);sample=next(w for w in win if float(w['bps'])>0)
    left=float(sample['window_start_active_s']);right=float(sample['window_end_active_s']);elapsed=0.;correct=0
    for r in raw:
        duration=(datetime.fromisoformat(r['acquisition_iso'])-datetime.fromisoformat(r['trial_start_iso'])).total_seconds() if r['outcome']=='acquired' else float(r['trial_duration_sec'])
        elapsed+=duration
        if left<elapsed<=right+1e-8:correct+=r['outcome']=='acquired'
        if elapsed>right:break
    assert correct>0
    assert math.isclose(float(sample['bps']),correct*math.log2(49)/60,abs_tol=1e-7)
    target=root/'00_Config_1117_1120';target.mkdir(exist_ok=True)
    for name in ('config_1117_1120.html','config_1117_1120.json'):shutil.copy2(A/name,target/name)
    report['config_comparison']={'table':str((target/'config_1117_1120.html').relative_to(root)),'details':str((target/'config_1117_1120.json').relative_to(root))}
    (root/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    samples=[]
    for pattern in ('07_*/summary_all_runs/S05_mean_depth_7shank.png','08_*/summary_all_runs/S14_norm_median_depth_7shank.png','09_*/summary_all_runs/Run_BPS_TTT_summary_exclude1116.png','06_*/CSV*/summary_all_runs/Run_BPS_TTT_summary.png'):
        samples.extend(root.glob(pattern))
    canvas=Image.new('RGB',(500*len(samples),760),'white')
    for i,p in enumerate(samples):
        with Image.open(p) as im:canvas.paste(ImageOps.contain(im.convert('RGB'),(495,730)),(500*i,10))
    canvas.save(A/'report_final_preview.jpg')
    audit=dict(report=str(root),pngs=len(list(root.rglob('*.png'))),pdfs=len(list(root.rglob('*.pdf'))),
        neural_recordings=26,archive_runs=52,offline_behavior_runs=len(offline['runs']),online_behavior_runs=len(online['runs']),
        analyzed_offline_runs=sum(r.get('status')=='ok' for r in offline['runs']),bps_windows_checked=count,short_run_windows=partial,
        excluded_1116_summary_tables=len(scope_files),independent_check_run=str(run),sample_bps=float(sample['bps']),sample_window_end_active_s=right,sample_ttt=float(np.median(rt)))
    (A/'report_verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(audit,ensure_ascii=False))

if __name__=='__main__':main()
