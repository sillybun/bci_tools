"""Index existing artifacts without changing reports or recomputing neural models."""
from pathlib import Path
import csv,json,math,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_analysis_gui.report_run_pool import RunPool
from data_analysis_gui import storage
from data_analysis_gui.report_bga import bga_data_inputs
from data_analysis_gui.report_behavior import bundle_files,folder_info,behavior_run_inputs,CATEGORIES
from data_analysis_gui.core import behavior
from data_analysis_gui.sections.s13_online_behavior_v1 import rolling_bps

ROOT=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report')

def typed_rows(path):
    strings={'session','date','run_key','outcome','timing_source','invalid_reason'}
    integers={'block','trial','trial_start','trial_end','window_trials','window_attempts','segment_id','fitts_fully_contained_trials','grid_cells','attempts','successes','timeouts','incorrect','unknown'}
    with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(line for line in f if not line.startswith('#')))
    for r in rows:
        for k,v in r.items():
            if k in strings:continue
            if v in ('True','False'):r[k]=v=='True';continue
            if v=='':r[k]=math.nan;continue
            try:r[k]=int(float(v)) if k in integers and math.isfinite(float(v)) else float(v)
            except (ValueError,TypeError):pass
    return rows

def seed_bga(pool,root,report):
    for d in report.get('bga_daily',{}).get('days',[]):
        if not d.get('run_folder') or d['status'] not in ('ok','no_xdf'):continue
        folder=Path(d['run_folder']);inputs=bga_data_inputs(folder)
        current={v['path']:v['sha256'] for v in inputs['files'].values()}
        recorded={str(Path(v['path']).resolve()):v['sha256'] for v in d['sources']}
        table=root/d['files']['csv']
        if current!=recorded or not table.exists():continue
        def build(p):
            values=[math.nan]*1024;rows=typed_rows(table);assert len(rows)==1024
            for r in rows:
                ch=int(r['channel_0based']);assert 0<=ch<1024
                if r['observed']:values[ch]=float(r['max_code'])
            info={k:d[k] for k in ('status','run_folder','sources','notes')}
            return dict(values=values,info=info)
        pool.get('BGA_data',folder,inputs,build)

def seed_behavior(pool,root,report):
    for key in ('offline_behavior','online_behavior'):
        manifest=report.get(key,{})
        if manifest.get('method')!=rolling_bps.METHOD:continue
        recorded={v['folder']:v['files'] for v in manifest.get('sources',[])}
        for row in manifest.get('runs',[]):
            if row.get('status')!='ok':continue
            folder=Path(row['folder']);mode,date,_=folder_info(folder);files=bundle_files(folder,mode)
            if mode=='grid' and (folder/'task_config.json').exists():files['task_config']=folder/'task_config.json'
            label=f'{date[4:6]}-{date[6:]} {folder.name}';inputs=behavior_run_inputs(files,mode,label)
            old=recorded.get(str(folder),{})
            if {k:v['sha256'] for k,v in old.items()}!={k:v['sha256'] for k,v in inputs['sources'].items()}:continue
            category=(('06_在线脑控行为_Grid/CSV行为分析' if mode=='grid' else '05_在线脑控行为_CO8/CSV行为分析') if key=='online_behavior' else CATEGORIES[mode])
            source=root/category/'runs'/f'{date}_{folder.name}'
            if not all((source/name).exists() for name in ('trials.csv','block_metrics.csv','rolling_bps_windows.csv')):continue
            def build(p):
                trials=typed_rows(source/'trials.csv');windows=typed_rows(source/'rolling_bps_windows.csv');blocks=typed_rows(source/'block_metrics.csv')
                metric=behavior.metrics(trials,mode)
                assert len(trials)==row['trials'] and len(windows)==row['rolling_windows']
                assert math.isclose(metric['rt_median_s'],row['rt_median_s'],abs_tol=1e-8) or (math.isnan(metric['rt_median_s']) and row['rt_median_s'] is None)
                return dict(trials=trials,windows=windows,blocks=blocks,metric=metric,daily=rolling_bps.daily_summary(windows,mode,dates=[date])[0])
            pool.get('BPS_TTT_data',str(folder),inputs,build)

def main():
    pool=RunPool(ROOT/'individual_runs');seen=set();reports=[]
    paths=sorted(ROOT.glob('*/report.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:2]
    for path in paths:
        report=json.loads(path.read_text(encoding='utf-8'));reports.append(str(path.parent))
        for s in report.get('sources',[]):
            if s['section'] not in ('s05','s14','s10','s13'):continue
            source=Path(s['folder']);rp=source/'result.json'
            if not rp.exists():continue
            record=json.loads(rp.read_text(encoding='utf-8'));token=(s['section'],record['signature'])
            if token in seen:continue
            seen.add(token);inputs=record['inputs'];selected=[]
            for item in inputs['recordings']:
                matches=[r['path'] for r in report['selection'] if Path(r['path']).name==item['name']]
                matches=[p for p in matches if storage.recording_sha(p)==item['sha256']]
                if len(matches)==1:selected.append(matches[0])
            if len(selected)!=1:continue
            pool.section(dict(folder=str(source),record=record,job=dict(section=s['section'],recordings=selected)))
        seed_bga(pool,path.parent,report);seed_behavior(pool,path.parent,report)
        print('Indexed',path.parent.name,flush=True)
    audit=dict(reports=reports,**pool.manifest())
    (pool.root/'_meta/seed_history.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    from collections import Counter
    print(dict(Counter(e['kind'] for e in pool.events)),flush=True)

if __name__=='__main__':main()
