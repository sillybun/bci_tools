"""Normalize new sessions safely and prepare a 1110--1123 report excluding 1116."""
from pathlib import Path
from dataclasses import replace
import hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_analysis_gui import config as C
from data_analysis_gui.core.catalog import format_run_name,scan
from data_analysis_gui.reporting import report_plan

ROOT=C.DEFAULT_DATA_ROOT.resolve()
AUDIT=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/import_20261123')
AUDIT.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def normalize():
    if (AUDIT/'normalization_complete.json').exists():return
    plans=[];unchanged=[]
    for date in ('20261120','20261123'):
        day=(ROOT/date).resolve()
        folders=sorted((p for p in day.iterdir() if (p/'session.json').is_file()),key=lambda p:read(p/'session.json')['started_at'])
        for i,folder in enumerate(folders,1):
            session=read(folder/'session.json');config=read(folder/'task_config.json') if (folder/'task_config.json').is_file() else {}
            grid=config.get('GRID_TASK',{});online=bool(config.get('CONTROL',{}).get('BCI_ENABLED'))
            notes=[f'{grid["GRID_ROWS"]}x{grid["GRID_COLS"]}'] if grid else ['startup']
            if grid.get('NEARBY_TARGET_ASSIST_ENABLED'):notes.append('nearby-assist')
            if config:notes.append(str(config['GENERAL_TIMING']['DWELL_TIME_SEC']).replace('.','p')+'dwell')
            target=day/format_run_name(i,'Grid','online' if online else 'offline',notes)
            assert folder.resolve().parent==day and target.resolve().parent==day
            assert target==folder or not target.exists(),f'Target already exists: {target}'
            xdfs=sorted(folder.rglob('*.xdf'));moves=[]
            for j,p in enumerate(xdfs,1):
                expected=f'{target.name}-{date}-rec-{j:03d}-main-spike.xdf'
                moves.append(dict(old=p.relative_to(folder).as_posix(),new=p.with_name(expected).relative_to(folder).as_posix()))
            if target==folder and all(m['old']==m['new'] for m in moves):
                unchanged.append(dict(date=date,folder=str(folder),run=f'{i:03d}',xdf_count=len(xdfs)))
                continue
            files=[dict(path=p.relative_to(folder).as_posix(),sha256=sha(p)) for p in folder.rglob('*') if p.is_file()]
            plans.append(dict(date=date,old=str(folder),new=str(target),run=f'{i:03d}',online=online,
                config_available=bool(config),session=session['session_id'],files=files,xdf_moves=moves))
    dump(AUDIT/'normalization_plan.json',dict(rename=plans,already_normalized=unchanged))
    for item in plans:
        source,target=Path(item['old']),Path(item['new']);day=ROOT/item['date']
        assert source.resolve().parent==day.resolve() and target.resolve().parent==day.resolve()
        if source!=target:source.rename(target)
        for f in item['files']:assert sha(target/f['path'])==f['sha256'],target/f['path']
        recordings=[]
        for m in item['xdf_moves']:
            old,new=target/m['old'],target/m['new']
            if old!=new:
                assert not new.exists();old.rename(new)
            recordings.append(dict(path=new.relative_to(ROOT).as_posix(),segment='main',original_xdf=old.name))
        meta=dict(format='keaiduo-session-v1',run_id=item['run'],task='Grid',mode='online' if item['online'] else 'offline',
            condition='脑控' if item['online'] else ('摇杆手控 / 离线' if item['config_available'] else '空启动 / 控制方式未知'),
            condition_tag='bci' if item['online'] else 'joystick',session=item['session'],
            session_path=target.relative_to(ROOT).as_posix(),run_folder=target.name,recordings=recordings,
            analysis_window_policy='through_task_end')
        dump(target/'session_info.json',meta)
    dump(AUDIT/'normalization_complete.json',dict(renamed_sessions=len(plans),already_normalized=len(unchanged),raw_file_bytes_unchanged=True))
    print('NORMALIZED',len(plans),'sessions;',len(unchanged),'already canonical',flush=True)

def request():
    previous=read(AUDIT.parent/'import_20261120/report_request.json')
    params={j['section']:j['params'] for j in previous['jobs']}
    records,notice=scan(ROOT)
    selected=[r for r in records if '2026-11-10'<=r.date<='2026-11-23' and r.date!='2026-11-16']
    # Header-complete aborted XDFs may contain no usable cursor trajectory at all.
    from data_analysis_gui.core.xdf_io import load_streams
    for i,r in enumerate(selected):
        if not r.reason and r.date=='2026-11-23' and r.path.stat().st_size<1_000_000:
            streams,_=load_streams(r.path)
            cursor_samples=len(streams.get('cursortasks_cursor_state',{}).get('time_stamps',[]))
            if cursor_samples<2:
                selected[i]=replace(r,reason=f'中止的极短记录：cursor仅{cursor_samples}帧，无法进行神经任务分析；CSV行为另行保留')
    valid=[r for r in selected if not r.reason]
    offline=[];online=[]
    for day in sorted(ROOT.iterdir()):
        if not day.is_dir() or not day.name.isdigit() or not '20261110'<=day.name<='20261123' or day.name=='20261116':continue
        for p in sorted(day.glob('run-*')):
            if not p.is_dir() or not ((p/'trials.csv').exists() or (p/'behavior/trials.csv').exists()):continue
            cfg_path=p/'task_config.json'
            if not cfg_path.is_file():cfg_path=p/'config/task_config.json'
            cfg=read(cfg_path) if cfg_path.is_file() else {}
            meta=read(p/'session_info.json') if (p/'session_info.json').exists() else {}
            brain=cfg.get('CONTROL',{}).get('BCI_ENABLED',meta.get('mode')=='online' or '-online' in p.name)
            (online if brain else offline).append(str(p))
    jobs,report=report_plan(valid,lambda s:params.get(s.id,{}),behavior_folders=offline,online_behavior_folders=online,
        date_range=('2026-11-10','2026-11-23'),analysis_excluded_dates=['2026-11-16'],include_run_exports=False)
    report['skipped'] += [dict(category='神经分析',path=str(r.path),reason=r.reason) for r in selected if r.reason]
    req=dict(previous,jobs=[dict(section=s.id,version=s.version,params=p,recordings=[str(r.path) for r in rr]) for s,rr,p in jobs],
        report=report,font_percent=100,layout_path=str(C.DEFAULT_LAYOUT_XLSX))
    paths=[r['path'] for r in report['recordings']]+offline+online
    assert not any('20261116' in p or '2026-11-16' in p for p in paths)
    dump(AUDIT/'report_request.json',req)
    dump(AUDIT/'xdf_inventory.json',[dict(path=str(r.path),date=r.date,online=r.online,reason=r.reason) for r in selected])
    print('REPORT',report['name'],'valid XDF',len(valid),'jobs',len(jobs),'offline CSV',len(offline),'online CSV',len(online),'notice',notice,flush=True)
    for r in selected:
        if r.date in ('2026-11-20','2026-11-23'):print(r.date,r.path.name,'online',r.online,'reason',r.reason,flush=True)

if __name__=='__main__':normalize();request()
