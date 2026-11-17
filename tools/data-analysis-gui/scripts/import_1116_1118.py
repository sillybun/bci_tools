"""Normalize new session folders and XDF basenames; preserve original evidence bytes."""
from pathlib import Path
import json,csv,hashlib,re,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_analysis_gui.core.catalog import format_run_name
ROOT=Path('E:/AitBCI/data/keaiduo').resolve();AUDIT=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/import_20261116_1118');AUDIT.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main(days=('20261116','20261117','20261118')):
 if (AUDIT/'normalization_complete.json').exists():print('Already normalized');return
 plans=[]
 for day in days:
  sessions=sorted([p for p in (ROOT/day).iterdir() if p.is_dir() and (p/'session.json').exists()],key=lambda p:json.loads((p/'session.json').read_text(encoding='utf-8-sig'))['started_at'])
  for i,folder in enumerate(sessions,1):
   m=json.loads((folder/'session.json').read_text(encoding='utf-8-sig'));cfg=json.loads((folder/'task_config.json').read_text(encoding='utf-8-sig')) if (folder/'task_config.json').exists() else {}
   assert cfg.get('CONTROL',{}).get('BCI_ENABLED',False)==False
   mode=m['task_mode'];notes=[];g=cfg.get('GRID_TASK',{})
   if mode=='grid':notes=[f'{g["GRID_ROWS"]}x{g["GRID_COLS"]}']+(['nearby-assist'] if g.get('NEARBY_TARGET_ASSIST_ENABLED') else [])
   if cfg:notes.append(str(cfg['GENERAL_TIMING']['DWELL_TIME_SEC']).replace('.','p')+'dwell')
   else:notes.append('startup')
   name=format_run_name(i,'CO8' if mode=='centerout' else 'Grid','offline',notes);target=folder.with_name(name)
   assert folder.resolve().parent==ROOT/day and target.resolve().parent==ROOT/day and not target.exists()
   files=[]
   for f in folder.rglob('*'):
    if not f.is_file():continue
    rel=f.relative_to(folder);newrel=rel
    if f.suffix.lower()=='.xdf':newrel=rel.with_name(f'{name}-{day}-rec-001-main-spike.xdf')
    files.append(dict(old=str(rel),new=str(newrel),sha256=sha(f)))
   with (folder/'trials.csv').open(encoding='utf-8-sig') as f:trials=list(csv.DictReader(f))
   assert len({r['trial_num_overall'] for r in trials})==len(trials)
   assert len({r['session_timestamp_iso'] for r in trials})<=1
   plans.append(dict(old=str(folder),new=str(target),run=f'{i:03d}',date=day,original_session=m['session_id'],mode=mode,started_at=m['started_at'],config=cfg,files=files,trials=len(trials)))
 (AUDIT/'normalization_plan.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2),encoding='utf-8')
 for p in plans:
  old=Path(p['old']);target=Path(p['new']);old.rename(target)
  for f in p['files']:
   a=target/f['old'];b=target/f['new']
   if a!=b:assert not b.exists();a.rename(b)
   assert sha(b)==f['sha256']
  cfg=p['config'];g=cfg.get('GRID_TASK',{});condition='摇杆手控 / 离线'
  if p['mode']=='grid':condition+=f' / {g["GRID_ROWS"]}x{g["GRID_COLS"]} / nearby assist '+('on' if g.get('NEARBY_TARGET_ASSIST_ENABLED') else 'off')
  if cfg:condition+=f' / dwell {cfg["GENERAL_TIMING"]["DWELL_TIME_SEC"]} s'
  meta=dict(format='keaiduo-session-v1',run_id=p['run'],task='CO8' if p['mode']=='centerout' else 'Grid',mode='offline',condition=condition,condition_tag='joystick',session=p['original_session'],session_path=target.relative_to(ROOT).as_posix(),run_folder=target.name,recordings=[dict(path=(target/f['new']).relative_to(ROOT).as_posix(),segment='main',original_xdf=Path(f['old']).name) for f in p['files'] if f['old'].endswith('.xdf')],analysis_window_policy='through_task_end')
  (target/'session_info.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
  print(p['date'],p['run'],target.name,'trials',p['trials'],flush=True)
 (AUDIT/'normalization_complete.json').write_text(json.dumps(dict(sessions=len(plans),xdf=sum(f['new'].endswith('.xdf') for p in plans for f in p['files']),evidence_bytes_unchanged=True,plans=str(AUDIT/'normalization_plan.json')),indent=2),encoding='utf-8')
if __name__=='__main__':main()
