"""Behavior-only BPS/TTT from CSVs for new joystick sessions; no neural data required."""
from pathlib import Path
import sys,json,csv,math
from collections import defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_analysis_gui import config as C,storage
from data_analysis_gui.core import behavior
from data_analysis_gui.core.style import finish_figure
from data_analysis_gui.sections.s13_online_behavior_v1 import rolling_bps,co8_bps
from data_analysis_gui.sections.s13_online_behavior_v1.section import behavior_figure
from data_analysis_gui.sections.base import new_figure
ROOT=C.DEFAULT_DATA_ROOT;BASE=Path('E:/AitBCI/data_analysis_gui_output/keaiduo');AUDIT=BASE/'import_20261116_1118'
def write(p,rows):
 if not rows:return
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def save(fig,p):
 if '--resume' in sys.argv and p.with_suffix('.png').exists() and p.with_suffix('.pdf').exists():fig.clear();return
 finish_figure(fig,120);fig.savefig(p.with_suffix('.png'),dpi=250,bbox_inches='tight',facecolor='white');fig.savefig(p.with_suffix('.pdf'),bbox_inches='tight',facecolor='white');fig.clear()
def main():
 parent=BASE/'behavior';parent.mkdir(exist_ok=True)
 version=1
 while (parent/f'20261116_20261118_18runs_v{version}').exists():version+=1
 out=sorted(parent.glob('20261116_20261118_18runs_v*'))[-1] if '--resume' in sys.argv else parent/f'20261116_20261118_18runs_v{version}';out.mkdir(exist_ok=True)
 plans=json.loads((AUDIT/'normalization_plan.json').read_text(encoding='utf-8'));summary=[];by_mode=defaultdict(list);windows_by_mode=defaultdict(list);pool={};entries=defaultdict(list);sources=[]
 for p in plans:
  folder=Path(p['new']);mode=p['mode'];name=folder.name;date=p['date'];target=out/date/name;target.mkdir(parents=True,exist_ok=True)
  cfg=p['config'];g=cfg.get('GRID_TASK',{});condition=f'{g.get("GRID_ROWS","")}x{g.get("GRID_COLS","")} nearby={g.get("NEARBY_TARGET_ASSIST_ENABLED")} dwell={cfg.get("GENERAL_TIMING",{}).get("DWELL_TIME_SEC")}' if mode=='grid' else 'CO8 joystick'
  label=f'{date[4:6]}-{date[6:]} run-{p["run"]} | {condition}'
  if p['trials']==0:
   (target/'NO_TRIALS.txt').write_text('Startup session contains zero trial rows; BPS/TTT are unavailable. No XDF was used.\n',encoding='utf-8');summary.append(dict(date=date,run=p['run'],folder=str(folder),task=mode,trials=0,status='no trials'));continue
  files=dict(trials=folder/'trials.csv',log_0=folder/'logs/task.log')
  if mode=='centerout':files['blocks']=folder/'block_metrics.csv'
  trials=behavior._from_csv(files,mode)
  assert len(trials)==p['trials']
  if mode=='grid':assert all(np.isfinite(r['bits']) and abs(r['bits']-math.log2(g['GRID_ROWS']*g['GRID_COLS']))<1e-12 for r in trials)
  metrics=behavior.metrics(trials,mode);key=str(folder)
  windows=rolling_bps.windows(trials,mode,session=label,date=date,run=key);daily=rolling_bps.daily_summary(windows,mode,dates=[date])[0]
  grouped=defaultdict(list)
  for trial in trials:grouped[trial['block']].append(trial)
  blocks=[dict(block=b,**{k:v for k,v in behavior.metrics(ts,mode).items() if k!='bps'}) for b,ts in sorted(grouped.items())]
  save(behavior_figure(blocks,mode,label+' | Joystick behavior'),target/'Behavior_by_block')
  save(rolling_bps.rolling_figure(windows,mode,label+' | Joystick | 60-active-second BPS'),target/'Rolling_BPS_60s')
  fig=new_figure(9,5);ax=fig.subplots();rt=[r['rt_s'] for r in trials if r['outcome']=='acquired' and np.isfinite(r['rt_s'])]
  if rt:ax.hist(rt,bins=min(40,max(5,int(np.sqrt(len(rt))))),color='#c25e00');ax.axvline(np.median(rt),color='#333',ls='--',label=f'median {np.median(rt):.3f}s');ax.legend()
  ax.set(xlabel='Time to target (s; includes dwell)',ylabel='Successful trials',title=label+'\nJoystick TTT distribution');save(fig,target/'TTT_distribution')
  # BPS: 60s windows, 1s step; entire short runs use their actual duration.
  result=dict(date=date,run=p['run'],folder=str(folder),task=mode,trials=len(trials),status='ok',condition=condition,**{k:v for k,v in metrics.items() if k!='bps'},**{k:v for k,v in daily.items() if k!='date'},rolling_windows=len(windows))
  summary.append(result);write(target/'trials_analyzed.csv',trials);write(target/'block_metrics.csv',blocks);write(target/'rolling_bps_windows.csv',windows)
  (target/'summary.json').write_text(json.dumps(storage.jsonable(result),ensure_ascii=False,indent=2),encoding='utf-8')
  by_mode[mode].append(result);windows_by_mode[mode].extend(windows);pool[key]=trials;entries[mode].append(dict(run_key=key,date=date,session=f'{date[4:]} r{p["run"]}'))
  sources.append(dict(run=name,date=date,files={k:dict(path=str(v),sha256=storage.file_digest(v)) for k,v in files.items()}))
  print(date,name,'trials',len(trials),'BPS windows',len(windows),flush=True)
 for mode in ['grid','centerout']:
  if not entries[mode]:continue
  rows=rolling_bps.run_summary(windows_by_mode[mode],mode,runs=entries[mode],trials_by_run=pool);write(out/f'{mode}_run_BPS_TTT_summary.csv',rows)
  if mode=='grid':
   title,fig=rolling_bps.grid_run_figure(rows);fig._suptitle.set_text('Joystick Grid | Run BPS and TTT');save(fig,out/'Grid_run_BPS_TTT_summary')
  else:
   title,fig=rolling_bps.run_figure(rows,mode);fig._suptitle.set_text('Joystick CO8 | Run rolling BPS');save(fig,out/'CO8_run_BPS_summary')
 write(out/'all_run_summary.csv',summary)
 (out/'manifest.json').write_text(json.dumps(storage.jsonable(dict(source='behavior CSVs and logs; no XDF required or loaded',sessions=18,analyzed=17,empty=1,conditions='Joystick control; original nearby-target assistance and dwell recorded separately',method=rolling_bps.METHOD,sources=sources,runs=summary)),ensure_ascii=False,indent=2),encoding='utf-8')
 (AUDIT/'behavior_output.json').write_text(json.dumps(dict(folder=str(out)),indent=2),encoding='utf-8');print('DONE',out,flush=True)
if __name__=='__main__':main()

