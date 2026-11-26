"""CSV-backed offline behavior included in one-click reports, independent of XDF availability."""
from pathlib import Path
from collections import defaultdict
import csv,json
import numpy as np
from . import storage
from .core import behavior
from .core.archive_logs import archived_logs
from .core.catalog import RUN_PREFIX
from .core.run_config import configured_run_name
from .core.style import finish_figure
from .sections.base import new_figure
from .sections.s13_online_behavior_v1.section import behavior_figure
from .sections.s13_online_behavior_v1 import rolling_bps
CATEGORIES={'grid':'09_离线行为_Grid_BPS_TTT','centerout':'10_离线行为_CO8_BPS_TTT'}

def bundle_files(folder,mode):
    folder=Path(folder);trials=folder/'behavior/trials.csv'
    if not trials.is_file():trials=folder/'trials.csv'
    if not trials.is_file():return {}
    files={'trials':trials}
    if mode=='centerout' and trials.with_name('block_metrics.csv').is_file():files['blocks']=trials.with_name('block_metrics.csv')
    logs=archived_logs(folder/'logs','grid-task*.log')
    if not logs:logs=(archived_logs(folder/'logs','task.log') or archived_logs(folder/'logs','cursor_tasks_app.log'))[:1]
    files.update({f'log_{i}':p for i,p in enumerate(logs)})
    return files

def folder_info(folder):
    folder=Path(folder);meta={}
    for file in ('session_info.json','session.json'):
        p=folder/file
        if p.exists():meta.update(json.loads(p.read_text(encoding='utf-8-sig')))
    mode=meta.get('task_mode') or ('grid' if meta.get('task')=='Grid' or 'Grid' in folder.name else 'centerout')
    date=next((p.name for p in folder.parents if len(p.name)==8 and p.name.isdigit()),'')
    return mode,date,meta

def _compute_run_behavior(files,mode,label,date,folder):
    trials=behavior._from_csv(files,mode)
    if mode=='grid' and 'task_config' in files:
        cfg=json.loads(files['task_config'].read_text(encoding='utf-8-sig'));g=cfg.get('GRID_TASK',{});n=int(g.get('GRID_ROWS',0))*int(g.get('GRID_COLS',0))
        if n>1:
            for r in trials:
                if np.isfinite(r['bits']) and not np.isclose(r['bits'],np.log2(n)):raise ValueError(f'Grid dimensions disagree: {folder}')
                r['bits']=float(np.log2(n))
    block=defaultdict(list)
    for r in trials:block[r['block']].append(r)
    blocks=[dict(block=b,**{k:v for k,v in behavior.metrics(t,mode).items() if k!='bps'}) for b,t in sorted(block.items())]
    metric=behavior.metrics(trials,mode);win=rolling_bps.windows(trials,mode,session=label,date=date,run=str(folder));daily=rolling_bps.daily_summary(win,mode,dates=[date])[0]
    return dict(trials=trials,blocks=blocks,metric=metric,windows=win,daily=daily)


def _draw_run_behavior(out,data,mode,label,label_control,font_percent):
    out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        finish_figure(fig,font_percent)
        for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),dpi=220,bbox_inches='tight',facecolor='white')
        fig.clear()
    trials=data['trials']
    save(behavior_figure(data['blocks'],mode,label+f'\n{label_control} behavior'),'Behavior_by_block')
    save(rolling_bps.rolling_figure(data['windows'],mode,label+f'\n{label_control} BPS; 60 active seconds'),'Rolling_BPS_60s')
    fig=new_figure(9,5);ax=fig.subplots();rt=[r['rt_s'] for r in trials if r['outcome']=='acquired' and np.isfinite(r['rt_s'])]
    if rt:ax.hist(rt,bins=min(40,max(5,int(np.sqrt(len(rt))))),color='#c25e00');ax.axvline(np.median(rt),ls='--',color='#333')
    ax.set(xlabel='Time to target (s; includes dwell)',ylabel='Successful trials',title=label+f'\n{label_control} TTT distribution');save(fig,'TTT_distribution')
    for name,rows in [('trials',trials),('block_metrics',data['blocks']),('rolling_bps_windows',data['windows'])]:
        if rows:
            with (out/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    return {}


def behavior_run_inputs(files,mode,label):
    from .report_run_pool import file_inputs,dependencies
    from .sections.s13_online_behavior_v1 import co8_bps
    return dict(sources=file_inputs(files),mode=mode,label=label,
        code=dependencies(_compute_run_behavior,behavior.__file__,rolling_bps.__file__,co8_bps.__file__))


def render_offline_behavior(work,folders,font_percent,send,*,online=False,include_runs=True,pool=None):
    label_control='Online' if online else 'Offline'
    categories=({'grid':'06_在线脑控行为_Grid/CSV行为分析','centerout':'05_在线脑控行为_CO8/CSV行为分析'} if online else CATEGORIES)
    exported=[];summaries=[];sources=[];groups=defaultdict(list);windows=defaultdict(list);trials_pool={};run_entries=defaultdict(list)
    def write(p,rows):
        if not rows:return
        with p.open('w',encoding='utf-8-sig',newline='') as f:
            keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    def save(fig,path):
        finish_figure(fig,font_percent)
        for ext in ('png','pdf'):fig.savefig(path.with_suffix('.'+ext),dpi=220,bbox_inches='tight',facecolor='white')
        fig.clear();exported.append(str(path.with_suffix('.png').relative_to(work)))
    for folder in map(Path,dict.fromkeys(folders)):
        mode,date,meta=folder_info(folder)
        if mode not in categories:continue
        files=bundle_files(folder,mode)
        run_name=folder.name
        if match := RUN_PREFIX.match(folder.name):
            run_name=configured_run_name(folder,match['number'],match['task'],match['mode'],part=match['part'])
        label=f'{date[4:6]}-{date[6:]} {run_name}'
        out=work/categories[mode]/'runs'/f'{date}_{folder.name}'
        if include_runs:out.mkdir(parents=True,exist_ok=True)
        if not files:
            summaries.append(dict(folder=str(folder),date=date,status='no trials.csv'));continue
        with files['trials'].open(encoding='utf-8-sig') as f:raw=list(csv.DictReader(f))
        if not raw:
            if include_runs:(out/'NO_TRIALS.txt').write_text('No trial rows; BPS/TTT unavailable.',encoding='utf-8')
            summaries.append(dict(folder=str(folder),date=date,status='empty session',trials=0));continue
        config=folder/'task_config.json'
        if mode=='grid' and config.is_file():files['task_config']=config
        if pool:
            from .report_run_pool import dependencies
            inputs=behavior_run_inputs(files,mode,label)
            data,cached_data,hit=pool.get('BPS_TTT_data',str(folder),inputs,lambda p:_compute_run_behavior(files,mode,label,date,folder))
            send('message',message=f'BPS/TTT {"缓存复用" if hit else "首次计算"}: {date} {folder.name}')
        else:data=_compute_run_behavior(files,mode,label,date,folder)
        trials,blocks,metric,win,daily=(data[k] for k in ('trials','blocks','metric','windows','daily'))
        if include_runs:
            if pool:
                _,bundle,_=pool.get('BPS_TTT_figures',str(folder),dict(data=cached_data.name,control=label_control,code=dependencies(_draw_run_behavior,rolling_bps.rolling_figure,behavior_figure,Path(__file__).parent/'core/style.py')),lambda p:_draw_run_behavior(p,data,mode,label,label_control,font_percent))
                published=pool.publish(bundle,out)
                exported.extend(str(p.relative_to(work)) for p in published if p.suffix=='.png')
            else:
                _draw_run_behavior(out,data,mode,label,label_control,font_percent)
                exported.extend(str(p.relative_to(work)) for p in out.glob('*.png'))
        row=dict(folder=str(folder),date=date,task=mode,condition=meta.get('condition',label_control.lower()),trials=len(trials),status='ok',**{k:v for k,v in metric.items() if k!='bps'},**{k:v for k,v in daily.items() if k!='date'},rolling_windows=len(win))
        summaries.append(row);groups[mode].append(row);windows[mode].extend(win);trials_pool[str(folder)]=trials
        run_entries[mode].append(dict(run_key=str(folder),date=date,session=label))
        sources.append(dict(folder=str(folder),files={k:dict(path=str(p),sha256=storage.file_digest(p)) for k,p in files.items()}))
        send('message',message=f'{label_control} behavior: {date} {folder.name}, {len(trials)} trials, {len(win)} BPS windows')
    for mode in groups:
        out=work/categories[mode]/'summary_all_runs';out.mkdir(parents=True,exist_ok=True)
        rows=rolling_bps.run_summary(windows[mode],mode,runs=run_entries[mode],trials_by_run=trials_pool)
        title,fig=rolling_bps.run_figure(rows,mode,online=online);fig._suptitle.set_text(f'{label_control} {"Grid" if mode=="grid" else "CO8"} | Run BPS and TTT');save(fig,out/'Run_BPS_TTT_summary')
        write(out/'run_summary.csv',rows);write(out/'run_metrics.csv',groups[mode])
        ttt_sets=[('',rows)]
        if not online and any(r['date']=='20261116' for r in run_entries[mode]):
            entries=[r for r in run_entries[mode] if r['date']!='20261116']
            subset=[r for r in windows[mode] if r['date']!='20261116']
            filtered=rolling_bps.run_summary(subset,mode,runs=entries,trials_by_run=trials_pool)
            title,fig=rolling_bps.run_figure(filtered,mode,online=online)
            fig._suptitle.set_text(f'Offline {"Grid" if mode=="grid" else "CO8"} | Run BPS and TTT | excluding 1116')
            save(fig,out/'Run_BPS_TTT_summary_exclude1116');write(out/'Run_BPS_TTT_summary_exclude1116.csv',filtered)
            ttt_sets.append(('_exclude1116',filtered))
        if mode=='centerout':
            for suffix,time_rows in ttt_sets:
                fig=new_figure(max(10,len(time_rows)*.6),5);ax=fig.subplots();x=np.arange(len(time_rows));ax.errorbar(x,[r['rt_median_s'] for r in time_rows],yerr=[[r['rt_median_s']-r['rt_q1_s'] for r in time_rows],[r['rt_q3_s']-r['rt_median_s'] for r in time_rows]],fmt='o-',capsize=3,color='#c25e00');ax.set_xticks(x,[r['run'] for r in time_rows],rotation=60,ha='right',fontsize=7);ax.set(ylabel='Time to target (s)',title=f'{label_control} CO8 | TTT median and IQR'+(' | excluding 1116' if suffix else ''));save(fig,out/('Run_TTT_summary'+suffix))
    result=dict(folders=list(map(str,folders)),runs=summaries,sources=sources,pngs=exported,window_seconds=60,step_seconds=1,short_run_policy='ratio_actual_duration_only_if_entire_run_under_60s',method=rolling_bps.METHOD,scope=f'{label_control} behavior CSVs, including sessions without usable XDF; independent of neural trial filtering')
    if folders:(work/('online_behavior_manifest.json' if online else 'offline_behavior_manifest.json')).write_text(json.dumps(storage.jsonable(result),ensure_ascii=False,indent=2),encoding='utf-8')
    return result
