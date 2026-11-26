"""CSV-backed control-mode comparison, joined to saved configurations by run identity."""
from pathlib import Path
import csv, json, math, os, re, shutil, subprocess, tempfile
import numpy as np
from . import storage
from .core.run_config import configured_run_name, naming_config
from .report_behavior import folder_info, bundle_files, behavior_run_inputs, _compute_run_behavior
from .sections.s13_online_behavior_v1 import rolling_bps
from .sections.base import new_figure

CATEGORY = '11_脑控与离线摇杆_BPS_TTT'
WORKBOOK_NAME = 'behavior_config_stat.xlsx'
COLORS = {'BCI': '#0072B2', 'Joystick': '#D55E00', 'Unknown': '#999999'}

def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict): result.update(flatten(item, name))
        else: result[name] = item
    return result

def discover(paths, excluded=('20261116',)):
    """Include CSV-only days between selected dates; only canonical numbered runs."""
    days = set()
    for value in paths:
        p = Path(value)
        days.update(x for x in (p, *p.parents) if re.fullmatch(r'20\d{6}', x.name))
    result = []
    for root in {d.parent for d in days}:
        selected = sorted(d.name for d in days if d.parent == root)
        for day in sorted(root.iterdir()):
            if day.is_dir() and selected[0] <= day.name <= selected[-1] and re.fullmatch(r'20\d{6}', day.name) and day.name not in excluded:
                result.extend(p for p in day.glob('run-*') if p.is_dir() and re.match(r'run-\d+-',p.name))
    return sorted(set(result))

def config_path(folder):
    return next((p for p in (folder/'task_config.json',folder/'config/task_config.json') if p.is_file()), None)

def source_signature(paths):
    from .report_run_pool import file_inputs
    files = {}
    for folder in discover(paths):
        mode, _, _ = folder_info(folder)
        for key, path in bundle_files(folder,mode).items(): files[f'{folder}/{key}'] = path
        for p in (config_path(folder), folder/'session.json', folder/'session_info.json'):
            if p and p.is_file(): files[str(p)] = p
        files[str(folder)] = folder/'task_config.json' if (folder/'task_config.json').exists() else folder/'session_info.json'
    # Folder inventory also invalidates a previously missing-config/CSV result.
    return dict(folders=[str(p) for p in discover(paths)], files=file_inputs({k:p for k,p in files.items() if p.is_file()}))

def collect(folders, pool, progress=lambda s: None):
    rows, full, sources = [], [], []
    for folder in folders:
        folder = Path(folder); mode, day, meta = folder_info(folder)
        p = config_path(folder); cfg = json.loads(p.read_text(encoding='utf-8-sig')) if p else {}
        fields = flatten(cfg); ctl=cfg.get('CONTROL',{}); g=cfg.get('GRID_TASK',{}); timing=cfg.get('GENERAL_TIMING',{})
        task=meta.get('task') or ('Grid' if mode=='grid' else 'CO8')
        run=re.match(r'run-(\d+(?:-[a-z])?)-',folder.name)[1]
        date=f'{day[:4]}-{day[4:6]}-{day[6:]}'
        start=meta.get('basic_information',{}).get('start_time') or meta.get('started_at','')
        control=('BCI' if ctl.get('BCI_ENABLED') else 'Joystick' if ctl.get('JOYSTICK_ENABLED') else 'Unknown') if cfg else 'Unknown'
        decoder=cfg.get('REALTIME_DECODER',{}); model=decoder.get('ACTIVE_MODEL',''); update=decoder.get('ONLINE_UPDATE',{}); u=update.get('MODELS',{}).get(model.upper(),{})
        row=dict(config_key=f'{date}|{task}|{run}',date=date,task=task,run=run,control=control,mode=mode,start=start,status='missing_trials',
            grid=f'{g["GRID_ROWS"]}x{g["GRID_COLS"]}' if mode=='grid' and g.get('GRID_ROWS') and g.get('GRID_COLS') else None,dwell_s=timing.get('DWELL_TIME_SEC'),timeout_s=timing.get('TRIAL_TIMEOUT_SEC'),
            preview_next_target=g.get('PREVIEW_NEXT_TARGET') if mode=='grid' else 'n.a.',nearby_assist=g.get('NEARBY_TARGET_ASSIST_ENABLED') if mode=='grid' else 'n.a.',
            blocks=(g if mode=='grid' else cfg.get('CENTER_OUT_TASK',{})).get('NUM_BLOCKS'),joystick_gain=ctl.get('POS_SCALE_FACTOR'),
            bci_velocity_scale=ctl.get('BCI_VELOCITY_SCALE'),bci_smooth_alpha=ctl.get('BCI_VELOCITY_SMOOTH_ALPHA'),bci_cursor_mode=ctl.get('BCI_CURSOR_MODE'),
            configured_model=model,online_update=update.get('ENABLED'),active_model_update=u.get('ENABLED'),blocks_per_update=u.get('BLOCKS_PER_UPDATE'),
            min_blocks_for_refit=u.get('MIN_BLOCKS_FOR_REFIT'),blend_alpha=u.get('BLEND_ALPHA'),auto_swap=u.get('AUTO_SWAP'),
            config_source=str(p) if p else 'missing config',run_folder=str(folder))
        fields_for_name = naming_config(cfg)
        row.update(nearby_radius=fields_for_name['nearby_radius'],
                   nearby_probability=fields_for_name['nearby_probability'],
                   bci_target_assist_enabled=fields_for_name['bci_target_assist_enabled'],
                   bci_target_assist_speed=fields_for_name['bci_target_assist_speed'])
        number, _, part = run.partition('-')
        row['run_name'] = (configured_run_name(folder, number, task, 'online' if control == 'BCI' else 'offline',
                                               part=part or None)
                           if cfg and control != 'Unknown' else folder.name)
        files=bundle_files(folder,mode)
        if files:
            if mode=='grid' and (folder/'task_config.json').is_file(): files['task_config']=folder/'task_config.json'
            label=f'{day[4:6]}-{day[6:]} {folder.name}'; inputs=behavior_run_inputs(files,mode,label)
            try:
                data,cached,hit=pool.get('BPS_TTT_data',str(folder),inputs,lambda out:_compute_run_behavior(files,mode,label,day,folder))
                stat,=rolling_bps.run_summary(data['windows'],mode,runs=[dict(run_key=str(folder),date=day,session=label)],trials_by_run={str(folder):data['trials']})
                row.update({k:v for k,v in stat.items() if k not in ('date','run','run_key')})
                row.update(status='ok' if data['trials'] else 'empty_trials',trials=len(data['trials']),successes=sum(t['outcome']=='acquired' for t in data['trials']),windows=len(data['windows']))
                row['attempts']=data['metric']['attempts']
                row['success_rate']=data['metric']['success_pct']/100
                sources.append(dict(config_key=row['config_key'],pool=str(cached),cache_hit=hit,inputs=inputs))
            except (ValueError,KeyError) as exc:
                row.update(status='invalid_behavior',error=str(exc))
        rows.append(row);full.append(dict(config_key=row['config_key'],fields=fields,source=row['config_source']))
        progress(f'{date} {task} run{run}: {row["status"]}')
    rows.sort(key=lambda r:(r['date'],r['start'],r['run_folder']))
    for i,r in enumerate(rows,2):r['excel_row']=i;r['plot_id']=f'{i-1:02d}'
    full.sort(key=lambda r:next(i for i,x in enumerate(rows) if x['config_key']==r['config_key']))
    changes=[]; previous=None
    for r in full:
        if previous and previous['fields'] and r['fields']:
            for key in sorted(set(previous['fields'])|set(r['fields'])):
                a=previous['fields'].get(key,'未保存字段');b=r['fields'].get(key,'未保存字段')
                if a!=b and not key.startswith('SESSION.'):
                    changes.append(dict(config_key=r['config_key'],previous=previous['config_key'],field=key,before=a,after=b,
                                        kind='新增' if key not in previous['fields'] else
                                        '缺失' if key not in r['fields'] else '值变化'))
        if r['fields']:
            previous=r
    return dict(rows=rows,full_config=full,changes=changes,sources=sources,method=rolling_bps.METHOD)

def figure(rows,mode,title):
    mixed = mode is None
    if mixed:
        rows = [dict(row, **{f'combined_bps_{stat}': row.get(f'{prefix}_{stat}')
                            for stat in ('median', 'q1', 'q3', 'peak')})
                for row in rows
                for prefix in ('neuralink_bps' if row['mode'] == 'centerout' else 'bps',)]
    methods = [('combined_bps', 'BPS')] if mixed else rolling_bps.methods(mode)
    specs=[(k+'_median',k+'_q1',k+'_q3',label+' (bits/s)',k+'_peak') for k,label in methods]
    specs.append(('rt_median_s','rt_q1_s','rt_q3_s','TTT (s; includes dwell)',None))
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath

    labels = [f"{r['date'][5:]} {r.get('run_name', r['run'])} #{r['plot_id']}" for r in rows]
    measure = TextToPath()
    tick_depth = max((measure.get_text_width_height_descent(label, FontProperties(size=11), False)[0]
                      for label in labels), default=0) / 72
    footer_height = tick_depth + 1.3
    plot_height = len(specs)*2.9 + 1.4
    height = plot_height + footer_height
    fig = new_figure(max(12,len(rows)*.56),height)
    fig.set_layout_engine(None)
    axes = fig.subplots(len(specs),1,sharex=True)
    fig.subplots_adjust(bottom=footer_height/height,top=1-1.1/height,hspace=.35)
    fig.suptitle(title,fontsize=18)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],color=COLORS[c],marker=m,ls='',label=l) for c,m,l in [('BCI','o','脑控 BCI'),('Joystick','s','离线摇杆 Joystick')]]+[Line2D([],[],color='#777777',marker='*',ls='',label='BPS 窗口峰值')],loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,frameon=False)
    for ax,(key,q1,q3,label,peak) in zip(axes,specs):
        for i,r in enumerate(rows):
            v=r.get(key,math.nan);c=COLORS[r['control']]
            if isinstance(v,(int,float)) and math.isfinite(v):
                ax.errorbar(i,v,yerr=[[v-r[q1]],[r[q3]-v]],fmt='o' if r['control']=='BCI' else 's',color=c,capsize=3)
                if len(rows)<=15:ax.annotate(f'{v:.2f}',(i,v),xytext=(3,5),textcoords='offset points',fontsize=11,color=c)
            else:ax.text(i,.03,'n.a.',transform=ax.get_xaxis_transform(),rotation=90,ha='center',color='#777777',fontsize=9)
            p=r.get(peak,math.nan)
            if peak and isinstance(p,(int,float)) and math.isfinite(p):ax.scatter(i,p,marker='*',s=65,color=c,alpha=.7)
        ax.set_ylabel(label,fontsize=14);ax.tick_params(labelsize=11);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
        for i in range(1,len(rows)):
            if rows[i]['date']!=rows[i-1]['date']:ax.axvline(i-.5,color='#888888',ls='--',lw=.7)
    axes[-1].set_xticks(range(len(rows)),labels,rotation=90,ha='center',va='top')
    for tick,r in zip(axes[-1].get_xticklabels(),rows):tick.set_color(COLORS[r['control']])
    axes[-1].set_xlabel('日期 / 任务 / run；#编号对应 Excel 的图编号',fontsize=14)
    methods_note = 'BPS 采用各范式口径：Grid 用 C × log₂(N) / T，CO8 用 Neuralink-style。\n' if mixed else ''
    fig.text(.5,.015,methods_note+'60 s 有效尝试时长窗口，step 1 s；长 run 从 60 s 开始，整 run 不足 60 s 才按实际时长折算。\n中位数 ± IQR（非置信区间）；TTT 每个成功 trial 仅计一次。不同范式、dwell、辅助设置请分组比较。',ha='center',fontsize=11)
    return fig

def export_workbook(payload,path,*,previews=True):
    """Use the desktop's bundled Artifact Tool; never install dependencies in this repo."""
    node=os.environ.get('AITBCI_ARTIFACT_NODE'); modules=os.environ.get('AITBCI_ARTIFACT_MODULES')
    if not node:
        candidates=sorted(Path('C:/Program Files/WindowsApps').glob('OpenAI.CodexPrimaryRuntime.*/dependencies/node/bin/node.exe'),reverse=True)
        if not candidates:
            found=subprocess.run(['powershell','-NoProfile','-Command',"Get-AppxPackage '*CodexPrimaryRuntime*' | Select-Object -ExpandProperty InstallLocation"],capture_output=True,text=True)
            candidates=[Path(p.strip())/'dependencies/node/bin/node.exe' for p in found.stdout.splitlines() if p.strip()]
            candidates=sorted((p for p in candidates if p.is_file()),reverse=True)
        if not candidates:raise RuntimeError('未找到 Codex bundled Node；设置 AITBCI_ARTIFACT_NODE / AITBCI_ARTIFACT_MODULES')
        node=str(candidates[0])
    modules=modules or str(Path(node).parent.parent/'node_modules')
    with tempfile.TemporaryDirectory(prefix='bci-config-excel-') as tmp:
        tmp=Path(tmp);shutil.copy2(Path(__file__).with_name('control_comparison_excel.mjs'),tmp/'build.mjs')
        subprocess.run(['cmd','/c','mklink','/J',str(tmp/'node_modules'),modules],check=True,capture_output=True)
        (tmp/'source.json').write_text(json.dumps(storage.jsonable(payload),ensure_ascii=False,allow_nan=False,default=str),encoding='utf-8')
        try:
            completed=subprocess.run([node,str(tmp/'build.mjs'),str(tmp/'source.json'),str(Path(path).resolve()),*(['--no-previews'] if not previews else [])],capture_output=True,text=True,timeout=240,encoding='utf-8',errors='replace')
            if completed.returncode:raise RuntimeError('Excel export failed: '+completed.stderr[-6000:])
        finally:
            # Remove the junction itself before TemporaryDirectory cleanup.
            os.rmdir(tmp/'node_modules')

def render_comparison(work,report,font_percent,send,pool):
    paths=[r['path'] for r in report.get('recordings',report.get('selection',[]))]
    paths+=report.get('offline_behavior_folders',[])+report.get('online_behavior_folders',[])
    excluded={'20261116',*(x.replace('-','') for x in report.get('analysis_excluded_dates',[]))}
    folders=discover(paths,tuple(excluded))
    if report.get('run_filter') is not None:
        from .core.catalog import run_folder
        allowed = {folder.resolve() for p in paths if (folder := run_folder(Path(p))) is not None}
        folders = [folder for folder in folders if folder.resolve() in allowed]
    payload=collect(folders,pool,lambda msg:send('message',message=msg))
    return export_comparison(payload,work,font_percent)


def export_comparison(payload,work,font_percent):
    out=Path(work)/CATEGORY/'summary_all_runs'
    if not payload['rows']:return dict(category=CATEGORY,run_count=0,valid_runs=0,pngs=[],note='No numbered run folders in selection')
    out.mkdir(parents=True,exist_ok=True)
    clean=storage.jsonable(payload)
    # jsonable leaves nonfinite floats unchanged: represent unavailable values as null.
    def finite(x):
        if isinstance(x,dict):return {k:finite(v) for k,v in x.items()}
        if isinstance(x,list):return [finite(v) for v in x]
        return None if isinstance(x,float) and not math.isfinite(x) else x
    clean=finite(clean)
    (out/'comparison_data.json').write_text(json.dumps(clean,ensure_ascii=False,indent=2),encoding='utf-8')
    keys=list(dict.fromkeys(k for r in clean['rows'] for k in r))
    with (out/'run_statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(clean['rows'])
    plot = export_summary_figure(payload['rows'], out, font_percent)
    export_workbook(clean,out/WORKBOOK_NAME,previews=False)
    return dict(category=CATEGORY,run_count=len(payload['rows']),valid_runs=sum(r['status']=='ok' for r in payload['rows']),
                pngs=[str(plot.relative_to(work))],excel=str((out/WORKBOOK_NAME).relative_to(work)),method=rolling_bps.METHOD)


def export_summary_figure(rows, out, font_percent):
    from .core.style import finish_figure

    fig = figure(rows, None, '全部 run · 脑控与离线摇杆 BPS / TTT')
    finish_figure(fig, font_percent)
    plot = out / 'all_runs_BCI_Joystick_BPS_TTT.png'
    fig.savefig(plot, dpi=220, bbox_inches='tight', facecolor='white')
    fig.clear()
    (out/'index.html').write_text(
        '<meta charset="utf-8"><h1>脑控与离线摇杆 BPS / TTT</h1>'
        f'<p><a href="{WORKBOOK_NAME}">下载配置与统计 Excel</a></p>'
        f'<p><a href="{plot.name}">全部 run 汇总图</a></p>', encoding='utf-8')
    return plot
