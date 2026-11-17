"""Normalize the new day, tabulate saved configurations, prepare the existing report workflow."""
from pathlib import Path
import csv, hashlib, html, json, sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_analysis_gui.core.catalog import format_run_name,scan
from data_analysis_gui.reporting import report_plan
from data_analysis_gui import config as C

ROOT=C.DEFAULT_DATA_ROOT.resolve()
OUT=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/import_20261120')
OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def flatten(c,p=''):
    result={}
    for k,v in c.items():
        key=f'{p}.{k}' if p else k
        if isinstance(v,dict):result.update(flatten(v,key))
        else:result[key]=v
    return result
def normalize():
    audit=OUT/'normalization_complete.json'
    if audit.exists():return
    day=(ROOT/'20261120').resolve();plans=[]
    folders=sorted((p for p in day.iterdir() if (p/'session.json').is_file()),key=lambda p:read(p/'session.json')['started_at'])
    for i,p in enumerate(folders,1):
        m=read(p/'session.json');c=read(p/'task_config.json') if (p/'task_config.json').exists() else {};g=c.get('GRID_TASK',{})
        online=bool(c.get('CONTROL',{}).get('BCI_ENABLED'));mode='online' if online else 'offline'
        notes=[f'{g["GRID_ROWS"]}x{g["GRID_COLS"]}'] if g else ['startup']
        if g.get('NEARBY_TARGET_ASSIST_ENABLED'):notes.append('nearby-assist')
        if c:notes.append(str(c['GENERAL_TIMING']['DWELL_TIME_SEC']).replace('.','p')+'dwell')
        target=p.with_name(format_run_name(i,'Grid',mode,notes))
        assert p.resolve().parent==day and target.resolve().parent==day and not target.exists()
        files=[dict(path=str(f.relative_to(p)),sha256=sha(f)) for f in p.rglob('*') if f.is_file()]
        xdfs=list(p.rglob('*.xdf'));assert len(xdfs)<=1
        plans.append(dict(old=str(p),new=str(target),run=f'{i:03d}',online=online,config_available=bool(c),session=m['session_id'],files=files))
    (OUT/'normalization_plan.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2),encoding='utf-8')
    for item in plans:
        old=Path(item['old']);target=Path(item['new']);old.rename(target)
        recordings=[]
        for f in item['files']:
            p=target/f['path'];assert sha(p)==f['sha256']
            if p.suffix.lower()=='.xdf':
                q=p.with_name(f'{target.name}-20261120-rec-001-main-spike.xdf');assert not q.exists();p.rename(q)
                recordings.append(dict(path=q.relative_to(ROOT).as_posix(),segment='main',original_xdf=p.name))
        meta=dict(format='keaiduo-session-v1',run_id=item['run'],task='Grid',mode='online' if item['online'] else 'offline',
                  condition='脑控' if item['online'] else ('摇杆手控 / 离线' if item['config_available'] else '空启动 / 控制方式未知'),
                  condition_tag='bci' if item['online'] else 'joystick',session=item['session'],session_path=target.relative_to(ROOT).as_posix(),
                  run_folder=target.name,recordings=recordings,analysis_window_policy='through_task_end')
        (target/'session_info.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    audit.write_text(json.dumps(dict(sessions=len(plans),evidence_bytes_unchanged=True),indent=2),encoding='utf-8')

def config_table():
    rows=[];configs=[];workers=[]
    for day in range(17,21):
        folders=sorted((p for p in (ROOT/f'202611{day}').iterdir() if (p/'session.json').exists()),key=lambda p:read(p/'session.json')['started_at'])
        for p in folders:
            m=read(p/'session.json');c=read(p/'task_config.json') if (p/'task_config.json').exists() else {};f=flatten(c)
            control=c.get('CONTROL',{});g=c.get('GRID_TASK',{});dec=c.get('REALTIME_DECODER',{});active=dec.get('ACTIVE_MODEL','')
            update=dec.get('ONLINE_UPDATE',{});u=update.get('MODELS',{}).get(active.upper(),{})
            trialfile=p/'trials.csv'
            with trialfile.open(encoding='utf-8-sig') as stream:n=sum(1 for _ in csv.DictReader(stream))
            r=dict(date=f'2026-11-{day}',run=read(p/'session_info.json').get('run_id',p.name) if (p/'session_info.json').exists() else p.name,
                   start=m['started_at'][11:],control=('BCI' if control.get('BCI_ENABLED') else 'Joystick') if c else 'Unknown (startup)',
                   bci_enabled=control.get('BCI_ENABLED'),joystick_enabled=control.get('JOYSTICK_ENABLED'),
                   grid=f'{g.get("GRID_ROWS")}x{g.get("GRID_COLS")}' if g else '',blocks=g.get('NUM_BLOCKS'),trials_per_block=g.get('TRIALS_PER_BLOCK'),
                   dwell_s=c.get('GENERAL_TIMING',{}).get('DWELL_TIME_SEC'),timeout_s=c.get('GENERAL_TIMING',{}).get('TRIAL_TIMEOUT_SEC'),
                   nearby_assist=g.get('NEARBY_TARGET_ASSIST_ENABLED'),pos_scale_factor=control.get('POS_SCALE_FACTOR'),smooth_alpha=control.get('BCI_VELOCITY_SMOOTH_ALPHA'),
                   configured_model=active,online_update=update.get('ENABLED'),active_model_update=u.get('ENABLED'),blocks_per_update=u.get('BLOCKS_PER_UPDATE'),
                   min_blocks_for_refit=u.get('MIN_BLOCKS_FOR_REFIT'),blend_alpha=u.get('BLEND_ALPHA'),auto_swap=u.get('AUTO_SWAP'),
                   target_accept_radius=u.get('TARGET_ACCEPT_RADIUS'),target_intent_speed=u.get('TARGET_INTENT_SPEED'),require_improvement=u.get('REQUIRE_IMPROVEMENT','not set'),
                   channel_range=dec.get('SPIKE_SOURCE',{}).get('CHANNEL_RANGE'),model_ref=(m.get('decoder') or {}).get('model_ref'),
                   model_ref_error=(m.get('decoder') or {}).get('model_ref_error'),trials=n,xdf_count=len(list(p.rglob('*.xdf'))),
                   source=str(p/'task_config.json') if c else str(p/'session.json'),folder=str(p))
            rows.append(r)
            for wp in sorted(p.glob('online_update/*/refit_worker_config.json')):
                wc=read(wp);update_cfg=wc.get('online_update_config',{});model=update_cfg.get('models',{}).get(active,{})
                workers.append(dict(date=r['date'],run=r['run'],configured_model=active,blocks_per_update=model.get('blocks_per_update'),
                    min_blocks_for_refit=model.get('min_blocks_for_refit'),blend_alpha=model.get('blend_alpha'),auto_swap=model.get('auto_swap'),
                    target_accept_radius=model.get('target_accept_radius'),target_intent_speed=model.get('target_intent_speed'),require_improvement=model.get('require_improvement','not set'),source=str(wp)))
            if c:configs.append((r,f))
    keys=sorted(set().union(*(f for r,f in configs)));vary={k:[dict(date=r['date'],run=r['run'],value=f.get(k,'<missing>')) for r,f in configs] for k in keys if len({json.dumps(f.get(k,'<missing>'),sort_keys=True) for r,f in configs})>1}
    common={k:configs[0][1].get(k) for k in keys if k not in vary}
    data=dict(runs=rows,refit_worker_configs=workers,variable_fields=vary,common_fields=common,scope='Saved task configuration; model and refit settings do not prove external decoder execution or successful updates.')
    (OUT/'config_1117_1120.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    def cell(x):return html.escape('—' if x is None or x=='' else str(x))
    def table(fields,records=None):
        return '<table><thead><tr>'+''.join(f'<th>{cell(k)}</th>' for k in fields)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{cell(r.get(k))}</td>' for k in fields)+'</tr>' for r in (rows if records is None else records))+'</tbody></table>'
    text='<!doctype html><meta charset="utf-8"><title>1117–1120 Config 对照</title><style>body{font:14px Microsoft YaHei,Arial;margin:28px;color:#18232e}table{border-collapse:collapse;white-space:nowrap;margin:20px 0}th,td{padding:7px 10px;border:1px solid #ccd4df}th{background:#e6edf5;position:sticky;top:0}tr:nth-child(even){background:#f5f7fa}h1{font-size:24px}h2{font-size:20px}pre{white-space:pre-wrap}details{margin:18px 0}</style>'
    text+='<h1>1117–1120：逐 run config 对照</h1><p>1117–1119 均为手控；1120 同时有手控和脑控。表中 decoder / online update 为保存的配置，不能据此认定外部解码器实际使用了该模型、或更新已成功生效。空启动缺少 config，单独列出。</p>'
    text+='<p>共同设置（34 份有效 config）：Grid target_only、preview next target 开启、timeout 5 s、ITI 0 s；BCI target assist 关闭。SPIKE_SOURCE 为 LSL counts，CHANNEL_RANGE=1:512，WINDOW_MS=20 ms。JOYSTICK_ENABLED 始终为 true，因此另列 BCI_ENABLED，避免把配置中的脑控开关误读为关闭了摇杆。nearby assist 开启时 probability=0.9、radius=3。</p>'
    text+='<p>任务程序保存的 rolling metrics 配置为 window=60 s、interval=1 s、update_mode=trial。此次分析报告从原始 trial 时长重新计算 60 s 窗口、1 s step 的 BPS；长 run 从第 60 s 开始，仅整个 run 不足 60 s 时按实际有效时长折算。</p>'
    text+='<p>列说明：blocks 是计划 block 数，trials 是实际记录 trial 数；smooth_alpha 是 BCI_VELOCITY_SMOOTH_ALPHA；blocks_per_update / min_blocks_for_refit 是更新间隔和最低 block 数；blend_alpha 是新旧权重混合参数；auto_swap 是自动替换开关；target_accept_radius / target_intent_speed 是该 active model 的在线更新标签参数。手控 run 的模型字段仅表示保存的设置。</p>'
    text+='<h2>任务与控制</h2>'+table(['date','run','start','control','bci_enabled','joystick_enabled','grid','blocks','trials_per_block','dwell_s','timeout_s','nearby_assist','pos_scale_factor','smooth_alpha','trials','xdf_count'])
    text+='<h2>解码器与在线更新配置</h2>'+table(['date','run','configured_model','online_update','active_model_update','blocks_per_update','min_blocks_for_refit','blend_alpha','auto_swap','target_accept_radius','target_intent_speed','require_improvement','channel_range','model_ref','model_ref_error'])
    if workers:
        text+='<h2>在线更新 worker 保存的配置</h2><p>这是 refit_worker_config.json 中对应 active model 的设置；auto 若已解析则显示数值。配置文件存在仍不等于更新或权重切换已成功。</p>'+table(['date','run','configured_model','blocks_per_update','min_blocks_for_refit','blend_alpha','auto_swap','target_accept_radius','target_intent_speed','require_improvement','source'],workers)
    text+='<details><summary>全部变化字段（含显示及记录设置）</summary><pre>'+cell(json.dumps(vary,ensure_ascii=False,indent=2))+'</pre></details><details><summary>所有已保存 config 的共同设置</summary><pre>'+cell(json.dumps(common,ensure_ascii=False,indent=2))+'</pre></details>'
    text+='<h2>原始 config 来源</h2><ul>'+''.join(f'<li>{cell(r["date"])} run-{cell(r["run"])} <a href="{Path(r["source"]).as_uri()}">{cell(r["source"])}</a></li>' for r in rows)+'</ul>'
    (OUT/'config_1117_1120.html').write_text(text,encoding='utf-8')
    print('CONFIG',len(rows),'runs',len(vary),'varying fields',flush=True)

def request():
    old=read(OUT.parent/'import_20261119/BPS60s_step1s_request.json');params={j['section']:j['params'] for j in old['jobs']}
    recs,notice=scan(ROOT);selected=[r for r in recs if '2026-11-11'<=r.date<='2026-11-20'];valid=[r for r in selected if not r.reason]
    offline=[];online=[]
    for day in range(11,21):
        folder=ROOT/f'202611{day}'
        if not folder.exists():continue
        for p in sorted(folder.iterdir()):
            if not p.is_dir() or not ((p/'trials.csv').exists() or (p/'behavior/trials.csv').exists()):continue
            meta=read(p/'session_info.json') if (p/'session_info.json').exists() else {}
            cfg=read(p/'task_config.json') if (p/'task_config.json').exists() else {}
            is_online=cfg.get('CONTROL',{}).get('BCI_ENABLED',meta.get('mode')=='online' or '-online' in p.name)
            if is_online:
                if day!=16:online.append(str(p))
            else:offline.append(str(p))
    jobs,report=report_plan(valid,lambda s:params.get(s.id,{}),behavior_folders=offline,date_range=('2026-11-11','2026-11-20'),analysis_excluded_dates=['2026-11-16'])
    report['online_behavior_folders']=online
    groups={str(Path(p).resolve()) for p in offline+online}|{r['run_group'] for r in report['recordings']}
    report['run_count']=len(groups);report['name']=f'2026-11-11_2026-11-20_{len(groups)}runs'
    report['skipped'] += [dict(category='神经分析',path=str(r.path),reason=r.reason) for r in selected if r.reason and r.date!='2026-11-16']
    req=dict(old,jobs=[dict(section=s.id,version=s.version,params=p,recordings=[str(r.path) for r in recs]) for s,recs,p in jobs],report=report,font_percent=100,layout_path=str(C.DEFAULT_LAYOUT_XLSX))
    (OUT/'report_request.json').write_text(json.dumps(req,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'xdf_inventory.json').write_text(json.dumps([dict(path=str(r.path),date=r.date,online=r.online,reason=r.reason) for r in selected],ensure_ascii=False,indent=2),encoding='utf-8')
    print('REPORT',report['name'],'neural recordings',report['recording_count'],'jobs',len(jobs),'offline',len(offline),'online CSV',len(online),flush=True)
    for r in selected:
        if r.date=='2026-11-20':print(r.path.name,'online',r.online,'reason',r.reason,flush=True)

if __name__=='__main__':
    if '--table-only' in sys.argv:config_table()
    else:normalize();config_table();request()
