"""Apply the user's chronological November 11 run order and separate 8x8 parts."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis_gui.core.catalog import scan, describe
from data_analysis_gui.core.channels import read_channel_config
from data_analysis_gui.core.behavior import source_files
from data_analysis_gui.core.session import Session
from data_analysis_gui.core.xdf_split import digest
from data_analysis_gui.sections.s11_online_vs_offline_v1.model_replay import manifest

ROOT = Path('E:/AitBCI/data/keaiduo').resolve()
DAY = ROOT / '20261111'
NAMES = {
    '20261111-102023': 'run-001-CO8-offline',
    '20261111-104853': 'run-002-CO8-online',
    '20261111-112022': 'run-003-a-Grid-online-6x6',
    '20261111-113117': 'run-004-Grid-online-8x8',
    '20261111-114324': 'run-005-Grid-online-10x10',
    '20261111-115759': 'run-006-Grid-offline-12x12',
    '20261111-120908': 'run-007-Grid-offline-7x7',
    '20261111-122553': 'run-008-Grid-online-7x7--0p01dwell',
    '20261111-131426': 'run-009-CO8-offline-LFP',
}
EARLY_NAME = 'run-003-b-Grid-online-8x8'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def run_id(name):
    match = re.match(r'run-(\d{3})(?:-([a-z]))?-', name)
    return match[1] + (f'-{match[2]}' if match[2] else '')


def main():
    bundles = []
    for folder in DAY.iterdir():
        if not folder.is_dir() or not (folder / 'run_info.json').is_file():
            continue
        meta = load(folder / 'session_info.json')
        key = meta['session'].rsplit('_', 1)[-1]
        target = DAY / NAMES[key]
        if target != folder and target.exists():
            raise FileExistsError(target)
        bundles.append(dict(source=folder, target=target, meta=meta))
    if len(bundles) != 9 or (DAY / EARLY_NAME).exists():
        raise ValueError('Unexpected starting run structure')
    shared = next(p for p in bundles if p['meta']['session'].endswith('20261111-113117'))
    early = deepcopy(shared['meta'])
    early['recordings'] = [r for r in early['recordings'] if r['segment'] == 'early-split']
    shared['meta']['recordings'] = [r for r in shared['meta']['recordings'] if r['segment'] == 'main']
    if len(early['recordings']) != 1 or len(shared['meta']['recordings']) != 1:
        raise ValueError('Expected one early and one main 8x8 recording')
    partial = dict(source=shared['source'], target=DAY / EARLY_NAME, meta=early, copied_bundle=True)
    all_bundles = bundles + [partial]
    if len({p['target'] for p in all_bundles})!=10:
        raise ValueError('Conflicting chronological run names')
    file_plans, replacements = [], {}
    for p in all_bundles:
        target, meta = p['target'], p['meta']
        if not p.get('copied_bundle'):
            replacements[p['source'].relative_to(ROOT).as_posix()] = target.relative_to(ROOT).as_posix()
            replacements[p['source'].name] = target.name
        for i, row in enumerate(meta['recordings'], 1):
            source = ROOT / row['path']
            signal = 'lfp' if meta['task'] == 'CO8LFP' else 'spike'
            segment = 'early' if row['segment'] == 'early-split' else 'main' if row['segment'] == 'main' else ''
            label = f'-{segment}' if segment else ''
            filename = f'{target.name}-20261111-rec-{i:03d}{label}-{signal}.xdf'
            dest = target / 'xdf' / filename
            file_plans.append(dict(source=source, target=dest, bundle=p))
            replacements[source.relative_to(ROOT).as_posix()] = dest.relative_to(ROOT).as_posix()
            replacements[source.name] = filename
            row['path'] = dest.relative_to(ROOT).as_posix()
    rules = sorted(replacements.items(), key=lambda p:len(p[0]), reverse=True)

    def rewrite_text(text):
        for old,new in rules:
            text = text.replace(old,new).replace(old.replace('/', '\\'),new.replace('/', '\\'))
        return text

    def rewrite(value):
        if isinstance(value,str):return rewrite_text(value)
        if isinstance(value,list):return [rewrite(x) for x in value]
        if isinstance(value,dict):return {k:rewrite(v) for k,v in value.items()}
        return value

    token = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    audit = ROOT.parent / 'keaiduo_archive' / f'reorder_1111_audit_{token}'
    audit.mkdir(parents=True)
    backups, moves, created, copies = [], [], [], []

    def save(path,data):
        if not path.exists():created.append(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

    def backup(path):
        if not path.exists():return
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path('gui_preferences.json')
        target = audit/'metadata_before'/relative
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
            backups.append((path,target))

    # Establish the baseline from the current files, not from optional older
    # audit folders which the user may have archived or removed.
    previous=[]
    for item in file_plans:
        r=describe(item['source'],ROOT)
        row=dict(path=r.path.relative_to(ROOT).as_posix(),date=r.date,condition=r.condition)
        if not r.reason:
            s=Session.load(r)
            row.update(frames=len(s.counts),trials=len(s.trials))
            if r.online:
                models=manifest(r,{})
                row['model_weights']=len(models['weights']) if models else 0
            del s
        elif item['bundle']['meta']['task']!='CO8LFP':
            raise ValueError(r.reason)
        previous.append(row)
    expected = {rewrite_text(row['path']):row for row in previous if row['date']=='2026-11-11'}
    raw_hashes = {p['target'].relative_to(ROOT).as_posix():digest(p['source']) for p in file_plans}
    # Evidence outside xdf is moved with its original bundle, or copied exactly
    # into the early fragment, except the whole-session behavior CSV.
    evidence = []
    mutable = {'session_info.json','run_info.json','实验内容说明.txt','实验设计说明.txt'}
    for p in bundles:
        for file in p['source'].rglob('*'):
            if file.is_file() and file.parent.name!='xdf' and not(file.parent==p['source'] and file.name in mutable):
                evidence.append((p['target']/file.relative_to(p['source']),digest(file)))
    save(audit/'name_mapping.json',[dict(old=p['source'].relative_to(ROOT).as_posix(),
        new=p['target'].relative_to(ROOT).as_posix(),session=p['meta']['session'],
        segment='early' if p.get('copied_bundle') else 'main') for p in all_bundles])
    try:
        # Only verified duplicate evidence is newly created. The main bundle
        # retains the complete behavior CSV; the early bundle uses XDF events.
        for directory in shared['source'].iterdir():
            if not directory.is_dir() or directory.name in ('xdf','behavior'):
                continue
            for source in directory.rglob('*'):
                if not source.is_file():continue
                target = partial['target']/source.relative_to(shared['source'])
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target);created.append(target)
                sha = digest(source)
                if digest(target)!=sha:raise ValueError('Early-bundle copy mismatch')
                copies.append(dict(source=rewrite_text(source.relative_to(ROOT).as_posix()),
                    target=target.relative_to(ROOT).as_posix(),sha256=sha))
        for p in bundles:
            source,target=p['source'],p['target']
            if source.resolve().parent!=DAY or target.resolve().parent!=DAY:
                raise ValueError('Run move escaped November 11')
            for row in [r for r in file_plans if r['bundle']['source']==source]:
                if row['source'].resolve().parent!=source/'xdf':
                    raise ValueError('XDF source escaped its original run')
                intermediate=(row['target'] if row['bundle'].get('copied_bundle')
                              else source/'xdf'/row['target'].name)
                allowed = row['bundle']['target']/'xdf' if row['bundle'].get('copied_bundle') else source/'xdf'
                if intermediate.resolve().parent!=allowed:
                    raise ValueError('XDF destination escaped its planned run')
                if intermediate==row['source']:continue
                intermediate.parent.mkdir(parents=True,exist_ok=True)
                if intermediate.exists():raise FileExistsError(intermediate)
                row['source'].rename(intermediate);moves.append((row['source'],intermediate))
            if source!=target:
                source.rename(target);moves.append((source,target))
        final_meta=[]
        for p in all_bundles:
            folder=p['target'];meta=rewrite(p['meta'])
            backup(folder/'session_info.json');backup(folder/'run_info.json')
            meta.update(run_folder=folder.name,session_path=folder.relative_to(ROOT).as_posix(),
                run_id=run_id(folder.name),chronological_number=run_id(folder.name).split('-')[0],
                naming_schema='20261111-chronological-run-number',
                split_part=run_id(folder.name).split('-')[1] if '-' in run_id(folder.name) else None)
            if meta['session'].endswith('20261111-113117'):
                meta['associated_recording_bundles']=[EARLY_NAME,NAMES['20261111-113117']]
                meta['recording_scope']='early_fragment' if p.get('copied_bundle') else 'main_recording'
                meta['role']='early_fragment' if p.get('copied_bundle') else 'primary'
                meta['split_session']=bool(p.get('copied_bundle'))
                meta['coverage_note']=('与 run-004 属于同一原始 8x8 session，仅含开头约8秒。'
                    '本片段不使用完整会话的200试次CSV，行为指标按本XDF事件计算。'
                    if p.get('copied_bundle') else
                    '与 run-003-b 属于同一原始8x8 session，此处为原 GRID_run-002 主录制。'
                    '完整行为CSV共200试次，不代表本XDF覆盖全部200试次。')
                if p.get('copied_bundle'):
                    meta['source_behavior_reference']='20261111/run-004-Grid-online-8x8/behavior/trials.csv'
                    text=folder/'behavior'/'README.txt'
                    text.parent.mkdir(parents=True,exist_ok=True)
                    text.write_text(meta['coverage_note']+'\n',encoding='utf-8');created.append(text)
            save(folder/'session_info.json',meta)
            info=dict(format='keaiduo-run-v1',run=folder.name,run_id=meta['run_id'],session=meta['session'],
                      session_count=1,session_path='.',condition=meta['condition'],storage_layout='run-root')
            save(folder/'run_info.json',info)
            for filename in ('实验内容说明.txt','实验设计说明.txt'):
                file=folder/filename;backup(file)
                if meta['session'].endswith('20261111-113117'):
                    text=(folder.name+'\n原始 session：'+meta['session']+'\n'+meta['coverage_note']+'\n\n'
                        '本目录的 XDF：\n'+'\n'.join(Path(r['path']).name for r in meta['recordings'])+
                        '\n\n完整会话实验参数（不等于当前片段覆盖量）：\n'+json.dumps(meta['basic_information'],ensure_ascii=False,indent=2))
                else:text=rewrite_text(file.read_text(encoding='utf-8-sig'))
                if not file.exists():created.append(file)
                file.write_text(text,encoding='utf-8')
            final_meta.append(meta)
        index=DAY/'session_index.json';backup(index)
        data=load(index)
        supplementary=[e for e in data['sessions'] if not e.get('run_folder')]
        data['sessions']=supplementary+sorted(final_meta,key=lambda e:e['run_id'])
        data.update(session_count=len({e['session'] for e in data['sessions']}),
            recording_bundle_count=len(data['sessions']),run_count=10,xdf_count=10,
            recorded_session_count=len({e['session'] for e in final_meta}))
        save(index,data)
        index=DAY/'file_manifest.json';backup(index)
        data=rewrite(load(index))
        for row in data:
            for copy in copies:
                if copy['source'] in row.get('destinations',[]) and copy['target'] not in row['destinations']:
                    row['destinations'].append(copy['target'])
        save(index,data)
        table='\n'.join(e['run_folder']+' → '+e['session'] for e in sorted(final_meta,key=lambda e:e['run_id']))
        for name in ('整理说明.txt','实验内容说明.txt','实验设计说明.txt','会话与参数变化说明.txt'):
            file=DAY/name;backup(file)
            file.write_text('1111当前按时间顺序命名：\n'+table+
                '\nrun-003-b 与 run-004 为同一8x8 session的不同录制片段，分别保存；run-007为offline。\n\n'
                +rewrite_text(file.read_text(encoding='utf-8-sig')),encoding='utf-8')
        guide=ROOT/'归档命名与目录说明.txt'
        if guide.exists():
            backup(guide)
            text=rewrite_text(guide.read_text(encoding='utf-8-sig'))
            text=re.sub(r'^1111 的 run-001-b-Grid 目录整合[^\n]*',
                '1111的8x8早期片段单独存于run-003-b，原GRID_run-002主录制位于run-004；两份XDF分开保存。',text,flags=re.M)
            guide.write_text(text+'\n1111按用户指定的当日完整时间顺序编号，见该日整理说明和 '+str(audit)+'\n',encoding='utf-8')
        prefs=Path(__file__).resolve().parents[1]/'.local/preferences.json'
        if prefs.exists():backup(prefs);save(prefs,rewrite(load(prefs)))
        checks=[]
        for path,sha in raw_hashes.items():
            if digest(ROOT/path)!=sha:raise ValueError('XDF content changed')
        for path,sha in evidence:
            if digest(path)!=sha:raise ValueError('Evidence content changed')
        for row in copies:
            if digest(ROOT/row['target'])!=row['sha256']:raise ValueError('Copied evidence changed')
        records,notice=scan(DAY)
        if notice or len(records)!=10:raise ValueError('Incorrect recording inventory')
        for r in records:
            relative=r.path.relative_to(ROOT).as_posix();prior=expected[relative]
            meta=load(r.path.parent.parent/'session_info.json')
            if r.run!=meta['run_id'] or r.condition!=prior['condition']:raise ValueError('Run identity/condition changed')
            read_channel_config(r.channel_config)
            row=dict(path=relative,run_id=r.run,date=r.date,condition=r.condition,status='LFP-only')
            if not r.reason:
                s=Session.load(r)
                if len(s.counts)!=prior['frames'] or len(s.trials)!=prior['trials']:
                    raise ValueError('Data/trial reconstruction changed')
                row.update(status='loaded',frames=len(s.counts),trials=len(s.trials),
                    complete_trials=sum(t['complete'] for t in s.trials),
                    trial_end_events=sum(e[0]=='T_TRIAL_END' for e in s.events))
                if r.online:
                    models=manifest(r,{})
                    row['model_weights']=len(models['weights']) if models else 0
                    if row['model_weights']!=prior['model_weights']:raise ValueError('Model association changed')
                if r.run=='003-b' and source_files(r):raise ValueError('Early fragment must not use the whole-session CSV')
                del s
            elif meta['task']!='CO8LFP':raise ValueError(r.reason)
            checks.append(row)
        save(audit/'copied_sidecars.json',copies);save(audit/'validation.json',checks)
        save(audit/'completion.json',dict(status='verified',run_count=10,xdf_count=10,
            run007_mode='offline',source_xdf_bytes_unchanged=True,shared_8x8_session_split_into_two_recording_bundles=True))
        print('COMPLETE',audit,flush=True)
        for row in sorted(checks,key=lambda r:r['run_id']):print(row,flush=True)
    except Exception:
        for file,saved in reversed(backups):shutil.copy2(saved,file)
        for old,new in reversed(moves):new.rename(old)
        # Only files newly copied/generated under the new early-fragment folder
        # are removed; original XDFs have already been restored by the move log.
        early_root=DAY/EARLY_NAME
        for file in reversed(created):
            if file.is_relative_to(early_root) and file.exists():file.unlink()
        if early_root.exists():
            for directory in sorted((p for p in early_root.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
                directory.rmdir()
            early_root.rmdir()
        raise


if __name__=='__main__':main()
