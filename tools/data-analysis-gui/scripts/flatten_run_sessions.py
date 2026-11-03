"""Lift each normalized run's sole session contents into that run's root.

Only the four explicitly requested dates are changed. Original evidence bytes
are moved, duplicate generated documents are archived, metadata is backed up,
and the new paths are written to current indexes and GUI preferences.
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_analysis_gui.core.catalog import scan
from data_analysis_gui.core.channels import read_channel_config
from data_analysis_gui.core.session import Session
from data_analysis_gui.core.xdf_split import digest
from data_analysis_gui.sections.s11_online_vs_offline_v1.model_replay import manifest

ROOT = Path('E:/AitBCI/data/keaiduo').resolve()
DAYS = ('20261106', '20261109', '20261110', '20261111')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    plans, files = [], []
    for day in DAYS:
        for run in sorted((ROOT / day).iterdir()):
            if not run.is_dir() or '_run-' not in run.name:
                continue
            children = list((run / 'sessions').iterdir())
            if len(children) != 1 or not children[0].is_dir():
                raise ValueError(f'Expected exactly one session: {run}')
            source = children[0]
            if not source.resolve().is_relative_to(ROOT / day) or source.is_symlink():
                raise ValueError('Session source escaped the requested day')
            for child in source.iterdir():
                target = run / child.name
                if target.exists() and not (child.is_file() and target.is_file()
                                             and digest(child) == digest(target)):
                    raise ValueError(f'Conflicting run-root contents: {target}')
            old, new = source.relative_to(ROOT).as_posix(), run.relative_to(ROOT).as_posix()
            plans.append(dict(source=source, run=run, old=old, new=new))
            for file in source.rglob('*'):
                if file.is_file():
                    target = run / file.relative_to(source)
                    mutable = file.parent == source and file.name in (
                        'session_info.json', '实验内容说明.txt', '实验设计说明.txt')
                    files.append(dict(old=file.relative_to(ROOT).as_posix(),
                        new=target.relative_to(ROOT).as_posix(), sha256=digest(file), mutable=mutable))
    if len(plans) != 28:
        raise ValueError(f'Unexpected run count: {len(plans)}')
    token = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    audit = ROOT / f'flatten_runs_audit_{token}'
    audit.mkdir()
    backups, operations = {}, []

    def save_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

    def backup(path):
        key = str(path.resolve())
        if key not in backups:
            relative = (path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path('gui_preferences.json'))
            target = audit / 'metadata_before' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            backups[key] = target

    def rewrite_text(text):
        for plan in plans:
            text = text.replace(plan['old'], plan['new'])
            text = text.replace(plan['old'].replace('/', '\\'), plan['new'].replace('/', '\\'))
        return text

    def rewrite(value):
        if isinstance(value, str):
            return rewrite_text(value)
        if isinstance(value, list):
            return [rewrite(x) for x in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    save_json(audit / 'move_plan.json', [{k: str(v) for k, v in p.items()} for p in plans])
    save_json(audit / 'file_hashes_before.json', files)
    previous_checks = read_json(sorted(ROOT.glob('normalization_audit_*/validation.json'))[-1])
    expected = {rewrite_text(r['path']): r for r in previous_checks}
    try:
        for plan in plans:
            source, run = plan['source'], plan['run']
            for child in list(source.iterdir()):
                target = run / child.name
                if child.resolve().parent != source.resolve() or target.resolve().parent != run.resolve():
                    raise ValueError('Move target escaped the run')
                if target.exists():
                    # The two generated experiment-content documents are equal;
                    # preserve the redundant copy in the audit, not in the run.
                    target = audit / 'duplicate_documents' / child.relative_to(ROOT)
                    target.parent.mkdir(parents=True, exist_ok=True)
                child.rename(target)
                operations.append((child, target))
        for plan in plans:
            run = plan['run']
            for filename in ('session_info.json', 'run_info.json'):
                file = run / filename
                backup(file)
                data = rewrite(read_json(file))
                data['storage_layout'] = 'run-root'
                if filename == 'session_info.json':
                    data['session_path'] = plan['new']
                else:
                    data['session_path'] = '.'
                save_json(file, data)
            for filename in ('实验内容说明.txt', '实验设计说明.txt'):
                file = run / filename
                backup(file)
                file.write_text(rewrite_text(file.read_text(encoding='utf-8-sig')), encoding='utf-8')
        for day in DAYS:
            for filename in ('session_index.json', 'file_manifest.json'):
                file = ROOT / day / filename
                backup(file)
                save_json(file, rewrite(read_json(file)))
            for filename in ('整理说明.txt', '实验内容说明.txt', '实验设计说明.txt', '会话与参数变化说明.txt'):
                file = ROOT / day / filename
                backup(file)
                text = rewrite_text(file.read_text(encoding='utf-8-sig'))
                text = text.replace(' / sessions / session-日期-时间', '')
                file.write_text(text, encoding='utf-8')
        guide = ROOT / '归档命名与目录说明.txt'
        backup(guide)
        text = rewrite_text(guide.read_text(encoding='utf-8-sig'))
        text = text.replace('/sessions/session-日期-时间/', '/')
        text = text.replace('session 内按 ', 'run 内按 ')
        text += ('\n已移除各 run 的 sessions/session-日期-时间 两层目录。'
                 'session 身份保存在 run 根目录的 session_info.json，不再额外嵌套文件夹。\n'
                 f'本次路径变更与修改前索引备份：{audit}\n')
        guide.write_text(text, encoding='utf-8')
        prefs = Path(__file__).resolve().parents[1] / '.local' / 'preferences.json'
        if prefs.exists():
            backup(prefs)
            save_json(prefs, rewrite(read_json(prefs)))
        # Every original data/evidence file must still be byte-identical. Only
        # the explicitly listed generated annotations and indexes can change.
        for row in files:
            if not row['mutable'] and digest(ROOT / row['new']) != row['sha256']:
                raise ValueError(f'File changed during flattening: {row["new"]}')
        checks = []
        for day in DAYS:
            recordings, notice = scan(ROOT / day)
            if notice:
                raise ValueError(notice)
            for r in recordings:
                relative = r.path.relative_to(ROOT).as_posix()
                if relative not in expected:
                    raise ValueError(f'Unexpected flattened recording: {relative}')
                baseline = expected[relative]
                read_channel_config(r.channel_config)
                if r.condition != baseline['condition'] or r.date != baseline['date']:
                    raise ValueError('Condition/date metadata changed')
                row = dict(path=relative, date=r.date, condition=r.condition, status='LFP-only')
                if not r.reason:
                    session = Session.load(r)
                    if len(session.counts) != baseline['frames'] or len(session.trials) != baseline['trials']:
                        raise ValueError('Flattening changed analysis samples or trial reconstruction')
                    row.update(status='loaded', frames=len(session.counts), trials=len(session.trials))
                    if r.online:
                        models = manifest(r, {})
                        weights = len(models['weights']) if models else 0
                        if weights != baseline['model_weights']:
                            raise ValueError('Model bundle changed')
                        row['model_weights'] = weights
                    del session
                elif 'CO8LFP' not in relative:
                    raise ValueError(r.reason)
                checks.append(row)
            print('VERIFIED_FLAT', day, 'xdf=',len(recordings), flush=True)
        if len(checks) != 30:
            raise ValueError('Recording count changed')
        # These are verified empty directories; no recursive deletion is used.
        for plan in plans:
            plan['source'].rmdir()
            (plan['run'] / 'sessions').rmdir()
        save_json(audit / 'validation.json', checks)
        save_json(audit / 'completion.json', dict(status='verified', run_count=len(plans), xdf_count=len(checks),
            layout='date/run/{xdf,behavior,config,logs,initial_weights,refit,...}',
            session_wrapper_directories_removed=True, evidence_bytes_unchanged=True,
            historical_audits='Earlier normalization audits retained verbatim as history; current paths use this move map'))
        print('COMPLETE', audit, flush=True)
    except Exception:
        for original, saved in reversed(list(backups.items())):
            shutil.copy2(saved, original)
        for source, target in reversed(operations):
            source.parent.mkdir(parents=True, exist_ok=True)
            target.rename(source)
        raise


if __name__ == '__main__':
    main()
