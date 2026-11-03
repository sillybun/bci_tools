"""Apply run-NNN[-part]-task-mode-note naming to the four normalized days."""
from collections import defaultdict
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_analysis_gui.core.catalog import RUN_PREFIX, format_run_name, scan
from data_analysis_gui.core.channels import read_channel_config
from data_analysis_gui.core.session import Session
from data_analysis_gui.core.xdf_split import digest
from data_analysis_gui.sections.s11_online_vs_offline_v1.model_replay import manifest

ROOT = Path('E:/AitBCI/data/keaiduo').resolve()
DAYS = ('20261106', '20261109', '20261110', '20261111')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def compact_condition(text):
    return text.replace('自由手脑控 / 无辅助', '脑控').replace(' / 无目标辅助', '')


def plan_names(root=ROOT, days=DAYS, *, remove_split_note=False, remove_common_notes=False, remove_manual_note=False):
    plans, split_groups = [], defaultdict(list)
    for day in days:
        for folder in sorted((root / day).iterdir()):
            if not folder.is_dir() or not (folder / 'run_info.json').is_file():
                continue
            meta = read_json(folder / 'session_info.json')
            if remove_split_note or remove_common_notes or remove_manual_note:
                match = RUN_PREFIX.match(folder.name)
                if not match:
                    raise ValueError('Expected run-first names')
                def simplify(value):
                    if remove_manual_note:
                        return value.replace('-manual', '') if match['mode'].lower() == 'offline' else value
                    if remove_common_notes:
                        return value.replace('-arm-free', '').replace('-no-assist', '')
                    return value.replace('-split-session', '').replace('-early-split-', '-early-').replace('-startup-split-', '-startup-')
                name = simplify(folder.name)
                plan = dict(folder=folder, target=folder.with_name(name), name=name, day=day, meta=meta,
                    number=match['number'], part=match['part'], task=match['task'], mode=match['mode'],
                    run_id=match['number'] + (f'-{match["part"]}' if match['part'] else ''), files=[])
                if plan['target'] != folder and plan['target'].exists():
                    raise FileExistsError(plan['target'])
                for row in meta['recordings']:
                    source = root / row['path']
                    filename = simplify(source.name)
                    plan['files'].append(dict(source=source, filename=filename,
                        target=plan['target'] / 'xdf' / filename))
                plans.append(plan)
                continue
            number = re.search(r'_run-([^_]+)', folder.name)
            if not number:
                raise ValueError(f'Run is not in the expected pre-migration naming scheme: {folder}')
            mode = meta['mode']
            suffix = folder.name.split(f'_{mode}_', 1)[1] if f'_{mode}_' in folder.name else ''
            notes = suffix.split('_') if suffix else []
            plan = dict(folder=folder, day=day, meta=meta, number='001' if meta['role'] == 'test' else number[1],
                        part=None, task='CO8' if meta['task'] == 'CO8LFP' else meta['task'], mode=mode, notes=notes)
            split = [r for r in meta['recordings'] if 'split' in r['segment']]
            if split:
                original = split[0]['original_xdf']
                origin_number = re.search(r'_run-(\d{3})', Path(original).stem)[1]
                plan['number'] = origin_number
                plan['split_origin'] = original
                split_groups[day, original].append(plan)
            plans.append(plan)
    if remove_split_note or remove_common_notes or remove_manual_note:
        if len({p['target'] for p in plans}) != len(plans):
            raise ValueError('Conflicting simplified names')
        return plans
    for group in split_groups.values():
        for i, plan in enumerate(sorted(group, key=lambda p: p['meta']['session'].rsplit('_', 1)[-1])):
            plan['part'] = chr(ord('a') + i)
    targets = set()
    for plan in plans:
        if plan['task'] == 'CO8' and plan['mode'] == 'online' and '无目标辅助' in plan['meta']['condition']:
            if 'no-assist' not in plan['notes']:
                plan['notes'].append('no-assist')
        if plan['meta']['session'].endswith('20261111-113117'):
            plan['notes'].append('with-run-002')
        name = format_run_name(plan['number'], plan['task'], plan['mode'], plan['notes'], part=plan['part'])
        plan['name'] = name
        plan['run_id'] = f'{int(plan["number"]):03d}' + (f'-{plan["part"]}' if plan['part'] else '')
        plan['target'] = plan['folder'].with_name(name)
        if plan['target'].exists() or plan['target'] in targets:
            raise FileExistsError(plan['target'])
        targets.add(plan['target'])
        plan['files'] = []
        for row in plan['meta']['recordings']:
            source = root / row['path']
            serial = re.search(r'_rec-(\d{3})_', source.stem)[1]
            signal = 'lfp' if plan['meta']['task'] == 'CO8LFP' else 'spike'
            filename = f'{name}-{plan["day"]}-rec-{serial}-{row["segment"]}-{signal}.xdf'
            plan['files'].append(dict(source=source, filename=filename,
                target=plan['target'] / 'xdf' / filename))
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remove-split-note', action='store_true')
    parser.add_argument('--remove-common-notes', action='store_true')
    parser.add_argument('--remove-manual-note', action='store_true')
    args = parser.parse_args()
    plans = plan_names(remove_split_note=args.remove_split_note, remove_common_notes=args.remove_common_notes,
                       remove_manual_note=args.remove_manual_note)
    if len(plans) != 28:
        raise ValueError('Expected exactly 28 normalized runs')
    token = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    audit = ROOT / f'run_naming_audit_{token}'
    audit.mkdir()
    operations, backups, hashes, substitutions = [], {}, [], {}
    for plan in plans:
        old, new = plan['folder'].relative_to(ROOT), plan['target'].relative_to(ROOT)
        substitutions[old.as_posix()] = new.as_posix()
        substitutions[plan['folder'].name] = plan['name']
        for row in plan['files']:
            substitutions[row['source'].relative_to(ROOT).as_posix()] = row['target'].relative_to(ROOT).as_posix()
            substitutions[row['source'].name] = row['filename']
        for file in plan['folder'].rglob('*'):
            if not file.is_file():
                continue
            target = plan['target'] / file.relative_to(plan['folder'])
            if file.parent.name == 'xdf':
                target = next(r['target'] for r in plan['files'] if r['source'] == file)
            mutable = file.parent == plan['folder'] and file.name in (
                'session_info.json', 'run_info.json', '实验内容说明.txt', '实验设计说明.txt')
            hashes.append(dict(old=file.relative_to(ROOT).as_posix(), new=target.relative_to(ROOT).as_posix(),
                               sha256=digest(file), mutable=mutable))
    replacements = sorted(substitutions.items(), key=lambda p: len(p[0]), reverse=True)

    def text_replace(text):
        for old, new in replacements:
            text = text.replace(old, new).replace(old.replace('/', '\\'), new.replace('/', '\\'))
        return text

    def rewrite(value):
        if isinstance(value, str):
            return text_replace(value)
        if isinstance(value, list):
            return [rewrite(x) for x in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    def save_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

    def backup(path):
        key = str(path.resolve())
        if key not in backups:
            relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path('gui_preferences.json')
            target = audit / 'metadata_before' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            backups[key] = target

    save_json(audit / 'file_hashes_before.json', hashes)
    save_json(audit / 'name_mapping.json', [dict(old=p['folder'].relative_to(ROOT).as_posix(),
        new=p['target'].relative_to(ROOT).as_posix(), run_id=p['run_id'], session=p['meta']['session'],
        split_origin=p.get('split_origin'),
        xdf=[dict(old=r['source'].relative_to(ROOT).as_posix(), new=r['target'].relative_to(ROOT).as_posix())
             for r in p['files']]) for p in plans])
    baseline_pattern = ('run_naming_audit_*/validation.json' if args.remove_split_note or args.remove_common_notes or args.remove_manual_note
                        else 'flatten_runs_audit_*/validation.json')
    baseline = read_json(sorted(ROOT.glob(baseline_pattern))[-1])
    expected = {text_replace(row['path'].replace('lfp-only', 'LFP') if args.remove_common_notes else row['path']): row
                for row in baseline}
    try:
        for plan in plans:
            folder, target = plan['folder'], plan['target']
            if folder.resolve().parent != ROOT / plan['day'] or target.resolve().parent != ROOT / plan['day']:
                raise ValueError('Run rename target escaped the requested day')
            for row in plan['files']:
                intermediate = row['source'].with_name(row['filename'])
                if intermediate == row['source']:
                    continue
                if intermediate.exists() or row['source'].resolve().parent != folder / 'xdf':
                    raise ValueError('Unexpected XDF rename target')
                row['source'].rename(intermediate)
                operations.append((row['source'], intermediate))
            if folder != target:
                folder.rename(target)
                operations.append((folder, target))
        for plan in plans:
            folder = plan['target']
            for filename in ('session_info.json', 'run_info.json'):
                file = folder / filename
                backup(file)
                meta = rewrite(read_json(file))
                meta['naming_schema'] = 'run-NNN[-part]-task-mode-note'
                meta['run_id'] = plan['run_id']
                meta['original_number_for_split'] = plan['number'] if plan['part'] else None
                meta['split_part'] = plan['part']
                if filename == 'session_info.json':
                    meta['run_folder'] = plan['name']
                    meta['session_path'] = folder.relative_to(ROOT).as_posix()
                    if args.remove_split_note:
                        meta['condition_tag'] = ' '.join(meta.get('condition_tag', '').replace('split-session', '').split())
                    if args.remove_common_notes:
                        meta.setdefault('condition_details', meta['condition'])
                        meta['condition'] = compact_condition(meta['condition'])
                        meta['condition_tag'] = ' '.join(meta.get('condition_tag', '').replace('arm free', '').replace('no assist', '').split())
                    if args.remove_manual_note and meta['mode'] == 'offline':
                        meta['condition_tag'] = ' '.join(x for x in meta.get('condition_tag', '').split() if x != 'manual')
                else:
                    meta['run'] = plan['name']
                    if args.remove_common_notes and 'condition' in meta:
                        meta.setdefault('condition_details', meta['condition'])
                        meta['condition'] = compact_condition(meta['condition'])
                save_json(file, meta)
            for filename in ('实验内容说明.txt', '实验设计说明.txt'):
                file = folder / filename
                backup(file)
                file.write_text(text_replace(file.read_text(encoding='utf-8-sig')), encoding='utf-8')
        for day in DAYS:
            for filename in ('session_index.json', 'file_manifest.json'):
                file = ROOT / day / filename
                backup(file)
                data = rewrite(read_json(file))
                if filename == 'session_index.json':
                    for entry in data['sessions']:
                        if entry.get('run_folder'):
                            entry.update(read_json(ROOT / entry['session_path'] / 'session_info.json'))
                    data['naming_schema'] = 'run-NNN[-part]-task-mode-note'
                save_json(file, data)
            mapping = '\n'.join(f'{p["name"]} → {p["meta"]["session"]}' for p in plans if p['day'] == day)
            for filename in ('整理说明.txt', '实验内容说明.txt', '实验设计说明.txt', '会话与参数变化说明.txt'):
                file = ROOT / day / filename
                backup(file)
                text = text_replace(file.read_text(encoding='utf-8-sig'))
                text = text.replace('任务_run-编号_online或offline_关键信息', 'run-编号[-拆分字母]-任务-online或offline-关键信息')
                text = text.replace('Grid run 按当天实际任务顺序统一编号；原编号、原任务名保存在 session_info.json。',
                    '普通 run 使用已有编号；拆分 run 保留来源编号并按 session 时间顺序增加 a/b/c；原编号、原任务名保存在 session_info.json。')
                file.write_text('当前归档命名：run-编号[-a/b/c]-CO8或Grid-online或offline-note\n'
                    '拆分字母按原 XDF 内 session 的时间顺序分配，保留来源 run 编号；历史原实验编号仍保留作证据。\n'
                    + mapping + '\n\n' + text, encoding='utf-8')
        guide = ROOT / '归档命名与目录说明.txt'
        backup(guide)
        text = text_replace(guide.read_text(encoding='utf-8-sig'))
        text = text.replace('任务_run-编号_online或offline_关键信息', 'run-编号[-拆分字母]-任务-online或offline-关键信息')
        text = text.replace('run 编号按当天同类任务的正式 session 时间顺序统一；历史编号保存在 session_info.json。',
            '普通 run 使用已有编号；拆分 run 保留来源编号并按 session 时间顺序增加 a/b/c；历史编号保存在 session_info.json。')
        text += ('\n当前 run 与 XDF 均采用 run 开头、横线分隔：run-001-CO8-online-note。\n'
                 '拆分保留原 run 编号：run-006-a-CO8-…、run-006-b-Grid-…、run-006-c-Grid-…。\n'
                 '1111 的 run-001-b-Grid 目录整合该 8x8 session 的早期拆分片段和原 run-002 主录制，'
                 '两个 XDF 用 rec-001/002 区分，不拼接。a/b/c 已标识拆分，名称和标签不再附加 split-session。\n'
                 'XDF 文件名在 run 名之后附加日期、rec 编号、片段类型及 spike/lfp，以便离开目录后仍能区分日期。\n'
                 f'新旧名称和修改前索引：{audit}\n')
        guide.write_text(text, encoding='utf-8')
        prefs = Path(__file__).resolve().parents[1] / '.local' / 'preferences.json'
        if prefs.exists():
            backup(prefs)
            save_json(prefs, rewrite(read_json(prefs)))
        for row in hashes:
            if not row['mutable'] and digest(ROOT / row['new']) != row['sha256']:
                raise ValueError('Evidence bytes changed: ' + row['new'])
        checks = []
        for day in DAYS:
            records, notice = scan(ROOT / day)
            if notice:
                raise ValueError(notice)
            for r in records:
                relative = r.path.relative_to(ROOT).as_posix()
                prior = expected[relative]
                meta = read_json(r.path.parent.parent / 'session_info.json')
                expected_condition = compact_condition(prior['condition']) if args.remove_common_notes else prior['condition']
                if r.run != meta['run_id'] or r.condition != expected_condition or r.date != prior['date']:
                    raise ValueError('Run identity, condition or date changed unexpectedly')
                read_channel_config(r.channel_config)
                row = dict(path=relative, run_id=r.run, date=r.date, condition=r.condition, status='LFP-only')
                if not r.reason:
                    session = Session.load(r)
                    if len(session.counts) != prior['frames'] or len(session.trials) != prior['trials']:
                        raise ValueError('Renaming changed loaded data or trials')
                    row.update(status='loaded', frames=len(session.counts), trials=len(session.trials))
                    if r.online:
                        model_files = manifest(r, {})
                        count = len(model_files['weights']) if model_files else 0
                        if count != prior['model_weights']:
                            raise ValueError('Model association changed')
                        row['model_weights'] = count
                    del session
                elif meta['task'] != 'CO8LFP':
                    raise ValueError(r.reason)
                checks.append(row)
            print('VERIFIED_NAMES', day, 'xdf=', len(records), flush=True)
        if len(checks) != 30:
            raise ValueError('XDF count changed')
        save_json(audit / 'validation.json', checks)
        save_json(audit / 'completion.json', dict(status='verified', run_count=28, xdf_count=30,
            naming_schema='run-NNN[-part]-task-mode-note', evidence_bytes_unchanged=True))
        print('COMPLETE', audit, flush=True)
    except Exception:
        for original, saved in reversed(list(backups.items())):
            shutil.copy2(saved, original)
        for old, new in reversed(operations):
            new.rename(old)
        raise


if __name__ == '__main__':
    main()
