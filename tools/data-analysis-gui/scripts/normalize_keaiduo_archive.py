"""Normalize the November 6/9/10/11 archive; build and verify before publishing.

The weekly package is read-only. The original November 6 folder is moved intact
to a named sibling archive after all four replacement days have been verified.
No XDF or scientific evidence file is edited: filenames and locations change,
and explicit JSON/text indexes describe new locations and source provenance.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_analysis_gui.core.catalog import describe, scan
from data_analysis_gui.core.channels import read_channel_config
from data_analysis_gui.core.session import Session, recording_window_policy
from data_analysis_gui.core.xdf_io import read_headers
from data_analysis_gui.core.xdf_split import digest
from data_analysis_gui.sections.s11_online_vs_offline_v1.model_replay import manifest


DATA = Path('E:/AitBCI/data')
ROOT = DATA / 'keaiduo'
WEEK = DATA / 'keaiduo_1109_1110_1111_整理包'
SPLIT = DATA / 'keaiduo_1109_1110_1111_session_split'
OLD = ROOT / '20261106'
DAYS = ('20261106', '20261109', '20261110', '20261111')
COMMON = Path('shared_metadata/package_20261109-20261111')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def stamp(session):
    return re.search(r'_(\d{8}-\d{6})$', session)[1]


def config_design(folder):
    file = folder / 'config' / 'task_config.json'
    if not file.is_file():
        return {}
    cfg = read_json(file)
    mode = cfg.get('TASK_INFO', {}).get('TASK_MODE')
    task = cfg.get('CENTER_OUT_TASK' if mode == 'centerout' else 'GRID_TASK', {})
    timing, control = cfg.get('GENERAL_TIMING', {}), cfg.get('CONTROL', {})
    return dict(task_parameters=dict(task_mode=mode, grid_rows=task.get('GRID_ROWS'),
        grid_cols=task.get('GRID_COLS'), planned_blocks=task.get('NUM_BLOCKS'),
        dwell_sec=timing.get('DWELL_TIME_SEC'), iti_sec=timing.get('ITI_DURATION_SEC'),
        trial_duration_sec=timing.get('TRIAL_DURATION_SEC')),
        speed_parameters=dict(bci_enabled=control.get('BCI_ENABLED'),
            bci_target_assist_effective=bool(control.get('BCI_ENABLED') and control.get('BCI_TARGET_ASSIST_ENABLED')),
            initial_target_assist_speed=control.get('BCI_TARGET_ASSIST_SPEED')),
        raw_task_config='config/task_config.json')


class Organizer:
    def __init__(self, resume_stage=None):
        for day in DAYS[1:]:
            if (ROOT / day).exists():
                raise FileExistsError(ROOT / day)
        self.token = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        if resume_stage:
            stage = Path(resume_stage).resolve()
            if stage.parent != DATA.resolve() or not stage.name.startswith('keaiduo_normalize_stage_'):
                raise ValueError('Unexpected resume staging path')
            backup = Path(read_json(stage / 'normalization_plan.json')['backup'])
            self.token = backup.name.removeprefix('20261106_before_normalization_')
            if not re.fullmatch(r'\d{8}_\d{6}', self.token):
                raise ValueError('Invalid resume archive timestamp')
        self.backup = DATA / 'keaiduo_archive' / f'20261106_before_normalization_{self.token}'
        if self.backup.exists() or (ROOT / COMMON).exists():
            raise FileExistsError('Archive destination already exists')
        self.stage = stage if resume_stage else Path(tempfile.mkdtemp(prefix='keaiduo_normalize_stage_', dir=DATA))
        self.inventory = {str(p.resolve()): dict(source=self.source_name(p), bytes=p.stat().st_size,
                          sha256=None, destinations=[], retained_at_source=False)
                          for base in (WEEK, OLD) for p in base.rglob('*') if p.is_file()}
        self.hashes, self.entries, self.recordings = {}, {}, []
        self.design = read_json(WEEK / 'experiment_design.json')
        self.source_sessions = {s['session']: s for s in read_json(WEEK / 'session_index.json')}
        self.split_rows = read_json(SPLIT / 'split_manifest.json')['sessions']
        self.old_index = read_json(OLD / 'recordings_index.json')

    def source_name(self, path):
        path = Path(path).resolve()
        return str(self.backup / path.relative_to(OLD)) if path.is_relative_to(OLD) else str(path)

    def sha(self, path):
        key = str(Path(path).resolve())
        if key not in self.hashes:
            self.hashes[key] = digest(path)
        return self.hashes[key]

    def copy(self, source, relative):
        source = Path(source).resolve()
        target = self.stage / relative
        if not target.resolve().is_relative_to(self.stage):
            raise ValueError('Copy destination escaped staging root')
        sha = self.sha(source)
        if target.exists():
            if digest(target) != sha:
                raise ValueError(f'Conflicting evidence files: {source} -> {relative}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if digest(target) != sha:
                raise ValueError(f'Copy verification failed: {source}')
        item = self.inventory.get(str(source))
        if item is not None:
            item['sha256'] = sha
            if relative.as_posix() not in item['destinations']:
                item['destinations'].append(relative.as_posix())
        return target

    def tree(self, source, relative):
        for file in sorted(source.rglob('*')):
            if file.is_file():
                self.copy(file, relative / file.relative_to(source))

    def retain(self, file, reason):
        item = self.inventory[str(file.resolve())]
        item.update(sha256=self.sha(file), retained_at_source=True, reason=reason)

    def plan(self):
        counters = Counter()
        for s in sorted(self.design['sessions'], key=lambda s: stamp(s['session'])):
            name = s['session']
            source = self.source_sessions[name]
            day = stamp(name)[:8]
            params = s['task_parameters']
            task = ('CO8LFP' if params['task_label'] == 'CO8-LFP' else
                    'Grid' if params['task_mode'] == 'grid' else 'CO8')
            mode = 'online' if source['bci_enabled'] else 'offline'
            grid = f'{params["grid_rows"]}x{params["grid_cols"]}' if task == 'Grid' else ''
            is_main = bool(source['xdf_names'])
            split = any(row['session'] == name for row in self.split_rows)
            notes = [grid] if grid else (['manual'] if mode == 'offline' else [])
            if split:
                notes.append('split-session')
            if not s['completed_behavior_trial_rows']:
                notes.append('startup')
            if task == 'CO8LFP':
                notes.append('lfp-only')
            if task == 'Grid' and params['dwell_sec'] == 0.01:
                notes.append('dwell-0p01')
            if is_main:
                counters[day, task] += 1
                run = f'{task}_run-{counters[day, task]:03d}_{mode}' + ('_' + '_'.join(notes) if notes else '')
                base = Path(day) / run / 'sessions' / ('session-' + stamp(name))
            else:
                run = None
                base = Path(day) / 'supplementary' / 'sessions' / ('session-' + stamp(name))
            condition = ('脑控' if mode == 'online' else '手控 / 离线')
            if grid:
                condition += ' / ' + grid
            if mode == 'online':
                condition += ' / 无目标辅助'
            tag = ' '.join(x for x in (grid, 'split-session' if split else '',
                                      'no assist' if mode == 'online' else 'manual') if x)
            self.entries[name] = dict(format='keaiduo-session-v1', session=name, date=day,
                task=task, mode=mode, run_folder=run, session_path=base.as_posix(),
                condition=condition, condition_tag=tag, basic_information=s,
                original_locations=source['locations'], split_session=split,
                analysis_window_policy=None, role='primary' if is_main else 'supplementary', recordings=[])
        for row in self.old_index['primary_runs']:
            n = row['run']
            name = row['session']
            condition = {'001': '手控 / 离线', '002': '自由手脑控 / 无辅助',
                         '003': '手束缚脑控 / 初始辅助 200 px/s', '004': '手束缚脑控 / 初始辅助 100 px/s'}[n]
            tags = {'001': 'manual', '002': 'arm free no assist', '003': 'restrained assist200',
                    '004': 'restrained assist100'}
            notes = {'001': 'manual', '002': 'arm-free_no-assist', '003': 'restrained_assist-200',
                     '004': 'restrained_assist-100'}
            mode = 'offline' if n == '001' else 'online'
            run = f'CO8_run-{n}_{mode}_{notes[n]}'
            basic = config_design(OLD / row['directory'])
            basic.update(archive_evidence=row)
            self.entries[name] = dict(format='keaiduo-session-v1', session=name, date='20261106',
                task='CO8', mode=mode, run_folder=run,
                session_path=(Path('20261106') / run / 'sessions' / ('session-' + stamp(name))).as_posix(),
                condition=condition, condition_tag=tags[n], basic_information=basic,
                original_locations=[self.source_name(OLD / row['directory'])], split_session=False,
                analysis_window_policy='through_task_end', role='primary', recordings=[])
        for p in (OLD / 'test').rglob('*.xdf'):
            sessions = {h.findtext('desc/session_name') for h in read_headers(p)
                        if h.findtext('name') == 'cursortasks_events'}
            if len(sessions) != 1:
                raise ValueError('Morning test has ambiguous session identity')
            name = next(iter(sessions))
            run = 'CO8_run-test001_offline_morning-test'
            self.entries[name] = dict(format='keaiduo-session-v1', session=name, date='20261106',
                task='CO8', mode='offline', run_folder=run,
                session_path=(Path('20261106') / run / 'sessions' / ('session-' + stamp(name))).as_posix(),
                condition='早上测试 / 手控离线', condition_tag='morning test',
                basic_information={'scope': '早上测试，不计入四组正式实验'},
                original_locations=[self.source_name(p.parent)], split_session=False,
                analysis_window_policy=recording_window_policy(describe(p, OLD)), role='test', recordings=[])
        for row in self.old_index['supplementary_task_sessions']:
            name = row['session']
            basic = config_design(OLD / row['directory'])
            control = basic.get('speed_parameters', {}).get('bci_enabled')
            self.entries[name] = dict(format='keaiduo-session-v1', session=name, date='20261106',
                task='Grid' if name.startswith('grid-') else 'CO8',
                mode='online' if control is True else 'offline' if control is False else 'unknown',
                run_folder=None, session_path=(Path('20261106/supplementary/sessions') / ('session-' + stamp(name))).as_posix(),
                condition='补充会话，未分配正式实验条件', condition_tag='supplementary',
                basic_information=dict(**basic, archive_evidence=row),
                original_locations=[self.source_name(OLD / row['directory'])], split_session=False,
                analysis_window_policy=None, role='supplementary', recordings=[])
        write_json(self.stage / 'normalization_plan.json', dict(backup=str(self.backup),
                   target=str(ROOT), sessions=list(self.entries.values())))
        for day in DAYS:
            print(day, [(e['run_folder'], e['session']) for e in self.entries.values()
                        if e['date'] == day and e['run_folder']], flush=True)

    def session_tree(self, source, entry, source_group):
        base = Path(entry['session_path'])
        for file in sorted(source.rglob('*')):
            if not file.is_file() or file.suffix.lower() == '.xdf':
                continue
            relative = file.relative_to(source)
            if relative.parts[0] == 'xdf':
                self.copy(file, base / 'source_metadata' / source_group / 'recordings' / relative)
                continue
            if relative.parts[0] == 'training_evidence':
                target = base.parents[1] / 'training_and_deployment' / Path(*relative.parts[1:])
            elif len(relative.parts) == 1 and file.suffix.lower() == '.txt':
                target = base / 'source_notes' / source_group / relative
            elif len(relative.parts) == 1 and file.suffix.lower() == '.json':
                target = base / 'source_metadata' / source_group / relative
            elif len(relative.parts) == 1 and file.suffix.lower() == '.log':
                target = base / 'logs' / relative
            else:
                if len(relative.parts) > 2 and relative.parts[0] == 'refit' and re.fullmatch(
                        r'\d{8}_\d{6}', relative.parts[1]):
                    relative = Path('refit', *relative.parts[2:])
                target = base / relative
            self.copy(file, target)

    def add_recording(self, source, session, kind, *, original_xdf=None, split_provenance=None):
        e = self.entries[session]
        number = re.search(r'_run-([^_]+)', e['run_folder'])[1]
        serial = len(e['recordings']) + 1
        suffix = 'lfp' if e['task'] == 'CO8LFP' else 'spike'
        filename = (f'sub-keaiduo_ses-{e["date"]}_task-{e["task"]}_run-{number}_mode-{e["mode"]}'
                    f'_rec-{serial:03d}_{kind}_{suffix}.xdf')
        relative = Path(e['session_path']) / 'xdf' / filename
        self.copy(source, relative)
        row = dict(path=relative.as_posix(), segment=kind, source=self.source_name(source),
                   source_sha256=self.sha(source), original_xdf=self.source_name(original_xdf or source))
        if split_provenance:
            row['split_provenance'] = split_provenance
        e['recordings'].append(row)
        self.recordings.append(row)

    def build_week(self):
        # Co-locate repeated copies of the same actual session; JSON/text source
        # evidence remains distinguished by its original recording group.
        for name, source in self.source_sessions.items():
            e = self.entries[name]
            for location in source['locations']:
                folder = WEEK / location
                group = folder.parents[1].name if 'sessions' in folder.parts else 'supplementary'
                self.session_tree(folder, e, group)
        split_sources = {str(Path(r['source']).resolve()) for r in self.split_rows}
        for row in self.split_rows:
            original = Path(row['source']).resolve()
            file = SPLIT / row['output']
            kind = 'startup-split' if self.entries[row['session']]['basic_information'].get(
                'completed_behavior_trial_rows') == 0 else 'early-split' if row['session'].endswith('20261111-113117') else 'split-session'
            self.add_recording(file, row['session'], kind, original_xdf=original, split_provenance=row)
        for day in DAYS[1:]:
            source_day = WEEK / day[-4:]
            for group in source_day.iterdir():
                if not group.is_dir() or not (group / 'recording_index.json').exists():
                    continue
                index = read_json(group / 'recording_index.json')
                for xdf in index['xdfs']:
                    file = WEEK / xdf['package_path']
                    if str(file.resolve()) in split_sources:
                        self.retain(file, 'Replaced in active tree by separately verified per-session derived XDFs; original retained unchanged')
                    else:
                        if len(xdf['sessions']) != 1:
                            raise ValueError(f'Unsplit multi-session XDF: {file}')
                        self.add_recording(file, xdf['sessions'][0], 'main')
                for file in group.rglob('*'):
                    if not file.is_file() or file.relative_to(group).parts[0] in ('xdf', 'sessions'):
                        continue
                    self.copy(file, Path(day) / 'archive_metadata' / 'recording_groups' / group.name / file.relative_to(group))
            for key in ('shared_logs', 'training_and_deployment'):
                self.tree(source_day / key, Path(day) / key)
            supp = source_day / 'supplementary'
            for file in supp.rglob('*'):
                if file.is_file() and file.relative_to(supp).parts[0] != 'task_sessions':
                    self.copy(file, Path(day) / 'supplementary' / file.relative_to(supp))
        # Cross-day originals belong once in shared_metadata, not in a random day.
        for file in WEEK.rglob('*'):
            if file.is_file() and file.relative_to(WEEK).parts[0] not in ('1109', '1110', '1111'):
                self.copy(file, COMMON / 'source_global' / file.relative_to(WEEK))

    def build_old(self):
        for row in self.old_index['primary_runs']:
            e = self.entries[row['session']]
            source = OLD / row['directory']
            self.session_tree(source, e, source.name)
            for file in sorted((source / 'xdf').rglob('*.xdf'), key=lambda p: ('old1' not in p.stem, p.name)):
                self.add_recording(file, e['session'], 'early' if 'old1' in file.stem else 'main')
        for file in (OLD / 'test').rglob('*.xdf'):
            e = next(e for e in self.entries.values() if e['date'] == '20261106' and e['role'] == 'test')
            self.session_tree(file.parent, e, 'morning_test')
            self.add_recording(file, e['session'], 'test')
            for ini in (OLD / 'test').glob('*.ini'):
                self.copy(ini, Path(e['session_path']) / 'acquisition' / ini.name)
        for row in self.old_index['supplementary_task_sessions']:
            self.session_tree(OLD / row['directory'], self.entries[row['session']], 'supplementary')
        for file in OLD.rglob('*'):
            if not file.is_file():
                continue
            rel = file.relative_to(OLD)
            if rel.parts[0].startswith('CO8_run-') or rel.parts[0] == 'test':
                continue
            if rel.parts[:2] == ('supplementary', 'task_sessions'):
                continue
            if rel.parts[:2] == ('shared_config', 'acquisition') and file.suffix.lower() == '.log':
                target = Path('20261106/shared_logs/acquisition/neuralinsight') / file.name
            elif len(rel.parts) == 1:
                target = Path('20261106/archive_metadata/source_original') / rel
            else:
                target = Path('20261106') / rel
            self.copy(file, target)

    def text_for(self, e):
        lines = [f'{e["date"]} / {e["run_folder"] or "supplementary"}', f'原始任务 session：{e["session"]}',
                 f'控制与条件：{e["condition"]}', f'任务：{e["task"]}；控制方式：{e["mode"]}',
                 f'材料位置：{e["session_path"]}', '']
        basic = e['basic_information']
        for title, key in (('任务参数', 'task_parameters'), ('速度/辅助参数', 'speed_parameters'),
                           ('refit 参数', 'refit_parameters')):
            if basic.get(key):
                lines.extend([title + '：', json.dumps(basic[key], ensure_ascii=False, indent=2), ''])
        for key in ('initial_weight_id', 'final_weight_id', 'weight_transition', 'execution_status',
                    'compare_to_previous_retained_session', 'changes_vs_previous_retained_session',
                    'compare_to_previous_xdf_session', 'changes_vs_previous_xdf_session', 'data_coverage_note'):
            if basic.get(key) is not None:
                lines.append(f'{key}：{basic[key]}')
        old = basic.get('archive_evidence', {})
        if old:
            lines.extend(['1106 归档证据：', json.dumps(old, ensure_ascii=False, indent=2)])
        lines.extend(['', 'XDF 文件：'])
        lines.extend(f'{r["segment"]}：{Path(r["path"]).name}' for r in e['recordings'])
        lines.extend(['', '本 run 只对应上述一个 session。同一 session 的 early/main 分段不拼接、不删除。',
            '行为 CSV 和 refit 材料可覆盖完整会话，XDF 可能只覆盖其中一段；不能按 CSV 试次数推定 XDF 覆盖。',
            'split-session 表示来自多会话 XDF 的已验证拆分；原始大文件在原包中保留，session 间空闲数据未丢失。',
            '原始配置、日志和权重内容未改写。source_notes/source_metadata 内保留历史说明及原路径；以 session_info.json 的新路径为准。'])
        return '\n'.join(lines) + '\n'

    def daily_source_text(self, filename, day):
        text = (WEEK / filename).read_text(encoding='utf-8-sig')
        headings = list(re.finditer(r'^【(1109|1110|1111)[^\n]*】', text, re.M))
        blocks = [text[m.start():headings[i + 1].start() if i + 1 < len(headings) else len(text)]
                  for i, m in enumerate(headings) if m[1] == day[-4:]]
        if not blocks:
            raise ValueError(f'No day sections in {filename}: {day}')
        common = ''
        if filename == '实验设计说明.txt':
            common = text[text.index('参数解释与共同设置'):text.index('关键调整时间线')]
            common += '本日关键调整时间线\n\n' + '\n\n'.join(
                f'{x["when"]} {x["recording"]}\n{x["change"]}' for x in self.design['key_change_timeline']
                if x['when'].startswith(day[-4:])) + '\n\n'
        return (f'{day} {filename.removesuffix(".txt")}\n\n'
                '按原整理包的对应日期段拆出，比较对象涉及其他日期时保留其上下文。\n'
                '目录已统一命名；下文原 run 名用于对应历史实验记录，新旧路径见 session_index.json 和 file_manifest.json。\n\n'
                + common + '\n'.join(blocks))

    def documents(self):
        for e in self.entries.values():
            base = self.stage / e['session_path']
            base.mkdir(parents=True, exist_ok=True)
            initial = list((base / 'initial_weights').rglob('*.pkl'))
            if len(initial) == 1 and initial[0].name != 'INITIAL_position_velocity_kf.pkl':
                # Keep original evidence file and add the GUI's documented alias.
                alias = base / 'initial_weights' / 'INITIAL_position_velocity_kf.pkl'
                shutil.copy2(initial[0], alias)
                if digest(alias) != digest(initial[0]):
                    raise ValueError('Initial-weight alias hash mismatch')
            write_json(base / 'session_info.json', e)
            text = self.text_for(e)
            for filename in ('实验内容说明.txt', '实验设计说明.txt'):
                (base / filename).write_text(text, encoding='utf-8')
            if e['run_folder']:
                (base.parents[1] / '实验内容说明.txt').write_text(text, encoding='utf-8')
                write_json(base.parents[1] / 'run_info.json', dict(format='keaiduo-run-v1',
                           run=e['run_folder'], session=e['session'], session_count=1,
                           session_path=str(Path(e['session_path']).relative_to(Path(e['date']) / e['run_folder'])),
                           condition=e['condition']))
        for day in DAYS:
            entries = [e for e in self.entries.values() if e['date'] == day]
            folder = self.stage / day
            write_json(folder / 'session_index.json', dict(date=day, session_count=len(entries),
                recorded_session_count=sum(bool(e['recordings']) for e in entries),
                xdf_count=sum(len(e['recordings']) for e in entries), sessions=entries))
            text = '\n'.join([day + ' 每日整理说明', '',
                '结构：日期 / 任务_run-编号_online或offline_关键信息 / sessions / session-日期-时间。',
                '一个 run 只对应一个 session；同一 session 的多个 XDF 用 rec-001/002 与 early/main 区分。',
                'Grid run 按当天实际任务顺序统一编号；原编号、原任务名保存在 session_info.json。',
                '补充任务保留在 supplementary/sessions，不推定为正式实验；同日公用材料放 shared_config/shared_logs/training_and_deployment。',
                '共享跨天原始索引和会话总表只保留一份：../' + COMMON.as_posix(), '',
                *[f'{e["run_folder"] or "supplementary"} / session-{stamp(e["session"])} / {e["condition"]} / XDF={len(e["recordings"])}'
                  for e in sorted(entries, key=lambda x: stamp(x['session']))]]) + '\n'
            (folder / '整理说明.txt').write_text(text, encoding='utf-8')
            if day != '20261106':
                for filename in ('实验内容说明.txt', '实验设计说明.txt', '会话与参数变化说明.txt'):
                    (folder / filename).write_text(self.daily_source_text(filename, day), encoding='utf-8')
                write_json(folder / 'experiment_design.json', dict(scope={'date': day, 'sessions': len(entries)},
                    parameter_definitions=self.design['parameter_definitions'],
                    key_change_timeline=[x for x in self.design['key_change_timeline'] if x['when'].startswith(day[-4:])],
                    sessions=[s for s in self.design['sessions'] if s['day'] == day[-4:]]))
                write_json(folder / 'refit_index.json', [s for s in read_json(WEEK / 'refit_index.json')
                    if s['session_id'].startswith(day)])
            else:
                original_text = (OLD / '实验设计说明.txt').read_text(encoding='utf-8-sig')
                for filename in ('实验内容说明.txt', '实验设计说明.txt', '会话与参数变化说明.txt'):
                    (folder / filename).write_text(text + '\n1106 原实验设计与归档说明\n\n' + original_text,
                                                   encoding='utf-8')

    def check_accounting(self):
        missing = [row['source'] for row in self.inventory.values()
                   if not row['destinations'] and not row['retained_at_source']]
        if missing:
            raise ValueError('Unaccounted original files: ' + '\n'.join(missing[:30]))
        for day in DAYS:
            rows = [row for row in self.inventory.values()
                    if any(p.startswith(day + '/') for p in row['destinations'])]
            write_json(self.stage / day / 'file_manifest.json', rows)
        write_json(self.stage / 'source_file_accounting.json', list(self.inventory.values()))

    def validate(self):
        checks = []
        for day in DAYS:
            for run in (self.stage / day).iterdir():
                if run.is_dir() and '_run-' in run.name:
                    sessions = list((run / 'sessions').iterdir())
                    if len(sessions) != 1:
                        raise ValueError(f'Run must have exactly one session: {run}')
            recordings, notice = scan(self.stage / day)
            if notice or len(recordings) != sum(len(e['recordings']) for e in self.entries.values() if e['date'] == day):
                raise ValueError('Unexpected recording scan result')
            for r in recordings:
                ids = {h.findtext('desc/session_name') for h in read_headers(r.path)
                       if h.findtext('name') in ('cursortasks_events', 'cursortasks_cursor_state')}
                metadata = read_json(r.path.parent.parent / 'session_info.json')
                if ids != {metadata['session']} or not r.date:
                    raise ValueError(f'Session or date mismatch: {r.path}')
                cfg = read_channel_config(r.channel_config)
                row = dict(path=r.path.relative_to(self.stage).as_posix(), session=metadata['session'],
                           date=r.date, condition=r.condition, config=Path(r.channel_config).relative_to(self.stage).as_posix(),
                           config_good=len(cfg['good']))
                if metadata['task'] == 'CO8LFP':
                    if not r.reason.startswith('缺少数据流：'):
                        raise ValueError('Expected LFP-only recording to be flagged by spike analysis')
                    row['status'] = 'LFP-only; preserved, excluded from spike sections'
                else:
                    if r.reason:
                        raise ValueError(r.reason)
                    session = Session.load(r)
                    row.update(status='loaded', frames=len(session.counts), trials=len(session.trials),
                               complete_trials=sum(t['complete'] for t in session.trials))
                    if r.online:
                        models = manifest(r, {})
                        row['model_weights'] = len(models['weights']) if models else 0
                    del session
                checks.append(row)
            print('VERIFIED', day, 'runs=',sum(e['date'] == day and bool(e['run_folder']) for e in self.entries.values()),
                  'xdf=',len(recordings), flush=True)
        write_json(self.stage / 'validation.json', checks)
        return checks

    def publish(self, checks):
        # Exact, prevalidated directory targets; no recursive deletion or moving
        # computed trees outside the explicitly named data/archive directories.
        if OLD.resolve() != (DATA / 'keaiduo' / '20261106').resolve():
            raise ValueError('Unexpected original-day path')
        if self.backup.parent.resolve() != (DATA / 'keaiduo_archive').resolve():
            raise ValueError('Unexpected backup parent')
        for day in DAYS:
            if (self.stage / day).resolve().parent != self.stage.resolve():
                raise ValueError('Unexpected staged-day target')
            if day != '20261106' and (ROOT / day).exists():
                raise FileExistsError(ROOT / day)
        self.backup.parent.mkdir(parents=True, exist_ok=True)
        inplace = False
        try:
            OLD.rename(self.backup)
        except PermissionError:
            # A Windows process can hold the day directory itself as its cwd.
            # Keep that directory node, archive its verified children instead.
            self.backup.mkdir()
            moved = []
            try:
                for child in list(OLD.iterdir()):
                    target = self.backup / child.name
                    if child.resolve().parent != OLD.resolve() or target.resolve().parent != self.backup.resolve():
                        raise ValueError('Unexpected in-place archival target')
                    child.rename(target)
                    moved.append(target)
            except Exception:
                for target in reversed(moved):
                    target.rename(OLD / target.name)
                self.backup.rmdir()
                raise
            inplace = True
        installed = []
        inplace_installed = []
        try:
            for day in DAYS:
                if day == '20261106' and inplace:
                    staged_day = self.stage / day
                    for child in list(staged_day.iterdir()):
                        target = OLD / child.name
                        if child.resolve().parent != staged_day.resolve() or target.resolve().parent != OLD.resolve():
                            raise ValueError('Unexpected in-place installation target')
                        child.rename(target)
                        inplace_installed.append(target)
                else:
                    (self.stage / day).rename(ROOT / day)
                installed.append(day)
            (ROOT / COMMON).parent.mkdir(parents=True, exist_ok=True)
            (self.stage / COMMON).rename(ROOT / COMMON)
        except Exception:
            for day in reversed(installed):
                if day != '20261106' or not inplace:
                    (ROOT / day).rename(self.stage / day)
            if inplace:
                for target in reversed(inplace_installed):
                    target.rename(self.stage / '20261106' / target.name)
                for child in list(self.backup.iterdir()):
                    child.rename(OLD / child.name)
                self.backup.rmdir()
            else:
                self.backup.rename(OLD)
            raise
        audit = ROOT / f'normalization_audit_{self.token}'
        audit.mkdir()
        for name in ('normalization_plan.json', 'source_file_accounting.json', 'validation.json'):
            (self.stage / name).rename(audit / name)
        prefs = Path(__file__).resolve().parents[1] / '.local' / 'preferences.json'
        if prefs.exists():
            shutil.copy2(prefs, audit / 'preferences_before.json')
            data = read_json(prefs)
            mapping = {}
            for e in self.entries.values():
                for row in e['recordings']:
                    src = Path(row['source'])
                    if src.is_relative_to(self.backup):
                        mapping[(Path('20261106') / src.relative_to(self.backup)).as_posix()] = row['path']
            if Path(data.get('data_root', '')).resolve() == ROOT.resolve():
                single = data.get('single', {})
                if single.get('recording') in mapping:
                    single['recording'] = mapping[single['recording']]
                single['recordings'] = [mapping.get(p, p) for p in single.get('recordings', [])]
                cross = data.get('cross', {})
                cross['recordings'] = [mapping.get(p, p) for p in cross.get('recordings', [])]
                write_json(prefs, data)
        for item in self.inventory.values():
            if digest(item['source']) != item['sha256']:
                raise ValueError(f'Original source hash changed: {item["source"]}')
            for output in item['destinations']:
                if digest(ROOT / output) != item['sha256']:
                    raise ValueError(f'Final published hash mismatch: {output}')
        for row in self.recordings:
            if digest(ROOT / row['path']) != row['source_sha256']:
                raise ValueError(f'Published XDF differs from verified source: {row["path"]}')
        report = dict(status='verified', original_1106_backup=str(self.backup),
                      weekly_source_unchanged=str(WEEK), split_source_unchanged=str(SPLIT),
                      source_file_count=len(self.inventory), session_count=len(self.entries),
                      xdf_count=len(self.recordings), one_session_per_run=True, days=list(DAYS),
                      audit=str(audit), validation=checks)
        write_json(audit / 'completion.json', report)
        (ROOT / '归档命名与目录说明.txt').write_text(
            '统一归档：20261106、20261109、20261110、20261111\n\n'
            '日期/任务_run-编号_online或offline_关键信息/sessions/session-日期-时间/\n'
            '一个 run 严格对应一个 session。session 内按 xdf、behavior、config、logs、initial_weights、refit 分类。\n'
            'CO8=八方向；Grid 的 6x6/8x8/10x10/12x12/7x7 为网格规模；CO8LFP 为仅有 LFP 的手控 CO8。\n'
            'split-session=来自原多会话 XDF 的拆分；同 session 的 early/main 片段分别保存，不拼接。\n'
            'run 编号按当天同类任务的正式 session 时间顺序统一；历史编号保存在 session_info.json。\n'
            '1106 assist-200/100 表示初始目标辅助速度，可随 block 衰减；1109–1111 不据配置推断手是否束缚。\n'
            '补充会话不强行并入正式 run；公用材料放 shared_config、shared_logs、training_and_deployment。\n'
            '每日报告和 session_index.json 为当前目录索引；原有脚本/JSON 中的历史绝对路径原样保留作证据。\n'
            '共享跨天原始说明、索引、会话总表存放 shared_metadata/package_20261109-20261111。\n'
            f'原 1106 完整目录备份：{self.backup}\n原三天整理包完整保留：{WEEK}\n'
            f'逐文件新旧路径与 SHA-256 核对：{audit}\n', encoding='utf-8')
        print('PUBLISHED', json.dumps({k:v for k,v in report.items() if k != 'validation'}, ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-stage', type=Path)
    parser.add_argument('--publish-stage', type=Path)
    args = parser.parse_args()
    if args.publish_stage:
        stage = args.publish_stage.resolve()
        if stage.parent != DATA.resolve() or not stage.name.startswith('keaiduo_normalize_stage_'):
            raise ValueError('Unexpected publish staging path')
        plan = read_json(stage / 'normalization_plan.json')
        organizer = Organizer.__new__(Organizer)
        organizer.stage = stage
        organizer.backup = Path(plan['backup'])
        organizer.token = organizer.backup.name.removeprefix('20261106_before_normalization_')
        if not re.fullmatch(r'\d{8}_\d{6}', organizer.token) or organizer.backup.exists():
            raise ValueError('Invalid or existing archival target')
        rows = read_json(stage / 'source_file_accounting.json')
        organizer.inventory = {r['source']: r for r in rows}
        organizer.entries = {e['session']: e for day in DAYS
                             for e in read_json(stage / day / 'session_index.json')['sessions']}
        organizer.recordings = [r for e in organizer.entries.values() for r in e['recordings']]
        checks = read_json(stage / 'validation.json')
        if len(checks) != len(organizer.recordings):
            raise ValueError('Staging validation is incomplete')
        for row in rows:
            file = Path(row['source'])
            if file.is_relative_to(organizer.backup):
                file = OLD / file.relative_to(organizer.backup)
            if digest(file) != row['sha256']:
                raise ValueError('Original source changed since validation')
            for relative in row['destinations']:
                if digest(stage / relative) != row['sha256']:
                    raise ValueError('Staged evidence changed since validation')
        for row in organizer.recordings:
            if digest(stage / row['path']) != row['source_sha256']:
                raise ValueError('Staged XDF changed since validation')
        organizer.publish(checks)
        return
    organizer = Organizer(args.resume_stage)
    print('STAGE', organizer.stage, flush=True)
    organizer.plan()
    organizer.build_week()
    organizer.build_old()
    organizer.documents()
    organizer.check_accounting()
    checks = organizer.validate()
    organizer.publish(checks)


if __name__ == '__main__':
    main()
