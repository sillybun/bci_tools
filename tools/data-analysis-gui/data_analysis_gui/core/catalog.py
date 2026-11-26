"""Discover recordings under a data root using XDF headers only."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from ..config import DECODER_STREAM, REQUIRED_STREAMS
from .archive_logs import archived_logs
from .archive_names import archived_recording_alias
from .xdf_io import read_headers, spike_stream_aliases

DATA_TYPES = ('CO8', 'Grid', 'CO8 online', 'Grid online')
TASK_OF_TYPE = {'CO8': 'centerout', 'Grid': 'grid', 'CO8 online': 'centerout', 'Grid online': 'grid'}
RUN_PREFIX = re.compile(r'^run-(?P<number>\d{3})(?:-(?P<part>[a-z]))?-'
                        r'(?P<task>CO8LFP|CO8|Grid)-(?P<mode>online|offline)(?:-|$)', re.I)


def format_run_name(number, task, mode, notes=(), *, part=None):
    """Canonical run-first archive name; split parts share their parent run ID."""
    notes = tuple('LFP' if note == 'lfp-only' else note
                  for note in notes if note not in ('arm-free', 'no-assist', 'arm-free-no-assist')
                  and not (mode == 'offline' and note == 'manual')
                  and (part is None or note != 'split-session'))
    if not re.fullmatch(r'\d{1,3}', str(number)) or not 1 <= int(number) <= 999:
        raise ValueError('Run number must be 001..999')
    task = {'co8': 'CO8', 'grid': 'Grid', 'co8lfp': 'CO8LFP'}.get(str(task).lower())
    if task is None or mode not in ('online', 'offline'):
        raise ValueError('Unsupported task or control mode')
    if part is not None and not re.fullmatch(r'[a-z]', part):
        raise ValueError('Split part must be one lowercase letter')
    if any(not re.fullmatch(r'[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*', note) for note in notes):
        raise ValueError('Run notes must be safe hyphen-separated labels')
    tokens = ['run', f'{int(number):03d}'] + ([part] if part else []) + [task, mode, *notes]
    return '-'.join(tokens)


@dataclass(frozen=True)
class Recording:
    path: Path
    relative: str
    date: str                 # yyyy-mm-dd from the nearest dated folder
    task_mode: str            # centerout | grid | unknown
    online: bool
    run: str
    reason: str = ''          # non-empty: unusable, with explanation
    has_fpga_clock: bool = False
    header_info: dict = field(default_factory=dict, compare=False, hash=False)
    condition: str = ''       # explicit experiment condition from the archive folder

    @property
    def key(self) -> str:
        return self.relative

    @property
    def data_type(self) -> str:
        base = {'centerout': 'CO8', 'grid': 'Grid'}.get(self.task_mode, '?')
        return f'{base} online' if self.online else base

    @property
    def channel_config(self) -> str:
        return day_config_path(self.path)

    @property
    def run_name(self) -> str:
        folder = run_folder(self.path)
        if folder is None:
            return f'run-{self.run}'
        match = RUN_PREFIX.match(folder.name)
        if match:
            from .run_config import configured_run_name
            return configured_run_name(folder, match['number'], match['task'], match['mode'],
                                       part=match['part'])
        return folder.name

    @property
    def label(self) -> str:
        suffix = f' · {self.condition}' if self.condition else ''
        return f'{self.date} · {self.data_type} · {self.run_name}{suffix}'


def run_folder(path: Path) -> Path | None:
    """Archive run ancestor, including recordings nested in early_segment_old1."""
    return next((p for p in Path(path).parents
                 if RUN_PREFIX.match(p.name) or re.fullmatch(r'(?:CO8|Grid)_run-[^_]+_.+', p.name, re.I)), None)


def experiment_condition(path: Path) -> str:
    metadata = session_metadata(path)
    if isinstance(metadata.get('condition'), str):
        return metadata['condition']
    folder = run_folder(path)
    if folder is None:
        return ''
    name = folder.name.lower()
    if name.endswith('_manual'):
        return '手控 / 离线'
    if name.endswith('_arm_free_bci'):
        return '自由手脑控 / 无辅助'
    match = re.search(r'_restrained_bci_assist(\d+(?:\.\d+)?)$', name)
    return f'手束缚脑控 / 初始辅助 {match[1]} px/s' if match else ''


def session_metadata(path: Path) -> dict:
    """Explicit annotations in the normalized one-session-per-run archive."""
    path = Path(path)
    if path.parent.name.lower() != 'xdf':
        return {}
    try:
        data = json.loads((path.parent.parent / 'session_info.json').read_text(encoding='utf-8-sig'))
        return data if isinstance(data, dict) and data.get('format') == 'keaiduo-session-v1' else {}
    except (OSError, ValueError):
        return {}


def day_folder(path: Path) -> Path | None:
    """Nearest dated ancestor, including MMDD folders in multi-day packages."""
    for folder in Path(path).parents:
        if re.fullmatch(r'\d{8}|\d{4}-\d{2}-\d{2}', folder.name):
            return folder
        if re.fullmatch(r'\d{4}', folder.name):
            try:
                datetime.strptime('2000' + folder.name, '%Y%m%d')
                return folder
            except ValueError:
                pass
    return None


def day_config_path(path) -> str:
    """Prefer a day-local INI, otherwise a day-local Neuralinsight log."""
    folder = day_folder(Path(path))
    if folder is None:
        return ''
    date = folder.name.replace('-', '')
    locations = ((Path(path).parent.parent / 'acquisition',) if Path(path).parent.name.lower() == 'xdf' else ())
    locations += (folder, folder / 'shared_config' / 'acquisition',
                 folder / 'shared_logs' / 'acquisition' / 'neuralinsight')
    for location in locations:
        for name in ('configs.ini', 'config.ini', f'configs{date[-4:]}.ini', f'configs{date}.ini'):
            candidate = location / name
            if candidate.is_file():
                return str(candidate)
        candidates = [p for p in location.glob('config*.ini') if p.is_file()]
        if len(candidates) == 1:
            return str(candidates[0])
    for location in locations:
        standard = archived_logs(location, 'Neuralinsight.log')
        if len(standard) == 1:
            return str(standard[0])
        candidates = [p for p in location.glob('Neuralinsight_*.log') if p.is_file()
                      and re.fullmatch(r'Neuralinsight_\d{8}', p.stem, re.I)
                      and (p.stem[-8:] == date if len(date) == 8 else p.stem[-4:] == date)]
        if len(candidates) == 1:
            return str(candidates[0])
    return ''


def recording_date(path: Path) -> str:
    for folder in Path(path).parents:
        if re.fullmatch(r'\d{8}|\d{4}-\d{2}-\d{2}', folder.name):
            try:
                return datetime.strptime(folder.name.replace('-', ''), '%Y%m%d').strftime('%Y-%m-%d')
            except ValueError:
                continue
    return ''


def _run_label(path: Path) -> str:
    for name in [p.name for p in path.parents] + [path.stem]:
        if match := RUN_PREFIX.match(name):
            return match['number'] + (f'-{match["part"].lower()}' if match['part'] else '')
    match = re.search(r'_run-([^_]+)', path.stem)
    return match[1] if match else path.stem


def describe(path: Path, root: Path) -> Recording:
    date = recording_date(path)
    owner = run_folder(path)
    canonical = RUN_PREFIX.match(owner.name) if owner else None
    alias = archived_recording_alias(path)
    run = (alias.get('previous_run') if not canonical else None) or _run_label(path)
    relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
    name_online = ('online' in path.stem.lower() or 'online' in path.parent.name.lower() or
                   (canonical is not None and canonical['mode'].lower() == 'online') or
                   (owner is not None and canonical is None and
                    bool(re.search(r'(?:^|[-_])online(?:[-_]|$)', owner.name, re.I))) or
                   bool(alias.get('previous_online')))
    try:
        headers = read_headers(path)
    except (OSError, ValueError, ET.ParseError) as exc:
        return Recording(path, relative, date, 'unknown', name_online, run, f'无法读取 XDF：{exc}')
    names = {h.findtext('name') for h in headers}
    aliases = spike_stream_aliases(names)
    names.update(aliases.values())
    modes = {h.findtext('desc/task_mode') for h in headers
             if h.findtext('name') in ('cursortasks_events', 'cursortasks_cursor_state')}
    modes.discard(None)
    task_mode = next(iter(modes)) if len(modes) == 1 else 'unknown'
    online = DECODER_STREAM in names or name_online
    reason = ''
    missing = [n for n in REQUIRED_STREAMS if n not in names]
    if missing:
        reason = '缺少数据流：' + ', '.join(missing)
    elif task_mode not in ('centerout', 'grid'):
        reason = f'未支持或不一致的任务类型：{sorted(modes) or "缺失"}'
    info, fpga = {}, False
    for h in headers:
        source = h.findtext('name')
        if source in aliases.values():
            continue
        name = aliases.get(source, source)
        if name == 'spike_counts':
            info['channel_count'] = h.findtext('channel_count')
            info['window_ms'] = h.findtext('desc/window_ms')
            if not reason and h.findtext('channel_count') != '512':
                reason = '当前分析需要 512 通道计数流'
        if name == 'spike_counts_metadata':
            labels = [c.findtext('label') for c in h.findall('desc/channels/channel')]
            fpga = 'device_timestamp_100us' in labels
    metadata = session_metadata(path)
    if isinstance(metadata.get('condition_tag'), str):
        info['condition_tag'] = metadata['condition_tag']
    if metadata and metadata.get('analysis_window_policy') in (
            None, 'through_task_end', 'through_task_end_or_first_25_min'):
        info['analysis_window_policy'] = metadata.get('analysis_window_policy')
    return Recording(path, relative, date, task_mode, online, run, reason, fpga, info,
                     experiment_condition(path))


def scan(root) -> tuple[list[Recording], str]:
    """All .xdf files under root, sorted by date then path. Never raises for bad files."""
    try:
        root = Path(root).expanduser().resolve()
    except (OSError, ValueError):
        return [], '数据目录无效'
    if not root.is_dir():
        return [], f'数据目录不存在：{root}'
    paths, errors = [], []
    for folder, dirs, files in os.walk(root, onerror=lambda e: errors.append(str(e))):
        dirs.sort()
        paths += [Path(folder) / f for f in sorted(files) if f.lower().endswith('.xdf')]
    recordings = [describe(p, root) for p in paths]
    recordings.sort(key=lambda r: (r.date, r.relative))
    notice = f'{len(errors)} 个目录无法访问，已跳过' if errors else ''
    return recordings, notice


def filter_type(recordings, data_types) -> list[Recording]:
    wanted = set(data_types)
    return [r for r in recordings if not r.reason and r.data_type in wanted]
