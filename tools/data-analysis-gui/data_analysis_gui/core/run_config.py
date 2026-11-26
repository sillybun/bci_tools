"""Shared naming fields from the saved CursorTasks configuration."""
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import json
import re


def _fraction(value):
    try:
        number = Decimal(str(value))
        if number.is_finite() and 0 <= number < 1:
            return format(number.normalize(), 'f').partition('.')[2] or '0'
    except InvalidOperation:
        pass
    return ''


def naming_config(config):
    grid = config.get('GRID_TASK', {})
    control = config.get('CONTROL', config.get('control', config))
    control = {key.upper(): value for key, value in control.items()}
    timing = config.get('GENERAL_TIMING', {})
    return dict(grid_rows=grid.get('GRID_ROWS'), grid_cols=grid.get('GRID_COLS'),
                joystick_gain=control.get('POS_SCALE_FACTOR'),
                nearby_assist=grid.get('NEARBY_TARGET_ASSIST_ENABLED'),
                nearby_radius=grid.get('NEARBY_TARGET_ASSIST_RADIUS'),
                nearby_probability=grid.get('NEARBY_TARGET_ASSIST_PROBABILITY'),
                dwell_s=timing.get('DWELL_TIME_SEC'),
                preview_next_target=grid.get('PREVIEW_NEXT_TARGET'),
                bci_velocity_scale=control.get('BCI_VELOCITY_SCALE'),
                bci_target_assist_enabled=control.get('BCI_TARGET_ASSIST_ENABLED'),
                bci_target_assist_speed=control.get('BCI_TARGET_ASSIST_SPEED'))


def config_tags(config, task, online):
    fields = naming_config(config)
    tags = []
    if task.lower() == 'grid' and fields['grid_rows'] and fields['grid_cols']:
        tags.append(f"{fields['grid_rows']}x{fields['grid_cols']}")
    if speed := _fraction(fields['joystick_gain']):
        tags.append(f'sp{speed}')
    if online and fields['bci_velocity_scale'] is not None:
        try:
            scale = Decimal(str(fields['bci_velocity_scale']))
            if scale.is_finite() and scale > 0 and scale != 1 and scale * 100 == int(scale * 100):
                tags.append(f'bc{int(scale * 100)}')
        except InvalidOperation:
            pass
    try:
        dwell = Decimal(str(fields['dwell_s']))
        centiseconds = dwell * 100
        if dwell.is_finite() and dwell >= 0 and centiseconds == int(centiseconds):
            tags.append(f'dw{int(centiseconds):03d}')
    except InvalidOperation:
        pass
    if task.lower() == 'grid' and fields['preview_next_target'] is True:
        tags.append('nextT')
    if task.lower() == 'grid' and fields['nearby_assist'] is True and fields['nearby_radius'] in (1, 2, 3):
        tags.append(f"nearby{fields['nearby_radius']}")
    if fields['bci_target_assist_enabled'] is True:
        try:
            speed = Decimal(str(fields['bci_target_assist_speed']))
            if speed.is_finite() and speed >= 0:
                tags.append('ass' + format(speed.normalize(), 'f').replace('.', 'p'))
        except InvalidOperation:
            pass
    return tags


@lru_cache(maxsize=512)
def _read_config(path, mtime_ns, size):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def configured_run_name(folder, number, task, mode, *, part=None):
    from .catalog import RUN_PREFIX, format_run_name
    cfg = next((p for p in (folder / 'task_config.json', folder / 'config/task_config.json') if p.is_file()), None)
    try:
        config = {}
        if cfg is not None:
            stat = cfg.stat()
            config = _read_config(cfg, stat.st_mtime_ns, stat.st_size)
        match = RUN_PREFIX.match(folder.name)
        tail = folder.name[match.end():] if match else ''
        tail = re.sub(r'(?i)(?<![a-z0-9])assist-(\d+)(?![a-z0-9])', r'ass\1', tail)
        if config:
            tail = re.sub(r'(?i)(?<![a-z0-9])nearby-assist(?![a-z0-9])', '', tail)
        patterns = (r'\d+x\d+', r'sp\d+', r'bc\d+', r'dw\d+', r'nextT', r'nearby\d+', r'ass\d+(?:p\d+)?')

        def tag_rank(tag):
            return next((i for i, pattern in enumerate(patterns) if re.fullmatch(pattern, tag, re.I)), len(patterns))

        tags = config_tags(config, task, mode == 'online')
        notes = []
        assist_known = naming_config(config)['bci_target_assist_enabled'] is not None
        for note in filter(None, tail.split('-')):
            rank = tag_rank(note)
            if rank < len(patterns):
                if not config or (rank == len(patterns)-1 and not assist_known):
                    tags.append(note)
            elif not config or not re.fullmatch(r'\d+p\d+dwell', note, re.I):
                notes.append(note)
        tags = sorted(dict.fromkeys(tags), key=tag_rank)
        base = format_run_name(number, task, mode, tags, part=part)
        return '-'.join([base, *notes])
    except (OSError, ValueError, TypeError, AttributeError):
        return folder.name


def suggest_run_name(folder):
    """Suggest a canonical name for an existing run or dated CursorTasks folder."""
    from .catalog import RUN_PREFIX, format_run_name

    if match := RUN_PREFIX.match(folder.name):
        return configured_run_name(folder, match['number'], match['task'], match['mode'], part=match['part'])
    if not re.fullmatch(r'\d{8}', folder.parent.name):
        return folder.name

    def config_path(candidate):
        return next((p for p in (candidate / 'task_config.json', candidate / 'config/task_config.json')
                     if p.is_file()), None)

    try:
        path = config_path(folder)
        if path is None:
            return folder.name
        stat = path.stat()
        config = _read_config(path, stat.st_mtime_ns, stat.st_size)
        meta_path = folder / 'session.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8-sig')) if meta_path.is_file() else {}
        task = {'grid': 'Grid', 'centerout': 'CO8'}.get(meta.get('task_mode'))
        if task is None:
            task = next((name for prefix, name in (('grid-task', 'Grid'), ('center-out', 'CO8'))
                         if folder.name.lower().startswith(prefix)), None)
        control = config.get('CONTROL', {})
        if task is None or not isinstance(control.get('BCI_ENABLED'), bool):
            return folder.name
        siblings = sorted(p for p in folder.parent.iterdir() if p.is_dir())
        reserved = {int(match['number']) for item in siblings if (match := RUN_PREFIX.match(item.name))}
        unnumbered = [item for item in siblings if not RUN_PREFIX.match(item.name) and
                      (item == folder or (item / 'session.json').is_file() or config_path(item))]
        number = list(i for i in range(1, 1000) if i not in reserved)[unnumbered.index(folder)]
        online = control['BCI_ENABLED']
        return format_run_name(number, task, 'online' if online else 'offline', config_tags(config, task, online))
    except (OSError, ValueError, TypeError, AttributeError, IndexError):
        return folder.name
