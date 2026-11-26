"""Write a read-only migration plan and configuration history for a keaiduo archive."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis_gui.control_comparison import flatten
from data_analysis_gui.core.catalog import RUN_PREFIX, format_run_name, scan
from data_analysis_gui.core.run_config import config_tags, configured_run_name, naming_config
from data_analysis_gui.core.xdf_io import read_headers, spike_stream_aliases


def write_csv(path, rows, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else {}


def identify_signal(path):
    if re.search(r'lfp', str(path), re.IGNORECASE):
        return 'LFP', 'explicit LFP marker retained'
    try:
        names = {h.findtext('name') or '' for h in read_headers(path)}
    except (ValueError, OSError, SyntaxError) as exc:
        return '', f'header unreadable: {exc}'
    if 'spike_counts' in names or 'spike_counts' in spike_stream_aliases(names).values():
        return 'spike', 'XDF spike count stream'
    if any('lfp' in name.lower() for name in names):
        return 'LFP', 'XDF LFP stream'
    return '', 'no identifiable spike or LFP stream; inspect companion role'


def audit(root, out):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f'Data directory not found: {root}')
    out.mkdir(parents=True, exist_ok=True)
    records, notice = scan(root)
    signals = []
    for record in records:
        path = record.path
        signal, evidence = identify_signal(path)
        new_name = re.sub(r'(?i)(?<![a-z0-9])eeg(?![a-z0-9])', signal, path.name) if signal else path.name
        signals.append(dict(source=str(path), proposed_filename=new_name, signal=signal,
                            evidence=evidence, gui_status=record.reason or 'ok'))
    configs, changes, names = [], [], []
    prior = None
    for day in sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r'\d{8}', p.name)):
        candidates = {p for p in day.iterdir() if p.is_dir() and
                      (RUN_PREFIX.match(p.name) or (p / 'session.json').is_file())}
        candidates.update(p.parent.parent if p.parent.name == 'config' else p.parent
                          for p in day.rglob('task_config.json'))
        reserved = {int(m['number']) for p in candidates if (m := RUN_PREFIX.match(p.name))}
        next_number = (i for i in range(1, 1000) if i not in reserved)
        proposed = Counter()
        for folder in sorted(candidates):
            source = next((p for p in (folder / 'task_config.json', folder / 'config/task_config.json')
                           if p.is_file()), None)
            cfg = read_json(source) if source else {}
            meta = read_json(folder / 'session.json')
            control = cfg.get('CONTROL', {})
            match = RUN_PREFIX.match(folder.name)
            affiliated = [r for r in records if r.path.is_relative_to(folder) and not r.reason]
            modes = {r.task_mode for r in affiliated}
            task = match['task'] if match else {'grid': 'Grid', 'centerout': 'CO8'}.get(meta.get('task_mode') or (next(iter(modes)) if len(modes) == 1 else ''))
            online = (bool(control['BCI_ENABLED']) if 'BCI_ENABLED' in control else
                      match['mode'].lower() == 'online' if match else None)
            number = int(match['number']) if match else next(next_number)
            part = match['part'] if match else None
            proposal = ''
            if cfg and task and online is not None:
                mode = 'online' if online else 'offline'
                proposal = (configured_run_name(folder, number, task, mode, part=part) if match else
                            format_run_name(number, task, mode, config_tags(cfg, task, online)))
                proposed[proposal.casefold()] += 1
            fields = flatten(cfg)
            configs.append(dict(date=day.name, folder=str(folder), config_source=str(source or ''),
                                naming=naming_config(cfg), fields=fields))
            if fields and prior:
                previous = prior['fields']
                for key in sorted(set(previous) | set(fields)):
                    if key.startswith('SESSION.') or (key in previous and key in fields and previous[key] == fields[key]):
                        continue
                    changes.append(dict(date=day.name, folder=str(folder), previous_folder=prior['folder'],
                                        field=key, kind='added' if key not in previous else
                                        'missing' if key not in fields else 'changed',
                                        before=previous.get(key), after=fields.get(key)))
            if fields:
                prior = configs[-1]
            names.append(dict(date=day.name, source=str(folder), proposed_run=proposal,
                              status='review' if proposal else 'insufficient config or task identity'))
        for row in names:
            if row['date'] == day.name and proposed[row['proposed_run'].casefold()] > 1:
                row['status'] = 'collision: resolve before rename'
    write_csv(out / 'run_rename_plan.csv', names, ('date', 'source', 'proposed_run', 'status'))
    write_csv(out / 'xdf_rename_plan.csv', signals, ('source', 'proposed_filename', 'signal', 'evidence', 'gui_status'))
    write_csv(out / 'config_changes.csv', changes, ('date', 'folder', 'previous_folder', 'field', 'kind', 'before', 'after'))
    (out / 'config_inventory.json').write_text(json.dumps(dict(scan_notice=notice, configs=configs), ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{len(names)} archive folders, {len(records)} XDFs, {sum(bool(x["fields"]) for x in configs)} configs')
    print(Counter(row['status'] for row in names), Counter(row['signal'] for row in signals), Counter(row['kind'] for row in changes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.data, args.out)
