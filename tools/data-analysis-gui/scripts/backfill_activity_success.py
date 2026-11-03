"""Add a behavioral success-rate axis without recomputing cached neural statistics."""
import csv
from collections import defaultdict
from datetime import datetime
import gc
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis_gui import storage
from data_analysis_gui.core import behavior
from data_analysis_gui.core.catalog import describe, run_folder
from data_analysis_gui.core.session import Session
from data_analysis_gui.core.filters import TrialFilter, apply_filter
from data_analysis_gui.core.style import finish_figure
from data_analysis_gui.reporting import REPORT_ROOT
from data_analysis_gui.sections.x01_channel_stability_v1.section import ChannelStability, tuning_counts


def main():
    root = REPORT_ROOT.resolve()
    audit = root.parent / 'report_edit_backups' / ('activity_success_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    audit.mkdir(parents=True)
    manifests = [(path, path.read_bytes()) for path in sorted(root.glob('*/report.json'))]
    selections = {r['path']: r for _, raw in manifests for r in json.loads(raw)['selection']}
    cached = {}
    for i, path in enumerate(selections, 1):
        rec = describe(Path(path), Path('E:/AitBCI/data/keaiduo'))
        files = behavior.source_files(rec)
        try:
            trials = behavior._from_csv(files, rec.task_mode) if files else None
        except (KeyError, ValueError, OSError, TypeError, csv.Error):
            trials = None
        if trials is not None:
            source = str(files['trials'])
        else:
            session = Session.load(rec)
            trials, source, _ = behavior.collect(session)
            del session
            gc.collect()
        folder = run_folder(rec.path)
        identity = str(folder.resolve()) if folder else (rec.date, rec.task_mode, rec.online, rec.run or rec.key)
        cached[path] = dict(recording=rec, trials=trials, source=source, identity=identity)
        print(f'Behavior {i}/{len(selections)}: {rec.date} {rec.data_type} run-{rec.run}', flush=True)
    staged = []
    tuning_cache = {}
    for manifest_path, raw in manifests:
        manifest = json.loads(raw)
        report = manifest_path.parent
        stage = audit / 'staged' / report.name
        stage.mkdir(parents=True)
        selected = [cached[r['path']] for r in manifest['selection']]
        groups = defaultdict(list)
        for entry in selected:
            groups[entry['identity']].append(entry)
        summaries = {}
        for identity, entries in groups.items():
            csv_entries = [e for e in entries if e['source'] != 'XDF event markers']
            chosen = csv_entries[:1] if csv_entries else entries
            summaries[identity] = dict(behavior.block_success_summary([r for e in chosen for r in e['trials']]),
                                       success_source='; '.join(dict.fromkeys(e['source'] for e in chosen)))
        pngs, all_rows = [], []
        for source in manifest['sources']:
            if source['section'] != 'x01':
                continue
            folder = Path(source['folder'])
            record = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
            rows = storage.read_table(folder / record['tables']['direction_tuning_by_session'], limit=100000)
            inputs = record['inputs']['recordings']
            assert len(rows) == len(inputs)
            for row, item in zip(rows, inputs):
                entry, = [e for e in selected if e['recording'].path.name == item['name']]
                for field in ('active', 'direction_tuned'):
                    row[field] = None if row[field] in ('', 'None', 'nan') else float(row[field])
                row['transmitted'] = int(row['transmitted'])
                if int(row['trials']) < 16:
                    criteria = record['inputs']['criteria']
                    cache_key = (str(entry['recording'].path), storage.fingerprint(criteria))
                    if cache_key not in tuning_cache:
                        neural = Session.load(entry['recording'])
                        cohort = apply_filter(neural, TrialFilter.from_dict(criteria))
                        tuning_cache[cache_key] = tuning_counts(cohort.window_rates, cohort.directions)
                        del cohort, neural
                        gc.collect()
                    row.update(tuning_cache[cache_key])
                    print('Recomputed sparse record: ' + row['session'] + ' ' + str(tuning_cache[cache_key]), flush=True)
                row.update(summaries[entry['identity']])
            fig = ChannelStability.activity(rows)
            finish_figure(fig, manifest['font_percent'])
            relative, = [p for p in source['pngs'] if p.endswith('_Electrode_Activity_and_Direction_Tuning.png')]
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, dpi=storage.PNG_DPI, facecolor='white')
            fig.clear()
            table = target.with_name(target.stem + '_with_success.csv')
            with table.open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
                writer.writeheader()
                writer.writerows(rows)
            pngs.append(relative)
            all_rows.extend(rows)
        manifest['activity_success_overlay'] = dict(
            method='Right axis: overall average success percentage per run (successes / known attempts), with block Q1–Q3 drawn independently. '
                   'Success / known attempts, including timeouts/failures; independent of neural trial filters. '
                   'Complete CSV preferred over early fragments of the same run; otherwise recorded XDF trials only. '
                   'Channel counts below the former 16-trial threshold are recomputed without that threshold; ANOVA still requires two directions and positive residual degrees of freedom. Other cached channel counts are preserved.', pngs=pngs, data=all_rows)
        (stage / 'report.json').write_text(json.dumps(storage.jsonable(manifest), ensure_ascii=False, indent=2), encoding='utf-8')
        staged.append((report, stage, raw))
    assert all((report / 'report.json').read_bytes() == raw for report, _, raw in staged)
    for report, stage, _ in staged:
        for path in stage.rglob('*'):
            if not path.is_file():
                continue
            target = report / path.relative_to(stage)
            assert target.resolve().is_relative_to(root)
            if target.exists():
                before = audit / 'before' / report.name / path.relative_to(stage)
                before.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, before)
            shutil.copy2(path, target)
            assert storage.file_digest(path) == storage.file_digest(target)
        print('Updated ' + report.name, flush=True)
    (audit / 'completion.json').write_text(json.dumps(dict(reports=len(staged)), indent=2), encoding='utf-8')
    print(str(audit), flush=True)


if __name__ == '__main__':
    main()
