"""Add current rolling/per-run BPS views to existing report selections."""
import csv
from collections import defaultdict
import gc
import json
from datetime import datetime
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis_gui import storage
from data_analysis_gui.core import behavior
from data_analysis_gui.core.catalog import describe
from data_analysis_gui.core.crossday import recording_label
from data_analysis_gui.core.session import Session
from data_analysis_gui.core.style import finish_figure
from data_analysis_gui.reporting import CATEGORIES, REPORT_ROOT, _grid_behavior
from data_analysis_gui.sections.s13_online_behavior_v1 import rolling_bps, co8_bps


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)


def main():
    root = REPORT_ROOT.resolve()
    audit = root.parent / 'report_edit_backups' / ('bps_backfill_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    audit.mkdir(parents=True)
    manifests = [(p, p.read_bytes()) for p in sorted(root.glob('*/report.json'))]
    selections = {}
    for _, raw in manifests:
        for row in json.loads(raw)['selection']:
            if row['online'] and row['task_mode'] in behavior.TASK_LABELS:
                selections[row['path']] = row
    data = {}
    for index, path in enumerate(selections, 1):
        rec = describe(Path(path), Path('E:/AitBCI/data/keaiduo'))
        files = behavior.source_files(rec)
        try:
            trials = behavior._from_csv(files, rec.task_mode) if files else None
        except (KeyError, ValueError, OSError, TypeError, csv.Error):
            trials = None
        if trials is not None:
            source, note = str(files['trials']), ''
        else:
            session = Session.load(rec)
            trials, source, note = behavior.collect(session)
            del session
            gc.collect()
        label = recording_label(rec)
        key = rolling_bps.run_key(rec)
        windows = rolling_bps.windows(trials, rec.task_mode, session=label, date=rec.date, run=key)
        data[path] = dict(recording=rec, source=source, note=note, trials=trials, windows=windows,
                          run_key=key, date=rec.date, session=label)
        print(f'Loaded {index}/{len(selections)}: {rec.date} run-{rec.run}, {len(trials)} trials, {len(windows)} windows', flush=True)
    staged_reports = []
    rendered = {}
    for manifest_path, raw in manifests:
        report = manifest_path.parent
        manifest = json.loads(raw)
        stage = audit / 'staged' / report.name
        stage.mkdir(parents=True)
        generated, coverage, obsolete = [], [], []
        selected = [data[row['path']] for row in manifest['selection'] if row['path'] in data]
        for mode, category in [('centerout', CATEGORIES[4]), ('grid', CATEGORIES[5])]:
            entries = [entry for entry in selected if entry['recording'].task_mode == mode]
            if not entries:
                continue
            full_csv_runs = {entry['run_key'] for entry in entries if entry['source'] != 'XDF event markers'}
            pooled = []
            trial_pool = defaultdict(list)
            for entry in entries:
                rec = entry['recording']
                identity = storage.fingerprint([str(rec.path)])[:8]
                prior = list((report / category).glob(f'*_{identity}_*Rolling_BPS.png'))
                relative = (prior[0].relative_to(report) if prior else
                            Path(category) / f'S13_{storage.slug(entry["session"])}_{identity}_Rolling_BPS.png')
                cache_key = (str(rec.path), manifest['font_percent'])
                if cache_key not in rendered:
                    cached = audit / 'rendered' / f'{identity}_{manifest["font_percent"]}.png'
                    cached.parent.mkdir(exist_ok=True)
                    fig = rolling_bps.rolling_figure(entry['windows'], mode, f'{entry["session"]} · 60-active-second rolling BPS')
                    finish_figure(fig, manifest['font_percent'])
                    fig.savefig(cached, dpi=storage.PNG_DPI, facecolor='white')
                    fig.clear()
                    rendered[cache_key] = cached
                dest = stage / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(rendered[cache_key], dest)
                generated.append(str(relative))
                excluded = entry['source'] == 'XDF event markers' and entry['run_key'] in full_csv_runs
                if not excluded:
                    pooled.extend(entry['windows'])
                    trial_pool[entry['run_key']].extend(entry['trials'])
                coverage.append(dict(path=str(rec.path), run=entry['session'], source=entry['source'],
                                     trials=len(entry['trials']), windows=len(entry['windows']),
                                     excluded_from_run_summary=excluded, note=entry['note']))
            summaries = rolling_bps.run_summary(pooled, mode, runs=entries, trials_by_run=trial_pool)
            task = behavior.TASK_LABELS[mode]
            for peak in ((None,) if mode == 'grid' else (False, True)):
                excluded_dates = manifest.get('presentation_updates', {}).get('grid_bps_summary_excluded_dates', []) if mode == 'grid' else []
                plotted = [dict(r) for r in summaries if r['date'] not in excluded_dates]
                if mode == 'grid':
                    overrides = manifest.get('presentation_updates', {}).get('grid_bps_peak_overrides', {})
                    for row in plotted:
                        if row['date'] in overrides:
                            row['bps_peak'] = float(overrides[row['date']])
                title, fig = rolling_bps.run_figure(plotted, mode, peak=peak)
                finish_figure(fig, manifest['font_percent'])
                suffix = 'Median_IQR_and_Peak' if mode == 'grid' else 'Peak' if peak else 'Median_and_IQR'
                prior = list((report / category).glob(f'*Run_Rolling_BPS_{suffix}.png'))
                relative = (prior[0].relative_to(report) if prior else
                            Path(category) / f'X08_{task}_Run_Rolling_BPS_{suffix}.png')
                fig.savefig(stage / relative, dpi=storage.PNG_DPI, facecolor='white')
                fig.clear()
                generated.append(str(relative))
            tables = Path(category) / '统计数据'
            write_csv(stage / tables / 'Run_Rolling_BPS_Summary.csv', summaries)
            if mode == 'grid':
                write_csv(stage / tables / 'Grid_BPS_Summary_Plotted_Runs.csv', plotted)
            write_csv(stage / tables / 'Rolling_BPS_Windows.csv', pooled)
            (stage / tables / 'Rolling_BPS_Method.txt').write_text(rolling_bps.METHOD, encoding='utf-8-sig')
        # Retire legacy block/whole-session BPS views, keeping only 60-second BPS.
        for source in manifest['sources']:
            keep = []
            for relative in source.get('pngs', []):
                if relative.endswith(('_Run_Time_to_Target.png', '_Online_Grid_BPS_by_Method.png', '_Online_Grid_BPS.png',
                                      '_Online_CO8_BPS_by_Method.png')):
                    obsolete.append(relative)
                else:
                    keep.append(relative)
            source['pngs'] = keep
            if source['section'] != 's13':
                continue
            record = json.loads((Path(source['folder']) / 'result.json').read_text(encoding='utf-8'))
            name = record['inputs']['recordings'][0]['name']
            matching = [entry for entry in selected if entry['recording'].path.name == name]
            if len(matching) != 1 or matching[0]['recording'].task_mode != 'grid':
                continue
            entry = matching[0]
            identity = storage.fingerprint([str(entry['recording'].path)])[:8]
            for old in (report / CATEGORIES[5]).glob(f'*_{identity}_Grid_Behavior.png'):
                relative = old.relative_to(report)
                dest = stage / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                _grid_behavior(dict(record=record, folder=source['folder'], label=entry['session']),
                               manifest['font_percent'], dest)
                generated.append(str(relative))
        manifest.setdefault('methods', {})['Grid_Behavior.png'] = 'Block success rate and successful-trial acquisition-time median/IQR; BPS is separately calculated in rolling 60 active seconds.'
        for old in (report / CATEGORIES[5]).glob('*Run_Time_to_Target.png'):
            relative = str(old.relative_to(report))
            if relative not in obsolete:
                obsolete.append(relative)
        metadata = dict(method=rolling_bps.METHOD, scope='Original report selection; per-run, not daily.',
                        pngs=generated, sources=coverage, retired_pngs=obsolete,
                        window_seconds=60, clock='cumulative_active_attempt_seconds')
        (stage / 'rolling_bps_backfill.json').write_text(json.dumps(storage.jsonable(metadata), ensure_ascii=False, indent=2), encoding='utf-8')
        manifest['rolling_bps_backfill'] = metadata
        (stage / 'report.json').write_text(json.dumps(storage.jsonable(manifest), ensure_ascii=False, indent=2), encoding='utf-8')
        staged_reports.append((report, stage, raw))
        print(f'Staged {report.name}: {len(generated)} PNGs', flush=True)
    # Check that no report was regenerated concurrently before publishing any changes.
    assert all((report / 'report.json').read_bytes() == raw for report, _, raw in staged_reports)
    completed = []
    for report, stage, _ in staged_reports:
        for source in stage.rglob('*'):
            if not source.is_file():
                continue
            target = report / source.relative_to(stage)
            assert target.resolve().is_relative_to(root)
            if target.exists():
                backup = audit / 'before' / report.name / source.relative_to(stage)
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            assert storage.file_digest(source) == storage.file_digest(target)
        for old in (report / CATEGORIES[5]).rglob('*Run_Rolling_BPS_*.png'):
            if old.name.endswith(('Median_and_IQR.png', '_Peak.png')) and not old.name.endswith('Median_IQR_and_Peak.png'):
                assert old.resolve().is_relative_to(root)
                archived = audit / 'retired' / report.name / old.relative_to(report)
                archived.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old), str(archived))
        metadata = json.loads((report / 'rolling_bps_backfill.json').read_text(encoding='utf-8'))
        assert all((report / relative).is_file() for relative in metadata['pngs'])
        for relative in metadata.get('retired_pngs', []):
            old = report / relative
            if old.exists():
                assert old.resolve().is_relative_to(root)
                dest = audit / 'retired' / report.name / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old), str(dest))
        completed.append(dict(report=report.name, pngs=len(metadata['pngs'])))
    (audit / 'completion.json').write_text(json.dumps(completed, indent=2), encoding='utf-8')
    print(json.dumps(dict(completed=completed, backup=str(audit)), ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
