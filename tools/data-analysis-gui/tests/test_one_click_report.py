"""Report selection, queue isolation, cache reuse and PNG collection."""
import json
from pathlib import Path
import queue

import pytest

from data_analysis_gui import storage, worker
from data_analysis_gui import config as C
from data_analysis_gui.core.catalog import Recording
from data_analysis_gui.core.filters import TrialFilter
from data_analysis_gui.gui.task_queue import TaskQueue
from data_analysis_gui.reporting import CATEGORIES, SPATIAL_CATEGORIES, report_plan
from data_analysis_gui.sections.base import SectionResult, new_figure


@pytest.fixture
def report_request(tmp_path, monkeypatch):
    from PIL import Image
    reference=tmp_path/'BGA_shank_projection.png'
    Image.new('RGB',(60,40),'white').save(reference)
    monkeypatch.setattr(C,'BGA_SHANK_PROJECTION',reference)
    recordings = []
    for i, (mode, online) in enumerate((('centerout', False), ('grid', False),
                                       ('centerout', True), ('grid', True)), 1):
        path = tmp_path / f'run-{i}.xdf'
        path.write_bytes(bytes([i]))
        recordings.append(Recording(path, path.name, f'2026-11-0{i}', mode, online, str(i)))
    lookup = {str(r.path): r for r in recordings}
    monkeypatch.setattr(worker, 'describe', lambda path, root: lookup[str(path)])
    jobs, report = report_plan(recordings, lambda section: section.defaults(),include_run_exports=True)
    report['root'] = str(tmp_path / 'reports')
    request = dict(data_root=str(tmp_path), output_root=str(tmp_path / 'normal-results'),
                   criteria=TrialFilter().to_dict(), font_percent=120, report=report,
                   jobs=[dict(section=s.id, version=s.version, recordings=[str(r.path) for r in recs],
                              params=s.resolve_params(raw)) for s, recs, raw in jobs])
    return request, recordings


def test_report_plan_and_queue_keep_one_frozen_report(report_request):
    request, recordings = report_request
    lookup = {str(r.path): r for r in recordings}
    for job, category in zip(request['jobs'], request['report']['categories']):
        selected = [lookup[path] for path in job['recordings']]
        if category == CATEGORIES[3]:
            assert all(not r.online for r in selected)
        elif category in CATEGORIES[4:6]:
            assert all(r.online for r in selected)
            assert all(r.task_mode == ('centerout' if category == CATEGORIES[4] else 'grid') for r in selected)
    manager = TaskQueue(queue.Queue())
    added = manager.submit(request, None)
    assert len(added) == 1 and len(added[0].request['jobs']) == len(request['jobs'])
    assert not manager.submit(request, None)
    request['report']['name'] = 'changed'
    assert added[0].request['report']['name'] == '2026-11-01_2026-11-04_4runs'
    assert added[0].label == '一键报告 · 2026-11-01_2026-11-04_4runs'


def seed_cache(request):
    planned, errors = worker.plan(request)
    assert not errors
    source_pngs = {}
    for job in planned:
        sid = job['section'].id
        title = 'Online Behavior by Block' if sid == 's13' else 'Source figure'
        if sid == 'x08':
            task = 'CO8' if job['recordings'][0].task_mode == 'centerout' else 'Grid'
            title = f'Online {task} Run Rolling BPS ' + ('Summary' if task == 'Grid' else 'Median and IQR')
        fig = new_figure(2, 1)
        fig.subplots().plot([0, 1], [0, 1])
        result = SectionResult()
        result.add_figure(title, fig, method='test input')
        if sid == 'x01':
            from data_analysis_gui.sections.x01_channel_stability_v1.chip_layout import append_chip_layout
            append_chip_layout(result)
        if sid == 'x08' and task == 'CO8':
            peak = new_figure(2, 1)
            peak.subplots().plot([0, 1], [0, 1])
            result.add_figure(f'Online {task} Run Rolling BPS Peak', peak, method='test input')
        if sid == 's10':
            result.summary.update(n_trials=64, ridge_r2_mean=.5, ridge_r2_mean_ci=[.4, .6],
                                  velocity_kf_r2_mean=.7, velocity_kf_r2_mean_ci=[.65, .75])
        if sid == 's13':
            result.tables['block_metrics'] = [
                dict(block=i, attempts=20, successes=18, bps=1.2, success_pct=90,
                     rt_median_s=2, rt_q1_s=1.5, rt_q3_s=2.5) for i in (1, 2)
            ]
        record = storage.save_result(result, job['folder'], job['signature'], job['inputs'],
                                     job['section'], job['recordings'])
        for item in record['figures']:
            path = job['folder'] / item['files']['png']
            source_pngs[path] = path.read_bytes()
        fig.clear()
    return source_pngs


def test_report_cached_sources_copies_and_derivatives_preserve_originals(report_request, monkeypatch):
    request, _ = report_request
    sources = seed_cache(request)
    monkeypatch.setattr(worker, 'get_session', lambda r: pytest.fail('cache hit must not recompute'))
    events = []
    monkeypatch.setattr(worker, 'emit', lambda event, **values: events.append(dict(event=event, **values)))
    assert worker.run(request) == 0
    completed = [e for e in events if e['event'] == 'complete']
    assert len(completed) == 1
    folder = Path(completed[0]['folder'])
    assert folder.name == '2026-11-01_2026-11-04_4runs_v1'
    assert {p.name for p in folder.iterdir() if p.is_dir()} == set(CATEGORIES + SPATIAL_CATEGORIES)
    manifest = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    assert manifest['complete'] and not manifest['failures']
    assert manifest['run_count'] == 4 and manifest['recording_count'] == 4
    assert manifest['report_version'] == 1
    copied = [folder / p for entry in manifest['sources'] for p in entry['pngs']]
    assert len(copied) == len(sources)
    assert all(path.read_bytes() == original for path, original in sources.items())
    assert all(path.read_bytes() in sources.values() for path in copied)
    assert (folder / CATEGORIES[3] / 'Velocity.png').is_file()
    assert list((folder / CATEGORIES[1]).glob('*BGA_Chip_Layout_Mapping.png'))
    assert not list((folder / CATEGORIES[1] / '补充计算图').glob('*BGA_Chip_Layout_Mapping.png'))
    assert not list((folder / CATEGORIES[4]).glob('CO8_Behavior_Overview_*.png'))
    assert list((folder / CATEGORIES[5]).glob('*_Grid_Behavior.png'))
    assert list((folder / CATEGORIES[4]).glob('*Run_Rolling_BPS_Median_and_IQR.png'))
    assert list((folder / CATEGORIES[4]).glob('*Run_Rolling_BPS_Peak.png'))
    assert list((folder / CATEGORIES[5]).glob('*Run_Rolling_BPS_Summary.png'))
    assert worker.run(request) == 0
    assert (folder.parent / (folder.name.removesuffix('_v1') + '_v2') / 'report.json').is_file()
    assert (folder / 'report.json').is_file()


def test_report_partial_failure_keeps_other_categories(report_request, monkeypatch):
    request, _ = report_request
    seed_cache(request)
    # Invalidate one input's cached result. The worker must report that failure and keep going.
    planned, _ = worker.plan(request)
    failed = next(job for job in planned if job['section'].id == 'x02')
    (failed['folder'] / 'result.json').unlink()
    monkeypatch.setattr(worker, 'get_session', lambda r: (_ for _ in ()).throw(RuntimeError('test failure')))
    events = []
    monkeypatch.setattr(worker, 'emit', lambda event, **values: events.append(dict(event=event, **values)))
    assert worker.run(request) == 1
    assert not any(e['event'] == 'complete' for e in events)
    error, = [e for e in events if e['event'] == 'error']
    folder = Path(error['folder'])
    manifest = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    assert not manifest['complete'] and len(manifest['failures']) == 1
    assert (folder / CATEGORIES[3] / 'Velocity.png').is_file()
    assert list((folder / CATEGORIES[5]).glob('*_Grid_Behavior.png'))
    assert list((folder / CATEGORIES[4]).glob('*Run_Rolling_BPS_Median_and_IQR.png'))
    assert list((folder / CATEGORIES[4]).glob('*Run_Rolling_BPS_Peak.png'))
    assert list((folder / CATEGORIES[5]).glob('*Run_Rolling_BPS_Summary.png'))


def test_report_counts_multiple_xdf_in_one_run_once(tmp_path):
    folder = tmp_path / '20261106' / 'run-002-CO8-online'
    early = Recording(folder / 'xdf/early.xdf', 'early.xdf', '2026-11-06', 'centerout', True, '002')
    main = Recording(folder / 'xdf/main.xdf', 'main.xdf', '2026-11-06', 'centerout', True, '002')
    split = Recording(tmp_path / '20261106/run-002-a-CO8-online/xdf/split.xdf',
                      'split.xdf', '2026-11-06', 'centerout', True, '002-a')
    _, report = report_plan([early, main, split], lambda section: section.defaults())
    assert report['run_count'] == 2
    assert report['recording_count'] == 3
    assert report['name'] == '2026-11-06_2026-11-06_2runs'
    assert report['include_run_exports'] is False


def test_summary_only_report_uses_cached_run_data_but_exports_no_run_figures(report_request,monkeypatch):
    request,_=report_request
    seed_cache(request)
    request['report']['include_run_exports']=False
    monkeypatch.setattr(worker,'get_session',lambda r:pytest.fail('summary must reuse valid full caches'))
    events=[]
    monkeypatch.setattr(worker,'emit',lambda event,**values:events.append(dict(event=event,**values)))
    assert worker.run(request)==0
    root=Path(next(e['folder'] for e in events if e['event']=='complete'))
    images=list(root.rglob('*.png'));assert images
    assert all('summary_all_runs' in p.relative_to(root).parts for p in images)
    assert not list(root.rglob('runs'))
    assert (root/'02_通道稳定性分析/BGA_daily_first_run').is_dir()
    manifest=json.loads((root/'report.json').read_text(encoding='utf-8'))
    assert manifest['include_run_exports'] is False
    assert all(not s['pngs'] for s in manifest['sources'] if s['section'] in ('s05','s14','s10','s13'))


def test_cold_summary_cache_is_separate_from_full_figure_cache(report_request):
    request,_=report_request
    full,errors=worker.plan(request);assert not errors
    brief,errors=worker.plan(dict(request,report_summary_only=True));assert not errors
    for a,b in zip(full,brief):
        if a['section'].mode=='single':
            assert a['signature']!=b['signature']
            assert b['inputs']['figure_export']=='tables_only'
        else:assert a['signature']==b['signature']


def test_reports_reuse_pool_when_original_single_cache_is_removed(report_request,monkeypatch):
    import shutil
    request,_=report_request;seed_cache(request)
    assert worker.run(request)==0
    jobs,errors=worker.plan(request);assert not errors
    base=Path(request['output_root']).resolve()
    for j in jobs:
        if j['section'].mode=='single':
            assert j['folder'].resolve().is_relative_to(base)
            shutil.rmtree(j['folder'])
    monkeypatch.setattr(worker,'get_session',lambda r:pytest.fail('must load preserved run from shared pool'))
    assert worker.run(request)==0
