"""Section cache scope and worker failure isolation."""

import json
from dataclasses import replace

import pytest

from data_analysis_gui import storage, worker
from data_analysis_gui.core.filters import TrialFilter
from data_analysis_gui.core.synthetic import make_session
from data_analysis_gui.sections import versions


@pytest.fixture
def cache_project(tmp_path, monkeypatch):
    session = make_session(n_blocks=2)
    source = tmp_path / 'recording.xdf'
    source.write_bytes(b'recording identity')
    recording = replace(session.recording, path=source)
    sections = {sid: versions(sid)[-1] for sid in ('s02', 's03', 's04', 's12')}
    folders = {}
    # Include source sections consulted by S12's existing analysis dependencies.
    for sid in (*sections, 's01', 's05', 's06', 's07'):
        folder = tmp_path / 'sections' / sid
        folder.mkdir(parents=True)
        folders[sid] = folder
        (folder / 'section.py').write_text('VERSION = 1\n')
    shared = tmp_path / 'sections' / 'shared_frame_loss.py'
    shared.write_text('VERSION = 1\n')
    monkeypatch.setattr(storage, 'PACKAGE', tmp_path)
    monkeypatch.setattr(storage, 'SHA_CACHE', tmp_path / 'sha.json')
    monkeypatch.setattr(storage, 'section_folder', lambda section: folders[section.id])
    monkeypatch.setattr(worker, 'describe', lambda path, root: recording)
    output = tmp_path / 'output'

    def location(sid):
        section = sections[sid]
        inputs = storage.signature_inputs(section, [recording], TrialFilter(), section.resolve_params({}))
        signature = storage.fingerprint(inputs)
        return storage.result_folder(output, section, [recording], signature), signature

    for sid in sections:
        folder, signature = location(sid)
        folder.mkdir(parents=True)
        figure = folder / 'figure.svg'
        figure.write_text('<svg/>')
        (folder / 'result.json').write_text(json.dumps({
            'format': storage.FORMAT_VERSION,
            'signature': signature,
            'digests': {figure.name: storage.file_digest(figure)},
        }))
    return session, sections, folders, shared, output, location


@pytest.mark.parametrize('changed', ['s02', 's03', 's04'])
def test_section_edit_preserves_unrelated_cached_figures(cache_project, changed):
    _, sections, folders, _, _, location = cache_project
    (folders[changed] / 'section.py').write_text('VERSION = 2\n')
    affected = {changed, 's12'}
    assert {sid for sid in sections if storage.load_cached(*location(sid)) is None} == affected


def test_shared_frame_loss_edit_invalidates_only_its_consumers(cache_project):
    _, sections, _, shared, _, location = cache_project
    shared.write_text('VERSION = 2\n')
    assert {sid for sid in sections if storage.load_cached(*location(sid)) is None} == {'s03', 's12'}


def test_worker_uses_dated_config_and_honors_explicit_override(tmp_path, monkeypatch):
    from data_analysis_gui.core.catalog import day_config_path

    day = tmp_path / '20261103'
    day.mkdir()
    source = day / 'recording.xdf'
    source.write_bytes(b'recording identity')
    automatic = day / 'configs1103.ini'
    automatic.write_text('good="1"')
    manual = tmp_path / 'manual.ini'
    manual.write_text('good="2"')
    recording = replace(make_session(n_blocks=2).recording, path=source)
    monkeypatch.setattr(storage, 'SHA_CACHE', tmp_path / 'sha.json')
    monkeypatch.setattr(worker, 'describe', lambda path, root: recording)
    assert day_config_path(source) == str(automatic)
    standard = day / 'configs.ini'
    standard.write_text('good="3"')
    assert day_config_path(source) == str(standard)
    standard.unlink()
    jobs = [dict(section=sid, version=versions(sid)[-1].version, recordings=[str(source)])
            for sid in ('s02', 's03')]
    request = dict(output_root=str(tmp_path / 'output'), jobs=jobs)
    before, errors = worker.plan(request)
    assert not errors
    for job in jobs:
        job['channel_config_path'] = str(manual)
    after, errors = worker.plan(request)
    assert not errors
    assert before[0]['channel_config_path'] == str(automatic)
    assert after[0]['channel_config_path'] == str(manual)
    assert before[0]['signature'] != after[0]['signature']
    assert before[1]['signature'] == after[1]['signature']


@pytest.mark.parametrize('stage', ['planning', 'analysis'])
def test_failed_section_does_not_prevent_next_cache_read(cache_project, monkeypatch, stage):
    session, sections, folders, _, output, location = cache_project
    old_s03 = location('s03')
    (folders['s03'] / 'section.py').write_text('VERSION = 2\n')
    events = []
    analyzed = []
    monkeypatch.setattr(worker, 'emit', lambda event, **values: events.append((event, values)))
    monkeypatch.setattr(worker, 'get_session', lambda recording: session)

    def fail_analysis(section, ctx, *, font_percent):
        analyzed.append(section.id)
        raise RuntimeError('section failed')

    monkeypatch.setattr(worker, 'run_section', fail_analysis)
    request = {'output_root': str(output), 'jobs': [
        {'section': sid, 'version': 'missing' if sid == 's03' and stage == 'planning' else sections[sid].version,
         'recordings': ['recording.xdf']} for sid in ('s03', 's02')
    ]}
    assert worker.run(request) == 1
    assert analyzed == (['s03'] if stage == 'analysis' else [])
    assert any(event == 'error' and data['task'].startswith('s03:') for event, data in events)
    assert any(event == 'complete' and data['task'].startswith('s02:') and data['cached']
               for event, data in events)
    assert storage.load_cached(*old_s03) is not None


def test_transmission_reuses_single_session_after_changing_selection(tmp_path, monkeypatch):
    from data_analysis_gui.runner import build_context, run_section
    from data_analysis_gui.session_cache import SessionCache
    from data_analysis_gui.sections.x02_transmission_stability_v1 import section as transmission

    def execute(section_id, indices, threshold=.15, refresh=False):
        section = versions(section_id)[-1]
        sessions = [make_session(seed=i, n_blocks=2, date=f'2026-01-0{i + 1}') for i in indices]
        cache = SessionCache(tmp_path, refresh=refresh)
        for i, session in zip(indices, sessions):
            cache.bind_session(session, f'{i:064x}')
        ctx = build_context(sessions, TrialFilter(), {'gap_threshold': threshold}, section, artifacts=cache)
        result = run_section(section, ctx)
        for _, fig in result.figures:
            fig.clear()
        return result, cache

    single, _ = execute('s03', [0])
    full, _ = execute('x02', [0, 1, 2])
    row = single.figure_data[0]['session_totals'][0]
    assert {key: full.tables['session_transmission'][0][key] for key in row} == row
    forced, refreshed = execute('x02', [0, 2], refresh=True)
    assert forced.tables['session_transmission'] == full.tables['session_transmission'][::2]
    assert refreshed.hits == 0 and refreshed.misses == 6

    def unexpected(*args):
        pytest.fail('Retained sessions should not need another FPGA calculation')

    monkeypatch.setattr(transmission, 'fpga_analysis', unexpected)
    reduced, warm = execute('x02', [0, 2])
    assert reduced.tables['session_transmission'] == full.tables['session_transmission'][::2]
    assert warm.hits >= 4 and warm.misses == 0
    changed, cache = execute('x02', [0, 2], threshold=.2)
    assert cache.misses == 2  # New gap totals reuse the original FPGA clock analysis.
    assert all(row['long_receive_gap_threshold_s'] == .2 for row in changed.tables['session_transmission'])
