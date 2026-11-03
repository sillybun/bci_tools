import json
from pathlib import Path
import queue
from types import SimpleNamespace

import pytest

from data_analysis_gui.exporting import export_batch, export_result
from data_analysis_gui.gui.task_queue import TaskQueue


@pytest.mark.parametrize('layout', ['run', 'figure'])
def test_batch_layout_preserves_two_runs_and_all_formats(tmp_path, layout):
    batch = export_batch(tmp_path, layout)
    folders = []
    for run in ('001', '002'):
        folder = tmp_path / run
        (folder / 'figures').mkdir(parents=True)
        files = {}
        for ext in ('png', 'pdf', 'svg', 'csv'):
            relative = f'figures/S02_01_Channel_saturation_zones.{ext}'
            (folder / relative).write_text(run + ext)
            files[ext] = relative
        record = dict(section=dict(id='s02', version='v1'), signature=run,
                      figures=[dict(title='Channel saturation zones', files=files)], tables={})
        recording = SimpleNamespace(date='2026-11-11', data_type='CO8', run=run,
                                    key=run, path=Path(run + '.xdf'))
        folders.append(folder)
        export_result(folder, record, [recording], batch)
    root = Path(batch['root'])
    pngs = list(root.rglob('*.png'))
    assert len(pngs) == 2
    assert len({p.parent for p in pngs}) == (1 if layout == 'figure' else 2)
    assert {p.read_text() for p in pngs} == {'001png', '002png'}
    manifests = [json.loads(p.read_text()) for p in root.rglob('*.json')]
    assert len(manifests) == 2
    for manifest in manifests:
        assert len(manifest['files']) == 4
        assert all((root / f['exported']).is_file() for f in manifest['files'])
    assert all((folder / 'figures/S02_01_Channel_saturation_zones.png').is_file() for folder in folders)
    assert export_batch(tmp_path, layout)['root'] != batch['root']


def test_export_rejects_paths_outside_cached_result(tmp_path):
    folder = tmp_path / 'cache'
    folder.mkdir()
    (tmp_path / 'outside.png').write_text('outside')
    record = dict(section=dict(id='s02', version='v1'), signature='sig',
                  figures=[dict(title='test', files=dict(png='../outside.png'))])
    rec = SimpleNamespace(date='2026-11-11', data_type='CO8', run='001', key='one', path=Path('one.xdf'))
    batch = export_batch(tmp_path, 'figure')
    with pytest.raises(ValueError, match='越界'):
        export_result(folder, record, [rec], batch)
    assert not list(Path(batch['root']).rglob('*.png'))


def test_batch_export_does_not_duplicate_running_compute():
    manager = TaskQueue(queue.Queue())
    request = dict(jobs=[dict(section='s02', recordings=['one.xdf'])],
                   export=dict(root='first', layout='figure'))
    assert len(manager.submit(request, None)) == 1
    request['export']['root'] = 'second'
    assert not manager.submit(request, None)
    request['export_only'] = True
    assert len(manager.submit(request, None)) == 1


def test_export_only_uses_cache_even_when_recompute_is_checked(tmp_path, monkeypatch):
    from data_analysis_gui import worker

    folder = tmp_path / 'cache'
    (folder / 'figures').mkdir(parents=True)
    (folder / 'figures/test.png').write_text('cached image')
    recording = SimpleNamespace(date='2026-11-11', data_type='CO8', run='001',
                                key='one', path=Path('one.xdf'), label='one')
    section = SimpleNamespace(id='s02', display='S02')
    record = dict(section=dict(id='s02', version='v1'), signature='sig',
                  figures=[dict(title='test', files=dict(png='figures/test.png'))])
    monkeypatch.setattr(worker, 'plan', lambda request: ([dict(
        task='s02:one.xdf', section=section, recordings=[recording], folder=folder, signature='sig')], []))
    monkeypatch.setattr(worker.storage, 'load_cached', lambda *args: record)
    monkeypatch.setattr(worker, 'get_session', lambda *args: pytest.fail('Must export without loading XDF'))
    batch = export_batch(tmp_path, 'figure')
    events = []
    request = dict(output_root=str(tmp_path), export=batch, export_only=True, recompute=True)
    assert worker.run_jobs(request, lambda event, **values: events.append((event, values))) == 0
    assert any(event == 'complete' and values['cached'] for event, values in events)
    assert len(list(Path(batch['root']).rglob('*.png'))) == 1
