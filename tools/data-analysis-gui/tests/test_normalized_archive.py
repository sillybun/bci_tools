"""Normalized run/session annotations must survive directory and file renaming."""
import json
import xml.etree.ElementTree as ET
import pytest

from data_analysis_gui.core import catalog
from data_analysis_gui.core.crossday import recording_label
from data_analysis_gui.core.session import recording_window_policy


@pytest.mark.parametrize('flat', [False, True])
def test_annotations_preserve_condition_label_and_analysis_window(tmp_path, monkeypatch, flat):
    session = tmp_path / '20261109' / 'Grid_run-002_online_8x8_split-session'
    if not flat:
        session = session / 'sessions' / 'session-20261109-141002'
    xdf = session / 'xdf' / 'sub-keaiduo_run-002_mode-online_rec-001.xdf'
    session.mkdir(parents=True)
    metadata = dict(format='keaiduo-session-v1', condition='脑控 / 8x8 / 无目标辅助',
                    condition_tag='8x8 split-session no assist', analysis_window_policy=None)
    (session / 'session_info.json').write_text(json.dumps(metadata), encoding='utf-8')
    headers = [ET.fromstring(f'<info><name>{name}</name><channel_count>512</channel_count>'
                            '<desc><task_mode>grid</task_mode></desc></info>')
               for name in (*catalog.REQUIRED_STREAMS, 'cursortasks_decoder_velocity')]
    monkeypatch.setattr(catalog, 'read_headers', lambda _: headers)
    r = catalog.describe(xdf, tmp_path)
    assert r.date == '2026-11-09' and r.run == '002' and not r.reason
    assert r.condition == metadata['condition']
    assert '8x8 split-session no assist' in recording_label(r)
    assert 'manual' not in recording_label(r)
    assert recording_window_policy(r) is None  # Renaming must not introduce task-end cropping.


def test_local_acquisition_snapshot_takes_precedence_over_shared_day_config(tmp_path):
    day = tmp_path / '20261106'
    session = day / 'CO8_run-test001_offline_morning-test' / 'sessions' / 'session-20261106-110512'
    xdf = session / 'xdf' / 'recording.xdf'
    day.mkdir()
    shared = day / 'configs.ini'
    shared.write_text('good=1')
    local = session / 'acquisition' / 'configs1106.ini'
    local.parent.mkdir(parents=True)
    local.write_text('good=2')
    assert catalog.day_config_path(xdf) == str(local)


def test_unrecognized_metadata_does_not_replace_legacy_condition(tmp_path):
    folder = tmp_path / '20261106' / 'CO8_run-003_restrained_BCI_assist200' / 'sessions' / 'session-test'
    folder.mkdir(parents=True)
    (folder / 'session_info.json').write_text('{"condition": "unverified"}')
    assert '初始辅助 200' in catalog.experiment_condition(folder / 'xdf' / 'r.xdf')


def test_complete_behavior_csv_is_assigned_only_to_main_recording(tmp_path):
    from types import SimpleNamespace
    from data_analysis_gui.core.behavior import source_files

    session = tmp_path / 'sessions' / 'session-20261111-113117'
    xdf = session / 'xdf'
    xdf.mkdir(parents=True)
    early, main = xdf / 'early.xdf', xdf / 'main.xdf'
    early.write_bytes(b'early')
    main.write_bytes(b'main')
    trials = session / 'behavior' / 'trials.csv'
    trials.parent.mkdir()
    trials.write_text('header\n')
    (session / 'session_info.json').write_text(json.dumps(dict(format='keaiduo-session-v1',
        recordings=[dict(path='xdf/early.xdf', segment='early-split'),
                    dict(path='xdf/main.xdf', segment='main')])))
    assert source_files(SimpleNamespace(path=main, task_mode='grid'))['trials'] == trials
    assert source_files(SimpleNamespace(path=early, task_mode='grid')) == {}


def test_unannotated_multiple_recordings_do_not_guess_a_csv_assignment(tmp_path):
    from types import SimpleNamespace
    from data_analysis_gui.core.behavior import source_files

    xdf = tmp_path / 'xdf'
    xdf.mkdir()
    for name in ('a.xdf', 'b.xdf'):
        (xdf / name).write_bytes(b'xdf')
    trials = tmp_path / 'behavior' / 'trials.csv'
    trials.parent.mkdir()
    trials.write_text('header\n')
    assert source_files(SimpleNamespace(path=xdf / 'a.xdf', task_mode='grid')) == {}


def test_flat_run_keeps_model_lookup_scoped_to_its_own_metadata(tmp_path):
    from types import SimpleNamespace
    from data_analysis_gui.sections.s11_online_vs_offline_v1.model_replay import resolve_run

    day = tmp_path / '20261109'
    run = day / 'custom-run'
    run.mkdir(parents=True)
    (run / 'session_info.json').write_text(json.dumps({'format': 'keaiduo-session-v1'}))
    other = day / 'other-run' / 'refit'
    other.mkdir(parents=True)
    (other / 'online_update_swap_history.jsonl').write_text('')
    r = SimpleNamespace(path=run / 'xdf' / 'r.xdf')
    assert resolve_run(r) is None


def test_explicit_empty_note_does_not_mislabel_online_recording_as_manual(tmp_path):
    r = catalog.Recording(tmp_path / 'run-002-CO8-online' / 'xdf' / 'r.xdf', 'r.xdf',
        '2026-11-09', 'centerout', True, '002', header_info={'condition_tag': ''}, condition='脑控')
    label = recording_label(r)
    assert 'manual' not in label and 'no assist' not in label and label.endswith('run002')

def test_behavior_bundle_supports_new_root_csv_and_task_log(tmp_path):
    from types import SimpleNamespace
    from data_analysis_gui.core.behavior import source_files
    folder=tmp_path/'20261118'/'run-004-Grid-offline-8x8'
    (folder/'xdf').mkdir(parents=True);(folder/'logs').mkdir()
    xdf=folder/'xdf/recording.xdf';xdf.write_bytes(b'xdf')
    trials=folder/'trials.csv';trials.write_text('header\n')
    log=folder/'logs/task.log';log.write_text('Grid auto-scale enabled: rows=8, cols=8\n')
    files=source_files(SimpleNamespace(path=xdf,task_mode='grid'))
    assert files['trials']==trials and files['log_0']==log
