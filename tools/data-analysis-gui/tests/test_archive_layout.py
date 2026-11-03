"""Legacy and packaged run discovery, conditions and recording coverage."""
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from data_analysis_gui.core import catalog
from data_analysis_gui.core.crossday import recording_label, recording_style, session_label_color
from data_analysis_gui.core.session import Session, recording_window_policy
from data_analysis_gui.core.synthetic import make_session


@pytest.mark.parametrize('folder,online,condition,tag', [
    ('CO8_run-001_manual', False, '手控 / 离线', 'manual'),
    ('CO8_run-002_arm_free_BCI', True, '自由手脑控 / 无辅助', 'arm free'),
    ('CO8_run-003_restrained_BCI_assist200', True, '手束缚脑控 / 初始辅助 200 px/s', 'restrained assist200'),
    ('CO8_run-004_restrained_BCI_assist100', True, '手束缚脑控 / 初始辅助 100 px/s', 'restrained assist100'),
])
def test_archive_discovery(tmp_path, monkeypatch, folder, online, condition, tag):
    path = tmp_path / '20261106' / folder / 'xdf' / 'sub-keaiduo_run-003_spike.xdf'
    names = list(catalog.REQUIRED_STREAMS) + (['cursortasks_decoder_velocity'] if online else [])
    headers = [ET.fromstring(f'<info><name>{name}</name><channel_count>512</channel_count>'
                            '<desc><task_mode>centerout</task_mode></desc></info>') for name in names]
    monkeypatch.setattr(catalog, 'read_headers', lambda _: headers)
    r = catalog.describe(path, tmp_path)
    assert not r.reason
    assert r.condition == condition and r.online == online
    assert r.data_type == ('CO8 online' if online else 'CO8')
    assert catalog.filter_type([r], ['CO8', 'CO8 online']) == [r]
    assert tag in recording_label(r)
    assert condition in r.label
    if 'restrained' in tag:
        assert session_label_color(recording_label(r)) == recording_style(r)[1]
        assert recording_style(r)[1] != '#bd5021'


def test_config_lookup_preserves_legacy_priority_and_handles_nested_segment(tmp_path):
    day = tmp_path / '20261106'
    shared = day / 'shared_config' / 'acquisition' / 'configs1106.ini'
    shared.parent.mkdir(parents=True)
    shared.write_text('[Channels]\ngood=1')
    path = day / 'CO8_run-002_arm_free_BCI' / 'xdf' / 'early_segment_old1' / 'recording.xdf'
    assert catalog.day_config_path(path) == str(shared)
    assert catalog.experiment_condition(path) == '自由手脑控 / 无辅助'
    legacy = day / 'configs.ini'
    legacy.write_text('[Channels]\ngood=2')
    assert catalog.day_config_path(path) == str(legacy)
    legacy.unlink()
    shared.unlink()
    for name in ('configs1001.ini', 'configs1002.ini'):
        shared.with_name(name).write_text('')
    assert catalog.day_config_path(path) == ''  # never select an ambiguous acquisition config


@pytest.mark.parametrize('archived,session_end,expected_stop', [
    (True, False, 1700), (True, True, 1600), (False, False, 1500),
])
def test_long_archive_preserves_tail_and_legacy_test_policy(
        tmp_path, monkeypatch, archived, session_end, expected_stop):
    from data_analysis_gui.core import session as module

    path = tmp_path / '20261106' / ('CO8_run-003_restrained_BCI_assist200' if archived else 'test') / 'xdf' / 'r.xdf'
    r = replace(make_session(n_blocks=1).recording, path=path, date='2026-11-06')
    assert recording_window_policy(r) == ('through_task_end' if archived else 'through_task_end_or_first_25_min')
    times = np.array([0., 1500., 1600., 1700.])
    def stream(values, name):
        return dict(time_stamps=times.copy(), time_series=values, info={'name': [name]})
    events = [['T_TRIAL_START', '', '', ''], ['T_TRIAL_END', '', '', ''],
              ['T_SESSION_END' if session_end else 'T_TRIAL_START', '', '', ''], ['T_TRIAL_END', '', '', '']]
    streams = {
        'spike_counts': stream(np.zeros((4, 512)), 'spike_counts'),
        'spike_counts_metadata': stream(np.zeros((4, 5)), 'spike_counts_metadata'),
        'cursortasks_cursor_state': stream(np.zeros((4, 12)), 'cursortasks_cursor_state'),
        'cursortasks_events': stream(events, 'cursortasks_events'),
        'cursortasks_decoder_velocity': stream(np.zeros((4, 2)), 'cursortasks_decoder_velocity'),
    }
    monkeypatch.setattr(module, 'load_streams', lambda _: (streams, []))
    monkeypatch.setattr(module, 'electrode_ids', lambda _: np.arange(1, 513))
    s = Session.load(r)
    for values in (s.neural_times, s.cursor_times, s.event_times, s.metadata_times, s.decoder_times):
        assert values[-1] == expected_stop


def test_nested_segment_resolves_only_own_refit(tmp_path):
    from data_analysis_gui.sections.s11_online_vs_offline_v1.model_replay import resolve_run

    day = tmp_path / '20261106'
    run = day / 'CO8_run-002_arm_free_BCI'
    history = run / 'refit' / 'online_update_swap_history.jsonl'
    history.parent.mkdir(parents=True)
    history.write_text('')
    other = day / 'CO8_run-003_restrained_BCI_assist200' / 'refit' / history.name
    other.parent.mkdir(parents=True)
    other.write_text('')
    r = replace(make_session(n_blocks=1).recording,
                path=run / 'xdf' / 'early_segment_old1' / 'r.xdf')
    assert resolve_run(r) == history.parent
    history.unlink()
    assert resolve_run(r) is None
