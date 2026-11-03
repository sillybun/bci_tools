"""INI/log equivalence, saved classification validation and day-local discovery."""
from dataclasses import replace
import json

import pytest

from data_analysis_gui.core.catalog import day_config_path, day_folder
from data_analysis_gui.core.channels import INI_KEYS, read_channel_config


def log_record(good=(0, 4), uncertain=(2,), time='12:16:09.293', **overrides):
    fields = dict(good_channel_ids=list(good), uncertain_channel_ids=list(uncertain),
                  bad_channel_ids=sorted(set(range(1024)) - set(good) - set(uncertain)),
                  crosstalk_channel_ids=[4])
    fields.update(overrides)
    return (f'2026-11-09 {time}+08:00 [Filter] [Info] filter.quick.results_applied: '
            + '; '.join(f'{key}={json.dumps(value)}' for key, value in fields.items()))


def test_ini_and_log_produce_identical_channel_lists(tmp_path):
    ini = tmp_path / 'config.ini'
    log = tmp_path / 'Neuralinsight.log'
    log.write_text(log_record(), encoding='utf-8-sig')
    cfg = read_channel_config(log)
    ini.write_text('[Channels]\n' + '\n'.join(
        f'{key}="{",".join(str(i - 1) for i in cfg[key])}"' for key in INI_KEYS), encoding='utf-8-sig')
    assert read_channel_config(ini) == {key: cfg[key] for key in INI_KEYS}
    assert cfg['good'] == [1, 5]
    assert cfg['crosstalk_channels'] == [5]
    assert cfg['last_programmed_channels'] == cfg['last_selected_channels'] == []
    assert cfg['_source']['line'] == 1
    assert cfg['_source']['timestamp'] == '2026-11-09 12:16:09.293+08:00'


def test_log_uses_last_saved_classification_and_ignores_other_channel_events(tmp_path):
    log = tmp_path / 'Neuralinsight_20261109.log'
    log.write_text(log_record() + '\n' + log_record(good=(1, 4), time='12:16:10.000')
                   + '\n[Signal] channel_ids=[999]; good_count=1\n', encoding='utf-8')
    cfg = read_channel_config(log)
    assert cfg['good'] == [2, 5]
    assert cfg['_source']['line'] == 2
    assert cfg['_source']['selection'] == 'last_results_applied'


@pytest.mark.parametrize('bad_record', [
    'filter.quick.results_applied: good_channel_ids=[0]',  # incomplete latest result
    log_record(good_channel_ids=[0, 0]),
    log_record(good_channel_ids=[1024]),
    log_record(good_channel_ids=[True]),
    log_record(bad_channel_ids=[0, 3]),  # overlapping and incomplete partition
    log_record() + '; good_count=999',
    log_record().replace('good_channel_ids=[0, 4]', 'good_channel_ids=[0,]'),
])
def test_invalid_latest_result_is_not_silently_replaced_with_older_one(tmp_path, bad_record):
    log = tmp_path / 'Neuralinsight.log'
    log.write_text(log_record() + '\n' + bad_record, encoding='utf-8')
    with pytest.raises(ValueError):
        read_channel_config(log)


def test_log_without_classification_is_rejected(tmp_path):
    log = tmp_path / 'Neuralinsight.log'
    log.write_text('config.loaded: path=configs.ini\nchannel_map.apply_succeeded: result=success')
    with pytest.raises(ValueError, match='results_applied'):
        read_channel_config(log)


@pytest.mark.parametrize('day_name', ['1109', '20261109', '2026-11-09'])
def test_discovery_prefers_ini_and_falls_back_to_day_log(tmp_path, day_name):
    day = tmp_path / day_name
    source = day / 'CO8_run-001' / 'xdf' / 'r.xdf'
    log = day / 'shared_logs' / 'acquisition' / 'neuralinsight' / 'Neuralinsight_20261109.log'
    log.parent.mkdir(parents=True)
    log.write_text(log_record())
    assert day_folder(source) == day
    assert day_config_path(source) == str(log)
    ini = day / 'shared_config' / 'acquisition' / 'config.ini'
    ini.parent.mkdir(parents=True)
    ini.write_text('good=0')
    assert day_config_path(source) == str(ini)
    standard = day / 'configs.ini'
    standard.write_text('good=1')
    assert day_config_path(source) == str(standard)


def test_discovery_ignores_other_days_and_ambiguous_years(tmp_path):
    day = tmp_path / '1109'
    day.mkdir()
    source = day / 'xdf' / 'r.xdf'
    (day / 'Neuralinsight_20261110.log').write_text('')
    assert day_config_path(source) == ''
    for year in (2025, 2026):
        (day / f'Neuralinsight_{year}1109.log').write_text('')
    assert day_config_path(source) == ''
    generic = day / 'Neuralinsight.log'
    generic.write_text('')
    assert day_config_path(source) == str(generic)


def test_log_content_participates_in_section_cache_signature(tmp_path, monkeypatch):
    from data_analysis_gui import storage
    from data_analysis_gui.core.filters import TrialFilter
    from data_analysis_gui.core.synthetic import make_session
    from data_analysis_gui.sections import versions

    day = tmp_path / '1109'
    day.mkdir()
    path = day / 'r.xdf'
    path.write_bytes(b'identity')
    log = day / 'Neuralinsight.log'
    log.write_text(log_record())
    monkeypatch.setattr(storage, 'SHA_CACHE', tmp_path / 'sha.json')
    r = replace(make_session(n_blocks=1).recording, path=path)
    section = versions('s08')[-1]
    def signature():
        return storage.signature_inputs(section, [r], TrialFilter(), section.resolve_params({}))
    before = signature()
    assert before['channel_config']['path'] == str(log)
    log.write_text(log_record(good=(1, 4)) + '\n')
    assert signature()['channel_config']['sha256'] != before['channel_config']['sha256']


def test_s08_uses_log_and_records_provenance(tmp_path):
    from data_analysis_gui.core.filters import TrialFilter
    from data_analysis_gui.core.synthetic import make_session
    from data_analysis_gui.runner import build_context, run_section
    from data_analysis_gui.sections import versions

    log = tmp_path / 'Neuralinsight.log'
    log.write_text(log_record())
    section = versions('s08')[-1]
    session = make_session(n_blocks=2)
    ctx = build_context([session], TrialFilter(), {'decode': False}, section, channel_config_path=str(log))
    result = run_section(section, ctx)
    assert result.summary['transmitted_by_tier']['good'] == 2
    assert result.summary['channel_config_source']['line'] == 1
    assert any('Neuralinsight' in note for note in result.notes)
    for _, figure in result.figures:
        figure.clear()
