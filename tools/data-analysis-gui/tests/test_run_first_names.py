from pathlib import Path

import pytest

from data_analysis_gui.core.catalog import _run_label, format_run_name, run_folder


@pytest.mark.parametrize('number,task,mode,notes,part,expected', [
    (1, 'CO8', 'offline', ['manual'], None, 'run-001-CO8-offline'),
    ('006', 'CO8', 'online', ['no-assist', 'split-session'], 'a', 'run-006-a-CO8-online'),
    (2, 'CO8', 'online', ['arm-free', 'no-assist'], None, 'run-002-CO8-online'),
    (6, 'grid', 'online', ['12x12', 'startup'], 'b', 'run-006-b-Grid-online-12x12-startup'),
])
def test_formatter_and_parser_agree(number, task, mode, notes, part, expected):
    name = format_run_name(number, task, mode, notes, part=part)
    assert name == expected
    path = Path('20261109') / name / 'xdf' / f'{name}-20261109-rec-001-main-spike.xdf'
    assert run_folder(path) == path.parent.parent
    assert _run_label(path) == f'{int(number):03d}' + (f'-{part}' if part else '')
    assert _run_label(Path(path.name)) == _run_label(path)


@pytest.mark.parametrize('number,task,mode,notes,part', [
    (0, 'CO8', 'online', [], None), (1000, 'CO8', 'online', [], None),
    (1, 'unknown', 'online', [], None), (1, 'CO8', 'unknown', [], None),
    (1, 'CO8', 'online', ['../data'], None), (1, 'CO8', 'online', [], 'ab'),
])
def test_formatter_rejects_ambiguous_or_unsafe_names(number, task, mode, notes, part):
    with pytest.raises(ValueError):
        format_run_name(number, task, mode, notes, part=part)


def test_legacy_underscore_filenames_keep_existing_run_identity():
    path = Path('20261106/CO8_run-002_arm_free_BCI/xdf/sub-keaiduo_run-002_spike.xdf')
    assert _run_label(path) == '002'


def test_simplifying_existing_split_names_preserves_identity_and_segment_order(tmp_path):
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location('rename_run_first',
        Path(__file__).resolve().parents[1] / 'scripts' / 'rename_run_first.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = tmp_path / '20261109' / 'run-006-a-CO8-online-split-session-no-assist'
    xdf = run / 'xdf' / (run.name + '-20261109-rec-001-split-session-spike.xdf')
    xdf.parent.mkdir(parents=True)
    xdf.write_bytes(b'source identity')
    (run / 'run_info.json').write_text('{}')
    (run / 'session_info.json').write_text(json.dumps({'recordings': [{'path': xdf.relative_to(tmp_path).as_posix()}]}))
    plans = module.plan_names(tmp_path, ['20261109'], remove_split_note=True)
    assert plans[0]['name'] == 'run-006-a-CO8-online-no-assist'
    assert plans[0]['run_id'] == '006-a'
    assert plans[0]['files'][0]['filename'] == 'run-006-a-CO8-online-no-assist-20261109-rec-001-spike.xdf'
    assert xdf.read_bytes() == b'source identity'  # Planning is read-only.
