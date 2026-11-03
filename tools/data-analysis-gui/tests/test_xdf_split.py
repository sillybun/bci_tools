"""Real XDF round trips: session identity, implicit stamps and duplicates."""
from pathlib import Path
import struct

import numpy as np
import pytest

from data_analysis_gui.core.xdf_split import _load, _varint, _write, split_recording


def recording(path):
    names = {1: ('events', 'session-A', 'string'), 2: ('events', 'session-B', 'string'),
             3: ('spike_counts', '', 'int16'), 4: ('spike_counts', '', 'int16')}
    with Path(path).open('wb') as file:
        file.write(b'XDF:')
        _write(file, 1, b'<info><version>1</version></info>')
        for sid, (name, owner, fmt) in names.items():
            header = (f'<info><name>{name}</name><type>test</type><channel_count>1</channel_count>'
                      f'<channel_format>{fmt}</channel_format><nominal_srate>10</nominal_srate>'
                      f'<desc><session_name>{owner}</session_name></desc></info>').encode()
            _write(file, 2, struct.pack('<I', sid) + header)
            _write(file, 4, struct.pack('<Idd', sid, 0., .003))
        for sid in (3, 4):
            # Split samples across chunks; second chunk starts with implicit time.
            for start, stop in ((0, 20), (20, 41)):
                rows = bytearray()
                for k in range(start, stop):
                    rows.extend(b'\x08' + struct.pack('<d', 0.) if k == 0 else b'\x00')
                    rows.extend(struct.pack('<h', k))
                _write(file, 3, struct.pack('<I', sid) + _varint(stop - start) + rows)
        for sid, times in ((1, (1., 1.3)), (2, (3., 3.3))):
            rows = bytearray()
            for k, stamp in enumerate(times):
                value = f'marker-{sid}-{k}'.encode()
                rows.extend(b'\x08' + struct.pack('<d', stamp) + _varint(len(value)) + value)
            _write(file, 3, struct.pack('<I', sid) + _varint(2) + rows)
        for sid in names:
            count = 41 if sid in (3, 4) else 2
            _write(file, 6, struct.pack('<I', sid) + f'<info><sample_count>{count}</sample_count></info>'.encode())


def test_split_retains_exact_values_timestamps_duplicate_streams_and_clock_offsets(tmp_path):
    source = tmp_path / 'source.xdf'
    recording(source)
    before = source.read_bytes()
    rows = split_recording(source, tmp_path / 'split')
    assert [r['session'] for r in rows] == ['session-A', 'session-B']
    original = _load(source)
    for i, row in enumerate(rows):
        streams = _load(row['output'])
        assert set(streams) == {i + 1, 3, 4}
        assert streams[i + 1]['time_series'] == original[i + 1]['time_series']
        assert np.array_equal(streams[3]['time_series'], streams[4]['time_series'])
        stamps = original[3]['time_stamps']
        keep = (stamps >= row['shared_window_start']) & (stamps <= row['shared_window_end'])
        assert np.array_equal(streams[3]['time_stamps'], stamps[keep])
        assert streams[3]['time_stamps'][0] > 0  # Must not infer from removed original sample zero.
        assert np.array_equal(streams[3]['time_series'], original[3]['time_series'][keep])
        assert streams[3]['footer']['info']['sample_count'] == [str(keep.sum())]
        assert row['verification'].startswith('all retained samples')
        assert (Path(row['output']).parent.parent / 'xdf_split_manifest.json').is_file()
    assert source.read_bytes() == before


def test_split_context_does_not_duplicate_shared_samples(tmp_path):
    source = tmp_path / 'source.xdf'
    recording(source)
    rows = split_recording(source, tmp_path / 'split', context_seconds=5)
    a, b = [_load(r['output'])[3]['time_stamps'] for r in rows]
    assert not set(a) & set(b)
    np.testing.assert_array_equal(np.r_[a, b], _load(source)[3]['time_stamps'])


def test_split_refuses_to_overwrite_outputs(tmp_path):
    source = tmp_path / 'source.xdf'
    recording(source)
    with pytest.raises(FileExistsError):
        split_recording(source, tmp_path)


def test_session_model_lookup_never_borrows_other_sessions(tmp_path):
    from types import SimpleNamespace
    from data_analysis_gui.sections.s11_online_vs_offline_v1.model_replay import resolve_run

    day = tmp_path / '20261109'
    own = day / 'CO8_run-006' / 'sessions' / 'task-A'
    other = day / 'CO8_run-006' / 'sessions' / 'task-B' / 'refit'
    other.mkdir(parents=True)
    (other / 'online_update_swap_history.jsonl').write_text('')
    r = SimpleNamespace(path=own / 'xdf' / 'r.xdf')
    assert resolve_run(r) is None
    refit = own / 'refit' / '20261109_134918'
    refit.mkdir(parents=True)
    (refit / 'online_update_swap_history.jsonl').write_text('')
    assert resolve_run(r) == refit
