"""S07 electrode overlap, inherited source tests and cache dependency contracts."""
from dataclasses import replace

import numpy as np
import pytest

from data_analysis_gui import storage
from data_analysis_gui.cli import _param_filter
from data_analysis_gui.core.filters import TrialFilter
from data_analysis_gui.core.synthetic import make_session
from data_analysis_gui.runner import build_context, run_section
from data_analysis_gui.sections import versions
from data_analysis_gui.sections.s07_baseline_change_v1.overlap import (
    PARTITIONS, classify, collect, overlap_rows, tier_rows,
)


def test_overlap_partitions_and_zero_denominators():
    groups = classify([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0],
                      [1, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0])
    assert groups['categories'].tolist() == ['both', 'direction_only', 'active_not_significant',
                                           'time_only', 'inactive', 'inactive']
    rows = {r['category']: r for r in overlap_rows(groups)}
    assert rows['active']['channels'] == 4
    assert rows['both_tuned']['pct_active'] == 25
    assert rows['direction_tuned']['pct_active'] == 50
    assert rows['direction_tuned']['pct_transmitted'] == pytest.approx(100 * 2 / 6)
    tiers = np.array(['good', 'good', 'uncertain', 'bad', 'bad', 'unlisted'])
    composition = tier_rows(groups, tiers)
    assert sum(r['channels'] for r in composition) == 6
    for tier in np.unique(tiers):
        assert sum(r['channels'] for r in composition if r['tier'] == tier) == (tiers == tier).sum()
    empty = classify(np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4))
    assert all(r['pct_active'] is None for r in overlap_rows(empty))
    assert all(r['percent'] is None for r in tier_rows(empty, np.full(4, 'good')) if r['tier'] != 'good')


def test_summary_matches_standalone_s05_s06_and_config(tmp_path):
    session = make_session(n_blocks=6)
    cache = {}
    s05, s06, s07 = [versions(sid)[-1] for sid in ('s05', 's06', 's07')]
    p5 = dict(alpha=.01, gain_control=True)
    p6 = dict(alpha=.03, baseline_start=-.2, tmin=-.3, tmax=1.5, bin=.1)
    sources = {'s05': dict(version='v1', params=p5), 's06': dict(version='v1', params=p6)}
    config = tmp_path / 'configs.ini'
    config.write_text('good="' + ','.join(map(str, range(200))) + '"\nuncertain="' +
                      ','.join(map(str, range(200, 300))) + '"\nbad="' +
                      ','.join(map(str, range(300, 450))) + '"\n')
    ctx = build_context([session], TrialFilter(), {'_source_sections': sources}, s07,
                        channel_config_path=str(config), cache=cache)
    direction = run_section(s05, build_context([session], TrialFilter(), p5, s05, cache=cache))
    time = run_section(s06, build_context([session], TrialFilter(), p6, s06, cache=cache))
    combined = run_section(s07, ctx)
    try:
        direction_rows = {r['electrode']: r for r in direction.tables['anova_channels']}
        time_rows = {r['electrode']: r for r in time.tables['psth_channels']}
        rows = combined.tables['tuning_overlap_channels']
        assert len(rows) == 512
        for row in rows:
            expected_d, expected_t = direction_rows[row['electrode']], time_rows[row['electrode']]
            assert row['direction_tuned'] == expected_d['significant']
            assert row['time_tuned'] == expected_t['significant']
            assert row['direction_q'] == expected_d['q']
            assert row['time_q'] == expected_t['q']
            assert row['both_tuned'] == (expected_d['significant'] and expected_t['significant'])
        assert combined.summary['tuning_direction_channels'] == direction.summary['significant_channels']
        assert combined.summary['tuning_time_channels'] == time.summary['responsive_channels']
        assert [title for title, _ in combined.figures[:2]] == [
            'Direction and time tuning overlap', 'Tuning composition by config tier']
        assert combined.summary['tuning_config_status'] == 'computed'
        assert sum(r['channels'] for r in combined.tables['tuning_config_composition']) == 512
        assert {r['tier'] for r in combined.tables['tuning_config_composition']} == {'good', 'uncertain', 'bad', 'unlisted'}
        assert all(r['category'] in PARTITIONS for r in combined.tables['tuning_config_composition'])
        assert combined.figures[2][0] == 'Direction-specific tuning counts'
        detailed = combined.tables['direction_specific_tuning_channels']
        for row in combined.tables['direction_specific_tuning_counts']:
            selected = [r for r in detailed if r['direction_deg'] == row['direction_deg']]
            assert row['preferred_channels'] == sum(r['preferred'] for r in selected)
            if not row['tested_time_channels']:
                assert row['time_channels'] is None and row['preferred_and_time_channels'] is None
                continue
            assert row['time_channels'] == sum(r['temporal_significant'] is True for r in selected)
            assert row['preferred_and_time_channels'] == sum(
                r['preferred'] and r['temporal_significant'] is True for r in selected)
            assert row['direction_and_time_channels'] == sum(
                r['direction_anova_significant'] and r['temporal_significant'] is True for r in selected)
    finally:
        for result in (direction, time, combined):
            for _, fig in result.figures:
                fig.clear()


def test_cli_source_parameters_and_signature_dependencies(tmp_path, monkeypatch):
    s07 = versions('s07')[-1]
    raw = _param_filter([('s05:alpha', '.01'), ('s06:bin', '.1')])(s07)
    params = s07.resolve_params(raw)
    assert params['_source_sections']['s05']['params']['alpha'] == .01
    assert params['_source_sections']['s06']['params']['bin'] == .1
    source = tmp_path / 'recording.xdf'
    source.write_bytes(b'identity for signature input test')
    recording = replace(make_session(n_blocks=2).recording, path=source)
    inputs = storage.signature_inputs(s07, [recording], TrialFilter(), params)
    assert set(inputs['source_sections']) == {'s05', 's06'}
    assert inputs['source_sections']['s06']['params']['bin'] == .1
    alternate = s07.resolve_params(_param_filter([('s05:alpha', '.02'), ('s06:bin', '.1')])(s07))
    assert storage.fingerprint(inputs) != storage.fingerprint(
        storage.signature_inputs(s07, [recording], TrialFilter(), alternate))
    source_digest = storage.source_digest
    monkeypatch.setattr(storage, 'source_digest', lambda p: 'changed s06 code' if p.name.startswith('s06_') else source_digest(p))
    changed = storage.signature_inputs(s07, [recording], TrialFilter(), params)
    assert inputs['code'] == changed['code']
    assert storage.fingerprint(inputs) != storage.fingerprint(changed)


def test_no_config_and_unavailable_source_are_explicit():
    s07 = versions('s07')[-1]
    session = make_session(n_blocks=4)
    result = run_section(s07, build_context([session], TrialFilter(), {}, s07, channel_config_path=''))
    try:
        assert result.summary['tuning_config_status'] == 'no_configs_ini'
        assert not any(title == 'Tuning composition by config tier' for title, _ in result.figures)
    finally:
        for _, fig in result.figures:
            fig.clear()
    params = {'_source_sections': {'s06': dict(params={'baseline_start': -100, 'baseline_end': -99})}}
    unavailable = run_section(s07, build_context([session], TrialFilter(), params, s07, channel_config_path=''))
    try:
        assert unavailable.summary['tuning_overlap_status'] != 'computed'
        assert 'tuning_direction_channels' not in unavailable.summary
        assert 'tuning_overlap_counts' not in unavailable.tables
    finally:
        for _, fig in unavailable.figures:
            fig.clear()


def test_gui_collects_saved_and_edited_source_parameters():
    from data_analysis_gui.gui.app import SingleView

    class FakeForm:
        def raw(self):
            return dict(alpha=.02, bin=.1)

    class View:
        _raw_params = SingleView._raw_params
        forms = {'s06_v1': FakeForm()}
        saved_params = {'s05_v1': dict(alpha=.01)}

    raw = View()._raw_params(versions('s07')[-1])
    resolved = versions('s07')[-1].resolve_params(raw)
    assert resolved['_source_sections']['s05']['params']['alpha'] == .01
    assert resolved['_source_sections']['s06']['params']['alpha'] == .02


def test_direction_specific_timing_ties_and_untestable_directions(monkeypatch):
    from data_analysis_gui.sections.s07_baseline_change_v1 import directional as D
    section = versions('s07')[-1]
    from types import SimpleNamespace
    ctx = build_context([make_session(n_blocks=2)], TrialFilter(), {}, section)
    ctx = replace(ctx, sessions=[SimpleNamespace(electrode_ids=np.arange(1, 4))])
    rng = np.random.default_rng(17)
    # Same mean response at two directions, opposite timing; third channel is silent.
    binned = rng.normal(10, .2, (24, 2, 3))
    binned[:12, 0, 0] += 5
    binned[12:, 1, 0] += 5
    binned[:, :, 2] = 0
    info = dict(post=np.array([0, 1]), centres=np.array([.05, .15]),
                valid=np.ones(24, bool), dirs=np.repeat([0, 1], 12),
                binned=binned, base=np.tile([10., 10., 0.], (24, 1)))
    means = np.full((8, 3), np.nan)
    means[:2] = [[12.5, 11, 0], [12.5, 10, 0]]
    monkeypatch.setattr(D, 'channel_anova', lambda ctx: ({'mean': means}, None, None))
    monkeypatch.setattr(D, 'temporal_statistics', lambda ctx: info)
    groups = dict(sources=ctx.params['_source_sections'], direction=np.array([True, True, False]))
    result = D.direction_statistics(ctx, groups)
    assert result['preferred'].tolist() == [-1, 0, -1]  # ties not forced into 0 degrees
    assert result['sig'][:2, 0].all()  # same channel counted under both directions
    assert np.allclose(result['peak_time'][:2, 0], [.05, .15])
    assert result['rows'][2]['time_channels'] is None  # absent trials are not zero tuning
    assert not result['eligible'][:, 2].any()  # silent pairs outside BH family
    assert result['unassigned'] == 1
