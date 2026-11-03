import math

import pytest

from data_analysis_gui.core import behavior
from data_analysis_gui.sections.s13_online_behavior_v1 import co8_bps, rolling_bps


def trials(durations):
    return [dict(trial=i, block=1, outcome='acquired', duration_s=t,
                 rt_s=t, outbound_s=t, bits=3) for i, t in enumerate(durations, 1)]


def test_60_active_second_windows_include_timeouts_and_update_each_trial():
    rows = trials([10] * 6 + [5])
    rows[1]['outcome'] = 'timeout'
    result = rolling_bps.windows(rows, 'grid')
    assert [(r['trial_start'], r['trial_end']) for r in result] == [(1, 6), (1, 7)]
    assert [r['active_attempt_seconds'] for r in result] == [60, 60]
    assert result[0]['bps'] == pytest.approx(5 * 3 / 60)
    assert result[1]['bps'] == pytest.approx(6 * 3 / 60)
    assert result[1]['partial_first_trial']
    assert (result[1]['window_start_active_s'], result[1]['window_end_active_s']) == (5, 65)
    assert result[0]['timeouts'] == 1


def test_warmup_and_zero_or_invalid_time_never_inflate_peak():
    assert rolling_bps.windows(trials([10] * 5), 'grid') == []
    assert rolling_bps.windows(trials([0] * 10), 'grid') == []
    rows = trials([30, 30, math.nan, 30, 30])
    result = rolling_bps.windows(rows, 'grid')
    assert [r['trial_end'] for r in result] == [2, 5]
    assert result[0]['segment_id'] != result[1]['segment_id']
    rows = trials([30] * 4)
    rows[2]['trial'], rows[3]['trial'] = 4, 5
    result = rolling_bps.windows(rows, 'grid')
    assert [r['trial_end'] for r in result] == [2, 5]


def test_window_left_boundary_excludes_completion_at_exact_cutoff():
    result = rolling_bps.windows(trials([20] * 4), 'grid')
    assert [(r['trial_start'], r['trial_end']) for r in result] == [(1, 3), (2, 4)]
    assert all(r['bps'] == pytest.approx(9 / 60) for r in result)


def test_partial_attempt_time_is_clipped_and_fitts_uses_only_contained_trials():
    grid, = rolling_bps.windows(trials([30, 40]), 'grid')
    assert grid['bps'] == pytest.approx(6 / 60)
    assert grid['active_attempt_seconds'] == 60
    co8, = rolling_bps.windows(trials([30, 40]), 'centerout')
    assert co8['fitts_bps'] == pytest.approx(3 / 40)
    assert co8['fitts_fully_contained_trials'] == 1


def test_daily_distribution_uses_only_full_windows_and_keeps_peak_source():
    a = rolling_bps.windows(trials([10] * 7), 'grid', session='A', date='2026-11-09')
    b = rolling_bps.windows(trials([20] * 3), 'grid', session='B', date='2026-11-09')
    daily = rolling_bps.daily_summary(a + b, 'grid', dates=['2026-11-10'])
    row = daily[0]
    assert row['windows'] == 3 and row['sessions'] == 2
    assert row['bps_median'] == pytest.approx(.3)
    assert row['bps_q1'] == pytest.approx(.225) and row['bps_q3'] == pytest.approx(.3)
    assert row['bps_peak_session'] == 'A'
    assert (row['bps_peak_trial_start'], row['bps_peak_trial_end']) == (1, 6)
    assert row['bps_peak_end_active_s'] == 60
    assert daily[1]['bps_n'] == 0 and math.isnan(daily[1]['bps_peak'])


def test_same_day_runs_stay_separate_and_fragments_do_not_share_windows():
    a = rolling_bps.windows(trials([10] * 6), 'grid', session='A-main', date='2026-11-09', run='A')
    fragment = rolling_bps.windows(trials([20] * 3), 'grid', session='A-fragment', date='2026-11-09', run='A')
    b = rolling_bps.windows(trials([30] * 2), 'grid', session='B', date='2026-11-09', run='B')
    rows = rolling_bps.run_summary(a + fragment + b, 'grid', runs=[
        dict(run_key='A', session='run-001', date='2026-11-09'),
        dict(run_key='A', session='run-001', date='2026-11-09'),
        dict(run_key='B', session='run-002', date='2026-11-09'),
        dict(run_key='C', session='run-003', date='2026-11-09')])
    assert len(rows) == 3
    assert rows[0]['windows'] == 2 and rows[0]['bps_median'] == pytest.approx(.225)
    assert rows[0]['bps_peak_session'] == 'A-main'
    assert rows[1]['bps_median'] == rows[1]['bps_peak'] == pytest.approx(.1)
    assert rows[2]['bps_n'] == 0 and math.isnan(rows[2]['bps_peak'])


def test_co8_run_behavior_uses_complete_csv_without_double_counting(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from data_analysis_gui.core.catalog import Recording
    from data_analysis_gui.core.filters import TrialFilter
    from data_analysis_gui.sections.base import AnalysisContext
    from data_analysis_gui.sections.x08_online_behavior_v1.section import SECTION
    from data_analysis_gui.sections.s13_online_behavior_v1 import run_behavior

    folder = tmp_path / '20261111' / 'run-002-CO8-online' / 'xdf'
    records = [Recording(folder / f'{part}.xdf', f'{part}.xdf', '2026-11-11',
                         'centerout', True, '002') for part in ('early', 'main')]
    complete = trials([1, 2, 9, 10])
    complete[-1]['outcome'] = 'timeout'
    monkeypatch.setattr(behavior, 'source_files', lambda r:
                        {'trials': tmp_path / 'trials.csv'} if r.path.stem == 'main' else {})
    monkeypatch.setattr(behavior, 'collect', lambda s:
                        (complete, 'CSV', '') if s.recording.path.stem == 'main'
                        else (trials([100]), 'XDF event markers', ''))
    monkeypatch.setattr(co8_bps, 'add_xdf_attempt_times', lambda s, t, source: t)
    sessions = [SimpleNamespace(recording=r, task_mode='centerout') for r in records]
    result = SECTION.run(AnalysisContext(sessions, [], TrialFilter(), SECTION.defaults()))
    row, = result.tables['co8_run_behavior']
    assert row['attempts'] == 4 and row['successes'] == 3
    assert row['success_pct'] == 75 and row['rt_median_s'] == 2
    assert row['rt_q1_s'] == 1.5 and row['rt_q3_s'] == 5.5
    title, fig = result.figures[0]
    assert title == run_behavior.TITLE and len(fig.axes) == 2
    assert fig.axes[0].get_ylabel() == 'Success rate (%)'
    assert fig.axes[1].get_ylabel() == 'Time to target (s)'


def test_csv_success_uses_acquisition_not_feedback_or_next_start(tmp_path):
    path = tmp_path / 'trials.csv'
    path.write_text(
        'task_mode,session_timestamp_iso,trial_num_overall,block_num,outcome,trial_start_iso,acquisition_iso,trial_end_iso,trial_duration_sec\n'
        'grid,one,1,1,acquired,2026-11-09T10:00:00,2026-11-09T10:00:02,2026-11-09T10:00:12,12\n'
        'grid,one,2,1,timeout,2026-11-09T10:01:00,,2026-11-09T10:01:15,10\n', encoding='utf-8')
    rows = behavior._from_csv({'trials': path}, 'grid')
    assert rows[0]['duration_s'] == 2
    assert rows[1]['duration_s'] == 10


def test_xdf_miss_stops_before_feedback_and_never_uses_trial_end():
    from types import SimpleNamespace
    session = SimpleNamespace(trials=[dict(trial_index=0, complete=True, outcome='miss',
                                          miss_time=10, duration=14)])
    result = co8_bps.add_xdf_attempt_times(session, [dict(trial=1)], 'XDF event markers')
    assert result[0]['duration_s'] == 10
    del session.trials[0]['miss_time']
    result = co8_bps.add_xdf_attempt_times(session, [dict(trial=1)], 'XDF event markers')
    assert math.isnan(result[0]['duration_s'])


def test_reconstruction_records_first_miss_before_delayed_trial_end():
    from types import SimpleNamespace
    import numpy as np
    from data_analysis_gui.core.session import reconstruct_trials

    session = SimpleNamespace(event_times=[1., 11., 12., 15.],
                              events=[['T_TRIAL_START'], ['MISS'], ['MISS'], ['T_TRIAL_END']],
                              active_cursor=lambda row: np.array([], dtype=int))
    row, = reconstruct_trials(session)
    assert row['miss_time'] == 10
    assert row['end'] == 15


def test_grid_uses_achieved_bitrate_and_combines_median_iqr_with_peak():
    windows = rolling_bps.windows(trials([10] * 7), 'grid', session='run-001', date='2026-11-09')
    rows = rolling_bps.run_summary(windows, 'grid', runs=[
        dict(run_key='run-001', session='run-001', date='2026-11-09')],
        trials_by_run={'run-001': trials([10] * 7)})
    assert rolling_bps.methods('grid') == [('bps', 'BPS = C × log₂(N) / T')]
    _, fig = rolling_bps.run_figure(rows, 'grid')
    try:
        assert len(fig.axes) == 2
        assert fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[1])
        assert fig.get_figwidth() / fig.get_figheight() == pytest.approx(297 / 210)
        assert {t.get_text() for t in fig.axes[0].get_legend().get_texts()} == {'BPS · median and IQR', 'BPS · peak'}
        peak, = [line for line in fig.axes[0].lines if line.get_label() == 'BPS · peak']
        assert peak.get_color() == '#bd5021'
        assert peak.get_ydata()[0] == pytest.approx(.3)
        assert fig.axes[1].get_ylabel() == 'Time to target (s)'
        assert fig.axes[1].containers[0].lines[0].get_ydata()[0] == 10
        assert fig.axes[1].containers[0].lines[0].get_color() == '#c25e00'
    finally:
        fig.clear()
    fig = rolling_bps.rolling_figure(windows, 'grid', 'Grid rolling')
    try:
        assert len(fig.axes) == 1
    finally:
        fig.clear()


def test_run_rt_pools_successful_trials_once_not_overlapping_windows():
    original = trials([10] * 6)
    for row, rt in zip(original, [1, 2, 3, 4, 100, 999]):
        row['rt_s'] = rt
    original[-1]['outcome'] = 'timeout'
    fragment = trials([10, 10])
    fragment[0]['rt_s'], fragment[1]['rt_s'] = 10, 20
    windows = rolling_bps.windows(original, 'grid', session='A', date='2026-11-11', run='A')
    row, = rolling_bps.run_summary(windows, 'grid', runs=[dict(run_key='A', date='2026-11-11', session='A')],
                                  trials_by_run={'A': original + fragment})
    assert row['rt_n'] == 7
    assert row['rt_median_s'] == 4
    assert row['rt_q1_s'] == 2.5 and row['rt_q3_s'] == 15


def test_six_bps_ttt_reference_uses_verified_grid_size():
    for n, expected in [(36, math.log2(36)/6), (64, 1), (100, math.log2(100)/6),
                        (144, math.log2(144)/6), (49, math.log2(49)/6)]:
        assert rolling_bps.ttt_for_bps(n) == pytest.approx(expected)
    assert math.isnan(rolling_bps.ttt_for_bps(None))
    windows = rolling_bps.windows(trials([10] * 6), 'grid', session='A', date='2026-11-11', run='A')
    summary, = rolling_bps.run_summary(windows, 'grid', runs=[dict(run_key='A', date='2026-11-11', session='A')])
    assert summary['grid_cells'] == 8 and summary['ttt_for_6bps_s'] == .5
    _, fig = rolling_bps.grid_run_figure([summary])
    try:
        reference, = [line for line in fig.axes[1].lines if line.get_label() == '6 bits/s reference · 100% success']
        assert reference.get_ydata()[0] == .5
    finally:
        fig.clear()
