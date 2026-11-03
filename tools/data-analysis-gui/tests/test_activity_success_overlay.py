from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data_analysis_gui.core import behavior
from data_analysis_gui.core.catalog import Recording
from data_analysis_gui.sections.x01_channel_stability_v1.section import ChannelStability
from data_analysis_gui.sections.x01_channel_stability_v1.section import tuning_counts


def test_success_summary_is_distribution_of_block_rates_and_counts_timeouts():
    rows = [dict(block=1, outcome='acquired'), dict(block=1, outcome='timeout'),
            dict(block=1, outcome='unknown'), dict(block=2, outcome='acquired'),
            dict(block=None, outcome='acquired')]
    result = behavior.block_success_summary(rows)
    assert result['success_blocks'] == 2
    assert result['success_pct_mean'] == pytest.approx(75)
    assert result['success_pct_q1'] == 62.5 and result['success_pct_q3'] == 87.5
    assert np.isnan(behavior.block_success_summary([])['success_pct_mean'])


def test_same_run_prefers_full_csv_for_early_and_main_recordings(monkeypatch):
    folder = Path('20261106/run-002-CO8-online/xdf')
    sessions = [SimpleNamespace(recording=Recording(folder / name, name, '2026-11-06', 'centerout', True, '002'),
                                task_mode='centerout') for name in ['early.xdf', 'main.xdf']]
    monkeypatch.setattr(behavior, 'collect', lambda session:
                        ([dict(block=1, outcome='timeout')], 'XDF event markers', '')
                        if session is sessions[0] else
                        ([dict(block=1, outcome='acquired')], 'behavior/trials.csv', ''))
    result = behavior.run_success_summaries(sessions)
    assert len(result) == 2
    assert all(r['success_pct_mean'] == 100 for r in result)
    assert all(r['success_source'] == 'behavior/trials.csv' for r in result)


def test_activity_keeps_channel_bars_and_adds_percent_axis():
    row = dict(session='11-06 · online · CO8 · run003', date='2026-11-06', transmitted=512,
               active=335, direction_tuned=46, success_pct_mean=75,
               success_pct_q1=60, success_pct_q3=90)
    fig = ChannelStability.activity([row])
    try:
        assert len(fig.axes) == 2
        assert fig.axes[0].get_ylim() == (0, 512)
        assert fig.axes[1].get_ylim() == (0, 100)
        assert [p.get_height() for p in fig.axes[0].patches] == [335, 46]
        assert fig.axes[1].get_ylabel() == 'Block success rate (%)'
        assert fig.axes[1].lines[0].get_ydata()[0] == 75
    finally:
        fig.clear()


def test_average_can_be_below_block_iqr():
    trials = [dict(block=b, outcome='timeout' if b <= 6 else 'acquired')
              for b in range(1, 26) for _ in range(8)]
    row = behavior.block_success_summary(trials)
    assert row['success_pct_mean'] == 76
    assert row['success_pct_q1'] == row['success_pct_q3'] == 100
    fig = ChannelStability.activity([dict(session='run001', date='2026-11-09',
                                        transmitted=512, active=300, direction_tuned=100, **row)])
    try:
        assert fig.axes[1].lines[0].get_ydata()[0] == 76
    finally:
        fig.clear()


def test_tuning_counts_have_no_sixteen_trial_threshold():
    counts = tuning_counts([[1, 0], [2, 0], [5, 0], [6, 0]], [0, 0, 1, 1])
    assert counts['trials'] == 4 and counts['active'] == 1
    assert counts['anova_df_within'] == 2
    assert counts['direction_tuned'] is not None


def test_activity_remains_available_when_anova_has_no_residual_df():
    counts = tuning_counts([[1, 0], [2, 0], [5, 0], [6, 0]], [0, 1, 2, 3])
    assert counts['active'] == 1 and counts['direction_tuned'] is None
    assert counts['anova_df_within'] == 0
    fig = ChannelStability.activity([dict(session='run003', date='2026-11-09', transmitted=512, **counts)])
    try:
        assert fig.axes[0].patches[0].get_height() == 1
        assert any(text.get_text() == 'N/A' for text in fig.axes[0].texts)
        assert not any(text.get_text() == 'nan' for text in fig.axes[0].texts)
    finally:
        fig.clear()
