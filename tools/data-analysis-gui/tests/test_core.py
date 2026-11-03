import numpy as np

from data_analysis_gui.core import analysis as A
from data_analysis_gui.core.channels import column_tiers, id_grid, read_channel_config
from data_analysis_gui.core.features import velocity_bins
from data_analysis_gui.core.filters import TrialFilter, apply_filter
from data_analysis_gui.core.synthetic import make_session


def test_trial_reasons_and_success():
    s = make_session(seed=1, n_blocks=6, miss_every=10)
    c = apply_filter(s, TrialFilter())
    reasons = c.reason_counts()
    assert reasons['end_without_start'] == 1          # orphan end inserted by the generator
    assert reasons['not_success'] == 48 // 10
    assert reasons['overtime'] == 48 // 10             # misses last 11 s
    assert c.n == 48 - 48 // 10
    relaxed = apply_filter(s, TrialFilter(success_only=False, overtime_enabled=False))
    assert relaxed.n == 48


def test_neural_gap_excludes_trial():
    s = make_session(seed=2, n_blocks=4, gap_at=None)
    start = s.trials[5]['start']
    s2 = make_session(seed=2, n_blocks=4, gap_at=start + 0.3)
    c = apply_filter(s2, TrialFilter())
    assert 'neural_gap_in_window' in c.reason_counts() or 'no_neural_in_window' in c.reason_counts()
    assert c.n == apply_filter(s, TrialFilter()).n - 1


def test_start_without_end_relaxation():
    s = make_session(seed=3, n_blocks=3)
    # drop the T_TRIAL_END of trial 4
    ends = [i for i, e in enumerate(s.events) if e[0] == 'T_TRIAL_END']
    drop = ends[5]
    s.events.pop(drop)
    s.event_times = np.delete(s.event_times, drop)
    s.__dict__.pop('trials', None)
    strict = apply_filter(s, TrialFilter(success_only=False))
    assert strict.reason_counts()['nested_start'] == 1
    relaxed = apply_filter(s, TrialFilter(success_only=False, exclude_incomplete=False))
    assert relaxed.n == strict.n + 1
    assert sum(r['inferred_end'] for r in relaxed.rows) == 1


def test_window_rates_match_direct_mean():
    s = make_session(seed=4, n_blocks=3)
    c = apply_filter(s, TrialFilter())
    i = 3
    lo, hi = np.searchsorted(s.neural_times, [c.starts[i] + .2, c.starts[i] + 1.2])
    assert np.allclose(c.window_rates[i], s.counts[lo:hi].mean(0) / 0.02)


def test_decoders_beat_chance_on_tuned_data():
    s = make_session(seed=5)
    c = apply_filter(s, TrialFilter())
    scores, op, folds, tested = A.ridge_classifier_cv(c.window_rates, c.directions, c.blocks)
    assert A.balanced_accuracy(c.directions[tested], scores.argmax(1)[tested]) > 0.4
    vb = velocity_bins(c)
    pred, f, _ = A.cv_velocity(vb.features, vb.velocity, vb.trial, vb.sequence, vb.block)
    m = f >= 0
    assert A.r2_score(vb.velocity[m], pred['ridge'][m]).mean() > 0.3
    assert A.r2_score(vb.velocity[m], pred['kf'][m]).mean() > 0.3


def test_folds_purge_adjacent_blocks():
    blocks = np.repeat(np.arange(1, 21), 8)
    for _, tr, te in A.block_folds(blocks):
        assert not set(blocks[tr]) & set(range(blocks[te].min() - 1, blocks[te].max() + 2))


def test_channel_config_and_grid(tmp_path):
    ini = tmp_path / 'configs.ini'
    ini.write_text('good="0,1,2"\nuncertain="3"\nbad="4,5"\n')
    cfg = read_channel_config(ini)
    assert cfg['good'] == [1, 2, 3]
    assert list(column_tiers([1, 4, 6, 9], cfg)) == ['good', 'uncertain', 'bad', 'unlisted']
    grid = id_grid([1, 129], [5.0, 7.0])
    assert grid.shape == (128, 8) and grid[0, 0] == 5 and grid[0, 1] == 7 and np.isnan(grid[1, 0])


def test_v2_criteria_ambiguous_flags_and_session(tmp_path):
    s = make_session(seed=7, n_blocks=12, miss_every=0)
    # make trial 3 ambiguous (second acquisition) and trial 9 off-grid
    starts = [i for i, e in enumerate(s.events) if e[0] == 'T_TRIAL_START']
    acq = next(i for i in range(starts[3], len(s.events)) if s.events[i][0] == 'T_TARGET_ACQUIRED')
    s.events.insert(acq + 1, ['T_TARGET_ACQUIRED', '13', '0', ''])
    s.event_times = np.insert(s.event_times, acq + 1, s.event_times[acq] + 0.01)
    s.__dict__.pop('trials', None)
    c = apply_filter(s, TrialFilter())
    assert c.reason_counts()['ambiguous_outcome'] == 1
    row = next(r for r in c.table if r['outcome'] == 'ambiguous')
    assert row['primary_reason'] == 'ambiguous_outcome' and row['checks']['outcome'] == 'fail'
    # synthetic trials acquire ~1.4-2.8 s: no too-fast flags, no acquired-before-window flags
    assert c.flag_counts().get('too_fast', 0) == 0
    flagged = apply_filter(s, TrialFilter(too_fast_seconds=5.0))
    assert flagged.flag_counts()['too_fast'] > 0 and flagged.n == c.n          # flag does not exclude
    excluded = apply_filter(s, TrialFilter(too_fast_seconds=5.0, too_fast_action='exclude'))
    assert excluded.n < c.n
    status = c.session_status()
    assert status['qualified'] and status['session_type'] == 'offline'
    strict = apply_filter(s, TrialFilter(min_velocity_trials=10_000))
    assert not strict.session_status()['qualified']
    # config round trip
    path = tmp_path / 'filter.json'
    TrialFilter(overtime_seconds=7.5).save(path)
    assert TrialFilter.load(path).overtime_seconds == 7.5


def test_velocity_bin_policy():
    s = make_session(seed=8, n_blocks=6)
    # remove cursor samples in the middle of one trial's window -> cursor gap
    c = apply_filter(s, TrialFilter())
    t0 = c.starts[2]
    keep = ~((s.cursor_times > t0 + 0.55) & (s.cursor_times < t0 + 0.8))
    s.cursor_times, s.cursor = s.cursor_times[keep], s.cursor[keep]
    c = apply_filter(s, TrialFilter())
    drop_trial = velocity_bins(c)
    drop_bin = velocity_bins(c, policy='drop_bin')
    assert 2 not in set(drop_trial.trial) and 2 in set(drop_bin.trial)
    assert not c.velocity_include[2] and 'velocity_cursor_gap' in c.velocity_audit['reasons'][2]
    assert drop_bin.n > drop_trial.n


def test_no_layout_means_index_grid():
    from data_analysis_gui.core.channels import read_layout
    assert read_layout('') is None            # unchecked: default index arrangement on 128 x 8
