"""Every registered section runs end to end on synthetic sessions."""
import matplotlib
matplotlib.use('Agg')

import pytest

from data_analysis_gui.core.filters import TrialFilter
from data_analysis_gui.core.synthetic import make_session
from data_analysis_gui import storage
from data_analysis_gui.runner import build_context, run_section
from data_analysis_gui.sections import LOAD_ERRORS, registry, sections

SINGLE = sections('single')
CROSS = sections('cross')


@pytest.fixture(scope='module')
def synthetic():
    return [make_session(seed=i, date=f'2026-01-0{i + 1}', online=(i == 0)) for i in range(3)]


def test_all_sections_import():
    registry(reload=True)
    assert not LOAD_ERRORS, LOAD_ERRORS


def test_s01_recording_raster_keeps_late_trials_for_each_electrode(synthetic):
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    section = next(s for s in SINGLE if s.id == 's01')
    session = synthetic[0]
    electrodes = list(map(int, session.electrode_ids[:2]))
    context = build_context([session], TrialFilter(), dict(
        electrode=','.join(map(str, electrodes)), recording_raster_direction='90°', max_trials=5), section)
    result = run_section(section, context)
    rows = context.cohort.directions == 2
    starts = context.cohort.starts[rows]
    assert len(starts) > 5
    pages = [(fig, data) for (title, fig), data in zip(result.figures, result.figure_data)
             if title.startswith('Direction raster over recording time')]
    assert len(pages) == 2
    try:
        for electrode, (fig, data) in zip(electrodes, pages):
            trials = data['trials']
            assert {trial['electrode'] for trial in trials} == {electrode}
            assert {trial['direction_deg'] for trial in trials} == {90}
            np.testing.assert_allclose([trial['elapsed_min'] for trial in trials],
                                       (starts - session.neural_times[0]) / 60)
            assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()
            FigureCanvasAgg(fig).draw()
    finally:
        for _, fig in result.figures:
            fig.clear()


def test_s03_counts_with_missing_metadata_use_timestamp_alignment():
    import numpy as np
    from data_analysis_gui.sections import versions

    section = versions('s03')[0]
    session = make_session(n_blocks=2, fpga=True, drop_every=0)
    keep = np.ones(len(session.neural_times), dtype=bool)
    keep[[20, -1]] = False
    session.metadata = session.metadata[keep]
    session.metadata_times = session.neural_times[keep]
    result = run_section(section, build_context([session], TrialFilter(), {}, section))
    assert result.summary['metadata_matched_frames'] == len(session.metadata)
    assert result.summary['fpga_frames_lost'] == 0
    overview = result.figure_data[0]
    assert len(overview['receive_intervals']['interval_ms']) == len(session.neural_times) - 1
    assert len(overview['fpga_clock_steps']['step_ms']) == len(session.metadata) - 2


@pytest.mark.parametrize('fpga', [True, False], ids=['fpga', 'receive-only'])
def test_s03_merged_diagnostics(fpga):
    import numpy as np

    from data_analysis_gui.sections import versions

    available = versions('s03')
    assert [s.version for s in available] == ['v1']
    section = available[0]
    session = make_session(n_blocks=2, fpga=fpga, drop_every=400)
    ctx = build_context([session], TrialFilter(), {'max_gap_figures': 1}, section)
    result = run_section(section, ctx)
    try:
        titles = [title for title, _ in result.figures]
        assert not any(title.startswith(('Supplementary Inter-frame interval', 'Supplementary Acquisition-clock',
                                         'Supplementary Host timestamp')) for title in titles)
        assert 'Loss over time and per trial' in titles
        expected_primary = ['Transmission overview'] + (['Delivery delay overview'] if fpga else []) + [
            'Loss over time and per trial']
        assert titles[:len(expected_primary)] == expected_primary
        assert all(title.startswith('Supplementary ') for title in titles[len(expected_primary):])
        assert bool(result.summary['fpga_clock']) == fpga
        histogram = result.figure_data[0]['receive_interval_histogram']
        assert int(np.sum(histogram['count'])) == len(session.neural_times) - 1
        assert len(histogram['count']) == len(histogram['bin_left_ms']) == len(histogram['bin_right_ms'])
        overview_axes = result.figures[0][1].axes
        assert overview_axes[3].patches
        assert 'Supplementary Relative delivery delay over recording time' not in titles
        assert 'Aligned transport timeline' not in titles
        assert all(overview_axes[0].get_shared_x_axes().joined(overview_axes[0], ax)
                   for ax in overview_axes[1:3])
        if fpga:
            clock = session.metadata[:, 4]
            lost = np.maximum(np.rint(np.diff(clock) / 19.2).astype(int) - 1, 0)
            assert result.summary['fpga_frames_lost'] == int(lost.sum())
            assert result.summary['gap_figures_shown'] == 1
            assert len(result.tables['acquisition_gaps']) == int((lost > 0).sum())
            assert 'Supplementary Nonnegative relative delivery delay' not in titles
            delay_index = titles.index('Delivery delay overview')
            assert len(result.figures[delay_index][1].axes) == 3
            delay_data = result.figure_data[delay_index]
            assert delay_data['histogram_2ms']['count'].sum() == len(session.neural_times)
            assert sum(row['n_frames'] for row in delay_data['windows']) == len(session.neural_times)
        else:
            assert 'acquisition_gaps' not in result.tables
            assert 'UNKNOWN' in result.figures[0][1]._suptitle.get_text()
        _check_guide(section, result)
    finally:
        for _, fig in result.figures:
            fig.clear()


@pytest.mark.parametrize('section', SINGLE, ids=lambda s: s.id)
def test_single_day(section, synthetic, tmp_path):
    config = ''
    if 'channel_config' in section.requires:
        config = tmp_path / 'configs.ini'
        config.write_text('good="' + ','.join(map(str, range(0, 300))) + '"\n'
                          'uncertain="' + ','.join(map(str, range(300, 400))) + '"\n'
                          'bad="' + ','.join(map(str, range(400, 1024))) + '"\n')
    ctx = build_context(synthetic[:1], TrialFilter(), {}, section, channel_config_path=str(config))
    if section.id == 's14':
        from test_decoder_weights import write_depth_csv
        # Synthetic shank coordinates keep this contract test independent of local data files.
        import numpy as np
        channels = np.arange(1024)
        ctx.layout = dict(source='synthetic-shank-table', rows=128, columns=8, kind='shank_table',
                          position=np.column_stack([channels % 128, channels // 128]))
        ctx.params['depth_csv'] = str(write_depth_csv(tmp_path/'depth.csv'))
    result = run_section(section, ctx)
    assert result.figures, 'a section must produce at least one figure'
    if section.id == 's04':
        from matplotlib.patches import Ellipse

        ellipses = [patch for _, fig in result.figures for ax in fig.axes for patch in ax.patches
                    if isinstance(patch, Ellipse)]
        assert ellipses
        assert all(patch.get_facecolor()[3] == 0 for patch in ellipses)
        assert [name for name, _ in result.figures[:4]] == ['Dimensionality overview','PCA', 'UMAP embedding', 'TDR']
        assert all(name.startswith('Supplementary ') for name, _ in result.figures[4:])
        assert result.summary['time_variable'] == 'elapsed_seconds'
        assert result.summary['tdr_kinematics_computed']
        predictions = result.tables['tdr_kinematics_predictions']
        assert [r['trial_id'] for r in predictions if r['model'] == 'direction'] == [
            r['trial_id'] for r in predictions if r['model'] == '+ kinematics']
    _check_guide(section, result)
    _save(result, section, ctx, tmp_path)


@pytest.mark.parametrize('section', CROSS, ids=lambda s: s.id)
def test_cross_day(section, synthetic, tmp_path):
    ctx = build_context(synthetic, TrialFilter(), {}, section)
    if section.id == 'x09':
        import numpy as np
        ch = np.arange(1024)
        ctx.layout = dict(source='synthetic-shank-table', rows=128, columns=8, kind='shank_table',
                          position=np.column_stack([ch % 128, ch // 128]))
    result = run_section(section, ctx)
    assert result.figures
    if section.id == 'x05':
        assert result.summary['decoders'] == ['ridge', 'velocity_kf']
        assert [title for title, _ in result.figures[1:3]] == [
            'Within-day velocity and position', 'Within-day velocity by axis']
        for data, metrics in zip(result.figure_data[1:3], (('Velocity', 'Position'), ('Velocity X', 'Velocity Y'))):
            assert {row['metric'] for row in data['scores']} == set(metrics)
            assert len(data['scores']) == len(synthetic) * len(metrics) * 2
    _check_guide(section, result)
    _save(result, section, ctx, tmp_path)


def test_online_behavior_keeps_tasks_separate_and_counts_timeouts(monkeypatch):
    from types import SimpleNamespace

    from data_analysis_gui.core import behavior
    from data_analysis_gui.core.catalog import Recording
    from data_analysis_gui.sections.base import AnalysisContext

    def session(mode, online, run):
        from pathlib import Path

        rec = Recording(Path(f'{run}.xdf'), run, '2026-11-04', mode, online, run)
        return SimpleNamespace(recording=rec, task_mode=mode)

    trials = [dict(outcome='acquired', rt_s=2, bits=3, outbound_s=1.5, duration_s=2),
              dict(outcome='timeout', rt_s=10, bits=3, outbound_s=10, duration_s=10)]
    monkeypatch.setattr(behavior, 'collect', lambda s: (trials, 'test', ''))
    section = next(s for s in CROSS if s.id == 'x08')
    sessions = [session('grid', True, '001'), session('centerout', True, '002'),
                session('centerout', False, '003')]
    result = run_section(section, AnalysisContext(sessions, [], TrialFilter(), section.defaults()))
    assert [title for title, _ in result.figures] == [
        'Online CO8 Run Success and Time to Target',
        'Online CO8 Run Rolling BPS Median and IQR', 'Online CO8 Run Rolling BPS Peak',
        'Online Grid Behavior',
        'Online Grid Run Rolling BPS Summary']
    co8, = result.tables['co8_sessions']
    grid, = result.tables['grid_sessions']
    assert 'bps' not in co8 and 'bps' not in grid
    assert result.tables['co8_run_rolling_bps'][0]['neuralink_bps_n'] == 0
    assert result.tables['grid_run_rolling_bps'][0]['bps_n'] == 0
    assert co8['success_pct'] == grid['success_pct'] == 50
    assert co8['rt_median_s'] == grid['rt_median_s'] == 2


def test_x07_stability_requires_all_date_pairs_and_breaks_at_missing_dates():
    import numpy as np
    from data_analysis_gui.sections.x07_tuning_stability_v1.section import (
        daily_profiles, excluded_dates, one_day_sensitivity, stability,
    )

    angles = np.arange(8) * np.pi / 4
    days = []
    for day, phase in enumerate((0, np.pi / 4, np.pi / 2)):
        tested = np.array([True, day != 1, True])
        curves = np.stack([np.cos(angles - phase), np.cos(angles), np.cos(angles)], axis=1)
        curves[:, ~tested] = np.nan
        sign = np.array([1, 1, 1 if day < 2 else -1])
        days.append((curves, tested, tested.copy(), sign))
    # The first electrode has good adjacent pairs but incompatible endpoints.
    # The second has good endpoints, separated by a missing selected date.
    # The third has identical profiles but a response-sign reversal.
    tested, _, longest, _, stable_pairs, _, compatible = stability(days, 'psth', 0.7, 45)
    assert tested.tolist() == [3, 2, 3]
    assert longest.tolist() == [2, 1, 2]
    assert stable_pairs.tolist() == [2, 1, 1]
    dates = ['2026-10-26', '2026-10-27', '2026-11-05']
    robust, skipped, diagnostics = one_day_sensitivity(
        np.stack([day[2] for day in days]), compatible, longest, np.flatnonzero(tested >= 2), dates, 'psth')
    assert robust.tolist() == [2, 2, 2]
    assert skipped.tolist() == ['', '2026-10-27', '']
    assert diagnostics[1]['newly_stable_2plus'] == 1
    assert excluded_dates('1026，11-05', dates) == {'2026-10-26', '2026-11-05'}

    first = days[0]
    second = (first[0] + 2, first[1], np.array([False, True, True]), first[3])
    daily, _, tuned, _ = daily_profiles([first, second])
    np.testing.assert_allclose(daily, first[0] + 1)
    assert tuned.tolist() == [False, True, True]


def test_x06_explicit_reference_tests_earlier_and_later_recordings(synthetic):
    section = next(s for s in CROSS if s.id == 'x06')
    reference = synthetic[1].recording.key
    ctx = build_context(synthetic, TrialFilter(), dict(reference=reference, decoders='ridge'), section)
    result = run_section(section, ctx)
    rows = result.tables['transfer']
    assert [(row['ref_index'], row['test_index'], row['days_elapsed']) for row in rows] == [
        (1, 0, -1), (1, 2, 1)]
    assert result.summary['reference'] == reference


def _save(result, section, ctx, folder):
    assert len(result.methods) == len(result.figures)
    inputs = dict(format=storage.FORMAT_VERSION, section=section.id, version=section.version, code='test',
                  recordings=[dict(name=s.recording.path.name, sha256='test') for s in ctx.sessions],
                  criteria=ctx.criteria.to_dict(), params=ctx.params)
    record = storage.save_result(result, folder / section.id, 'sig', inputs, section,
                                 [s.recording for s in ctx.sessions])
    for figure in record['figures']:
        for ext in ('png', 'svg', 'pdf', 'csv', 'json'):
            assert (folder / section.id / figure['files'][ext]).is_file()
        assert all(path.startswith(f'figures/{section.id.upper()}_')
                   for path in (*figure['files'].values(), *figure['data_files'].values()))
        assert figure['method'], f'{section.id}: figure without method text: {figure["title"]}'
    assert storage.load_cached(folder / section.id, 'sig') is not None


def _check_guide(section, result):
    import re

    from data_analysis_gui.sections.guide import figure_guide, load_guide
    guide = load_guide(section)
    assert guide, f'{section.id}: guide.md missing'
    assert '**一句话总结：**' in guide.split('## 图', 1)[0]
    assert result.summary.get('one_sentence_summary'), f'{section.id}: missing computed one-sentence readout'
    for title, _ in result.figures:
        part = figure_guide(guide, title)
        assert part, f'{section.id}: guide.md has no "## 图 n：{title}" section'
        for heading in ('计算公式', '结果解读', '质量评价标准'):
            assert heading in part, f'{section.id} · {title}: guide section lacks "{heading}"'
        formula = part.split('### 计算公式', 1)[-1].split('### 结果解读', 1)[0]
        assert '**符号**' in formula and '**步骤**' in formula,             f'{section.id} · {title}: 计算公式 needs a **符号** list and **步骤**'
        interpretation = part.split('### 结果解读', 1)[1].split('### 质量评价标准', 1)[0]
        criteria = part.split('### 质量评价标准', 1)[1]
        labels = lambda value: re.findall(r'^- \*\*(.+?)\*\*：', value, flags=re.M)
        assert labels(interpretation) and labels(interpretation) == labels(criteria), (
            f'{section.id} · {title}: interpretations and criteria must cover the same subplot groups')
