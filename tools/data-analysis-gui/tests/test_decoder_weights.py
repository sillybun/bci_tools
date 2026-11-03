"""Mapping identity, coefficient units, and actual neural-to-velocity sensitivity."""
import zipfile
import csv
from pathlib import Path

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from data_analysis_gui import config
from data_analysis_gui.core.channels import read_layout, layout_grid
from data_analysis_gui.decoders import DecoderData, get
from data_analysis_gui.sections.s14_decoder_weights_v1.section import (
    SECTION, channel_rows, depth_rows, fit_weights, steady_gain)


def write_layout(path, *, duplicate=False, shared=False):
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    workbook = f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets><sheet name="shank" sheetId="1" r:id="rId1"/></sheets></workbook>'
    rels = '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>'
    headers = ''.join(f'<c r="{c}1" t="s"><v>{i}</v></c>' if shared else
                      f'<c r="{c}1" t="inlineStr"><is><t>{name}</t></is></c>'
                      for i, (c, name) in enumerate(zip('ABC', ('Channel#', 'Shank', 'Elec#'))))
    rows = [f'<row r="1">{headers}</row>']
    for n in range(1024):
        # Channel identity deliberately disagrees with both row and position order.
        ch = (n * 17 + 4) % 1024
        values = (ch if not duplicate or n != 1 else 4, n // 128 + 1, n % 128 + 1)
        rows.append(f'<row r="{n+2}">' + ''.join(f'<c r="{c}{n+2}"><v>{v}</v></c>'
                                                      for c, v in zip('ABC', values)) + '</row>')
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('xl/workbook.xml', workbook)
        archive.writestr('xl/_rels/workbook.xml.rels', rels)
        archive.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{ns}"><sheetData>{"".join(rows)}</sheetData></worksheet>')
        if shared:
            archive.writestr('xl/sharedStrings.xml', f'<sst xmlns="{ns}">' + ''.join(
                f'<si><t>{s}</t></si>' for s in ('Channel#', 'Shank', 'Elec#')) + '</sst>')
    return path


def write_depth_csv(path):
    positions = read_layout()['position']
    inverse = {(int(r),int(c)):ch for ch,(r,c) in enumerate(positions)}
    names = [name for i in range(1,9) for name in (f'shank{i}_ID',f'shank{i}_depth_um')]
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(names)
        for row in range(128):
            writer.writerow([v for col in range(8) for v in
                             (inverse[(row,col)],row*50+col*100-(106850 if col==0 else 0))])
    return path


@pytest.mark.parametrize('shared', [False, True])
def test_new_mapping_joins_channel_identity_not_row_number(tmp_path, shared):
    layout = read_layout(write_layout(tmp_path / 'mapping.xlsx', shared=shared))
    assert layout['kind'] == 'shank_table'
    assert tuple(layout['position'][4]) == (0, 0)
    assert tuple(layout['position'][21]) == (1, 0)
    # XDF index 0 -> electrode ID 5 -> Channel# 4 -> shank 1 / Elec# 1.
    grid = layout_grid(layout, [5, 22], [3., -2.])
    assert grid[0, 0] == 3 and grid[1, 0] == -2
    assert np.isfinite(grid).sum() == 2


def test_bad_layout_rejects_duplicate_channel(tmp_path):
    with pytest.raises(ValueError, match='重复'):
        read_layout(write_layout(tmp_path / 'bad.xlsx', duplicate=True))


def test_preferences_upgrade_only_previous_default(tmp_path, monkeypatch):
    p = tmp_path / 'preferences.json'
    monkeypatch.setattr(config, 'PREFERENCES_PATH', p)
    import json
    p.write_text(json.dumps({'layout_path': str(config.LEGACY_LAYOUT_XLSX)}))
    assert config.load_preferences()['layout_path'] == str(config.DEFAULT_LAYOUT_XLSX)
    p.write_text(json.dumps({'layout_path': str(tmp_path / 'custom.xlsx')}))
    assert config.load_preferences()['layout_path'] == str(tmp_path / 'custom.xlsx')


def training_data():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(300, 4)) * [1, 4, 2, 1]
    x[:, 3] = 7  # Explicit dropped constant.
    v = x[:, :3] @ np.array([[1., -.2], [.1, .5], [-.3, .4]]) + rng.normal(size=(300, 2)) * .5
    return DecoderData(features=x, velocity=v, position=np.zeros((300, 2)), sequence=np.zeros(300, int),
                       trial=np.repeat(np.arange(6), 50), bin_seconds=.1, electrode_ids=np.array([22, 5, 40, 90]))


@pytest.mark.parametrize('name', ['ridge', 'velocity_kf'])
def test_weights_predict_finite_difference_and_standardized_units(name):
    data = training_data()
    params = {'lam_ridge': .1, 'lam_kf': .001, 'q_shrinkage': .5}
    raw = fit_weights(data, name, params, 'per_feature_unit')
    sd = fit_weights(data, name, params, 'per_SD')
    np.testing.assert_allclose(sd['weights'][:3], raw['weights'][:3] * data.features.std(0)[:3, None])
    np.testing.assert_array_equal(raw['weights'][3], [0., 0.])
    decoder = get(name)
    model = decoder.fit(data, params)
    changed = data.features.copy()
    changed[-1, 1] += .01
    delta = decoder.predict(model, data.with_features(changed))[-1] - decoder.predict(model, data)[-1]
    np.testing.assert_allclose(delta, raw['weights'][1] * .01, rtol=1e-7, atol=1e-10)
    if name == 'velocity_kf':
        gain, _ = steady_gain(model['par'])
        assert gain.shape == (2, 3)
        assert not np.allclose(gain.T, model['par']['C'])


def test_missing_sites_are_not_zero_and_depth_denominator_tracks_coverage(tmp_path):
    layout = read_layout(write_layout(tmp_path / 'mapping.xlsx'))
    rows = channel_rows(layout, np.array([22, 5]), np.array([[3., 4.], [0., 0.]]), [True, False])
    assert len(rows) == 1024
    assert next(r for r in rows if r['electrode_id'] == 22)['norm'] == 5
    assert next(r for r in rows if r['electrode_id'] == 5)['status'] == 'constant_dropped'
    assert np.isnan(next(r for r in rows if r['electrode_id'] == 1)['norm'])
    depth = depth_rows(rows, 16)
    assert depth[0]['n_sites'] == 128 and depth[0]['n_transmitted'] == 2
    assert depth[0]['norm_mean_abs'] == 2.5 and depth[0]['norm_weight_share'] == 1
    assert np.isnan(depth[1]['norm_mean_abs'])


@pytest.mark.parametrize('decoder', ['ridge', 'velocity_kf'])
def test_s14_six_figures_with_s10_parameters_and_exact_exported_values(tmp_path, decoder):
    from data_analysis_gui.core.synthetic import make_session
    from data_analysis_gui.core.filters import TrialFilter
    from data_analysis_gui.runner import build_context, run_section
    from data_analysis_gui.sections.guide import load_guide, figure_guide
    params = {'decoder': decoder, '_source_sections': {
        's10': {'version': 'v1', 'params': {'bin': .2, 'n_bins': 4, 'lam_ridge': .3}}}}
    params['depth_csv']=str(write_depth_csv(tmp_path/'depth.csv'))
    session = make_session(n_blocks=3)
    ctx = build_context([session], TrialFilter(), params, SECTION,
                        layout_path=write_layout(tmp_path / 'mapping.xlsx'))
    result = run_section(SECTION, ctx)
    try:
        assert len(result.figures) == 7
        assert result.summary['source_section']['params']['bin'] == .2
        assert result.summary['source_section']['params']['lam_ridge'] == .3
        rows = result.tables['channel_weights']
        for row in rows:
            if row['transmitted']:
                assert row['norm'] == pytest.approx(np.hypot(row['vx'], row['vy']))
        guide = load_guide(SECTION)
        for (title, figure), payload, method in zip(result.figures[:6], result.figure_data[:6], result.methods[:6]):
            assert payload['channel_weights'] == rows
            assert 'not CV' in method
            assert '### 质量评价标准' in figure_guide(guide, title)
            FigureCanvasAgg(figure).draw()
        assert result.figures[6][0] == 'Active channel depth distribution'
        assert sum(r['single_unit_count'] for r in result.tables['active_depth_bins']) == int(np.any(session.counts>0,axis=0).sum())
        FigureCanvasAgg(result.figures[6][1]).draw()
    finally:
        for _, figure in result.figures:
            figure.clear()


def test_s14_refuses_index_grid_as_depth():
    from data_analysis_gui.sections.base import AnalysisContext
    with pytest.raises(ValueError, match='新天枢'):
        SECTION.run(AnalysisContext([], [], None, SECTION.resolve_params()))


def test_all_six_plots_use_absolute_reversed_hot_0_25_without_position_groups(tmp_path):
    from data_analysis_gui.sections.base import SectionResult
    from data_analysis_gui.sections.s14_decoder_weights_v1.section import append_weight_figures
    layout = read_layout(write_layout(tmp_path / 'map.xlsx'))
    rows = channel_rows(layout, np.array([5, 22]), np.array([[-30., 4.], [0., -3.]]), [True, True])
    site = next(r for r in rows if r['electrode_id'] == 5)
    assert site['vx'] == 30. and site['vx_signed'] == -30.
    result = SectionResult()
    append_weight_figures(result, layout, rows, 'test', 'velocity / input SD', 'test')
    try:
        assert len(result.figures) == 6
        for _, fig in result.figures:
            assert len(fig.axes) == 2  # Only the site map and its colorbar.
            ax = fig.axes[0]
            im = ax.images[0]
            assert im.cmap.name == 'hot_r' and im.get_clim() == (0., 25.)
            assert im.origin == 'lower' and ax.get_ylim() == (-.5, 127.5)
            def luminance(value):
                return np.dot(im.cmap(im.norm(value))[:3], [.2126, .7152, .0722])
            assert luminance(0) > luminance(12.5) > luminance(25)
            values = im.get_array().compressed()
            assert np.all(values >= 0)
            assert not any('Position groups' in a.get_title() for a in fig.axes)
        image = result.figures[0][1].axes[0].images[0].get_array()
        assert image[0, 0] == 30.  # No clipping of stored or numeric values.
        assert '30.00' in [t.get_text() for t in result.figures[1][1].axes[0].texts]
    finally:
        for _, fig in result.figures:
            fig.clear()


def test_cross_weight_statistics_exclude_missing_retain_zeros_and_average_norm_first(tmp_path):
    from data_analysis_gui.sections.x09_decoder_weights_v1.section import aggregate_result
    layout = read_layout(write_layout(tmp_path / 'mapping.xlsx'))
    a = channel_rows(layout, np.array([5, 22]), np.array([[3.,4.], [0.,0.]]), [True,False])
    b = channel_rows(layout, np.array([5]), np.array([[0.,10.]]), [True])
    c = channel_rows(layout, np.array([5]), np.array([[9.,0.]]), [True])
    result = aggregate_result(layout, [a,b,c], ['a','b','c'], 'velocity / input SD')
    try:
        assert len(result.figures) == 6
        rows = {r['channel_0based']:r for r in result.tables['channel_statistics']}
        assert rows[4]['vx_median'] == 3 and rows[4]['vx_average'] == 4
        assert rows[4]['norm_median'] == 9 and rows[4]['norm_average'] == 8
        assert rows[21]['n_runs_observed'] == 1 and rows[21]['norm_average'] == 0
        assert rows[21]['n_runs_constant'] == 1
        assert rows[0]['n_runs_observed'] == 0 and np.isnan(rows[0]['vx_median'])
        assert result.summary['n_channels_all_runs'] == 1
        limits = tuple(result.summary['color_limits'])
        assert limits[0] == 0 and limits[1] >= 9 and limits[1] < 25
        for _, fig in result.figures:
            assert len(fig.axes) == 2 and fig.axes[0].images[0].origin == 'lower'
            assert fig.axes[0].images[0].get_clim() == limits
            FigureCanvasAgg(fig).draw()
    finally:
        for _, fig in result.figures:
            fig.clear()


def test_depth_counts_each_nonzero_channel_once_and_uses_depth_bin_boundaries():
    from data_analysis_gui.sections.s14_decoder_weights_v1.depth_distribution import distribution
    depths={ch:dict(channel_0based=ch,csv_shank=1,depth_original_um=d,depth_offset_um=0,depth_um=d)
            for ch,d in [(4,95),(1,100),(9,100),(21,200)]}
    bins,rows=distribution([5,2,10,22],np.array([3,0,1,2]),depths,100)
    assert [r['single_unit_count'] for r in bins]==[1,1,1]
    assert [r['n_transmitted'] for r in bins]==[1,2,1]
    assert sum(r['active'] for r in rows)==3  # Not 3+1+2 = 6 events/codes.
    assert next(r for r in rows if r['electrode_id']==2)['active'] is False


def test_depth_distribution_uses_whole_run_even_for_a_single_transient_frame(tmp_path):
    from types import SimpleNamespace
    from data_analysis_gui.sections.base import SectionResult
    from data_analysis_gui.sections.s14_decoder_weights_v1.depth_distribution import append_distribution
    session=SimpleNamespace(counts=np.array([[0,0,3],[1,0,0],[2,0,0]]),
                            electrode_ids=np.array([1,2,3]),label='test')
    result=SectionResult()
    append_distribution(result,session,path=write_depth_csv(tmp_path/'depth.csv'))
    try:
        assert result.summary['active_depth_channels']==2
        assert sum(r['single_unit_count'] for r in result.tables['active_depth_bins'])==2
        assert len(result.figures)==1
    finally:
        result.figures[0][1].clear()

def test_shank1_depth_offset_preserves_latest_mapping_order(tmp_path):
    from data_analysis_gui.sections.s14_decoder_weights_v1.depth_distribution import load_depths, distribution
    path=write_depth_csv(tmp_path/'depth.csv')
    depths=load_depths(path)
    positions=read_layout()['position']
    for ch,d in depths.items():
        assert d['shank']==int(positions[ch,1])+1
        assert d['elec_index']==int(positions[ch,0])+1
        assert d['depth_offset_um']==(106850 if d['shank']==1 else 0)
        assert d['depth_um']==d['depth_original_um']+d['depth_offset_um']
    shank1=[ch for ch,d in depths.items() if d['shank']==1]
    bins,rows=distribution([ch+1 for ch in shank1],[3]*128,depths,100)
    assert len(rows)==128 and sum(r['single_unit_count'] for r in bins)==128
