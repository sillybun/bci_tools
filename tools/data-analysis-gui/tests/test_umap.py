"""UMAP reproducibility, small-cohort handling and S04 integration."""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from data_analysis_gui.core.filters import TrialFilter
from data_analysis_gui.core.synthetic import make_session
from data_analysis_gui.runner import build_context, run_section
from data_analysis_gui.sections.s04_dimensionality_v1.section import SECTION, umap_embedding


def test_umap_reproducible_and_caps_neighbors():
    x = np.random.default_rng(7).normal(size=(24, 8))
    x[:, -1] = 3.0
    first = umap_embedding(x, n_neighbors=100, seed=42)
    second = umap_embedding(x, n_neighbors=100, seed=42)
    assert first['n_neighbors'] == 23
    assert first['active_channels'] == 7
    assert first['scores'].shape == (24, 2)
    assert np.isfinite(first['scores']).all()
    np.testing.assert_array_equal(first['scores'], second['scores'])


@pytest.mark.parametrize('x', [np.ones((16, 4)), np.zeros((2, 4)),
                             np.full((16, 4), np.nan)])
def test_umap_rejects_unusable_features(x):
    with pytest.raises(ValueError, match='UMAP'):
        umap_embedding(x)


@pytest.mark.parametrize('enabled', [True, False])
def test_s04_umap_toggle_and_trial_identity(enabled):
    session = make_session(n_blocks=6)
    ctx = build_context([session], TrialFilter(),
                        {'n_perm': 0, 'umap_enabled': enabled}, SECTION)
    result = run_section(SECTION, ctx)
    try:
        titles = [title for title, _ in result.figures]
        assert titles[0]=='Dimensionality overview'
        overview=result.figures[0][1]
        panels=overview._overview_panels
        assert len(panels)==6
        FigureCanvasAgg(overview).draw()
        for top, bottom in zip(panels[:3], panels[3:]):
            upper, lower = top.get_position(), bottom.get_position()
            np.testing.assert_allclose([upper.x0, upper.x1], [lower.x0, lower.x1])
            assert upper.y0 > lower.y1
        tdr = next(fig for title, fig in result.figures if title == 'TDR')
        np.testing.assert_array_equal(panels[-1].collections[0].get_offsets(),
                                      tdr.axes[1].collections[0].get_offsets())
        assert 'PCA' in panels[0].get_title() and 'TDR' in panels[2].get_title()
        assert '%' in panels[0].get_xlabel() and 'Residual PC1' in panels[3].get_xlabel()
        assert 'residual var.' in panels[3].get_ylabel()
        assert 'Supplementary Residual PCA held-out projections' in titles
        assert result.tables['residual_pca_heldout_trials']
        assert ('UMAP embedding' in titles) == enabled
        assert ('umap' in result.summary) == enabled
        assert ('umap_projections' in result.tables) == enabled
        if enabled:
            rows = result.tables['umap_projections']
            ok = np.isfinite(ctx.cohort.window_rates).all(1)
            np.testing.assert_array_equal([r['trial_id'] for r in rows], ctx.cohort.trial_ids[ok])
            np.testing.assert_array_equal([r['block'] for r in rows], ctx.cohort.blocks[ok])
            np.testing.assert_array_equal([r['direction_deg'] for r in rows], ctx.cohort.directions[ok] * 45)
            assert np.isfinite([[r['UMAP1'], r['UMAP2']] for r in rows]).all()
            np.testing.assert_allclose(panels[1].get_xlim(),panels[4].get_xlim())
            np.testing.assert_allclose(panels[1].get_ylim(),panels[4].get_ylim())
            np.testing.assert_allclose([[r['umap1'],r['umap2']] for r in result.figure_data[0]['umap']],
                                       [[r['UMAP1'],r['UMAP2']] for r in rows])
        else:
            assert any('disabled' in t.get_text() for t in panels[1].texts)
    finally:
        for _, fig in result.figures:
            fig.clear()
