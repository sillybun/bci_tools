"""Channel coverage, source-significance selection and independent y-axis regression."""
import numpy as np

from data_analysis_gui.core.channel_atlas import append_anova_atlas, append_psth_atlas, channel_pages
from data_analysis_gui.gui.widgets import group_series
from data_analysis_gui.sections.base import SectionResult
from data_analysis_gui.sections.s06_psth_v1.section import smooth


def test_page_coverage_order_and_empty_significant_set():
    ids = np.arange(65, 0, -1)
    sig = np.zeros(65, bool)
    sig[[0, 20, 64]] = True
    q = np.ones(65)
    q[sig] = [.02, .001, .01]
    pages = list(channel_pages(ids, sig, q))
    assert [(name, page, total, len(order)) for name, page, total, order in pages] == [
        ('all channels', 1, 2, 64), ('all channels', 2, 2, 1), ('significant channels', 1, 1, 3)]
    assert ids[np.r_[pages[0][3], pages[1][3]]].tolist() == list(range(1, 66))
    assert pages[2][3].tolist() == [20, 64, 0]
    empty = list(channel_pages(ids, np.zeros(65, bool), q))[-1]
    assert empty[0] == 'significant channels' and len(empty[3]) == 0


def test_atlas_panels_keep_their_own_limits_and_y_labels():
    ids = np.array([90, 12])
    means = np.column_stack([np.linspace(1, 2, 8), np.linspace(100, 200, 8)])
    result = SectionResult()
    info = dict(st=dict(mean=means, sem=np.ones((8,2))*.1, p=np.array([.001,.9])),
                q=np.array([.002,.9]), sig=np.array([True,False]), active=np.ones(2,bool),
                mean_hz=np.array([1.5,150.]))
    append_anova_atlas(result, info, ids, 'test record; q<0.05')
    try:
        fig = result.figures[0][1]
        a, b = fig.axes[:2]
        assert len(fig.axes) == 64
        assert not a.get_shared_y_axes().joined(a,b)
        assert a.get_ylim()[0] > 90 and b.get_ylim()[1] < 3
        assert all(label.get_visible() for ax in (a,b) for label in ax.get_yticklabels())
        assert [r['electrode'] for r in result.figure_data[1]['channel_tuning']][0] == 90
        groups = group_series([dict(title=title) for title,_ in result.figures])
        assert len(groups) == 2
        a.set_ylim(100, 300)
        assert b.get_ylim()[1] < 3
    finally:
        for _,fig in result.figures:
            fig.clear()


def test_psth_atlas_uses_temporal_selection_and_exports_raw_and_smoothed():
    rng = np.random.default_rng(1)
    info = dict(binned=rng.uniform(0,20,(8,5,2)), dirs=np.arange(8), centres=np.linspace(-.2,.6,5),
                q=np.array([.9,.001]), p_ch=np.array([.9,.0001]), sig=np.array([False,True]),
                active=np.ones(2,bool), sigma=1)
    p = dict(align='trial_start', tmin=-.3,tmax=.7,baseline_start=-.3,baseline_end=0)
    result = SectionResult()
    append_psth_atlas(result,info,np.array([90,12]),p,'test record',smooth)
    try:
        all_rows = result.figure_data[0]['direction_psth']
        sig_rows = result.figure_data[1]['direction_psth']
        assert {r['electrode'] for r in all_rows} == {90,12}
        assert {r['electrode'] for r in sig_rows} == {12}
        first = all_rows[0]
        assert first['raw_bin_0_t_-0.2000s'] == info['binned'][0,0,1]
        assert first['plotted_bin_0_t_-0.2000s'] != first['raw_bin_0_t_-0.2000s']
        a,b = result.figures[0][1].axes[:2]
        assert len(a.lines) == 9  # eight directions and the event marker
        assert not a.get_shared_y_axes().joined(a,b)
        assert a.get_yticklabels() and b.get_yticklabels()
    finally:
        for _,fig in result.figures:
            fig.clear()
