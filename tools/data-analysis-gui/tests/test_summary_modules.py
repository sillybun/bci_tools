"""Source panels, settings and unavailable results in the single-day overview."""
from matplotlib.backends.backend_agg import FigureCanvasAgg

from data_analysis_gui.core.filters import TrialFilter
from data_analysis_gui.core.synthetic import make_session
from data_analysis_gui.runner import build_context, run_section
from data_analysis_gui.sections import versions
from data_analysis_gui.sections.s12_summary_v1.modules import TITLES


def test_overview_has_five_modules_and_source_tuning_results(tmp_path):
    section = versions('s12')[-1]
    session = make_session(n_blocks=4)
    config = tmp_path / 'configs.ini'
    config.write_text('good="0,1,2"\nuncertain="3,4"\nbad="5,6"\n')
    params = dict(n_perm=0, decoders='ridge', channel_impact=False,
                  _source_sections={'s04': {'params': {'umap_enabled': False, 'n_perm': 0}},
                                    's05': {'params': {'alpha': .01}}})
    ctx = build_context([session], TrialFilter(), params, section, channel_config_path=str(config))
    result = run_section(section, ctx)
    try:
        assert result.summary['overview_modules'] == list(TITLES)
        fig = result.figures[0][1]
        success, saturation, transmission, dimension, composition = fig.subfigs
        assert 'task success rate' in success._suptitle.get_text()
        assert 'channel saturation zones' in saturation._suptitle.get_text()
        assert 'Transmission overview' in transmission._suptitle.get_text()
        assert len(dimension._overview_panels) == 6
        assert 'Mutually exclusive' in composition.axes[0].get_title()
        data = result.figure_data[0]
        assert data['s01_task_success']
        assert 's03_receive_intervals' in data and 's04_residual_pca' in data
        assert not data['s04_umap']
        composition_rows = data['s07_disjoint_composition']
        assert sum(r['channels'] for r in composition_rows) == 512
        assert {r['tier'] for r in composition_rows} == {'good', 'uncertain', 'bad', 'unlisted'}
        tuning = result.tables['overview_tuned_channels']
        assert result.summary['overview_direction_tuned'] == sum(r['direction_tuned'] for r in tuning)
        assert [title for title, _ in result.figures] == ['Overview', 'Supplementary decoding overview']
        FigureCanvasAgg(fig).draw()
    finally:
        for _, fig in result.figures:
            fig.clear()


def test_small_cohort_does_not_invent_tuning_or_loss():
    section = versions('s12')[-1]
    session = make_session(n_blocks=1, fpga=False)
    result = run_section(section, build_context(
        [session], TrialFilter(), dict(n_perm=0, channel_impact=False), section))
    try:
        assert result.summary['overview_modules'] == list(TITLES)
        assert result.summary['overview_transmitted'] == 512
        assert not result.summary['fpga_clock']
        assert 'overview_direction_tuned' not in result.summary
        assert 'overview_pca_pc12_variance' not in result.summary
        assert 's07_disjoint_composition' not in result.figure_data[0]
        text = ' '.join(t.get_text() for ax in result.figures[0][1].axes for t in ax.texts)
        assert 'Unavailable' in text and 'unknown' in text
    finally:
        for _, fig in result.figures:
            fig.clear()
