"""All guide variants and metric-derived one-sentence result summaries."""
import re

import pytest

from data_analysis_gui.sections import sections
from data_analysis_gui.sections.guide import figure_guide, figure_titles, load_guide
from data_analysis_gui.sections.result_summaries import result_sentence

SECTIONS = sections('single') + sections('cross')


@pytest.mark.parametrize('section', SECTIONS, ids=lambda s: s.id)
def test_each_documented_figure_has_matching_panel_guidance(section):
    guide = load_guide(section)
    intro = guide.split('## 图', 1)[0]
    assert '**一句话总结：**' in intro
    assert '### 怎么看' not in guide and '### 什么样是好的数据' not in guide
    for title in figure_titles(guide):
        part = figure_guide(guide, title)
        assert '### 结果解读' in part and '### 质量评价标准' in part
        interpretation, criteria = part.split('### 结果解读', 1)[1].split('### 质量评价标准', 1)
        labels = lambda text: re.findall(r'^- \*\*(.+?)\*\*：', text, flags=re.M)
        assert labels(interpretation) == labels(criteria)
        assert labels(interpretation)


@pytest.mark.parametrize('section', SECTIONS, ids=lambda s: s.id)
def test_all_section_summaries_handle_missing_metrics(section):
    sentence = result_sentence(section.id, {})
    assert sentence.endswith('。')
    assert 'nan' not in sentence.lower() and 'None' not in sentence
    assert '未计算' in sentence or '未知' in sentence


def test_missing_clock_does_not_report_zero_frame_loss():
    text = result_sentence('s03', dict(n_frames=100, n_receive_gaps=2, fpga_clock=False))
    assert '100' in text and '2 个' in text and '未知' in text
    assert '缺失 0 帧' not in text
    reset = result_sentence('s03', dict(n_frames=100, n_receive_gaps=2, fpga_clock=True,
                                      fpga_frames_lost=3, fpga_drop_rate=.03, fpga_nonpositive_steps=1))
    assert '缺失 3 帧' in reset and '不覆盖' in reset


def test_metric_readout_changes_and_permutation_absence_is_explicit():
    before = result_sentence('s09', dict(n_trials=100, balanced_accuracy=.25, within_45deg=.5, p_value=None))
    after = result_sentence('s09', dict(n_trials=100, balanced_accuracy=.75, within_45deg=.9, p_value=.01))
    assert '25.0%' in before and '显著性未计算' in before
    assert '75.0%' in after and 'p=0.0100' in after
    assert before != after


def test_grid_and_online_readouts_preserve_interpretation_limits():
    grid = result_sentence('s09', dict(grid_n_targets=36, grid_target_r2_xy=.5,
                                     grid_identity_balanced_accuracy=.06, grid_identity_chance=1/36))
    assert 'Grid' in grid and '36' in grid and '12.5%' not in grid
    online = result_sentence('s10', dict(decoders=['ridge'], n_common_tested_bins=100,
                                        ridge_r2_mean=.6, recording='CO8 online'))
    assert '0.60' in online and '闭环输出' in online


@pytest.mark.parametrize('section_id', ['s09', 's12'])
def test_grid_result_and_all_generated_figures_have_guidance(section_id):
    from data_analysis_gui.core.filters import TrialFilter
    from data_analysis_gui.core.synthetic import make_session
    from data_analysis_gui.runner import build_context, run_section

    section = next(s for s in SECTIONS if s.id == section_id)
    session = make_session(task_mode='grid')
    context = build_context([session], TrialFilter(), {'n_perm': 0}, section)
    result = run_section(section, context)
    try:
        assert 'Grid' in result.summary['one_sentence_summary']
        guide = load_guide(section)
        for title, _ in result.figures:
            assert '### 质量评价标准' in figure_guide(guide, title)
    finally:
        for _, figure in result.figures:
            figure.clear()
