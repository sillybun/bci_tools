from io import BytesIO
from matplotlib.figure import Figure
from data_analysis_gui.core.style import finish_figure


def test_fixed_roles_are_idempotent_and_ignore_legacy_percent():
    f=Figure();ax=f.subplots();ax.plot([0,1],[0,1],label='Legend');ax.legend()
    ax.set(title='Panel',xlabel='X',ylabel='Y');f.suptitle('Title')
    annotation=ax.text(.5,.5,'Numeric cell',fontsize=3.5)
    for percent in [120,80,160]:finish_figure(f,percent)
    assert f._suptitle.get_fontsize()==ax.title.get_fontsize()==18
    assert ax.xaxis.label.get_fontsize()==ax.yaxis.label.get_fontsize()==14
    assert all(t.get_fontsize()==11 for t in ax.get_xticklabels()+ax.get_yticklabels())
    assert ax.get_legend().get_texts()[0].get_fontsize()==11
    assert annotation.get_fontsize()==3.5


def test_direct_exports_also_enforce_fixed_fonts_and_colorbar_labels():
    f=Figure();ax=f.subplots();image=ax.imshow([[0,1],[2,3]])
    ax.set_title('Small',fontsize=4);cb=f.colorbar(image,ax=ax);cb.set_label('Scale',fontsize=4)
    f.savefig(BytesIO(),format='svg')
    assert ax.title.get_fontsize()==18 and cb.ax.yaxis.label.get_fontsize()==14
    assert cb.ax.get_yticklabels()[0].get_fontsize()==11


def test_schematics_keep_legacy_fonts_idempotently():
    from data_analysis_gui.sections.session_overview import session_overview
    f,_=session_overview([])
    finish_figure(f);legend=f.axes[0].get_legend()
    assert legend.get_texts()[0].get_fontsize()==34.8
    finish_figure(f)
    assert legend.get_texts()[0].get_fontsize()==34.8

def test_run_labels_stay_on_one_line():
    from data_analysis_gui.core.crossday import recording_ticks
    from data_analysis_gui.sections.base import new_figure

    f = new_figure(6, 5)
    ax = f.subplots()
    label = '11-26 · online · Grid · run-006-Grid-online-7x7-sp042-bc150-dw001-nextT-nearby3'
    ax.legend([], [])
    recording_ticks(ax, [0], [label.replace(' · Grid', '\nGrid')])
    f.suptitle(label + '\nActive channels')
    finish_figure(f)
    assert '\n' not in ax.get_xticklabels()[0].get_text()
    assert ax.get_xticklabels()[0].get_rotation() == 90
    assert f._suptitle.get_text() == label + ' Active channels'
    assert not f._suptitle.get_wrap()


def test_missing_glyph_fails_export_and_preserves_existing_file(tmp_path):
    import pytest

    destination = tmp_path / 'figure.png'
    destination.write_bytes(b'previous result')
    figure = Figure()
    figure.suptitle('Unassigned Unicode: \u0378')
    with pytest.raises(RuntimeError, match='缺字'):
        figure.savefig(destination)
    assert destination.read_bytes() == b'previous result'
    assert list(tmp_path.iterdir()) == [destination]


def test_chinese_exports_render_and_svg_does_not_require_viewer_font():
    from xml.etree import ElementTree as ET

    from matplotlib import rcParams
    import pytest

    if not set(rcParams['font.family']) & {
        'Microsoft YaHei', 'Noto Sans CJK SC', 'PingFang SC', 'SimHei', 'WenQuanYi Zen Hei',
    }:
        pytest.skip('No installed Chinese font')
    figure = Figure()
    ax = figure.subplots()
    ax.set(title='脑控与离线摇杆', xlabel='日期')
    for fmt, magic in [('png', b'\x89PNG'), ('pdf', b'%PDF'), ('svg', b'<?xml')]:
        buffer = BytesIO()
        figure.savefig(buffer, format=fmt)
        payload = buffer.getvalue()
        assert payload.startswith(magic)
        if fmt == 'svg':
            root = ET.fromstring(payload)
            ns = '{http://www.w3.org/2000/svg}'
            assert list(root.iter(ns + 'use'))
            assert not list(root.iter(ns + 'text'))
