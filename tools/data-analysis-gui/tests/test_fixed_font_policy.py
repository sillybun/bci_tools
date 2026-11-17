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
