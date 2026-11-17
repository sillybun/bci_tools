from pathlib import Path
import numpy as np
from data_analysis_gui.report_bga import first_run,first_run_status


def test_first_run_is_not_replaced_by_later_recorded_run(tmp_path):
    first=tmp_path/'run-001-Grid-offline';first.mkdir()
    later=tmp_path/'run-002-Grid-offline';later.mkdir();(later/'later.xdf').write_bytes(b'later')
    assert first_run(tmp_path)==first
    values,info=first_run_status(first)
    assert info['status']=='no_xdf' and np.isnan(values).all()


def test_spike_only_run_uses_explicit_channel_identity_and_zero_is_observed(tmp_path,monkeypatch):
    from data_analysis_gui import report_bga
    path=tmp_path/'spikes.xdf';path.write_bytes(b'spike-only')
    stream={'info':{'desc':[{'channel_ids':['3,1']}]},'time_series':np.array([[0,1],[0,3]]),'time_stamps':np.array([1.,2.])}
    monkeypatch.setattr(report_bga,'load_streams',lambda p:({'spike_counts':stream},[]))
    values,info=first_run_status(tmp_path)
    assert values[0]==3 and values[2]==0 and np.isnan(values[1])
    assert info['status']=='ok' and info['sources'][0]['frames']==2


def test_reference_style_uses_solid_shank_colors_and_preserves_typography():
    from matplotlib.colors import to_rgba
    from data_analysis_gui.report_bga import bga_figure,SHANK_COLORS
    from data_analysis_gui.core.style import finish_figure
    layout={'bga_position':np.tile([[1,1]],(1024,1)),
            'position':np.column_stack([np.arange(1024)%128,np.arange(1024)%8])}
    values=np.full(1024,np.nan);values[:4]=[0,1,2,3]
    fig=bga_figure(layout,values,'20261111','run-006-Grid-offline','ok')
    finish_figure(fig)
    points=fig.axes[0].collections[0];colors=points.get_facecolors()
    assert np.allclose(colors[0],to_rgba('#000000'))
    assert np.allclose(colors[1],to_rgba(SHANK_COLORS[1]))
    assert np.allclose(colors[2],to_rgba(SHANK_COLORS[2]))
    assert np.allclose(colors[3],to_rgba(SHANK_COLORS[3]))
    assert np.allclose(colors[4],to_rgba('#BFC4CA'))
    assert len(points.get_edgecolors())==0
    assert fig._suptitle.get_text()=='Tianshu 3.0 - BGA Package - Run 006'
    assert fig._suptitle.get_fontsize()==24
    assert fig.axes[0].xaxis.label.get_fontsize()==20
    assert fig.axes[0].texts[0].get_fontsize()==5.8
    assert [t.get_text() for t in fig.legends[0].get_texts()][-2:]==['All codes = 0 (1)','Not transmitted (1020)']
