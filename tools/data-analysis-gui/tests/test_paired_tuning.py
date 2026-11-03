"""Paired channel atlas: source selection and directional-time highlight semantics."""
from types import SimpleNamespace

import numpy as np

from data_analysis_gui.sections.base import SectionResult
from data_analysis_gui.sections.s07_baseline_change_v1 import paired


def test_paired_curves_highlight_within_direction_time_not_preference(monkeypatch):
    ctx=SimpleNamespace(session=SimpleNamespace(electrode_ids=np.array([100,20,30]),label='test'))
    # The helper uses dataclasses.replace on the production context. Only source
    # test outputs are supplied here, isolating which flags the plot consumes.
    monkeypatch.setattr(paired,'replace',lambda ctx,**kw:ctx)
    mean=np.arange(24,dtype=float).reshape(8,3)
    monkeypatch.setattr(paired,'channel_anova',lambda ctx:({'mean':mean,'sem':np.ones((8,3))},None,None))
    times=np.array([-.1,.1,.2,.3,.4])
    binned=np.arange(16*5*3,dtype=float).reshape(16,5,3)
    monkeypatch.setattr(paired,'temporal_statistics',lambda ctx:dict(binned=binned,centres=times,
                        valid=np.ones(16,bool),dirs=np.repeat(np.arange(8),2),sigma=0))
    p6=dict(align='trial_start',bin=.1,baseline_start=-.2,baseline_end=0,tmin=-.2,tmax=.5)
    groups=dict(sources={'s05':{'params':{}},'s06':{'params':p6}},
                direction=np.array([True,False,False]),temporal=np.array([False,True,False]),
                rows=[dict(direction_q=.01,time_q=.8),dict(direction_q=.8,time_q=.01),dict(direction_q=.9,time_q=.9)])
    sig=np.zeros((8,3),bool);sig[0,0]=True;sig[1,1]=True;sig[2,2]=True
    directional=dict(sig=sig,eligible=np.ones((8,3),bool),q=np.where(sig,.001,.8),
                     preferred=np.array([3,-1,-1]))
    result=SectionResult()
    paired.append_paired(ctx,result,groups,directional)
    try:
        assert result.summary['paired_tuning_channels']==2
        rows=result.figure_data[0]['paired_channel_curves']
        assert {r['electrode'] for r in rows}=={20,100}  # pooled source union
        assert [(r['electrode'],r['direction_deg']) for r in rows if r['highlighted']]==[(20,45),(100,0)]
        assert [(r['electrode'],r['direction_deg']) for r in rows if r['preferred']]==[(100,135)]
        fig=result.figures[0][1]
        assert len(fig.axes)==64  # 32 cells x two axes, blanks hidden
        assert not fig.axes[0].get_shared_y_axes().joined(fig.axes[0],fig.axes[1])
        right=fig.axes[3]  # electrode 100's PSTH
        curves=[line for line in right.lines if len(line.get_xdata())==len(times)]
        bold=[line for line in curves if line.get_linewidth()>1]
        assert len(bold)==1 and bold[0].get_color()==paired.style.direction_color(0)
        expected=binned[:2,:,0].mean(axis=0)
        np.testing.assert_array_equal(bold[0].get_ydata(),expected)
        assert fig._series_pdf_name==paired.TITLE
    finally:
        for _,fig in result.figures:fig.clear()
