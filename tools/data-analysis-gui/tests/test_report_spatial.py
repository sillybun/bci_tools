"""Spatial report aggregation, missingness, and physical/depth coordinate contracts."""
import numpy as np
from data_analysis_gui.report_spatial import channel_statistics,daily_channel_statistics,depth_bins,sites_figure
from data_analysis_gui.reporting import report_plan,SPATIAL_CATEGORIES
from data_analysis_gui.core.catalog import Recording
from data_analysis_gui.core.channels import read_layout
from data_analysis_gui.sections.s14_decoder_weights_v1.depth_distribution import load_depths
from data_analysis_gui import config as C

def test_binary_summary_excludes_missing_and_untested_and_retains_ties():
    runs=[[dict(channel_0based=0,tuning=1),dict(channel_0based=1,tuning=1)],
          [dict(channel_0based=0,tuning=0),dict(channel_0based=1,tuning=0,tested='False')]]
    stats,n=channel_statistics(runs,['tuning'])
    assert stats['mean'][0,0]==stats['median'][0,0]==.5 and n[0,0]==2
    assert stats['mean'][1,0]==1 and n[1,0]==1
    assert np.isnan(stats['mean'][2,0]) and n[2,0]==0

def test_weights_keep_observed_zero_and_exclude_untransmitted():
    runs=[[dict(channel_0based=0,vx=0),dict(channel_0based=1,vx=100,transmitted='False')],
          [dict(channel_0based=0,vx=4),dict(channel_0based=1,vx=3)]]
    stats,n=channel_statistics(runs,['vx'])
    assert stats['mean'][0,0]==2 and n[0,0]==2
    assert stats['mean'][1,0]==3 and n[1,0]==1

def test_7shank_exclusion_happens_before_profile_means():
    rows=[dict(shank=1,depth_um=100,vx=100,event_hz=40),dict(shank=2,depth_um=100,vx=2,event_hz=6)]
    eight=depth_bins(rows,['vx','event_hz']);seven=depth_bins(rows,['vx','event_hz'],True)
    assert eight[1]['vx']==51 and seven[1]['vx']==2 and seven[1]['event_hz']==6

def test_depth_plot_shanks_2_to_8_keep_channel_identity_and_lower_origin():
    layout=read_layout();depths=load_depths("", layout=layout);v=np.arange(1024,dtype=float)
    fig=sites_figure(layout,depths,v,'Test',seven=True,limits=(0,1023))
    ax=fig.axes[0]
    assert list(ax.get_xticks())==list(range(2,9)) and ax.get_xlim()==(1.5,8.5)
    assert ax.get_ylim()[1]>ax.get_ylim()[0] and len(ax.collections)==7
    for col,mesh in zip(range(1,8),ax.collections):
        ids=[ch for ch,(_,c) in enumerate(layout['position']) if c==col]
        ids=sorted(ids,key=lambda ch:depths[ch]['depth_um'])
        np.testing.assert_equal(np.asarray(mesh.get_array()).ravel(),v[ids])
    fig.clear()

def test_report_plans_all_run_s05_s14_and_forces_dual_x01(tmp_path):
    recs=[Recording(tmp_path/f'{i}.xdf',f'{i}.xdf','2026-11-18','grid',False,str(i)) for i in range(2)]
    jobs,report=report_plan(recs,lambda s:dict(s.defaults(),lam_ridge=7) if s.id=='s10' else s.defaults())
    assert len([s for s,r,p in jobs if s.id=='s05'])==2
    assert len([s for s,r,p in jobs if s.id=='s14'])==2
    assert next(p for s,r,p in jobs if s.id=='x01')['map_layout']=='both'
    for s,r,p in jobs:
        if s.id=='s14':assert p['_source_sections']['s10']['params']['lam_ridge']==7


def test_electrode_axis_is_one_to_128_and_values_follow_positions():
    layout=read_layout();depths=load_depths("", layout=layout)
    values=layout['position'][:,0].astype(float)
    fig=sites_figure(layout,depths,values,'Positions',depth=False,limits=(0,127))
    ax=fig.axes[0]
    assert ax.get_ylim()==(.5,128.5)
    assert list(ax.get_yticks())==[1,16,32,48,64,80,96,112,128]
    for mesh in ax.collections:
        np.testing.assert_equal(np.asarray(mesh.get_array()).ravel(),np.arange(128))
    fig.clear()


def test_x01_both_layouts_display_shared_one_based_axis():
    from data_analysis_gui.sections.x01_channel_stability_v1.section import SECTION
    for layout in (None,read_layout()):
        fig=SECTION.maps(layout,{'max':np.zeros((2,1024))},['run001','run002'],{'map_quantity':'max_code'})
        ax=fig.axes[0]
        assert ax.get_ylim()==(.5,128.5)
        assert ax.get_yticklabels()[0].get_text()=='1'
        assert ax.get_yticklabels()[-1].get_text()=='128'
        fig.clear()


def test_report_excluded_date_retains_explicit_offline_behavior(tmp_path):
    recs=[Recording(tmp_path/f'{i}.xdf',f'{i}.xdf',date,'grid',False,str(i))
          for i,date in enumerate(['2026-11-16','2026-11-17'])]
    folders=[str(tmp_path/'20261116'/'run-001'),str(tmp_path/'20261117'/'run-001')]
    jobs,report=report_plan(recs,lambda s:s.defaults(),behavior_folders=folders,analysis_excluded_dates=['2026-11-16'])
    assert all(r.date!='2026-11-16' for _,rows,_ in jobs for r in rows)
    assert report['analysis_excluded_dates']==['2026-11-16']
    assert report['offline_behavior_folders']==folders


def test_run_export_switch_preserves_summary_numbers(tmp_path,monkeypatch):
    from matplotlib.figure import Figure
    from data_analysis_gui import report_spatial as module
    from data_analysis_gui import storage
    monkeypatch.setattr(Figure,'savefig',lambda *a,**k:None)
    monkeypatch.setattr(module,'sites_figure',lambda *a,**k:Figure())
    entries=[]
    for i,significant in enumerate(('True','False')):
        source=tmp_path/f'source{i}';source.mkdir()
        (source/'channels.csv').write_text('electrode,active,tested,significant\n1,True,True,'+significant+'\n')
        entries.append(dict(job=dict(section='s05',recordings=[f'run{i}.xdf']),folder=str(source),record=dict(summary={'recording':f'run{i}'},tables={'anova_channels':'channels.csv'})))
    data=[]
    for include in (False,True):
        out=tmp_path/str(include);out.mkdir()
        meta=module.render_report_spatial(out,entries,dict(recordings=[dict(path=f"run{i}.xdf", date="2026-11-18", run_group=f"run{i}") for i in range(2)],include_run_exports=include),100,lambda *a,**k:None)
        data.append(storage.read_table(out/SPATIAL_CATEGORIES[0]/'summary_all_runs/channel_tuning_statistics.csv',limit=2000))
        assert bool(list(out.rglob('runs')))==include
        if not include:assert all('summary_all_runs' in p or 'summary_by_day' in p for p in meta['pngs'])
    assert data[0]==data[1]
    assert float(data[0][0]['tuning_mean'])==.5


def test_daily_weight_summary_excludes_outlier_after_combining_recordings():
    def recording(value):
        return [dict(channel_0based=0,vx=value)]
    runs = dict(a=[recording(0),recording(2)], b=[recording(2)],
                c=[recording(3)], d=[recording(100)], e=[recording(2)])
    stats, counts, exclusions = daily_channel_statistics(runs, ('vx',), trim_weights=True)
    assert stats['mean'][0,0] == stats['median'][0,0] == 2
    assert counts[0,0] == 4
    assert [(r['run'],r['value']) for r in exclusions] == [('d',100)]
