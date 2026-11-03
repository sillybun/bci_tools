"""Additional 7shank depth exports, excluding original Shank1 without renumbering."""
from pathlib import Path
import sys,json,csv
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0,str(Path(__file__).resolve().parent))
from export_s14_depth_summary import joined_rows,depth_figure,ROOT as SUMMARY
from export_depth_weights_and_firing import fig_profiles,KEYS
from export_s05_tuning_layouts import tuning_figure,read_layout,load_depths,C,storage
from data_analysis_gui.sections.s14_decoder_weights_v1.depth_distribution import figure as active_figure
SINGLE=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/single')
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(x for x in f if x.strip() and not x.startswith('#')))
def write(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def title7(fig):
 fig._suptitle.set_text(fig._suptitle.get_text().replace('Shank1 +106850 um','Shank 2-8')+' | 7shank')
def export(fig,dest,pdf=None,dpi=250):
 fig.savefig(dest.with_suffix('.png'),dpi=dpi);fig.savefig(dest.with_suffix('.pdf'))
 if pdf is not None:pdf.savefig(fig)
 fig.clear()
def base_manifest(out,source,count):
 (out/'7shank_manifest.json').write_text(json.dumps(dict(source=str(source),n_figures=count,layout=str(C.DEFAULT_LAYOUT_XLSX),layout_digest=storage.file_digest(C.DEFAULT_LAYOUT_XLSX),depth=str(C.DEFAULT_DEPTH_CSV),depth_digest=storage.file_digest(C.DEFAULT_DEPTH_CSV),included_shanks=list(range(2,9)),excluded_shanks=[1],numbering='Original Shank2..8 preserved',calculation='Saved channel statistics/decisions; filter Shank1 before aggregation; no model fitting or ANOVA recomputation'),ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 # Six channel-statistic heatmaps, same shared colour limits as 8shank version.
 rows=[r for r in joined_rows() if r['shank']!=1];assert len(rows)==896
 out=SUMMARY/'figures_depth_7shank';out.mkdir(exist_ok=True)
 limits=json.loads((SUMMARY/'result.json').read_text(encoding='utf-8'))['summary']['color_limits']
 with PdfPages(out/'Weighting_Projection_depth_7shank.pdf') as pdf:
  for i,(stat,component) in enumerate(((s,c) for s in ('median','average') for c in ('vx','vy','norm')),1):
   fig=depth_figure(rows,component,stat,limits);fig.axes[0].set_xlim(1.5,8.5);fig.axes[0].set_xticks(range(2,9));title7(fig)
   fig.texts[-1].set_text('7shank: Shank 2-8; Shank 1 excluded.')
   export(fig,out/f'X09_{i:02d}_{component}_weighting_{stat}_depth_7shank',pdf,300)
 write(out/'channel_statistics_with_depth.csv',rows)
 offsets=read(SUMMARY/'figures_depth/shank_depth_offsets.csv');write(out/'shank_depth_offsets.csv',[r for r in offsets if int(r['shank'])!=1]);base_manifest(out,SUMMARY/'figures_depth',6)
 print('6 summary heatmaps exported',flush=True)
 # Reaggregate weight and binary-frame-rate depth profiles after excluding Shank1.
 src=SINGLE/'S14/depth_weighting_and_firing';out=SINGLE/'S14/depth_weighting_and_firing_7shank';out.mkdir(exist_ok=True)
 for d in ['per_run','summary']:(out/d).mkdir(exist_ok=True)
 meta=json.loads((src/'manifest.json').read_text(encoding='utf-8'));raw=[r for r in read(src/'channel_weights_and_events.csv') if int(r['shank'])!=1]
 profiles=[]
 for run in meta['runs']:
  group=[r for r in raw if r['recording']==run['name']];bins=[]
  for left in range(0,8300,100):
   g=[r for r in group if left<=float(r['depth_um'])<left+100]
   bins.append(dict(recording=run['name'],depth_left_um=left,depth_right_um=left+100,depth_center_um=left+50,n_transmitted=len(g),
     **{k:float(np.mean([float(r[v]) for r in g])) if g else np.nan for k,v in zip(KEYS,['vx','vy','norm','event_hz'])},
     event_count_sum=sum(int(r['nonzero_frame_events']) for r in g),event_hz_sum=sum(float(r['event_hz']) for r in g)))
  assert sum(r['event_count_sum'] for r in bins)==sum(int(r['nonzero_frame_events']) for r in group)
  profiles.append(bins)
 with PdfPages(out/'per_run/All_45_runs_depth_profiles_7shank.pdf') as pdf:
  for run,bins in zip(meta['runs'],profiles):
   fig=fig_profiles(bins,run['label'],meta['limits']);title7(fig);export(fig,out/'per_run'/(run['name']+'_7shank'),pdf,220)
 summary=[]
 for stat,agg in [('median',np.median),('average',np.mean)]:
  bins=[]
  for j in range(len(profiles[0])):
   present=[p[j] for p in profiles if p[j]['n_transmitted']>0]
   bins.append(dict(statistic=stat,depth_center_um=profiles[0][j]['depth_center_um'],n_runs=len(present),**{k:float(agg([r[k] for r in present])) if present else np.nan for k in KEYS}))
  fig=fig_profiles(bins,'45 runs - '+stat);title7(fig);export(fig,out/'summary'/('Depth_profiles_'+stat+'_7shank'),dpi=300);summary.extend(bins)
 write(out/'channel_weights_and_events.csv',raw);write(out/'per_run_depth_bins.csv',[r for p in profiles for r in p]);write(out/'summary_depth_bins.csv',summary);base_manifest(out,src,47)
 print('45 run profiles and 2 summaries exported',flush=True)
 # Active-channel counts, not frame event counts: filter first, then histogram.
 src=SINGLE/'S14/S14_07_active_channel_depth_distribution';out=SINGLE/'S14/S14_07_active_channel_depth_distribution_7shank';out.mkdir(exist_ok=True)
 m=json.loads((src/'manifest.json').read_text(encoding='utf-8'));raw=[r for r in read(src/'active_depth_channels.csv') if int(r['shank'])!=1];allbins=[];groups=[]
 edges=np.arange(0,8400,100)
 for run in m['records']:
  g=[r for r in raw if r['recording']==run['name']];active=[float(r['depth_um']) for r in g if r['active']=='True'];counts,_=np.histogram(active,edges);trans,_=np.histogram([float(r['depth_um']) for r in g],edges)
  bins=[dict(recording=run['name'],depth_left_um=float(a),depth_right_um=float(b),depth_center_um=(a+b)/2,single_unit_count=int(n),n_transmitted=int(t)) for a,b,n,t in zip(edges[:-1],edges[1:],counts,trans)]
  assert sum(r['single_unit_count'] for r in bins)==len(active);groups.append(bins);allbins.extend(bins)
 ymax=max(1,int(np.ceil(max(r['single_unit_count'] for r in allbins)*1.15)))
 with PdfPages(out/'All_runs_active_channel_depth_distribution_7shank.pdf') as pdf:
  for run,bins in zip(m['records'],groups):
   fig=active_figure(bins,run['name'],y_max=ymax);ax=fig.axes[0];ax.set_title(ax.get_title()+' | 7shank');export(fig,out/(run['name']+'_7shank'),pdf,300)
 write(out/'active_depth_channels.csv',raw);write(out/'active_depth_bins.csv',allbins);base_manifest(out,src,45)
 print('45 active-channel histograms exported',flush=True)
 # S05 decisions unchanged for retained channels; x-axis labels stay 2..8.
 src=SINGLE/'S05';out=src/'tuning_layout_depth_7shank';out.mkdir(exist_ok=True);(out/'tables').mkdir(exist_ok=True)
 m=json.loads((src/'tuning_layout_manifest.json').read_text(encoding='utf-8'));layout=read_layout();depths=load_depths(C.DEFAULT_DEPTH_CSV)
 with PdfPages(out/'All_45_runs_S05_depth_7shank.pdf') as pdf:
  for run in m['runs']:
   rows=[r for r in read(src/'tables_tuning_layout'/(run['name']+'.csv')) if int(r['shank'])!=1]
   fig=tuning_figure(layout,depths,rows,run['name'],True);fig.axes[0].set_xlim(1.5,8.5);fig.axes[0].set_xticks(range(2,9));title7(fig);export(fig,out/(run['name']+'_7shank'),pdf)
   write(out/'tables'/(run['name']+'.csv'),rows)
 base_manifest(out,src/'tuning_layout_depth',45)
 print('DONE: all four 7shank sets; 143 PNGs, individual PDFs and combined PDFs',flush=True)
if __name__=='__main__':main()
