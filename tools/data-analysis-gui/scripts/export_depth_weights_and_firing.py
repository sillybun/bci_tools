"""Depth profiles of saved model weights and binary frame events for the same 45 runs."""
import csv,json,sys,shutil
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_analysis_gui import config as C,storage
from data_analysis_gui.core.catalog import describe
from data_analysis_gui.core.channels import read_layout
from data_analysis_gui.sections.s14_decoder_weights_v1.depth_distribution import load_depths
ROOT=Path('E:/AitBCI/data_analysis_gui_output/keaiduo')
OUT=ROOT/'single/S14/depth_weighting_and_firing'
KEYS=('vx_mean','vy_mean','norm_mean','event_hz_mean')
LABELS=('Mean |wx| / channel','Mean |wy| / channel','Mean norm / channel','Binary events / s / channel')
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(x for x in f if x.strip() and not x.startswith('#')))
def write(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def fig_profiles(rows,label,limits=None,stat=None):
 fig=Figure(figsize=(12,8),layout='constrained');axes=fig.subplots(2,2)
 for i,(ax,key,title) in enumerate(zip(axes.flat,KEYS,LABELS)):
  x=[r['depth_center_um'] for r in rows];y=[r[key] for r in rows]
  ax.plot(x,y,lw=1.7,color=['#0072bd','#d95319','#7e2f8e','#388443'][i])
  ax.set(xlim=(0,8300),xlabel='Depth (um)',ylabel=title,ylim=(0,limits[i] if limits else None))
  ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
 fig.suptitle(label+'\n100 um bins; Shank1 +106850 um',fontsize=13)
 return fig

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 for sub in ['per_run','summary']: (OUT/sub).mkdir(exist_ok=True)
 depths=load_depths(C.DEFAULT_DEPTH_CSV);position=read_layout()['position']
 assert sum(d['depth_original_um']<0 for d in depths.values())==128
 assert {int(position[ch,1])+1 for ch,d in depths.items() if d['depth_original_um']<0}=={1}
 source=json.loads((ROOT/'s14_hot_r_0_25_redraw.json').read_text(encoding='utf-8'))['completed']
 s02=json.loads((ROOT/'single/S02/manifest.json').read_text(encoding='utf-8'))['completed']
 codes={}
 for r in s02:
  cache=Path(r['cache']);m=json.loads((cache/'result.json').read_text(encoding='utf-8'))
  codes[m['inputs']['recordings'][0]['sha256']]=(cache,m,r)
 allrows=[];profiles=[];runs=[]
 for job in source:
  folder=Path(job['output']);m=json.loads((folder/'result.json').read_text(encoding='utf-8'));rec=m['inputs']['recordings'][0]
  cache,q,s=codes[rec['sha256']];assert q['summary']['scope']=='whole_recording'
  assert storage.load_cached(cache,q['signature']) is not None
  header=describe(Path(s['recording']),C.DEFAULT_DATA_ROOT).header_info
  frame_s=float(header.get('window_ms') or 20)/1000
  n=q['summary']['n_frames'];exposure=n*frame_s
  byid={int(r['electrode_id'])-1:r for r in read(cache/'tables/channel_codes.csv')}
  weights=read(folder/'tables/channel_weights.csv');rows=[]
  name=Path(s['pngs'][0]).stem
  for w in weights:
   if w['transmitted']!='True':continue
   ch=int(w['channel_0based']);d=depths[ch]
   c=byid[ch];counts=[int(c[f'n_code{k}']) for k in range(4)];assert sum(counts)==n
   events=sum(counts[1:]);assert events==n-counts[0]
   # Validate the integration interval against the independent S02 mean rate.
   assert abs(sum(k*v for k,v in enumerate(counts))/exposure-float(c['mean_hz']))<.000051
   r=dict(recording=name,channel_0based=ch,shank=int(position[ch,1])+1,csv_shank=d['csv_shank'],depth_um=d['depth_um'],
          vx=abs(float(w['vx_signed'])),vy=abs(float(w['vy_signed'])),norm=float(w['norm']),
          nonzero_frame_events=events,exposure_seconds=exposure,event_hz=events/exposure,n_frames=n)
   rows.append(r)
  bins=[]
  for left in range(0,8300,100):
   group=[r for r in rows if left<=r['depth_um']<left+100]
   bins.append(dict(recording=name,depth_left_um=left,depth_right_um=left+100,depth_center_um=left+50,n_transmitted=len(group),
       vx_mean=float(np.mean([r['vx'] for r in group])) if group else np.nan,
       vy_mean=float(np.mean([r['vy'] for r in group])) if group else np.nan,
       norm_mean=float(np.mean([r['norm'] for r in group])) if group else np.nan,
       event_hz_mean=float(np.mean([r['event_hz'] for r in group])) if group else np.nan,
       event_count_sum=sum(r['nonzero_frame_events'] for r in group),event_hz_sum=sum(r['event_hz'] for r in group)))
  assert sum(r['event_count_sum'] for r in bins)==sum(r['nonzero_frame_events'] for r in rows)
  allrows.extend(rows);profiles.append(bins);runs.append(dict(name=name,label=m['summary']['recording'],s02=str(cache),weights=str(folder),exposure_seconds=exposure,n_included=len(rows)))
 summary=[]
 for stat in ['median','average']:
  bins=[]
  for j in range(len(profiles[0])):
   present=[p[j] for p in profiles if p[j]['n_transmitted']>0]
   agg=np.median if stat=='median' else np.mean
   bins.append(dict(statistic=stat,depth_center_um=profiles[0][j]['depth_center_um'],n_runs=len(present),
     **{k:float(agg([r[k] for r in present])) if present else np.nan for k in KEYS}))
  summary.extend(bins)
 limits=[max(r[k] for p in profiles for r in p if np.isfinite(r[k]))*1.08 or 1 for k in KEYS]
 with PdfPages(OUT/'per_run/All_45_runs_depth_profiles.pdf') as pdf:
  for run,bins in zip(runs,profiles):
   fig=fig_profiles(bins,run['label'],limits);fig.savefig(OUT/'per_run'/(run['name']+'.png'),dpi=220);pdf.savefig(fig);fig.clear()
 for stat in ['median','average']:
  bins=[r for r in summary if r['statistic']==stat];fig=fig_profiles(bins,'45 runs - '+stat)
  fig.savefig(OUT/'summary'/('Depth_profiles_'+stat+'.png'),dpi=300);fig.savefig(OUT/'summary'/('Depth_profiles_'+stat+'.pdf'));fig.clear()
 write(OUT/'channel_weights_and_events.csv',allrows);write(OUT/'per_run_depth_bins.csv',[r for p in profiles for r in p]);write(OUT/'summary_depth_bins.csv',summary)
 write(OUT/'depth_mapping_audit.csv',[dict(**d,plot_shank=int(position[ch,1])+1) for ch,d in depths.items()])
 (OUT/'excluded_channels.csv').write_text('channel_0based,reason\n',encoding='utf-8-sig')
 (OUT/'manifest.json').write_text(json.dumps(dict(runs=runs,depth_source=str(C.DEFAULT_DEPTH_CSV),depth_digest=storage.file_digest(C.DEFAULT_DEPTH_CSV),layout=str(C.DEFAULT_LAYOUT_XLSX),layout_digest=storage.file_digest(C.DEFAULT_LAYOUT_XLSX),depth_mapping='CSV Channel IDs joined to latest mapping; same Shank1..8; Shank1 +106850 um',excluded='none',shank1_offset_um=106850,definition='one event per channel per received frame with code > 0; code 1,2,3 all count 1',exposure='received frames times XDF configured window_ms; gaps excluded',aggregation='100um bins, mean over transmitted retained channels including zero weights/events; no observations = missing; cross-run median/mean of bin means with equal run weight',weight_scope='S14 S10 training selection',firing_scope='S02 whole loaded recording',limits=limits),ensure_ascii=False,indent=2),encoding='utf-8')
 print('DONE: 45 per-run profiles, 2 summary profiles, joined tables:',OUT)
if __name__=='__main__':main()
