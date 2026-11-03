"""Mean of valid per-run S05 0/1 tuning decisions (tuned-run fraction)."""
from pathlib import Path
import sys,json
import numpy as np
from matplotlib.figure import Figure
from matplotlib import colormaps
from matplotlib.colors import Normalize
sys.path.insert(0,str(Path(__file__).resolve().parent))
from export_s05_tuning_layouts import C,storage,read_layout,load_depths,read,write,OUT

def main():
 m=json.loads((OUT/'tuning_layout_manifest.json').read_text(encoding='utf-8'));layout=read_layout();depths=load_depths(C.DEFAULT_DEPTH_CSV)
 observations={ch:[] for ch in range(1024)};transmitted=np.zeros(1024,int)
 for run in m['runs']:
  for r in read(OUT/'tables_tuning_layout'/(run['name']+'.csv')):
   ch=int(r['channel_0based']);transmitted[ch]+=1
   if r.get('tested','True')=='True':observations[ch].append(int(r['significant']=='True'))
 rows=[];grid=np.full((128,8),np.nan)
 for ch,values in observations.items():
  mean=float(np.mean(values)) if values else np.nan;row,col=map(int,layout['position'][ch]);grid[row,col]=mean
  rows.append(dict(channel_0based=ch,electrode_id=ch+1,shank=col+1,elec_index=row+1,depth_um=depths[ch]['depth_um'],n_runs_selected=len(m['runs']),n_runs_transmitted=int(transmitted[ch]),n_runs_tested=len(values),n_runs_tuned=sum(values),tuning_mean=mean if values else ''))
 assert sum(len(v) for v in observations.values())==44*512
 fig=Figure(figsize=(8.27,11.69));ax=fig.add_axes([.12,.12,.68,.75]);ax.set_facecolor('#d6d4cf')
 cmap=colormaps['hot_r'].with_extremes(bad='#d6d4cf');norm=Normalize(0,1);inverse={(int(r),int(c)):ch for ch,(r,c) in enumerate(layout['position'])}
 for col in range(8):
  ds=np.array([depths[inverse[(r,col)]]['depth_um'] for r in range(128)]);order=np.argsort(ds);centres=ds[order];edges=np.r_[centres[0]-25,(centres[:-1]+centres[1:])/2,centres[-1]+25]
  assert np.allclose((edges[:-1]+edges[1:])/2,centres)
  image=ax.pcolormesh([col+.5,col+1.5],edges,np.ma.masked_invalid(grid[order,col,None]),cmap=cmap,norm=norm,shading='flat')
 ax.set(xlim=(.5,8.5),ylim=(-25,8500),xticks=range(1,9),xlabel='Shank',ylabel='Depth (um)');ax.set_yticks(range(0,8500,1000));ax.spines[['top','right']].set_visible(False)
 cax=fig.add_axes([.84,.12,.025,.75]);cb=fig.colorbar(image,cax=cax,ticks=np.linspace(0,1,6));cb.set_label('Fraction of tested runs with direction tuning')
 fig.suptitle('S05 direction tuning - mean\n45 recordings; mean of valid per-run 0/1 decisions\nBH-FDR q < 0.05 within each run',fontsize=13,y=.96)
 fig.text(.5,.055,'0 = never tuned; 1 = tuned in every tested run; high = dark.',ha='center',fontsize=9)
 fig.text(.5,.035,'Grey = no valid test / no site. Untransmitted and untested runs excluded.',ha='center',fontsize=8)
 out=OUT/'tuning_layout_depth/summary';out.mkdir(exist_ok=True);stem='S05_direction_tuning_mean_depth'
 fig.savefig(out/(stem+'.png'),dpi=300);fig.savefig(out/(stem+'.pdf'));fig.clear();write(out/(stem+'.csv'),rows)
 (out/'mean_manifest.json').write_text(json.dumps(dict(source_manifest=str(OUT/'tuning_layout_manifest.json'),runs=m['runs'],definition='Per-channel mean of S05 binary tuning decisions = n_tuned/n_tested; missing/untransmitted/untested excluded; each tested run equally weighted',n_no_valid_test=int(np.isnan(grid).sum()),color_limits=[0,1],colormap='hot_r',layout=str(C.DEFAULT_LAYOUT_XLSX),layout_digest=storage.file_digest(C.DEFAULT_LAYOUT_XLSX),depth=str(C.DEFAULT_DEPTH_CSV),depth_digest=storage.file_digest(C.DEFAULT_DEPTH_CSV),shank1_offset_um=106850),ensure_ascii=False,indent=2),encoding='utf-8')
 print('DONE',out,'valid channels',int(np.isfinite(grid).sum()),'range',float(np.nanmin(grid)),float(np.nanmax(grid)))
if __name__=='__main__':main()
