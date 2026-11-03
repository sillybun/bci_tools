"""Median of per-run S05 binary tuning decisions, conditional on a valid test."""
from pathlib import Path
import sys,json,csv
import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from export_s05_tuning_layouts import C,storage,read_layout,load_depths,read,write,OUT

def main():
 m=json.loads((OUT/'tuning_layout_manifest.json').read_text(encoding='utf-8'));layout=read_layout();depths=load_depths(C.DEFAULT_DEPTH_CSV)
 observations={ch:[] for ch in range(1024)};transmitted=np.zeros(1024,int)
 for run in m['runs']:
  rows=read(OUT/'tables_tuning_layout'/(run['name']+'.csv'))
  for r in rows:
   ch=int(r['channel_0based']);transmitted[ch]+=1
   if r.get('tested','True')=='True':observations[ch].append(int(r['significant']=='True'))
 rows=[];grid=np.full((128,8),np.nan)
 for ch in range(1024):
  values=observations[ch];median=float(np.median(values)) if values else np.nan;row,col=map(int,layout['position'][ch]);grid[row,col]=median
  rows.append(dict(channel_0based=ch,electrode_id=ch+1,shank=col+1,elec_index=row+1,depth_um=depths[ch]['depth_um'],n_runs_selected=len(m['runs']),n_runs_transmitted=int(transmitted[ch]),n_runs_tested=len(values),n_runs_tuned=sum(values),tuning_fraction=sum(values)/len(values) if values else '',tuning_median=median if values else ''))
 assert sum(len(v) for v in observations.values())==44*512
 cmap=ListedColormap(['white','#efaa32','#c92929']).with_extremes(bad='#d6d4cf');norm=BoundaryNorm([-.25,.25,.75,1.25],3)
 fig=Figure(figsize=(8.27,11.69));ax=fig.add_axes([.12,.14,.82,.73]);ax.set_facecolor('#d6d4cf');inverse={(int(r),int(c)):ch for ch,(r,c) in enumerate(layout['position'])}
 for col in range(8):
  ds=np.array([depths[inverse[(r,col)]]['depth_um'] for r in range(128)]);order=np.argsort(ds);centres=ds[order];edges=np.r_[centres[0]-25,(centres[:-1]+centres[1:])/2,centres[-1]+25]
  assert np.allclose((edges[:-1]+edges[1:])/2,centres)
  ax.pcolormesh([col+.5,col+1.5],edges,np.ma.masked_invalid(grid[order,col,None]),cmap=cmap,norm=norm,shading='flat')
 ax.set(xlim=(.5,8.5),ylim=(-25,8500),xticks=range(1,9),xlabel='Shank',ylabel='Depth (um)');ax.set_yticks(range(0,8500,1000));ax.spines[['top','right']].set_visible(False)
 fig.suptitle('S05 direction tuning - median\n45 recordings; median of valid per-run 0/1 decisions\nBH-FDR q < 0.05 within each run',fontsize=13,y=.96)
 counts={v:int(np.sum(grid==v)) for v in [0,.5,1]}
 fig.legend(handles=[Patch(color='#c92929',label=f'Median 1: tuned in >50% of tested runs (n={counts[1]})'),Patch(color='#efaa32',label=f'Median 0.5: tuned in exactly 50% (n={counts[.5]})'),Patch(facecolor='white',edgecolor='#888',label=f'Median 0: tuned in <50% (n={counts[0]})'),Patch(color='#d6d4cf',label='No valid test / no site')],loc='lower center',bbox_to_anchor=(.5,.025),frameon=False,fontsize=9)
 out=OUT/'tuning_layout_depth/summary';out.mkdir(exist_ok=True);stem='S05_direction_tuning_median_depth'
 fig.savefig(out/(stem+'.png'),dpi=300);fig.savefig(out/(stem+'.pdf'));fig.clear();write(out/(stem+'.csv'),rows)
 (out/'median_manifest.json').write_text(json.dumps(dict(source_manifest=str(OUT/'tuning_layout_manifest.json'),runs=m['runs'],definition='Per-channel median of S05 binary significant decisions, excluding untransmitted and untested runs; 0.5 retained as a tie',counts={str(k):v for k,v in counts.items()},n_no_valid_test=int(np.isnan(grid).sum()),layout=str(C.DEFAULT_LAYOUT_XLSX),layout_digest=storage.file_digest(C.DEFAULT_LAYOUT_XLSX),depth=str(C.DEFAULT_DEPTH_CSV),depth_digest=storage.file_digest(C.DEFAULT_DEPTH_CSV),shank1_offset_um=106850),ensure_ascii=False,indent=2),encoding='utf-8')
 print('DONE',out,'counts',counts,'missing',int(np.isnan(grid).sum()),'tested coverage',min(len(v) for v in observations.values()),max(len(v) for v in observations.values()))
if __name__=='__main__':main()
