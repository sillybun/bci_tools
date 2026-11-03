"""Map S05 significant channels onto latest electrode and depth layouts for 45 runs."""
import sys,json,csv
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_analysis_gui import config as C,storage
from data_analysis_gui.core.channels import read_layout,layout_grid
from data_analysis_gui.core.catalog import describe
from data_analysis_gui.core.filters import TrialFilter
from data_analysis_gui.core.session import Session
from data_analysis_gui.core import analysis as A
from data_analysis_gui.runner import build_context
from data_analysis_gui.sections.s05_anova_v1.section import SECTION,channel_anova
from data_analysis_gui.sections.s14_decoder_weights_v1.depth_distribution import load_depths
ROOT=Path('E:/AitBCI/data_analysis_gui_output/keaiduo');OUT=ROOT/'single/S05'
def read(path):
 with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(line for line in f if line.strip() and not line.startswith('#')))
def write(path,rows):
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def tuning_figure(layout,depths,rows,label,depth):
 fig=Figure(figsize=(8.27,11.69));ax=fig.add_axes([.12,.12,.82,.75]);ax.set_facecolor('#d6d4cf')
 cmap=ListedColormap(['#ffffff','#c92929','#badfea']).with_extremes(bad='#d6d4cf');norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
 ids=np.array([int(r['electrode']) for r in rows]);sig=np.array([2 if r.get('tested')=='False' else int(r['significant']=='True') for r in rows],float)
 grid=layout_grid(layout,ids,sig)
 if depth:
  inv={(int(r),int(c)):ch for ch,(r,c) in enumerate(layout['position'])}
  for col in range(8):
   ds=np.array([depths[inv[(r,col)]]['depth_um'] for r in range(128)])
   order=np.argsort(ds);centres=ds[order];edges=np.r_[centres[0]-25,(centres[:-1]+centres[1:])/2,centres[-1]+25]
   ax.pcolormesh([col+.5,col+1.5],edges,np.ma.masked_invalid(grid[order,col,None]),cmap=cmap,norm=norm,shading='flat')
  ax.set(xlim=(.5,8.5),ylim=(-25,8500),ylabel='Depth (um)',xticks=range(1,9));ax.set_yticks(range(0,8500,1000))
 else:
  ax.imshow(np.ma.masked_invalid(grid),origin='lower',aspect='auto',cmap=cmap,norm=norm,extent=(.5,8.5,.5,128.5),interpolation='nearest')
  ax.set(xlim=(.5,8.5),ylim=(.5,128.5),ylabel='Electrode #',xticks=range(1,9));ax.set_yticks([1,16,32,48,64,80,96,112,128])
 ax.set_xlabel('Shank');ax.spines[['top','right']].set_visible(False)
 n=int(np.sum(sig==1));status=f'{n} tuned / {len(rows)} transmitted; BH-FDR q < 0.05' if np.all(sig!=2) else 'Not tested: fewer than 16 eligible trials'
 fig.suptitle(f'S05 direction tuning - {"Depth" if depth else "Electrode #"}\n{label}\n{status}',fontsize=14,y=.96)
 fig.legend(handles=[Patch(color='#c92929',label='Direction tuned'),Patch(facecolor='white',edgecolor='#888888',label='Transmitted; not significant'),Patch(color='#badfea',label='Transmitted; not tested'),Patch(color='#d6d4cf',label='Not transmitted / no site')],loc='lower center',bbox_to_anchor=(.5,.035),ncol=1,frameon=False,fontsize=10)
 return fig

def main():
 OUT.mkdir(exist_ok=True);(OUT/'tables_tuning_layout').mkdir(exist_ok=True)
 layout=read_layout();depths=load_depths(C.DEFAULT_DEPTH_CSV)
 historical=[]
 for path in ROOT.glob('single/*/s05_v1/*/result.json'):
  meta=json.loads(path.read_text(encoding='utf-8'));historical.append((path,meta))
 template=historical[0][1];criteria=template['inputs']['criteria'];params=template['inputs']['params'];assert params['alpha']==.05 and not params['gain_control']
 old={}
 for path,m in historical:
  if m['inputs']['criteria']==criteria and m['inputs']['params']==params:old[m['inputs']['recordings'][0]['sha256']]=(path,m)
 source=json.loads((ROOT/'single/S02/manifest.json').read_text(encoding='utf-8'))['completed'];entries=[]
 dirs={False:OUT/'tuning_layout_electrode_number',True:OUT/'tuning_layout_depth'}
 for d in dirs.values():d.mkdir(exist_ok=True)
 with PdfPages(dirs[False]/'All_45_runs_S05_electrode_number.pdf') as pdf_e,PdfPages(dirs[True]/'All_45_runs_S05_depth.pdf') as pdf_d:
  for i,run in enumerate(source,1):
   sm=json.loads((Path(run['cache'])/'result.json').read_text(encoding='utf-8'));rec=sm['inputs']['recordings'][0];name=Path(run['pngs'][0]).stem
   if rec['sha256'] in old:
    path,m=old[rec['sha256']];assert storage.load_cached(path.parent,m['signature']) is not None
    rows=read(path.parent/m['tables']['anova_channels']);provenance=str(path);n_trials=m['summary']['trials'];mode='historical'
   else:
    path=Path(run['recording']);assert storage.recording_sha(path)==rec['sha256']
    session=Session.load(describe(path,C.DEFAULT_DATA_ROOT));ctx=build_context([session],TrialFilter.from_dict(criteria),params,SECTION,layout_path=str(C.DEFAULT_LAYOUT_XLSX))
    x=ctx.cohort.window_rates
    x=x[np.isfinite(x).all(1)]
    if len(x)<16:
     rows=[dict(stored_column=j+1,electrode=int(e),active=str(bool(x[:,j].std()>1e-12)) if len(x) else 'False',F='',p='',q='',eta2='',significant='False',tested='False') for j,e in enumerate(session.electrode_ids)]
    else:
     st,x,directions=channel_anova(ctx)
     active=x.std(0)>1e-12;q=A.bh_fdr(st['p'],np.flatnonzero(active));sig=active&(q<params['alpha'])
     rows=[dict(stored_column=j+1,electrode=int(e),active=str(bool(active[j])),F=float(st['F'][j]),p=float(st['p'][j]),q=float(q[j]),eta2=float(st['eta2'][j]),significant=str(bool(sig[j])),tested='True') for j,e in enumerate(session.electrode_ids)]
    n_trials=len(x);provenance=str(path);mode='computed';del session,ctx,x
   assert len(rows)==512
   for r in rows:r.setdefault('tested','True')
   assert all((r['significant']=='True')==((r['active']=='True') and float(r['q'])<.05) for r in rows if r['tested']=='True')
   for r in rows:
    ch=int(r['electrode'])-1;row,col=map(int,layout['position'][ch]);r.update(channel_0based=ch,shank=col+1,elec_index=row+1,depth_um=depths[ch]['depth_um'])
   write(OUT/'tables_tuning_layout'/(name+'.csv'),rows)
   for depth,pdf in [(False,pdf_e),(True,pdf_d)]:
    fig=tuning_figure(layout,depths,rows,run['label'],depth);dest=dirs[depth]/name
    fig.savefig(dest.with_suffix('.png'),dpi=250);fig.savefig(dest.with_suffix('.pdf'));pdf.savefig(fig);fig.clear()
   entries.append(dict(name=name,n_trials=n_trials,n_significant=sum(r['significant']=='True' for r in rows),tested=all(r['tested']=='True' for r in rows),source=provenance,mode=mode))
   print(f'[{i}/45] {mode}: tuned={entries[-1]["n_significant"]} {name}',flush=True)
 (OUT/'tuning_layout_manifest.json').write_text(json.dumps(dict(runs=entries,criteria=criteria,params=params,layout=str(C.DEFAULT_LAYOUT_XLSX),layout_digest=storage.file_digest(C.DEFAULT_LAYOUT_XLSX),depth=str(C.DEFAULT_DEPTH_CSV),depth_digest=storage.file_digest(C.DEFAULT_DEPTH_CSV),shank1_offset_um=106850,orientation='Shank1 left, Shank8 right; lower origin',definition='S05 plain one-way ANOVA of original count-derived trial window rates; BH-FDR among active channels; q<0.05; gain control disabled',historical=sum(r['mode']=='historical' for r in entries)),ensure_ascii=False,indent=2),encoding='utf-8')
 print('DONE: 45 runs x 2 layouts, PNG/PDF and two combined PDFs',flush=True)
if __name__=='__main__':main()
