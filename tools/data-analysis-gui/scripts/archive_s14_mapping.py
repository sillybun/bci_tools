from pathlib import Path
import shutil,json,csv,sys
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from data_analysis_gui import storage
from data_analysis_gui.core.style import finish_figure
from data_analysis_gui.sections.base import SectionResult
from data_analysis_gui.sections.s14_decoder_weights_v1.section import channel_rows,append_weight_figures
single=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/single');archive=single/'old_mapping'
for name in ['S14','S14_05_norm_weighting_heatmap','S14_05_norm_weighting_heatmap_origin_lower']:
 src=single/name;dst=archive/name
 for f in src.rglob('*'):
  if f.is_file():
   target=dst/f.relative_to(src);target.parent.mkdir(parents=True,exist_ok=True)
   if not target.exists():shutil.copy2(f,target)
manifest=json.loads((single.parent/'s14_hot_r_0_25_redraw.json').read_text(encoding='utf-8'))
for job in manifest['completed'][:4]:
 folder=Path(job['output']);entry=json.loads((folder/'result.json').read_text(encoding='utf-8'));s=entry['summary']
 with (folder/'tables/channel_weights.csv').open(encoding='utf-8-sig') as f:raw=list(csv.DictReader(line for line in f if line.strip() and not line.startswith('#')))
 pos=np.empty((1024,2),int)
 for r in raw:pos[int(r['channel_0based'])]=(int(r['elec_index'])-1,int(r['shank'])-1)
 layout=dict(position=pos,rows=128,columns=8,row_label='Elec# (1 near fan-in; 128 tip)',column_label='Shank',kind='shank_table')
 selected=sorted((r for r in raw if r['transmitted']=='True'),key=lambda r:int(r['stored_column_0based']))
 ids=np.array([int(r['electrode_id']) for r in selected]);w=np.array([[float(r['vx_signed']),float(r['vy_signed'])] for r in selected]);keep=np.array([r['status']=='fitted' for r in selected])
 result=SectionResult();append_weight_figures(result,layout,channel_rows(layout,ids,w,keep),f'{s["recording"]}\n{s["decoder"]}; {s["weight_basis"]}; {s["n_bins"]} bins','velocity / input SD','Saved old mapping positions')
 name=folder.parents[1].name[:10]+'_'+Path(entry['inputs']['recordings'][0]['name']).stem
 for n,(title,fig) in enumerate(result.figures,1):
  target=archive/'S14'/('S14_'+str(n).zfill(2)+'_'+title.replace(' ','_'))/(name+'.png')
  finish_figure(fig,entry['inputs']['font_percent']);fig.savefig(target,dpi=storage.PNG_DPI,facecolor='white',bbox_inches='tight');fig.clear()
 print('Archived old layout:',name,flush=True)
print('Archive complete')
