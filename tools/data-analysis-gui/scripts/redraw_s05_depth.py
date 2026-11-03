"""Redraw S05 depth layouts from saved S05 decisions; never recompute the tests."""
from pathlib import Path
import json,sys
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0,str(Path(__file__).resolve().parent))
from export_s05_tuning_layouts import C,storage,read_layout,load_depths,read,write,tuning_figure,OUT,ROOT

def main():
 manifest=OUT/'tuning_layout_manifest.json';m=json.loads(manifest.read_text(encoding='utf-8'))
 source=json.loads((ROOT/'single/S02/manifest.json').read_text(encoding='utf-8'))
 labels={Path(r['pngs'][0]).stem:r['label'] for r in source['completed']}
 layout=read_layout();depths=load_depths(C.DEFAULT_DEPTH_CSV);out=OUT/'tuning_layout_depth'
 with PdfPages(out/'All_45_runs_S05_depth.pdf') as pdf:
  for i,run in enumerate(m['runs'],1):
   path=OUT/'tables_tuning_layout'/(run['name']+'.csv');rows=read(path)
   original=[(r['electrode'],r['significant'],r.get('tested','True'),r['q']) for r in rows]
   for r in rows:r['depth_um']=depths[int(r['electrode'])-1]['depth_um']
   assert original==[(r['electrode'],r['significant'],r.get('tested','True'),r['q']) for r in rows]
   write(path,rows)
   fig=tuning_figure(layout,depths,rows,labels[run['name']],True)
   fig.savefig(out/(run['name']+'.png'),dpi=250);fig.savefig(out/(run['name']+'.pdf'));pdf.savefig(fig);fig.clear()
   print(f'[{i}/45] S05 depth redrawn',flush=True)
 m.pop('shank8_offset_um',None);m.update(depth=str(C.DEFAULT_DEPTH_CSV),depth_digest=storage.file_digest(C.DEFAULT_DEPTH_CSV),shank1_offset_um=106850,depth_redraw='Saved S05 significance decisions; no ANOVA recomputation')
 manifest.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
