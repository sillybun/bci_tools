"""Add daily first-run chip maps to the requested report without redrawing old figures."""
from pathlib import Path
import json,sys,shutil,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
from data_analysis_gui.report_bga import render_daily_bga

ROOT=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report/2026-11-11_2026-11-20_52runs_v1')
AUDIT=ROOT.parent.parent/'import_20261120'/'bga_first_run_backfill'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    AUDIT.mkdir(exist_ok=True)
    assert not (ROOT/'bga_daily_manifest.json').exists(),'Daily BGA maps already present; inspect before rerunning.'
    original={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.png','.pdf','.svg','.csv')}
    (AUDIT/'original_figures.json').write_text(json.dumps(original,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(ROOT/'report.json',AUDIT/'report_before.json')
    report=json.loads((ROOT/'report.json').read_text(encoding='utf-8'))
    result=render_daily_bga(ROOT,report,100,lambda *a,**k:print(k.get('message',''),flush=True))
    report['bga_daily']=result
    (ROOT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    assert all(sha(ROOT/p)==h for p,h in original.items()),'An existing figure changed'
    (AUDIT/'verification.json').write_text(json.dumps(dict(existing_files_unchanged=len(original),days=result['days']),ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',len(result['days']),'daily maps;',len(original),'existing files unchanged',flush=True)
if __name__=='__main__':main()
