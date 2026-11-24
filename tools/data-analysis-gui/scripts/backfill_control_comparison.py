"""Backfill only the requested new comparison section; preserve all prior artifacts."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_analysis_gui.control_comparison import render_comparison,CATEGORY
from data_analysis_gui.report_run_pool import RunPool,atomic_json
from data_analysis_gui.core.style import apply

root=Path(sys.argv[1] if len(sys.argv)>1 else 'E:/AitBCI/data_analysis_gui_output/keaiduo/report/2026-11-10_2026-11-23_60runs_v1')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before={p:digest(p) for p in root.rglob('*') if p.is_file() and p.name!='report.json' and CATEGORY not in p.parts}
report=json.loads((root/'report.json').read_text(encoding='utf-8'))
pool=RunPool(root.parent/'individual_runs');apply()
result=render_comparison(root,report,100,lambda event,**kw:print(kw.get('message',''),flush=True),pool)
assert all(p.is_file() and digest(p)==sha for p,sha in before.items())
report['control_comparison']=result;atomic_json(root/'report.json',report)
print(json.dumps(dict(**result,preserved_files=len(before)),ensure_ascii=False),flush=True)
