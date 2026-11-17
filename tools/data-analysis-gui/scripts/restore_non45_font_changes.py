"""Undo this typography migration outside the explicitly retained 45-run report."""
from pathlib import Path
import json
import re
import shutil
import hashlib

ROOT=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report').resolve()
KEEP=ROOT/'2026-10-26_2026-11-11_45runs_V1'
AUDIT=ROOT.parent/'font_update_18_14_11'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    plan=json.loads((AUDIT/'restore_other_reports_plan.json').read_text(encoding='utf-8'))
    tasks=json.loads((AUDIT/'vector_tasks.json').read_text(encoding='utf-8'))
    native={Path(v).resolve() for v in tasks.values() if Path(v).resolve().is_relative_to(ROOT) and not Path(v).resolve().is_relative_to(KEEP)}
    dates={}
    for p in (AUDIT/'backup').glob('*.svg'):
        with p.open('r',encoding='utf-8') as f:s=f.read(4096)
        vb=re.search(r'viewBox="([^"]+)"',s);date=re.search(r'<dc:date>(.*?)</dc:date>',s)
        if vb and vb[1].startswith('0 0 ') and date:dates.setdefault(date[1],p)
    native_original={}
    for p in native:
        s=p.read_text(encoding='utf-8');date=re.search(r'<dc:date>(.*?)</dc:date>',s)
        assert date and date[1] in dates,str(p)
        native_original[p]=dates[date[1]]
    assert len(native_original)==7,len(native_original)
    for row in plan:
        p=Path(row['path']).resolve()
        assert p.is_relative_to(ROOT) and not p.is_relative_to(KEEP),p
        original=Path(row['original_png']);assert original.exists()
        if row['original_pdf']:assert Path(row['pdf_backup']).exists()
    for row in plan:
        p=Path(row['path']).resolve();original=Path(row['original_png'])
        shutil.copy2(original,p)
        assert digest(p)==original.stem
        pdf=p.with_suffix('.pdf');svg=p.with_suffix('.svg')
        if row['original_pdf']:shutil.copy2(row['pdf_backup'],pdf)
        elif pdf.exists():pdf.unlink()
        if svg in native_original:shutil.copy2(native_original[svg],svg)
        elif svg.exists():svg.unlink()
    result={'restored_pngs':len(plan),'restored_pdfs':sum(r['original_pdf'] for r in plan),'restored_native_svgs':len(native_original),'retained_report':str(KEEP)}
    (AUDIT/'restoration_complete.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(result)

if __name__=='__main__':main()
