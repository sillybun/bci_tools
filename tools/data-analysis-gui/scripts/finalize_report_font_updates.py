"""Rebuild combined PDFs and audit the report-wide typography update."""
from pathlib import Path
import collections
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pypdf import PdfReader,PdfWriter
from update_report_font_sizes import ROOT,AUDIT,LOG,sha,backup,is_active,is_schematic

# Only this historical report is authorized for typography migration.
ROOT = ROOT/'2026-10-26_2026-11-11_45runs_V1'


def main():
    logs=[json.loads(line) for line in LOG.read_text(encoding='utf-8').splitlines()]
    first={};last={}
    for x in logs:first.setdefault(x['path'],x);last[x['path']]=x
    restored=json.loads((AUDIT/'restored_schematics.json').read_text(encoding='utf-8'))
    assert all(sha(p)==first[p]['before'] for p in restored)
    protected=json.loads((AUDIT/'data_before.json').read_text(encoding='utf-8'))
    changed=[p for p,h in protected.items() if sha(p)!=h]
    assert not changed,changed
    pdf_updates=[];unmatched=[]
    for pdf in list(ROOT.rglob('*.pdf')):
        if not is_active(pdf) or pdf.with_suffix('.png').exists():continue
        n=len(PdfReader(pdf).pages)
        if n<=1:unmatched.append(str(pdf));continue
        images=sorted(p for p in pdf.parent.glob('*.png') if is_active(p))
        if 'Six_Channels_Direction_Spectrograms' in pdf.name:
            images=[p for p in images if p.name.startswith('LFP_Channel_')]
        if len(images)!=n or any(not p.with_suffix('.pdf').exists() for p in images):
            unmatched.append(str(pdf));continue
        before=backup(pdf);writer=PdfWriter()
        for p in images:writer.append(str(p.with_suffix('.pdf')))
        writer.write(str(pdf));writer.close()
        assert len(PdfReader(pdf).pages)==n
        pdf_updates.append(dict(path=str(pdf),pages=n,before=before,sha256=sha(pdf)))
    images=[p for p in ROOT.rglob('*.png') if is_active(p)]
    missing=[str(p) for p in images if str(p) not in last and str(p) not in restored]
    assert not missing,missing
    # Latest figure digests, keeping the first originals in the backup index.
    versions=collections.Counter(last[str(p)].get('layout_version') for p in images if not is_schematic(p))
    policy=dict(title_pt=18,axis_label_pt=14,tick_pt=11,legend_pt=11,schematics='Original PNGs restored exactly; future schematic typography preserved',
        chinese_font='Microsoft YaHei for Chinese SVG text; native fallback for Matplotlib',
        backup=str(AUDIT/'backup'),data_tables_unchanged=len(protected))
    for report in [ROOT]:
        local=[p for p in images if p.is_relative_to(report)]
        (report/'font_style_update.json').write_text(json.dumps(dict(policy=policy,pngs=len(local),schematics_preserved=sum(is_schematic(p) for p in local),
            analysis_figures_updated=sum(not is_schematic(p) for p in local)),ensure_ascii=False,indent=2),encoding='utf-8')
        metadata=report/'report.json'
        if metadata.exists():
            r=json.loads(metadata.read_text(encoding='utf-8'));r['font_policy']=policy;r['font_update_manifest']='font_style_update.json'
            metadata.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    result=dict(complete=not unmatched,png_count=len(images),analysis_pngs=sum(not is_schematic(p) for p in images),
        restored_schematics=len(restored),unchanged_csv_tables=len(protected),versions=dict(versions),
        combined_pdfs=pdf_updates,unmatched_pdfs=unmatched,policy=policy)
    (AUDIT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('combined_pdfs','policy')},ensure_ascii=False))


if __name__=='__main__':main()
