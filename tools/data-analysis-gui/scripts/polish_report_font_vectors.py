"""Keep 11pt atlas ticks readable and match figure-wide axis captions to 14pt."""
from pathlib import Path
import collections
import hashlib
import json
import os
import re
import shutil
import sys
from xml.etree import ElementTree as ET
from concurrent.futures import ProcessPoolExecutor,as_completed

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from update_report_font_sizes import ROOT,AUDIT,NS,cairosvg,backup,record,is_schematic


def transform(source):
    root=ET.fromstring(Path(source).read_bytes());parents={c:p for p in root.iter() for c in p}
    boxes={};rotated=[]
    for g in root.iter(NS+'g'):
        if not g.get('id','').startswith('axes_'):continue
        patch=next((p for p in g if p.get('id','').startswith('patch_')),None)
        path=patch.find(NS+'path') if patch is not None else None
        if path is None:continue
        points=re.findall(r'[ML]\s*([-\d.]+)\s+([-\d.]+)',path.get('d',''))
        if points:
            xs,ys=zip(*[(float(x),float(y)) for x,y in points]);boxes[g]=(min(xs),min(ys),max(xs),max(ys))
    for g,box in boxes.items():
        if box[2]-box[0]>=160:continue
        ticks=[]
        for t in g.iter(NS+'text'):
            node=t;chain=[]
            while node in parents:node=parents[node];chain.append(node.get('id',''))
            if any(x.startswith('xtick_') for x in chain) and 'x' in t.attrib and 'y' in t.attrib:ticks.append(t)
        if len(ticks)<5:continue
        for t in ticks:
            x=float(t.get('x'));y=float(t.get('y'))+3;t.set('y',str(y));t.set('transform',f'rotate(-45 {x} {y})')
            style=re.sub(r'text-anchor:[^;]+','text-anchor: end',t.get('style',''));t.set('style',style)
        rotated.append(g)
    rows=sorted({round(boxes[g][1],2) for g in rotated});extra=max(0,len(rows)-1)*6
    if rotated:
        for g,box in boxes.items():
            row=min(range(len(rows)),key=lambda i:abs(rows[i]-box[1]));dy=row*6
            g.set('transform',f'translate(0 {dy}) '+g.get('transform',''))
    labels={'Target direction / deg','Time / s','Time (s)','Time from target onset (s)','Target direction (deg)'}
    changed=bool(rotated)
    for t in root.iter(NS+'text'):
        node=t;chain=[]
        while node in parents:node=parents[node];chain.append(node.get('id',''))
        if any(x.startswith('axes_') for x in chain):continue
        content=''.join(t.itertext()).strip();role=t.get('data-font-role')
        if content in labels:
            t.set('style',re.sub(r'font-size:\s*[\d.]+(?:px|pt)?','font-size: 14px',t.get('style','')))
            t.set('data-font-role','axis_label');changed=True
        if extra and role!='figure-title':
            if 'rotate(-90' in t.get('transform',''):t.set('transform',f'translate(0 {extra/2}) '+t.get('transform',''))
            elif 'y' in t.attrib:t.set('y',str(float(t.get('y'))+extra))
    if not changed:return None
    if extra:
        vb=list(map(float,root.get('viewBox').split()));vb[3]+=extra;root.set('viewBox',' '.join(map(str,vb)))
        root.set('height',str(float(root.get('height').removesuffix('pt'))+extra)+'pt')
    content=ET.tostring(root,encoding='utf-8',xml_declaration=True);key=hashlib.sha256(content).hexdigest()
    cache=AUDIT/'polished';cache.mkdir(exist_ok=True);stem=cache/key
    if not stem.with_suffix('.png').exists():
        temporary=stem.with_name(stem.name+'_'+str(os.getpid()))
        temporary.with_suffix('.svg').write_bytes(content)
        cairosvg.svg2pdf(bytestring=content,write_to=str(temporary.with_suffix('.pdf')),dpi=72,background_color='white')
        cairosvg.svg2png(bytestring=content,write_to=str(temporary.with_suffix('.png')),dpi=300,background_color='white')
        from PIL import Image
        with Image.open(temporary.with_suffix('.png')) as im:im.save(temporary.with_suffix('.png'),dpi=(300,300))
        for ext in ['svg','pdf','png']:os.replace(temporary.with_suffix('.'+ext),stem.with_suffix('.'+ext))
    return str(stem)


def main():
    tasks=json.loads((AUDIT/'vector_tasks.json').read_text(encoding='utf-8'));errors=[];done=0
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(transform,Path(p).with_suffix('.svg')):Path(p) for p in tasks if not is_schematic(p)}
        for future in as_completed(futures):
            p=futures[future]
            try:
                value=future.result()
                if value:
                    stem=Path(value);before=backup(p)
                    for ext in ['png','pdf','svg']:shutil.copy2(stem.with_suffix('.'+ext),p.with_suffix('.'+ext))
                    record(p,'editable SVG typography',p.with_suffix('.svg'),before);done+=1
            except Exception as exc:errors.append(dict(path=str(p),error=repr(exc)))
    (AUDIT/'polish_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Polished',done,'errors',len(errors))


if __name__=='__main__':main()
