"""Restyle report figures from saved vectors/tables; keep source data and backups."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis_gui import storage

ROOT = Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report')
AUDIT = ROOT.parent/'font_update_18_14_11'
AUDIT.mkdir(exist_ok=True)
NATIVE = Path('C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/native/poppler/Library/bin')
_dll = os.add_dll_directory(str(NATIVE))
import cairosvg
NS = '{http://www.w3.org/2000/svg}'
ET.register_namespace('', NS[1:-1])
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
BACKUPS = AUDIT/'backup'
BACKUPS.mkdir(exist_ok=True)
LOG = AUDIT/'updates.jsonl'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(line for line in f if line.strip() and not line.startswith('#')))


def typed(rows):
    out = []
    for r in rows:
        r = dict(r)
        for k, v in r.items():
            if k in ('date', 'run', 'run_key', 'session', 'recording') or k.endswith('_session'):
                continue
            try:r[k] = float(v)
            except (ValueError, TypeError):pass
        out.append(r)
    return out


def backup(p):
    p = Path(p)
    if p.exists():
        h = sha(p)
        target = BACKUPS/(h+p.suffix.lower())
        if not target.exists():shutil.copy2(p, target)
        return h
    return None


def record(p, method, source=None, before=None):
    with LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(path=str(p), method=method, source=str(source or ''), before=before,
                               sha256=sha(p), titles_pt=18, labels_pt=14, ticks_pt=11,
                               layout_version='svg-cjk-v5' if method=='editable SVG typography' else 'wrapped-v2'), ensure_ascii=False)+'\n')


def is_active(p):
    return not any('style_before_' in x or x.startswith('.') for x in p.relative_to(ROOT).parts)


def is_schematic(p):
    return Path(p).name.startswith('X00_') or 'Session_Overview' in Path(p).name


def save(fig, png, source, *, dpi=300):
    from data_analysis_gui.core.style import finish_figure
    png = Path(png)
    before = backup(png)
    finish_figure(fig)
    for ext in ('png', 'pdf', 'svg'):
        target = png.with_suffix('.'+ext)
        if ext != 'png':backup(target)
        fig.savefig(target, dpi=dpi if ext == 'png' else None, facecolor='white')
    record(png, 'saved statistics redraw', source, before)
    fig.clear()


def source_vectors():
    tasks = {}
    for p in ROOT.glob('*/report.json'):
        m = json.loads(p.read_text(encoding='utf-8'))
        for source in m.get('sources', []):
            folder = Path(source['folder'])
            meta = folder/'result.json'
            if not meta.exists():continue
            data = json.loads(meta.read_text(encoding='utf-8'))
            for relative in source['pngs']:
                target = p.parent/relative
                if not target.exists():continue
                choices = [folder/f['files']['svg'] for f in data['figures']
                           if f['files'].get('svg') and target.name.endswith(Path(f['files']['png']).name)]
                if len(choices) == 1 and choices[0].exists():tasks[str(target)] = str(choices[0])
    # The moved S02 collections are exact copies of their cached section exports.
    for mpath in ROOT.glob('*/S02/manifest.json'):
        m = json.loads(mpath.read_text(encoding='utf-8'))
        for entry in m.get('completed', []):
            folder = Path(entry['cache'])
            data = json.loads((folder/'result.json').read_text(encoding='utf-8'))
            for old in entry['pngs']:
                old = Path(old)
                target = mpath.parent/old.parent.name/old.name
                for f in data['figures']:
                    if Path(f['files']['png']).stem == old.parent.name:
                        source = folder/f['files']['svg']
                        if target.exists() and source.exists():tasks[str(target)] = str(source)
    for p in ROOT.rglob('*.svg'):
        if is_active(p) and p.with_suffix('.png').exists():tasks.setdefault(str(p.with_suffix('.png')), str(p))
    return tasks


def normalize_svg(path):
    root = ET.fromstring(Path(path).read_bytes())
    parents = {c: p for p in root.iter() for c in p}
    vb = list(map(float, root.attrib['viewBox'].split()))
    width, height = vb[2:]
    axes_top = {}
    axes_bottom = {}
    def global_position(text):
        x=float(text.get('x','0').split()[0]);y=float(text.get('y','0').split()[0]);node=text
        while node is not None:
            operations=re.findall(r'(translate|rotate|scale|matrix)\(([^)]*)\)',node.get('transform',''))
            for op,args in reversed(operations):
                v=list(map(float,re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',args)))
                if op=='translate':x+=v[0];y+=v[1] if len(v)>1 else 0
                elif op=='scale':x*=v[0];y*=v[1] if len(v)>1 else v[0]
                elif op=='matrix':x,y=v[0]*x+v[2]*y+v[4],v[1]*x+v[3]*y+v[5]
                else:
                    angle=math.radians(v[0]);cx,cy=v[1:3] if len(v)>1 else (0,0)
                    x,y=cx+(x-cx)*math.cos(angle)-(y-cy)*math.sin(angle),cy+(x-cx)*math.sin(angle)+(y-cy)*math.cos(angle)
            node=parents.get(node)
        return x,y
    for g in root.iter(NS+'g'):
        if not g.get('id', '').startswith('axes_'):continue
        patch = next((p for p in g if p.get('id', '').startswith('patch_')), None)
        if patch is not None:
            path_node = patch.find(NS+'path')
            if path_node is not None:
                points = re.findall(r'[ML]\s*([-\d.]+)\s+([-\d.]+)', path_node.get('d', ''))
                if points:
                    axes_top[g] = min(float(y) for x, y in points)
                    axes_bottom[g] = max(float(y) for x, y in points)
    groups = collections.defaultdict(list)
    counts = collections.Counter()
    for text in root.iter(NS+'text'):
        chain = [];node = text
        while node in parents:
            node = parents[node];chain.append(node)
        ids = [x.get('id', '') for x in chain]
        axis = next((x for x in chain if x.get('id', '').startswith('axes_')), None)
        style = text.get('style', '')
        match = re.search(r'font-size:\s*([\d.]+)', style)
        old = float(match[1]) if match else 10
        x,y = global_position(text)
        if any(x.startswith(('xtick_', 'ytick_')) for x in ids):target, role = 11, 'tick'
        elif any(x.startswith('matplotlib.axis_') for x in ids):target, role = 14, 'axis_label'
        elif any(x.startswith('legend_') for x in ids):target, role = 11, 'legend'
        elif axis is None:
            if 'rotate(-90' in text.get('transform',''):
                target,role=14,'axis_label'
            else:target, role = (18, 'title') if y < height*.15 else (11, 'footer')
        elif axis in axes_top and y < axes_top[axis]-1:
            target, role = 18, 'title'
        elif axis in axes_bottom and y > axes_bottom[axis]+1:
            target, role = 11, 'external_label'
        else:
            target, role = min(old,11), 'data_annotation'
        if match:style = re.sub(r'font-size:\s*[\d.]+(?:px|pt)?', f'font-size: {target}px', style)
        else:style += f';font-size:{target}px'
        if re.search(r'[\u3400-\u9fff]', ''.join(text.itertext())):
            style=re.sub(r'font-family:[^;]+', "font-family: 'Microsoft YaHei'",style)
        text.set('style', style)
        text.set('data-font-role', 'figure-title' if axis is None and role=='title' else role)
        counts[role] += 1
        group = next((x for x in chain if x.get('id', '').startswith('text_')), parents.get(text))
        groups[group].append((text, target, role))
        if axis is None and role in ('title','footer') and not list(text):
            text.set('x',str(width/2));text.set('y',str(y));text.attrib.pop('transform',None)
            text.set('style',re.sub(r';?\s*text-anchor:[^;]+','',text.get('style',''))+';text-anchor:middle')
    # Figure headings may be longer at 18pt; wrap without changing their words.
    from PIL import ImageFont
    import textwrap
    fonts={}
    for group in groups.values():
        extras=[]
        for text,size,role in group:
            if role!='title' or 'x' not in text.attrib or list(text):continue
            family='msyh.ttc' if re.search(r'[\u3400-\u9fff]',text.text or '') else 'arial.ttf'
            if family not in fonts:fonts[family]=ImageFont.truetype('C:/Windows/Fonts/'+family,100)
            line=text.text or '';extent=fonts[family].getlength(line)*size/100
            if extent <= width*.94:continue
            chars=max(10,int(len(line)*width*.94/extent)-2)
            lines=textwrap.wrap(line,width=chars,break_long_words=True,break_on_hyphens=False)
            text.text=lines[0]
            parent=parents[text]
            for i,line in enumerate(lines[1:],1):
                clone=ET.Element(NS+'text',text.attrib);clone.text=line
                clone.set('y',str(float(text.get('y'))+i*size*1.2));parent.append(clone)
                extras.append((clone,size,role))
        group.extend(extras)
    # Multiline SVG text uses separate baselines; enlarge line spacing too.
    for group in groups.values():
        lines = [(t, size, role) for t, size, role in group if 'y' in t.attrib and not list(t)]
        if len(lines) < 2:continue
        ys = sorted({float(t.get('y')) for t, _, _ in lines})
        target = max(size for _, size, _ in lines)
        if len(ys) < 2 or min(b-a for a,b in zip(ys,ys[1:])) >= target*1.15:continue
        # Keep headings above their plots, including newly wrapped lines.
        original_last=min(max(ys),height*.08) if all(t.get('data-font-role')=='figure-title' for t,_,_ in lines) else max(ys)
        first = original_last-(len(ys)-1)*target*1.2 if all(role == 'title' for _,_,role in lines) else ys[0]
        remap = {y: first+i*target*1.2 for i,y in enumerate(ys)}
        for t, _, _ in lines:
            old_y = float(t.get('y'));new_y = remap[old_y];t.set('y', str(new_y))
            transform = t.get('transform', '')
            if transform.startswith('rotate(-0 ') or transform.startswith('rotate(0 '):
                t.set('transform', re.sub(r'([-\d.]+)\)$', f'{new_y})', transform))
    pad = 32
    for text in root.iter(NS+'text'):
        if 'y' in text.attrib:
            pad = max(pad, -float(text.get('y').split()[0])+25)
    root.set('viewBox', f'{-pad} {-pad} {width+2*pad} {height+2*pad}')
    root.set('width', f'{width+2*pad}pt');root.set('height', f'{height+2*pad}pt')
    return ET.tostring(root, encoding='utf-8', xml_declaration=True), dict(counts)


def render_vector(source):
    source = Path(source)
    if source.is_relative_to(ROOT):
        # Native report SVGs are also destinations. Recover their untouched
        # original from the content-addressed backup on a resumed run.
        dc='{http://purl.org/dc/elements/1.1/}'
        tree=ET.fromstring(source.read_bytes());date=tree.find('.//'+dc+'date')
        if date is not None:
            for candidate in BACKUPS.glob('*.svg'):
                raw=ET.fromstring(candidate.read_bytes());d=raw.find('.//'+dc+'date')
                if d is not None and d.text==date.text and raw.get('viewBox','').startswith('0 0 '):
                    source=candidate;break
    cache = AUDIT/'vectors';cache.mkdir(exist_ok=True)
    raw_key = sha(source)
    alias = cache/(raw_key+'.v5.key')
    if alias.exists():
        stem = cache/alias.read_text().strip()
        if stem.with_suffix('.png').exists():return str(stem)
    content, roles = normalize_svg(source)
    key = hashlib.sha256(content).hexdigest();stem = cache/key
    if not stem.with_suffix('.png').exists():
        stem.with_suffix('.svg').write_bytes(content)
        cairosvg.svg2pdf(bytestring=content, write_to=str(stem.with_suffix('.pdf')), dpi=72, background_color='white')
        cairosvg.svg2png(bytestring=content, write_to=str(stem.with_suffix('.png')), dpi=300, background_color='white')
        from PIL import Image
        with Image.open(stem.with_suffix('.png')) as im:im.save(stem.with_suffix('.png'),dpi=(300,300))
    alias.write_text(key)
    return str(stem)


def vectors():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    taskfile = AUDIT/'vector_tasks.json'
    tasks = json.loads(taskfile.read_text(encoding='utf-8')) if taskfile.exists() else source_vectors()
    taskfile.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding='utf-8')
    done = {}
    if LOG.exists():
        for line in LOG.read_text(encoding='utf-8').splitlines():
            item=json.loads(line)
            if item['method']=='editable SVG typography' and item.get('layout_version')=='svg-cjk-v5':done[item['path']]=item
    grouped = collections.defaultdict(list)
    for target,source in tasks.items():
        if is_schematic(target):continue
        if target in done and Path(target).exists() and sha(target)==done[target]['sha256']:continue
        grouped[source].append(target)
    errors=[];n=len(done)
    print('SVG pending unique sources',len(grouped),'completed',n,flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(render_vector,source):source for source in grouped}
        for future in as_completed(futures):
            source=futures[future]
            try:
                stem=Path(future.result())
                for target in grouped[source]:
                    target=Path(target);before=backup(target)
                    for ext in ('png','svg','pdf'):
                        dest=target.with_suffix('.'+ext)
                        if ext!='png':backup(dest)
                        shutil.copy2(stem.with_suffix('.'+ext),dest)
                    record(target,'editable SVG typography',source,before);n+=1
            except Exception as exc:errors.append(dict(source=source,error=str(exc)))
            if n%25==0:print(f'vectors {n}/{len(tasks)} errors={len(errors)}',flush=True)
    (AUDIT/'vector_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
    print('vectors DONE',n,'errors',len(errors),flush=True)


if __name__ == '__main__':
    vectors()
