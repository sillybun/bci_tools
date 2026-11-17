"""Immutable, checksummed per-run artifacts shared by independently selected reports."""
from contextlib import contextmanager
from pathlib import Path
import inspect,json,os,re,shutil,tempfile,time
from . import storage


def dependencies(*functions_or_paths):
    import hashlib
    return [dict(name=getattr(x,'__qualname__',str(x)),sha256=(
        hashlib.sha256(inspect.getsource(x).encode()).hexdigest() if callable(x)
        else storage.file_digest(Path(x)))) for x in functions_or_paths]


def file_inputs(files):
    return {name:dict(path=str(Path(p).resolve()),sha256=storage.recording_sha(p)) for name,p in sorted(files.items())}


def link_file(source,target):
    """Reports reference immutable pool files through hard links on the same volume."""
    source,target=Path(source),Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if os.path.samefile(source,target):return 'hardlink'
        target.unlink()
    try:os.link(source,target);return 'hardlink'
    except OSError:shutil.copy2(source,target);return 'copy'


def atomic_json(path,value):
    """Replace metadata without modifying any hard-linked original cache file."""
    path=Path(path);handle,name=tempfile.mkstemp(prefix='.metadata_',dir=path.parent)
    os.close(handle);temp=Path(name)
    try:
        temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=True,default=storage.jsonable),encoding='utf-8')
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)


def run_label(run):
    """A readable date + run, with task/control when the archive provides it."""
    path=Path(str(run));names=[p.name for p in (path,*path.parents)]
    date=next((n for n in names if re.fullmatch(r'20\d{6}|20\d{2}-\d{2}-\d{2}',n)),None)
    if date and '-' not in date:date=f'{date[:4]}-{date[4:6]}-{date[6:]}'
    from .core.catalog import RUN_PREFIX
    match=next((m for n in names if (m:=RUN_PREFIX.match(n))),None)
    if match:
        number=match['number']+('-'+match['part'] if match['part'] else '')
        extra=f'_{match["task"]}-{match["mode"]}'
    else:
        old=next((m for n in names if (m:=re.search(r'(?:^|_)run-(\d{1,3})(?:-([a-z])(?=[_-]))?',n,re.I))),None)
        number=(f'{int(old[1]):03d}'+('-'+old[2] if old[2] else '')) if old else storage.slug(path.name,30)
        task=next((t for t in ('CO8','Grid') if any(t.lower() in n.lower() or (t=='CO8' and 'center_out' in n.lower()) for n in names)),'')
        extra='_'+task if task else ''
    return f'{date or "undated"}_run-{number}{extra}'


def artifact_group(kind):
    if kind.startswith('section_'):return kind.removeprefix('section_').upper(),'analysis'
    if kind.upper().startswith('S05'):return 'S05','depth_layout'
    if kind.upper().startswith('S14'):return 'S14','depth_layout'
    if kind.startswith('BGA'):return 'BGA','statistics' if kind.endswith('_data') else 'plots'
    if kind.startswith('BPS_TTT'):return 'BPS_TTT','statistics' if kind.endswith('_data') else 'plots'
    if kind=='grid_behavior':return 'S13','behavior'
    return storage.slug(kind.upper(),30),'analysis'


def artifact_location(root,kind,run,signature):
    section,group=artifact_group(kind)
    return Path(root)/section/run_label(run)/group/signature[:20]


def organize_artifact(folder):
    """Images and their numeric inputs stay adjacent inside each immutable version."""
    folder=Path(folder);mapping={}
    for path in list(folder.rglob('*')):
        if not path.is_file() or path.name=='entry.json':continue
        relative=path.relative_to(folder).as_posix();suffix=path.suffix.lower()
        if relative=='result.json' or suffix=='.md':target=relative
        elif relative=='data.json':target='data/calculation.json'
        elif relative.startswith('data/'):target=relative
        elif suffix in ('.png','.pdf','.svg'):
            target=relative if relative.startswith('figures/') else 'figures/'+relative
        else:target='data/'+relative
        mapping[relative]=target
    for old,new in mapping.items():
        if old==new:continue
        source,target=folder/old,folder/new
        assert source.resolve().is_relative_to(folder.resolve()) and target.resolve().is_relative_to(folder.resolve())
        target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.exists(),f'Artifact path collision: {target}'
        source.rename(target)
    def translate(value):
        if isinstance(value,str):return mapping.get(value,value)
        if isinstance(value,list):return [translate(v) for v in value]
        if isinstance(value,dict):return {mapping.get(k,k):translate(v) for k,v in value.items()}
        return value
    for path in folder.rglob('*.figure.json'):
        value=json.loads(path.read_text(encoding='utf-8'));updated=translate(value)
        if updated!=value:atomic_json(path,updated)
    record_path=folder/'result.json'
    if record_path.exists():
        record=translate(json.loads(record_path.read_text(encoding='utf-8')))
        record['digests']={relative:storage.file_digest(folder/relative) for relative in record['digests']}
        atomic_json(record_path,record)
    for path in sorted((p for p in folder.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
        if not any(path.iterdir()):path.rmdir()
    return mapping


class RunPool:
    def __init__(self,root,*,refresh=False):
        marker=Path(root).parent/'.individual_runs_migration.lock'
        while marker.exists():time.sleep(.1)
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.events=[];self._validated={};self.refresh=refresh

    @contextmanager
    def _lock(self,key):
        folder=self.root/'_meta/locks';folder.mkdir(parents=True,exist_ok=True)
        with (folder/(key+'.lock')).open('a+b') as stream:
            stream.seek(0,2)
            if not stream.tell():stream.write(b'0');stream.flush()
            stream.seek(0)
            if os.name=='nt':
                import msvcrt
                while True:
                    try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1);break
                    except OSError:time.sleep(.1)
                try:yield
                finally:stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(stream,fcntl.LOCK_EX)
                try:yield
                finally:fcntl.flock(stream,fcntl.LOCK_UN)

    def index_run(self,folder):
        """Readable image/data index; indexes are not part of immutable artifacts."""
        import html
        from urllib.parse import quote
        run=Path(folder).parent.parent
        with self._lock('index_'+storage.fingerprint(str(run))):
            variants=[]
            for path in sorted(run.glob('*/*/entry.json')):
                entry=json.loads(path.read_text(encoding='utf-8'));base=path.parent
                files=[Path(p).as_posix() for p in entry['digests']]
                variants.append(dict(kind=entry['kind'],signature=entry['signature'],folder=base.relative_to(run).as_posix(),
                    figures=[p for p in files if Path(p).suffix=='.png'],data=[p for p in files if p.startswith('data/')]))
            atomic_json(run/'index.json',dict(run=run.name,variants=variants))
            body=['<!doctype html><meta charset="utf-8"><title>'+html.escape(run.name)+'</title>',
                '<style>body{font:15px Microsoft YaHei,Arial;margin:28px;color:#17212b}img{max-width:320px;max-height:300px;object-fit:contain} .grid{display:flex;flex-wrap:wrap;gap:20px}.item{max-width:340px;overflow-wrap:anywhere}a{color:#1763a0}details{margin:20px 0}li{margin:5px}</style>',
                '<h1>'+html.escape(run.name)+'</h1><p>图片与生成数据按版本保存。PNG 可直接预览；CSV/JSON 为计算数据，entry.json 记录输入、参数和校验信息。</p>']
            for v in variants:
                prefix=v['folder']+'/'
                def link(p,label):return '<a href="'+quote(prefix+p,safe='/')+'">'+html.escape(label)+'</a>'
                body.append('<details open><summary><b>'+html.escape(v['kind'])+'</b> · '+v['signature'][:12]+'</summary><p>'+link('entry.json','输入和参数')+'</p>')
                body.append('<details><summary>生成数据</summary><ul>'+''.join('<li>'+link(p,p)+'</li>' for p in v['data'])+'</ul></details><div class="grid">')
                for p in v['figures']:
                    stem=Path(p).stem
                    matched=[d for d in v['data'] if Path(d).suffix=='.csv' and Path(d).stem.startswith(stem)]
                    if not matched:matched=[d for d in v['data'] if Path(d).suffix in ('.csv','.json') and not d.endswith('.figure.json')]
                    body.append('<div class="item">'+link(p,'查看原图')+'<br><a href="'+quote(prefix+p,safe='/')+'"><img loading="lazy" src="'+quote(prefix+p,safe='/')+'"></a><p>'+html.escape(stem)+'</p>'+''.join(link(d,'数据：'+Path(d).name)+'<br>' for d in matched)+'</div>')
                body.append('</div></details>')
            text='\n'.join(body)
            target=run/'index.html'
            if not target.exists() or target.read_text(encoding='utf-8')!=text:target.write_text(text,encoding='utf-8')

    def _valid(self,folder):
        try:
            manifest=json.loads((folder/'entry.json').read_text(encoding='utf-8'))
            if manifest['signature'][:20]!=folder.name:return False
            for relative,digest in manifest['digests'].items():
                path=folder/relative
                if not path.resolve().is_relative_to(folder.resolve()):return False
                stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns)
                if self._validated.get(key)!=digest:
                    if storage.file_digest(path)!=digest:return False
                    self._validated[key]=digest
            return bool(manifest['digests'])
        except (OSError,ValueError,KeyError):return False

    def lookup(self,kind,run,inputs):
        if self.refresh:return None
        identity=dict(format='report-run-pool-v1',kind=kind,run=str(run),inputs=storage.jsonable(inputs))
        signature=storage.fingerprint(identity)
        folder=artifact_location(self.root,kind,run,signature)
        return folder if self._valid(folder) else None

    def get(self,kind,run,inputs,build):
        identity=dict(format='report-run-pool-v1',kind=kind,run=str(run),inputs=storage.jsonable(inputs))
        signature=storage.fingerprint(identity)
        folder=artifact_location(self.root,kind,run,signature);parent=folder.parent
        with self._lock(signature):
            hit=not self.refresh and self._valid(folder)
            if not hit:
                parent.mkdir(parents=True,exist_ok=True)
                temp=Path(tempfile.mkdtemp(prefix='.building_',dir=parent))
                try:
                    data=build(temp)
                    # Python cache preserves NaN; it must never turn missing observations into zero.
                    (temp/'data.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=True,default=storage.jsonable),encoding='utf-8')
                    organize_artifact(temp)
                    digests={str(p.relative_to(temp)):storage.file_digest(p) for p in temp.rglob('*') if p.is_file()}
                    (temp/'entry.json').write_text(json.dumps(dict(identity,signature=signature,digests=digests),ensure_ascii=False,indent=2),encoding='utf-8')
                    if folder.exists():
                        assert folder.resolve().is_relative_to(self.root)
                        shutil.rmtree(folder)
                    os.rename(temp,folder)
                finally:
                    if temp.exists():shutil.rmtree(temp)
            data=json.loads((folder/'data/calculation.json').read_text(encoding='utf-8'))
        self.events.append(dict(kind=kind,run=str(run),signature=signature,folder=str(folder),cached=hit))
        self.index_run(folder)
        return data,folder,hit

    def publish(self,folder,target,extensions=None):
        manifest=json.loads((Path(folder)/'entry.json').read_text(encoding='utf-8'));paths=[]
        for relative in manifest['digests']:
            if relative.replace('\\','/')=='data/calculation.json':continue
            if extensions and Path(relative).suffix not in extensions:continue
            parts=Path(relative).parts
            published=Path(*parts[1:]) if parts[0] in ('figures','data') else Path(relative)
            path=Path(target)/published
            mode=link_file(Path(folder)/relative,path);paths.append(path)
            if mode=='copy':self.events.append(dict(kind='copy_fallback',target=str(path)))
        return paths

    def section(self,entry):
        record=entry['record'];source=Path(entry['folder']);run=entry['job']['recordings'][0]
        def build(folder):
            for relative in [*record['digests'],'result.json']:
                if relative in record['digests'] and storage.file_digest(source/relative)!=record['digests'][relative]:
                    raise ValueError(f'Cached section file failed integrity check: {source/relative}')
                link_file(source/relative,folder/relative)
            return dict(original_folder=str(source),section=record['section'],signature=record['signature'])
        _,folder,hit=self.get('section_'+entry['job']['section'],run,dict(signature=record['signature']),build)
        return folder

    def manifest(self):
        return dict(root=str(self.root),hits=sum(bool(e.get('cached')) for e in self.events),
                    misses=sum(e.get('cached') is False for e in self.events),entries=self.events,
                    storage='Immutable signature folders; reports hard-link pool artifacts on the same volume, copy only when hard links are unavailable.')
