"""Reorganize the existing shared pool and update references without redrawing plots."""
from pathlib import Path
import json,shutil,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_analysis_gui import storage
from data_analysis_gui.report_run_pool import RunPool,artifact_location,organize_artifact,atomic_json

REPORT=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report').resolve()
OLD=REPORT/'_run_pool';NEW=REPORT/'individual_runs'
AUDIT=REPORT.parent/'import_20261120/individual_runs_migration'

def main():
    assert OLD.is_dir() and not NEW.exists(),'Expected the original pool and no existing destination'
    assert OLD.resolve().parent==REPORT and NEW.resolve().parent==REPORT
    entries=[]
    for path in sorted(OLD.glob('*/*/*/entry.json')):
        m=json.loads(path.read_text(encoding='utf-8'));target=artifact_location(NEW,m['kind'],m['run'],m['signature'])
        assert target.resolve().is_relative_to(NEW)
        entries.append(dict(old=path.parent,new=target,manifest=m))
    assert entries and len({str(e['new']) for e in entries})==len(entries)
    assert not list(OLD.rglob('.building_*')),'An artifact is currently being generated'
    AUDIT.mkdir(parents=True,exist_ok=True)
    assert not (AUDIT/'complete.json').exists(),'Migration already completed'
    report_json=list(REPORT.glob('*/report.json'))
    preview_json=list((REPORT.parent/'previews').glob('**/verification.json'))
    with zipfile.ZipFile(AUDIT/'metadata_before.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OLD.rglob('*.json'):z.write(p,'pool/'+p.relative_to(OLD).as_posix())
        for p in report_json:z.write(p,'reports/'+p.relative_to(REPORT).as_posix())
        for p in preview_json:z.write(p,'previews/'+p.relative_to(REPORT.parent/'previews').as_posix())
    (AUDIT/'plan.json').write_text(json.dumps([dict(old=str(e['old']),new=str(e['new'])) for e in entries],ensure_ascii=False,indent=2),encoding='utf-8')
    pool=RunPool(OLD)
    marker=REPORT/'.individual_runs_migration.lock';marker.write_text('Reorganizing existing cache; no computation.',encoding='utf-8')
    pairs=[];checked=0
    try:
        try:
            OLD.rename(NEW)
            moved_root=True
        except PermissionError:
            # Explorer may hold the root open; move verified artifact folders instead.
            NEW.mkdir()
            moved_root=False
        pool.root=NEW
        for e in entries:
            current=(NEW/e['old'].relative_to(OLD)) if moved_root else e['old']
            target=e['new'];target.parent.mkdir(parents=True,exist_ok=True)
            assert current.resolve().is_relative_to(NEW if moved_root else OLD) and target.resolve().is_relative_to(NEW)
            current.rename(target)
            mapping=organize_artifact(target)
            pairs.append((str(e['old']),str(target)))
            for old,new in mapping.items():pairs.append((str(e['old']/old),str(target/new)))
            for relative,digest in e['manifest']['digests'].items():
                old=Path(relative).as_posix();new=mapping.get(old,old)
                if Path(old).suffix!='.json':
                    assert storage.file_digest(target/new)==digest,f'Existing artifact changed: {target/new}'
                    checked+=1
        meta=NEW/'_meta';meta.mkdir(exist_ok=True)
        legacy=NEW if moved_root else OLD
        if (legacy/'_locks').exists():
            (legacy/'_locks').rename(meta/'locks')
            pairs.append((str(OLD/'_locks'),str(meta/'locks')))
        if (legacy/'seed_history.json').exists():
            (legacy/'seed_history.json').rename(meta/'seed_history.json')
            pairs.append((str(OLD/'seed_history.json'),str(meta/'seed_history.json')))
        pairs.append((str(OLD),str(NEW)))
        variants={}
        for old,new in pairs:
            variants[old]=new;variants[old.replace('\\','/')]=new.replace('\\','/')
        replacements=sorted(variants.items(),key=lambda x:len(x[0]),reverse=True)
        old_prefixes=(str(OLD),str(OLD).replace('\\','/'))
        def rewrite(value):
            if isinstance(value,str):
                if not value.startswith(old_prefixes):return value
                for old,new in replacements:
                    if value==old or value.startswith(old+'\\') or value.startswith(old+'/'):return new+value[len(old):]
                return value
            if isinstance(value,list):return [rewrite(v) for v in value]
            if isinstance(value,dict):return {k:rewrite(v) for k,v in value.items()}
            return value
        for p in [*NEW.rglob('*.json'),*report_json,*preview_json]:
            if p.name=='entry.json':continue
            before=json.loads(p.read_text(encoding='utf-8'));after=rewrite(before)
            # NaN is preserved; JSON serialization provides a stable comparison.
            if json.dumps(before,sort_keys=True)!=json.dumps(after,sort_keys=True):atomic_json(p,after)
        for e in entries:
            target=e['new'];entry=rewrite(e['manifest'])
            record_path=target/'result.json'
            if record_path.exists():
                record=json.loads(record_path.read_text(encoding='utf-8'))
                record['digests']={relative:storage.file_digest(target/relative) for relative in record['digests']}
                atomic_json(record_path,record)
            identity={k:entry[k] for k in ('format','kind','run','inputs')}
            assert storage.fingerprint(identity)==entry['signature'],'Unexpected change to analysis inputs'
            entry['digests']={p.relative_to(target).as_posix():storage.file_digest(p) for p in target.rglob('*') if p.is_file() and p.name!='entry.json'}
            atomic_json(target/'entry.json',entry)
            assert pool._valid(target),str(target)
            assert pool.lookup(entry['kind'],entry['run'],entry['inputs'])==target
            pool.index_run(target)
        # Remove only empty legacy containers after all artifacts are verified.
        for p in sorted((p for p in NEW.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
            if not any(p.iterdir()):p.rmdir()
        if OLD.exists():
            for p in sorted((p for p in OLD.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):
                if not any(p.iterdir()):p.rmdir()
            assert not any(OLD.iterdir()),'Legacy pool contains unexpected files'
            OLD.rmdir()
        result=dict(root=str(NEW),entries=len(entries),unchanged_artifact_files=checked,
                    reports_updated=len(report_json),sections=sorted(p.name for p in NEW.iterdir() if p.is_dir()),
                    recomputed_runs=0)
        atomic_json(AUDIT/'complete.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:marker.unlink(missing_ok=True)

if __name__=='__main__':main()
