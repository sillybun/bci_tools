"""Replace the redundant lfp-only filename label with LFP in the active archive."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis_gui.core.catalog import describe
from data_analysis_gui.core.xdf_split import digest

ROOT = Path('E:/AitBCI/data/keaiduo').resolve()
OLD = ROOT / '20261111' / 'run-001-CO8-offline-manual-lfp-only'
NEW = OLD.with_name(OLD.name.replace('lfp-only', 'LFP'))


def main():
    if not OLD.is_dir() or NEW.exists() or OLD.parent != ROOT / '20261111' or NEW.parent != OLD.parent:
        raise ValueError('Unexpected source or destination')
    token = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    audit = ROOT / f'lfp_label_audit_{token}'
    audit.mkdir()
    mapping = {OLD.relative_to(ROOT).as_posix(): NEW.relative_to(ROOT).as_posix(), OLD.name: NEW.name}
    files = list((OLD / 'xdf').glob('*.xdf'))
    hashes = {p.name.replace('lfp-only', 'LFP'): digest(p) for p in files}
    for p in files:
        mapping[p.name] = p.name.replace('lfp-only', 'LFP')
    backups, moves = [], []

    def replace(text):
        for old, new in sorted(mapping.items(), key=lambda x:len(x[0]), reverse=True):
            text = text.replace(old, new).replace(old.replace('/', '\\'), new.replace('/', '\\'))
        return text

    try:
        for file in files:
            target = file.with_name(file.name.replace('lfp-only', 'LFP'))
            file.rename(target)
            moves.append((file, target))
        OLD.rename(NEW)
        moves.append((OLD, NEW))
        edited = [NEW / x for x in ('run_info.json','session_info.json','实验内容说明.txt','实验设计说明.txt')]
        edited += [ROOT / '20261111' / x for x in ('session_index.json','file_manifest.json','整理说明.txt',
                                                   '实验内容说明.txt','实验设计说明.txt','会话与参数变化说明.txt')]
        edited.append(ROOT / '归档命名与目录说明.txt')
        prefs = Path(__file__).resolve().parents[1] / '.local' / 'preferences.json'
        if prefs.exists():
            edited.append(prefs)
        for file in edited:
            relative = file.relative_to(ROOT) if file.is_relative_to(ROOT) else Path('gui_preferences.json')
            saved = audit / 'metadata_before' / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, saved)
            backups.append((file, saved))
            text = replace(file.read_text(encoding='utf-8-sig'))
            if file.suffix == '.json':
                json.loads(text)
            file.write_text(text, encoding='utf-8')
        for filename, sha in hashes.items():
            file = NEW / 'xdf' / filename
            if digest(file) != sha:
                raise ValueError('XDF content changed')
            r = describe(file, ROOT)
            assert r.date == '2026-11-11' and r.run == '001'
            assert r.reason.startswith('缺少数据流：')  # LFP remains excluded from spike-only sections.
        meta = json.loads((NEW / 'session_info.json').read_text(encoding='utf-8'))
        assert all((ROOT / row['path']).is_file() for row in meta['recordings'])
        (audit / 'completion.json').write_text(json.dumps(dict(status='verified',old=str(OLD),new=str(NEW),
            xdf_sha256=hashes, data_content_unchanged=True),ensure_ascii=False,indent=2),encoding='utf-8')
        print('VERIFIED',NEW,flush=True)
    except Exception:
        for file, saved in reversed(backups):
            shutil.copy2(saved,file)
        for old,new in reversed(moves):
            new.rename(old)
        raise


if __name__=='__main__':
    main()
