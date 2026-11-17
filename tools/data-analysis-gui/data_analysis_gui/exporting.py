"""Collect cached results into a batch folder for run or figure comparisons."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from . import storage

EXPORT_LAYOUTS = {'按 run 归类': 'run', '按分析图归类': 'figure'}


def export_batch(output_root, layout):
    if layout not in EXPORT_LAYOUTS.values():
        raise ValueError(f'未知导出方式：{layout}')
    name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6]
    return dict(root=str(Path(output_root) / 'exports' / name), layout=layout)


def export_result(folder, record, recordings, batch):
    """Copy one completed result, with stable names across the batch's workers."""
    folder, root = Path(folder), Path(batch['root'])
    layout = batch['layout']
    if layout not in EXPORT_LAYOUTS.values():
        raise ValueError(f'未知导出方式：{layout}')
    sid = record['section']['id'].upper()
    labels = '_'.join(f'{r.date}_{r.data_type}_run-{r.run}' for r in recordings)
    selection = storage.fingerprint([r.key for r in recordings])[:8]
    run_name = storage.slug(labels, 45) + '_' + selection
    section_name = sid + '_' + record['section']['version']
    base = root / section_name if layout == 'figure' else root / run_name / section_name
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.export_', dir=root))
    manifest = dict(section=record['section'], recordings=[str(r.path) for r in recordings],
                    source=str(folder), signature=record['signature'], layout=layout, files=[])
    try:
        def copy(relative, target):
            source = (folder / relative).resolve()
            if not source.is_relative_to(folder.resolve()):
                raise ValueError(f'结果文件路径越界：{relative}')
            staged = work / target.relative_to(root)
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged)
            manifest['files'].append(dict(source=relative, exported=target.relative_to(root).as_posix()))

        def filename(relative, destination):
            original = Path(relative)
            budget = max(12, 235 - len(str(destination.resolve())) - len(run_name) - len(original.suffix) - 10)
            stem = storage.slug(original.stem, min(100, budget))
            return f'{run_name}_{stem}_{storage.fingerprint(relative)[:6]}{original.suffix}'

        for figure in record['figures']:
            title = figure['title'].removeprefix('Supplementary ')
            category = storage.slug(title, 50) + '_' + storage.fingerprint(title)[:6]
            destination = base / category if layout == 'figure' else base / 'figures'
            for kind, relative in figure['files'].items():
                if kind != 'json':
                    copy(relative, destination / filename(relative, destination))
            for relative in figure.get('data_files', {}).values():
                copy(relative, destination / filename(relative, destination))
        for relative in record.get('tables', {}).values():
            copy(relative, base / '_tables' / run_name / Path(relative).name)
        for relative in record.get('attachments', {}).values():
            copy(relative, base / '_attachments' / run_name / Path(relative).name)
        provenance = base / '_sources' / f'{run_name}.json'
        staged_manifest = work / provenance.relative_to(root)
        staged_manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.update(methods=[dict(title=f['title'], method=f.get('method', ''))
                                 for f in record['figures']], notes=record.get('notes', []))
        staged_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        # Each file is published atomically. Publish provenance last as completion marker.
        for staged in sorted(work.rglob('*')):
            if staged.is_file() and staged != staged_manifest:
                target = root / staged.relative_to(work)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, target)
        provenance.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_manifest, provenance)
    finally:
        shutil.rmtree(work)
    return root
