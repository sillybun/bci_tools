"""Cache-first result storage: search stored results first, compute only when missing.

Layout under the output root:

    single/<recording-slug>/<sid>_<ver>/<signature12>/
    cross/<n>rec_<recordings-hash8>/<sid>_<ver>/<signature12>/
        result.json                 # signature inputs, summary, notes, file digests
        figures/SID_NN_<title>.png|.svg|.pdf
        figures/SID_NN_<title>.csv      # plotted values + calculation method (# header)
        figures/SID_NN_<title>__<name>.csv   # explicit per-figure data tables
        figures/SID_NN_<title>.figure.json   # method, provenance, digests
        tables/<name>.csv

The signature covers: recording SHA-256s, section id/version, the source code of
the section folder, data_analysis_gui/core and data_analysis_gui/decoders, trial-filter criteria, section
parameters, layout workbook and configs.ini contents, and the export format
version. A stored result is used only if result.json matches and every listed file
still has its recorded digest; otherwise it is recomputed.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import threading

import numpy as np

from . import config as C
from .core.style import DEFAULT_FONT_PERCENT

FORMAT_VERSION = 'data-analysis-gui-result-v2'
FIGURE_FORMATS = ('png', 'svg', 'pdf')
PNG_DPI = 300
PACKAGE = Path(__file__).resolve().parent
_SHA_LOCK = threading.Lock()
SHA_CACHE = C.LOCAL_DIR / 'sha256_cache.json'


# ------------------------------------------------------------------ hashing
def file_digest(path, chunk=8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def recording_sha(path) -> str:
    """SHA-256 of a (large) recording, memoised by path + size + mtime."""
    path = Path(path).resolve()
    stat = path.stat()
    key = f'{path}|{stat.st_size}|{stat.st_mtime_ns}'
    with _SHA_LOCK:
        try:
            cache = json.loads(SHA_CACHE.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            cache = {}
        if key in cache:
            return cache[key]
    value = file_digest(path)
    with _SHA_LOCK:
        try:
            cache = json.loads(SHA_CACHE.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            cache = {}
        cache[key] = value
        SHA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        # Reports and cache importers can hash files in separate processes.
        # A shared .tmp name races on Windows; each writer owns its temp file.
        import tempfile,time
        handle,name=tempfile.mkstemp(prefix='sha_cache_',suffix='.tmp',dir=SHA_CACHE.parent)
        os.close(handle);tmp=Path(name)
        try:
            tmp.write_text(json.dumps(cache,indent=1),encoding='utf-8')
            for attempt in range(20):
                try:tmp.replace(SHA_CACHE);break
                except PermissionError:
                    if attempt==19:break  # Hash is valid even if the memo file is busy.
                    time.sleep(.025)
        finally:tmp.unlink(missing_ok=True)
    return value


def optional_digest(path) -> str:
    try:
        return file_digest(path) if path and Path(path).is_file() else ''
    except OSError:
        return ''


_SOURCE_CACHE: dict = {}


def source_digest(section_folder: Path) -> str:
    """Hash of the section folder plus shared code (core, decoders)."""
    folders = [Path(section_folder), PACKAGE / 'core', PACKAGE / 'decoders', PACKAGE / 'sections' / 'base.py']
    h = hashlib.sha256()
    for folder in folders:
        files = [folder] if folder.is_file() else sorted(folder.rglob('*.py')) if folder.exists() else []
        for f in files:
            stat = f.stat()
            key = (str(f), stat.st_mtime_ns, stat.st_size)
            if key not in _SOURCE_CACHE:
                _SOURCE_CACHE[key] = hashlib.sha256(f.read_bytes()).hexdigest()
            h.update(f.relative_to(PACKAGE).as_posix().encode())
            h.update(_SOURCE_CACHE[key].encode())
    return h.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def slug(text: str, limit=60) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text)).strip('_.')[:limit] or 'x'


# ------------------------------------------------------------------ signatures
def section_folder(section) -> Path:
    import importlib
    module = importlib.import_module(type(section).__module__)
    return Path(module.__file__).resolve().parent


def uses_channel_config(section):
    return section.mode == 'single' and (
        'channel_config' in section.requires or section.id in ('s02', 's07', 's12'))


def signature_inputs(section, recordings, criteria, params, layout_path='', *, channel_config_path=None,
                     font_percent=DEFAULT_FONT_PERCENT) -> dict:
    """Everything that determines a section result, including its effective config."""
    from .core.archive_names import SHORT_XDF
    from .core.catalog import day_config_path
    from .core.session import recording_window_policy
    config_path = ''
    if uses_channel_config(section) and recordings:
        config_path = (day_config_path(recordings[0].path) if channel_config_path is None
                       else str(channel_config_path))
    inputs = dict(format=FORMAT_VERSION, section=section.id, version=section.version, mode=section.mode,
                font_policy=dict(title_pt=18, axis_label_pt=14, tick_pt=11, schematics='preserve legacy style'),
                code=source_digest(section_folder(section)),
                recordings=[dict(name=r.path.name, sha256=recording_sha(r.path)) for r in recordings],
                criteria=criteria.to_dict(), params={k: params[k] for k in sorted(params)},
                font_percent=font_percent,
                layout=optional_digest(layout_path) or 'index-128x8',
                channel_config=dict(path=config_path, sha256=optional_digest(config_path)) if config_path else '')
    for source, recording in zip(inputs['recordings'], recordings):
        if SHORT_XDF.fullmatch(recording.path.name):
            source['relative_path'] = recording.relative
        if run_name := getattr(recording, 'run_name', ''):
            source['display_run_name'] = run_name
        if getattr(recording, 'condition', ''):
            source['condition'] = recording.condition
        window = recording_window_policy(recording)
        if window is not None:
            source['analysis_window_policy'] = window
    dependencies = getattr(section, 'cache_code_dependencies', ())
    if section.mode == 'cross':
        dependencies = (*dependencies, 'sections/session_overview.py')
    if dependencies:
        inputs['code_dependencies'] = {
            relative: source_digest(PACKAGE / relative) for relative in dependencies
        }
    if section.source_sections:
        from .sections import versions
        inputs['source_sections'] = {}
        resolved = section.resolve_params(params)['_source_sections']
        for sid, source in resolved.items():
            dependency = next(s for s in versions(sid) if s.version == source['version'])
            inputs['source_sections'][sid] = dict(version=dependency.version,
                                                  code=source_digest(section_folder(dependency)),
                                                  params=source['params'])
    if section.id == 's11' and recordings:
        from .sections.s11_online_vs_offline_v1.model_replay import signature_dependencies
        inputs['online_model_artifacts'] = signature_dependencies(recordings[0], params)
    if section.id == 's14':
        depth_path = section.resolve_params(params)['depth_csv']
        inputs['channel_depth_csv'] = dict(path=depth_path, sha256=optional_digest(depth_path))
    if section.id == 's15' and recordings:
        from .sections.s15_task_replay_v1.replay import source_files
        inputs['replay_sources'] = {key: dict(path=str(path), sha256=optional_digest(path))
                                   for key, path in source_files(recordings[0]).items()}
        inputs['replay_template'] = optional_digest(section_folder(section) / 'player.html')
    if section.id == 'x01':
        from .sections.x01_channel_stability_v1.section import config_candidates
        inputs['recording_configs'] = {path: optional_digest(path)
                                       for dated in config_candidates(recordings).values() for _, path in dated}
        inputs['chip_layout_reference'] = dict(path=str(C.BGA_SHANK_PROJECTION),
                                               sha256=optional_digest(C.BGA_SHANK_PROJECTION))
    if section.id in ('s13', 'x08'):
        from .core.behavior import source_files
        inputs['behavior_sources'] = [
            {key: dict(path=str(path), sha256=optional_digest(path))
             for key, path in source_files(recording).items()}
            for recording in recordings if recording.online
        ]
    if hasattr(section, 'extra_signature_inputs'):
        inputs['extra_sources'] = section.extra_signature_inputs(recordings, params)
    return inputs


def result_folder(root, section, recordings, signature: str) -> Path:
    root = Path(root)
    if section.mode == 'single':
        r = recordings[0]
        base = root / 'single' / slug(f'{r.date}_{r.data_type}_run-{r.run}_{r.path.stem}', 90)
    else:
        from .core.archive_names import SHORT_XDF
        names = sorted(r.relative if SHORT_XDF.fullmatch(r.path.name) else r.path.name for r in recordings)
        base = root / 'cross' / f'{len(recordings)}rec_{fingerprint(names)[:8]}'
    return base / f'{section.id}_{section.version}' / signature[:12]


# ------------------------------------------------------------------ lookup
def load_cached(folder: Path, signature: str) -> dict | None:
    """Stored result record if it matches the signature and all files are intact."""
    try:
        record = json.loads((Path(folder) / 'result.json').read_text(encoding='utf-8'))
        if record.get('signature') != signature or record.get('format') != FORMAT_VERSION:
            return None
        for rel, digest in record['digests'].items():
            if file_digest(Path(folder) / rel) != digest:
                return None
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


def exists_cached(folder: Path, signature: str) -> bool:
    """Cheap check for list status (no digest verification)."""
    try:
        record = json.loads((Path(folder) / 'result.json').read_text(encoding='utf-8'))
        return record.get('signature') == signature
    except (OSError, ValueError):
        return False


def load_cached_summary(folder: Path, signature: str) -> dict | None:
    """Read metrics from the signed result metadata without opening its figure files."""
    try:
        record = json.loads((Path(folder) / 'result.json').read_text(encoding='utf-8'))
        if record['signature'] == signature and record['format'] == FORMAT_VERSION:
            summary = record['summary']
            return summary if isinstance(summary, dict) else None
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


# ------------------------------------------------------------------ figure data
def _label(text):
    return '' if text is None else str(text)


def extract_plot_data(fig) -> list[dict]:
    """Long-format table of values drawn by lines, scatter, bars, images and meshes."""
    from matplotlib.collections import LineCollection, PathCollection, PolyCollection, QuadMesh
    from matplotlib.container import BarContainer
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse
    rows = []
    for p, ax in enumerate(fig.axes):
        if getattr(ax, '_colorbar', None) is not None or ax.get_label() == '<colorbar>':
            continue
        base = dict(panel=p, panel_title=_label(ax.get_title() or ax.get_title('left')),
                    x_label=_label(ax.get_xlabel()), y_label=_label(ax.get_ylabel()))
        for k, patch in enumerate(ax.patches):
            if isinstance(patch, Ellipse):
                rows.append(dict(base, artist='ellipse', series=_label(patch.get_gid() or patch.get_label()),
                                 item=k, x=patch.center[0], y=patch.center[1],
                                 extra=f'width={patch.width:.10g};height={patch.height:.10g};angle_deg={patch.angle:.10g}'))
        for k, line in enumerate(ax.get_lines()):
            if not isinstance(line, Line2D):
                continue
            x, y = np.asarray(line.get_xdata(), dtype=object), np.asarray(line.get_ydata(), dtype=object)
            if len(x) <= 2 and line.get_label().startswith('_child') and (
                    np.all(x == x[0]) or np.all(y == y[0]) if len(x) else True):
                kind = 'reference_line'
            else:
                kind = 'line'
            for i, (xv, yv) in enumerate(zip(x, y)):
                rows.append(dict(base, artist=kind, series=_label(line.get_label()), item=k, index=i, x=xv, y=yv))
        bar_patches = set()
        for k, container in enumerate(ax.containers):
            if isinstance(container, BarContainer):
                label = _label(container.get_label())
                for i, rect in enumerate(container.patches):
                    bar_patches.add(id(rect))
                    rows.append(dict(base, artist='bar', series=label, item=k, index=i,
                                     x=rect.get_x() + rect.get_width() / 2, y=rect.get_y() + rect.get_height(),
                                     value=rect.get_height(), extra=f'width={rect.get_width():.6g};bottom={rect.get_y():.6g}'))
        for k, coll in enumerate(ax.collections):
            label = _label(coll.get_label())
            if isinstance(coll, PathCollection):
                offsets = np.asarray(coll.get_offsets())
                values = coll.get_array()
                sizes = coll.get_sizes()
                for i, (xv, yv) in enumerate(offsets):
                    row = dict(base, artist='scatter', series=label, item=k, index=i, x=xv, y=yv)
                    if values is not None and len(values) == len(offsets):
                        row['value'] = values[i]
                    if len(sizes) == len(offsets) and len(sizes) > 1:
                        row['extra'] = f'size={sizes[i]:.4g}'
                    rows.append(row)
            elif isinstance(coll, QuadMesh):
                arr = np.ma.filled(np.ma.asarray(coll.get_array(), dtype=float), np.nan)
                arr = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr[None, :]
                for (i, j), v in np.ndenumerate(arr):
                    rows.append(dict(base, artist='mesh', series=label, item=k, index=i * arr.shape[1] + j,
                                     x=j, y=i, value=v))
            elif isinstance(coll, LineCollection):
                for s, seg in enumerate(coll.get_segments()):
                    for i, (xv, yv) in enumerate(seg):
                        rows.append(dict(base, artist='segment', series=label, item=k, index=i, x=xv, y=yv,
                                         extra=f'segment={s}'))
            elif isinstance(coll, PolyCollection):
                for s, path in enumerate(coll.get_paths()):
                    for i, (xv, yv) in enumerate(path.vertices):
                        rows.append(dict(base, artist='band', series=label, item=k, index=i, x=xv, y=yv,
                                         extra=f'polygon={s}'))
        for k, image in enumerate(ax.get_images()):
            if not isinstance(image, AxesImage):
                continue
            arr = np.ma.asarray(image.get_array())
            if arr.ndim == 3:   # RGB(A) image: values are colours, not data
                continue
            arr = np.ma.filled(arr.astype(float), np.nan)
            extent = image.get_extent()
            for (i, j), v in np.ndenumerate(arr):
                rows.append(dict(base, artist='image', series=_label(image.get_label()), item=k,
                                 index=i * arr.shape[1] + j, x=j, y=i, value=v,
                                 extra=f'row={i};col={j};extent={",".join(f"{e:.6g}" for e in extent)}'))
    return rows


def _table_rows(table) -> list[dict]:
    if isinstance(table, dict):
        columns = {k: np.asarray(v).ravel() if np.ndim(v) else np.asarray([v]) for k, v in table.items()}
        n = max((len(v) for v in columns.values()), default=0)
        return [{k: (v[i] if i < len(v) else '') for k, v in columns.items()} for i in range(n)]
    return list(table or [])


def _cell(v):
    v = jsonable(v)
    if isinstance(v, float):
        return f'{v:.10g}'
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return '' if v is None else v


def write_csv(path: Path, rows: list[dict], header_lines: list[str] = ()):
    columns = list(dict.fromkeys(k for r in rows for k in r)) or ['empty']
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        for line in header_lines:
            for part in str(line).splitlines() or ['']:
                f.write(f'# {part}\n')
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _cell(r.get(k, '')) for k in columns})


# ------------------------------------------------------------------ save
def save_result(result, folder: Path, signature: str, inputs: dict, section, recordings,
                seconds: float = 0.0) -> dict:
    """Write figures (png/svg/pdf + csv + figure.json), tables and result.json atomically."""
    folder = Path(folder)
    from .sections.result_summaries import result_sentence
    result.summary['one_sentence_summary'] = result_sentence(section.id, result.summary)
    tmp = folder.with_name(folder.name + '.partial')
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    (tmp / 'figures').mkdir(parents=True)
    (tmp / 'tables').mkdir()
    provenance = [f'section: {section.id.upper()} {section.version} · {section.title}',
                  f'section description: {section.description}',
                  'recordings: ' + '; '.join(f'{r.label} [{r.path.name}] sha256={x["sha256"]}'
                                             for r, x in zip(recordings, inputs['recordings'])),
                  'trial filter: ' + json.dumps(inputs['criteria'], ensure_ascii=False),
                  'parameters: ' + json.dumps(jsonable(inputs['params']), ensure_ascii=False),
                  f'code signature: {inputs["code"][:16]}  result signature: {signature[:16]}',
                  f'created: {_dt.datetime.now().isoformat(timespec="seconds")}']
    figures = []
    section_prefix = section.id.upper()
    methods = list(getattr(result, 'methods', [])) + [''] * len(result.figures)
    explicit = list(getattr(result, 'figure_data', [])) + [{}] * len(result.figures)
    for i, (title, fig) in enumerate(result.figures, 1):
        # Keep the display-only supplementary label out of Windows path budgets.
        stem = f'{section_prefix}_{i:02d}_{slug(title.removeprefix("Supplementary "), 49 - len(section_prefix))}'
        files = {}
        for ext in FIGURE_FORMATS:
            path = tmp / 'figures' / f'{stem}.{ext}'
            reference = getattr(fig, '_reference_image_path', None)
            if ext == 'png' and reference:
                import shutil
                shutil.copy2(reference, path)
            else:
                fig.savefig(path, dpi=PNG_DPI if ext == 'png' else None,
                            metadata={'Title': title} if ext in ('pdf', 'svg') else None)
            files[ext] = f'figures/{path.name}'
        method = methods[i - 1] or ('Values extracted from the drawn artists; see section description. '
                                    'No explicit method text was provided for this figure.')
        header = [f'figure: {title}', 'method:', *method.splitlines(), *provenance,
                  'columns: panel/panel_title = subplot; artist = line|scatter|bar|image|mesh|segment|band; '
                  'x,y = plotted coordinates (image/mesh: column,row); value = colour/bar value']
        path = tmp / 'figures' / f'{stem}.csv'
        # A supplied reference image has no measured pixel table to export.
        write_csv(path, [] if getattr(fig, '_reference_image_path', None) else extract_plot_data(fig), header)
        files['csv'] = f'figures/{path.name}'
        data_files = {}
        for name, table in (explicit[i - 1] or {}).items():
            path = tmp / 'figures' / f'{stem}__{slug(name, 40)}.csv'
            write_csv(path, _table_rows(table), [f'figure: {title}', f'data: {name}', 'method:', *method.splitlines(),
                                                 *provenance])
            data_files[name] = f'figures/{path.name}'
        meta = dict(title=title, method=method, files=files, data_files=data_files, provenance=provenance)
        path = tmp / 'figures' / f'{stem}.figure.json'
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        files['json'] = f'figures/{path.name}'
        figures.append(dict(title=title, method=method, files=files, data_files=data_files))
    # Channel atlases retain per-page exports and also provide one PDF per set.
    atlas_series = {}
    for index, (_, fig) in enumerate(result.figures):
        name = getattr(fig, '_series_pdf_name', None)
        if name:
            atlas_series.setdefault(name, []).append(index)
    if atlas_series:
        from matplotlib.backends.backend_pdf import PdfPages
        for name, indices in atlas_series.items():
            relative = f'figures/{section_prefix}_atlas_{slug(name, 59 - len(section_prefix))}.pdf'
            with PdfPages(tmp / relative, metadata={'Title': name}) as pdf:
                for index in indices:
                    pdf.savefig(result.figures[index][1])
            for index in indices:
                figures[index]['files']['series_pdf'] = relative
                meta_path = tmp / figures[index]['files']['json']
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                meta['files']['series_pdf'] = relative
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    tables = {}
    for name, rows in result.tables.items():
        if rows:
            path = tmp / 'tables' / f'{slug(name, 60)}.csv'
            write_csv(path, rows, [f'table: {name}', *provenance])
            tables[name] = f'tables/{path.name}'
    attachments = {}
    for name, content in getattr(result, 'attachments', {}).items():
        if not name or Path(name).name != name or name in ('.', '..') or '\\' in name:
            raise ValueError(f'无效的附件文件名：{name}')
        path = tmp / 'attachments' / name
        path.parent.mkdir(exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding='utf-8')
        attachments[name] = f'attachments/{name}'
    from .sections.guide import load_guide
    guide = load_guide(section)
    if guide:
        (tmp / 'GUIDE.md').write_text(guide, encoding='utf-8')
    digests = {p.relative_to(tmp).as_posix(): file_digest(p) for p in tmp.rglob('*') if p.is_file()}
    record = dict(format=FORMAT_VERSION, signature=signature, inputs=inputs,
                  section=dict(id=section.id, version=section.version, title=section.title, mode=section.mode),
                  recordings=[r.label for r in recordings], summary=jsonable(result.summary),
                  notes=list(result.notes), figures=figures, tables=tables, attachments=attachments, seconds=round(seconds, 2),
                  created=_dt.datetime.now().isoformat(timespec='seconds'), digests=digests)
    (tmp / 'result.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    if folder.exists():
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
    folder.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, folder)
    return record


def read_table(path: Path, limit=5000) -> list[dict]:
    with Path(path).open(encoding='utf-8-sig') as f:
        lines = [line for line in f if not line.startswith('#')]
    reader = csv.DictReader(io.StringIO(''.join(lines)))
    return [row for _, row in zip(range(limit), reader)]
