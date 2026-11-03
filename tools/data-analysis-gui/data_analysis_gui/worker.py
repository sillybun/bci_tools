"""Compute worker run in a child process by the GUI (and usable from the CLI).

    python -X utf8 -m data_analysis_gui.worker request.json

The request lists jobs (section + recordings + params). For each job the worker
first searches storage for a valid result with the same signature and only
computes when none exists (or `recompute` is set). Events are written to stdout as
lines starting with '@@ ' followed by JSON, which the GUI turns into the progress bar.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import traceback

import matplotlib
matplotlib.use('Agg')

from .core.catalog import describe
from .core.filters import TrialFilter
from .core.session import get_session
from .core.style import DEFAULT_FONT_PERCENT
from .runner import build_context, run_section
from .sections import registry, versions
from .session_cache import SessionCache
from . import storage


def emit(event, **values):
    print('@@ ' + json.dumps(dict(event=event, **storage.jsonable(values)), ensure_ascii=False), flush=True)


def find_section(section_id, version):
    for s in versions(section_id):
        if s.version == version:
            return s
    raise ValueError(f'未找到 section {section_id} {version}')


def plan(request) -> tuple[list[dict], list[dict]]:
    """Resolve each job independently and retain errors with their task IDs."""
    root = Path(request.get('data_root') or '.')
    criteria = TrialFilter.from_dict(request.get('criteria'))
    out, errors = [], []
    for job in request['jobs']:
        task = job['section'] + ':' + '+'.join(Path(p).name for p in job['recordings'])
        try:
            section = find_section(job['section'], job['version'])
            recordings = [describe(Path(p), root) for p in job['recordings']]
            params = section.resolve_params(job.get('params'))
            config = ''
            if storage.uses_channel_config(section):
                config = job['channel_config_path'] if 'channel_config_path' in job else recordings[0].channel_config
                if config and not Path(config).is_file():
                    raise ValueError(f'通道配置文件不存在：{config}')
            inputs = storage.signature_inputs(section, recordings, criteria, params, request.get('layout_path', ''),
                                              channel_config_path=config,
                                              font_percent=request.get('font_percent', DEFAULT_FONT_PERCENT))
            signature = storage.fingerprint(inputs)
            folder = storage.result_folder(request['output_root'], section, recordings, signature)
            out.append(dict(section=section, recordings=recordings, params=params, inputs=inputs,
                            signature=signature, folder=folder, criteria=criteria, task=task,
                            channel_config_path=config))
        except Exception as exc:
            errors.append(dict(task=task, message=str(exc), traceback=traceback.format_exc()))
    return out, errors


def run(request) -> int:
    if request.get('report'):
        from .reporting import run_report
        return run_report(request, run_jobs, emit)
    return run_jobs(request)


def run_jobs(request, send=None) -> int:
    send = emit if send is None else send
    registry()
    jobs, planning_errors = plan(request)
    send('plan', tasks=[dict(task=j['task'], section=j['section'].id, label=j['section'].display,
                             recordings=[r.label for r in j['recordings']]) for j in jobs])
    cache: dict = {}
    artifacts = SessionCache(request['output_root'], refresh=bool(request.get('recompute')))
    for error in planning_errors:
        send('error', **error)
    failures = len(planning_errors)
    def publish(job, record):
        if request.get('export'):
            from .exporting import export_result
            target = export_result(job['folder'], record, job['recordings'], request['export'])
            send('message', task=job['task'], message=f'本批次导出：{target}')

    for job in jobs:
        section, recordings = job['section'], job['recordings']
        send('start', task=job['task'], label=f'{section.display} · ' + ', '.join(r.label for r in recordings))
        start = time.time()
        try:
            record = (None if request.get('recompute') and not request.get('export_only')
                      else storage.load_cached(job['folder'], job['signature']))
            if record is not None:
                publish(job, record)
                send('complete', task=job['task'], folder=str(job['folder']), cached=True, seconds=0)
                continue
            if request.get('export_only'):
                raise ValueError('当前设置没有完整的已缓存结果，请先计算该记录的分析。')
            config = job['channel_config_path']
            if 'channel_config' in section.requires and not config:
                raise ValueError(f'该记录没有可用的通道配置（{recordings[0].path.parent}）')
            send('message', task=job['task'], message='加载记录')
            sessions = [get_session(r) for r in recordings]
            for session, source in zip(sessions, job['inputs']['recordings']):
                artifacts.bind_session(session, source['sha256'])
            hits, misses = artifacts.hits, artifacts.misses
            ctx = build_context(sessions, job['criteria'], None, section, request.get('layout_path'),
                                config, cache,
                                progress=lambda m, t=job['task']: send('message', task=t, message=str(m)),
                                artifacts=artifacts)
            ctx.params = job['params']
            result = run_section(section, ctx, font_percent=job['inputs']['font_percent'])
            send('message', task=job['task'], message=(f'Session 中间结果：复用 {artifacts.hits - hits} 项，'
                                                       f'计算 {artifacts.misses - misses} 项'))
            send('message', task=job['task'], message='保存 PNG / SVG / PDF / CSV')
            record = storage.save_result(result, job['folder'], job['signature'], job['inputs'], section, recordings,
                                         time.time() - start)
            for _, fig in result.figures:
                fig.clear()
            publish(job, record)
            send('complete', task=job['task'], folder=str(job['folder']), cached=False,
                 seconds=round(time.time() - start, 2))
        except Exception as exc:
            failures += 1
            send('error', task=job['task'], message=str(exc), traceback=traceback.format_exc())
    send('done', failures=failures)
    return 1 if failures else 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass
    request = json.loads(Path(argv[0]).read_text(encoding='utf-8'))
    return run(request)


if __name__ == '__main__':
    sys.exit(main())
