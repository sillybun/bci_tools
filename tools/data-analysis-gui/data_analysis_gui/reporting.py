"""Collect report PNGs while retaining normal section result directories."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from matplotlib.lines import Line2D

from . import config as C, storage
from .core.catalog import run_folder
from .core.crossday import recording_labels, recording_ticks
from .core.style import DEFAULT_FONT_PERCENT, finish_figure
from .sections import versions
from .sections.base import new_figure


REPORT_ROOT = Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report')
CATEGORIES = (
    '01_数据及实验范式示意图', '02_通道稳定性分析', '03_通讯稳定性分析',
    '04_离线解码分析', '05_在线脑控行为_CO8', '06_在线脑控行为_Grid',
)

SPATIAL_CATEGORIES = ('07_方向调谐_S05', '08_解码权重_S14')

def report_run_group(recording):
    """Count split run folders separately, but not multiple XDFs in one run."""
    folder = run_folder(recording.path)
    if folder is not None:
        return str(folder.resolve())
    return '|'.join((recording.date, recording.task_mode, str(recording.online),
                     recording.run or recording.key))


def report_plan(recordings, params_for, *, behavior_folders=None, online_behavior_folders=None, date_range=None, analysis_excluded_dates=(), include_run_exports=False):
    """Use all sessions for quality, offline sessions for CV, and online sessions for behavior."""
    requested_recordings = list(recordings)
    excluded_dates = set(analysis_excluded_dates)
    if online_behavior_folders is None:
        online_behavior_folders = []
        for rec in requested_recordings:
            folder = run_folder(rec.path) or rec.path.parent
            if rec.online and rec.date not in excluded_dates and ((folder/'trials.csv').is_file() or (folder/'behavior/trials.csv').is_file()):
                online_behavior_folders.append(str(folder))
    if behavior_folders is None:
        behavior_folders = []
        for rec in requested_recordings:
            folder = run_folder(rec.path) or rec.path.parent
            if not rec.online and ((folder/'trials.csv').is_file() or (folder/'behavior/trials.csv').is_file()):
                behavior_folders.append(str(folder))
    recordings = sorted((r for r in requested_recordings if r.date not in excluded_dates), key=lambda r: (r.date, r.key))
    if not recordings or any(not r.date for r in recordings):
        raise ValueError('一键报告需要选择日期已知的记录')
    jobs, categories, skipped = [], [], []

    def add(sid, recs, category):
        section = versions(sid)[-1]
        if section.mode == 'cross' and len(recs) < section.min_recordings:
            skipped.append(dict(category=category, reason=f'{sid.upper()} 至少需要 {section.min_recordings} 份记录'))
            return
        raw = dict(params_for(section))
        if sid == 'x01':
            raw['map_quantity'] = 'max_code'
            raw['map_layout'] = 'both'
        if sid == 's14':
            source = versions('s10')[-1]
            raw['_source_sections'] = {'s10': {'version': source.version, 'params': params_for(source)}}
            raw['depth_csv'] = str(C.DEFAULT_DEPTH_CSV)
        if sid == 's10' and not int(raw.get('n_boot', section.defaults()['n_boot'])):
            raw['n_boot'] = 200
        section.resolve_params(raw)
        jobs.append((section, recs, raw))
        categories.append(category)

    add('x00', recordings, CATEGORIES[0])
    add('x01', recordings, CATEGORIES[1])
    add('x02', recordings, CATEGORIES[2])
    offline = [r for r in recordings if not r.online and r.task_mode in ('centerout', 'grid')]
    if offline:
        add('x00', offline, CATEGORIES[3])
        for rec in offline:
            add('s10', [rec], CATEGORIES[3])
    else:
        skipped.append(dict(category=CATEGORIES[3], reason='没有所选的离线 CO8 / Grid 记录'))
    for mode, category in (('centerout', CATEGORIES[4]), ('grid', CATEGORIES[5])):
        online = [r for r in recordings if r.online and r.task_mode == mode]
        if not online:
            skipped.append(dict(category=category, reason='没有所选的该任务在线记录'))
        if online:
            add('x08', online, category)
        for rec in online:
            add('s13', [rec], category)
    for rec in recordings:
        add('s05', [rec], SPATIAL_CATEGORIES[0])
        add('s14', [rec], SPATIAL_CATEGORIES[1])
    behavior_folders = sorted(set(map(str,behavior_folders)))
    online_behavior_folders = sorted(set(map(str,online_behavior_folders)))
    run_count = len({report_run_group(r) for r in recordings} | {str(Path(p).resolve()) for p in behavior_folders+online_behavior_folders})
    first_date,last_date = date_range or (recordings[0].date,recordings[-1].date)
    return jobs, dict(
        name=f'{first_date}_{last_date}_{run_count}runs', offline_behavior_folders=behavior_folders, analysis_excluded_dates=sorted(excluded_dates), root=str(REPORT_ROOT),
        online_behavior_folders=online_behavior_folders,
        include_run_exports=bool(include_run_exports),
        run_count=run_count, recording_count=len(recordings),
        recordings=[dict(path=str(r.path), key=r.key, date=r.date, task_mode=r.task_mode,
                         online=r.online, label=label, run_group=report_run_group(r))
                    for r, label in zip(recordings, recording_labels(recordings))],
        categories=categories, skipped=skipped,
    )


def _save_png(fig, path, font_percent):
    finish_figure(fig, font_percent)
    fig.savefig(path, dpi=storage.PNG_DPI, bbox_inches='tight', facecolor='white')
    fig.clear()


def _velocity_figure(entries, font_percent, destination):
    fig = new_figure(max(11, 1.25 * len(entries)), 5.8)
    ax = fig.subplots()
    labels = [f'{entry["label"]} · n={entry["record"]["summary"].get("n_trials", "?")}' for entry in entries]
    for i in range(1, len(entries)):
        if entries[i]['date'] != entries[i - 1]['date']:
            ax.axvline(i - .5, color='#c8c8c8', ls='--', lw=.8)
    for offset, name, color in ((-.16, 'velocity_kf', '#c83327'), (.16, 'ridge', '#2176ae')):
        x = np.arange(len(entries)) + offset
        for i, entry in enumerate(entries):
            summary = entry['record']['summary']
            value = summary.get(f'{name}_r2_mean')
            if value is None or not math.isfinite(value):
                ax.text(i, .05, 'N/A', transform=ax.get_xaxis_transform(), ha='center', fontsize=8)
                continue
            ax.plot(x[i], value, 'o', color=color)
            interval = summary.get(f'{name}_r2_mean_ci')
            if interval and all(v is not None and math.isfinite(v) for v in interval):
                ax.vlines(x[i], interval[0], interval[1], color=color, lw=1.2)
                ax.hlines(interval, x[i] - .06, x[i] + .06, color=color, lw=1.2)
    ax.grid(axis='y', alpha=.2)
    ax.set(title='Velocity', ylabel='Held-out R²', xlim=(-.7, len(entries) - .3))
    ax.legend(handles=[Line2D([], [], ls='none', marker='o', color=color, label=label)
                       for color, label in (('#c83327', 'Vel-KF'), ('#2176ae', 'Ridge'))], frameon=False)
    recording_ticks(ax, range(len(labels)), labels, fontsize=9)
    _save_png(fig, destination / 'Velocity.png', font_percent)


def _grid_behavior(entry, font_percent, destination):
    """Grid block success and RT; BPS is shown separately using 60 active seconds."""
    record = entry['record']
    rows = storage.read_table(Path(entry['folder']) / record['tables']['block_metrics'], limit=1000000)
    if not rows:
        return
    numeric = ('success_pct', 'rt_median_s', 'rt_q1_s', 'rt_q3_s')
    for row in rows:
        for key in numeric:
            row[key] = float(row[key]) if row[key] not in ('', 'None') else np.nan
        for key in ('block', 'attempts', 'successes'):
            row[key] = int(row[key]) if row[key] else 0
    fig = new_figure(11, 6)
    axes = fig.subplots(2, 1, sharex=True)
    x = [row['block'] for row in rows]
    for ax, key, ylabel in zip(axes, ('success_pct', 'rt_median_s'),
                              ('Success rate (%)', 'Time to target (s)')):
        values = np.array([row[key] for row in rows])
        if key == 'rt_median_s':
            q1, q3 = np.array([row['rt_q1_s'] for row in rows]), np.array([row['rt_q3_s'] for row in rows])
            ax.errorbar(x, values, yerr=[values - q1, q3 - values], fmt='o-', color='#a65b24',
                        capsize=3, label='Median and IQR')
            ax.legend(frameon=False)
        else:
            ax.plot(x, values, 'o-', color='#286e9e')
        if key == 'success_pct':
            for block, value, row in zip(x, values, rows):
                if np.isfinite(value):
                    ax.annotate(f'{row["successes"]}/{row["attempts"]}', (block, value), (0, -5),
                                textcoords='offset points', ha='center', va='top', fontsize=6.5)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.2)
        if key == 'success_pct':
            ax.set_ylim(0, 105)
        else:
            ax.set_ylim(bottom=0)
    step = max(1, len(x) // 20)
    axes[-1].set_xticks(sorted(set(x[::step]) | {x[-1]}))
    axes[-1].set_xlabel('Task block')
    fig.suptitle(f'{entry["label"]} · Grid Behavior')
    _save_png(fig, destination, font_percent)


def assemble_report(report, entries, failures, font_percent, send):
    """Publish a fresh date-named folder, preserving previous report runs."""
    root = Path(report['root'])
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.report_', dir=root))
    include_runs=bool(report.get('include_run_exports',False))
    from .report_run_pool import RunPool,link_file,dependencies
    pool=RunPool(root/'individual_runs',refresh=bool(report.get('recompute')))
    manifest = dict(selection=report['recordings'], run_count=report.get('run_count'),
                    font_policy=dict(title_pt=18, axis_label_pt=14, tick_pt=11, schematics='preserve legacy style'),
                    recording_count=len(report['recordings']), font_percent=font_percent,
                    analysis_excluded_dates=report.get('analysis_excluded_dates', []),
                    include_run_exports=include_runs,
                    skipped=report['skipped'], failures=failures, sources=[], complete=False,
                    methods={
                        'Velocity.png': 'S10 held-out mean x/y velocity R² from five chronological block folds '
                                        'with ±1-block purge, Ridge and Vel-KF. Error bars are 95% block-bootstrap '
                                        'intervals from the same predictions. Only selected offline sessions.',
                        'Grid_Behavior.png': 'S13 block metrics. Success percentage over all '
                                             'known attempts including timeouts, and successful-trial acquisition '
                                             'time median and IQR. Only selected online Grid sessions.',
                    })
    offline = []
    labels = {str(Path(r['path'])): r for r in report['recordings']}
    try:
        for entry in entries:
            category = work / entry['category']
            category.mkdir(exist_ok=True)
            if not include_runs:
                category=category/'summary_all_runs'
                category.mkdir(exist_ok=True)
            record, folder = entry['record'], Path(entry['folder'])
            if record['section']['mode']=='single':
                folder=pool.section(entry)
                entry['folder']=str(folder)
                record=json.loads((folder/'result.json').read_text(encoding='utf-8'))
                entry['record']=record
            paths = entry['job']['recordings']
            key = storage.fingerprint(dict(paths=paths, params=entry['job'].get('params', {})))[:8]
            sid = entry['job']['section']
            session = labels.get(str(Path(paths[0]))) if len(paths) == 1 else None
            name = f'_{storage.slug(session["label"])}' if session else ''
            prefix = f'{sid.upper()}{name}_{key}'
            copied = []
            for figure in record['figures']:
                if not include_runs and record['section']['mode']=='single':
                    continue
                relative = figure['files'].get('png')
                if relative:
                    main = (
                        (sid == 'x00' and figure['title'] == 'Session Overview')
                        or (sid == 'x01' and figure['title'] in (
                            'Electrode Maps by Session', 'Electrode Maps by Session - default', 'Electrode Maps by Session - physical', 'Electrode Activity and Direction Tuning', 'BGA Chip Layout Mapping'))
                        or sid == 'x02'
                        or (sid == 'x08' and ('Run Rolling BPS' in figure['title'] or 'Run Time to Target' in figure['title']))
                        or (sid == 'x08' and figure['title'] == 'Online CO8 Run Success and Time to Target')
                        or (sid == 's13' and 'Rolling BPS' in figure['title'])
                        or (sid == 's13' and entry['category'] == CATEGORIES[4]
                            and figure['title'] == 'Online Behavior by Block')
                    )
                    if sid in ('s05', 's14'):
                        destination = category / 'runs' / storage.slug(session['label']) / '原分析图'
                    else:
                        destination = category if main else category / '补充计算图'
                    destination.mkdir(parents=True, exist_ok=True)
                    target = destination / f'{prefix}_{Path(relative).name}'
                    try:
                        link_file(folder / relative, target)
                    except OSError as exc:
                        failures.append(dict(stage='copy PNG', source=str(folder / relative), message=str(exc)))
                        continue
                    copied.append(str(target.relative_to(work)))
            manifest['sources'].append(dict(category=entry['category'], section=sid, folder=str(folder),
                                            signature=record['signature'], pngs=copied))
            if sid in ('s10', 's13'):
                selected = labels[str(Path(paths[0]))]
                entry = dict(entry, label=selected['label'], date=selected['date'])
                if sid == 's10':
                    offline.append(entry)
                elif include_runs and entry['category'] == CATEGORIES[5]:
                    try:
                        def build_grid(staged):
                            _grid_behavior(entry,font_percent,staged/'Grid_Behavior.png')
                            return {}
                        _,cached,_=pool.get('grid_behavior',paths[0],dict(source=record['signature'],label=entry['label'],style=dependencies(_grid_behavior,Path(__file__).parent/'core/style.py')),build_grid)
                        if (cached/'figures/Grid_Behavior.png').exists():link_file(cached/'figures/Grid_Behavior.png',category/f'{prefix}_Grid_Behavior.png')
                    except Exception as exc:
                        failures.append(dict(stage='Grid behavior', source=str(folder), message=str(exc)))
        if offline:
            send('message', message='生成离线解码汇总图')
            _velocity_figure(offline, font_percent, work / CATEGORIES[3] / ('' if include_runs else 'summary_all_runs'))
        from .report_spatial import render_report_spatial
        manifest['spatial'] = render_report_spatial(work, entries, report, font_percent, send,pool=pool)
        from .report_behavior import render_offline_behavior
        manifest['offline_behavior'] = render_offline_behavior(work, report.get('offline_behavior_folders', []), font_percent, send,include_runs=include_runs,pool=pool)
        manifest['online_behavior'] = render_offline_behavior(work, report.get('online_behavior_folders', []), font_percent, send, online=True,include_runs=include_runs,pool=pool)
        from .report_bga import render_daily_bga
        manifest['bga_daily'] = render_daily_bga(work,report,font_percent,send,pool=pool)
        from .control_comparison import render_comparison
        manifest['control_comparison'] = render_comparison(work,report,font_percent,send,pool)
        manifest['complete'] = not failures
    except Exception as exc:
        failures.append(dict(stage='report assembly', message=str(exc)))
        manifest['failures'] = failures
    manifest['run_pool']=pool.manifest()
    (work / 'report.json').write_text(json.dumps(storage.jsonable(manifest), ensure_ascii=False, indent=2),
                                     encoding='utf-8')
    # A fresh run avoids mixing old selections or stale PNGs under the same date range.
    for number in range(1, 10000):
        target = root / f'{report["name"]}_v{number}'
        if target.exists():
            continue
        try:
            manifest['report_version'] = number
            manifest['folder_name'] = target.name
            (work / 'report.json').write_text(
                json.dumps(storage.jsonable(manifest), ensure_ascii=False, indent=2), encoding='utf-8')
            os.rename(work, target)
            return target
        except FileExistsError:
            continue
    raise RuntimeError(f'报告目录已满，请检查 {root}')


def run_report(request, run_jobs, send):
    request = dict(request)
    if C.DEFAULT_LAYOUT_XLSX.is_file():
        request['layout_path'] = str(C.DEFAULT_LAYOUT_XLSX)
    report = request['report']
    request['report_summary_only']=not bool(report.get('include_run_exports',False))
    request['report_run_pool_root']=str(Path(report['root'])/'individual_runs')
    report['recompute']=bool(request.get('recompute'))
    report['layout_path'] = request.get('layout_path', '')
    if len(request['jobs']) != len(report['categories']):
        raise ValueError('报告任务与目录分组数量不一致')
    task = f'report:{report["name"]}'
    send('plan', tasks=[dict(task=task, section='report', label='一键报告')])
    send('start', task=task, label=f'一键报告 · {report["name"]}')
    entries, failures = [], []

    def progress(event, **values):
        if event == 'message':
            send('message', task=task, message=values['message'])

    for index, (job, category) in enumerate(zip(request['jobs'], report['categories']), 1):
        send('message', task=task, message=f'报告 {index}/{len(request["jobs"])} · {category} · {job["section"].upper()}')

        def receive(event, **values):
            if event == 'complete':
                folder = Path(values['folder'])
                record = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
                entries.append(dict(job=job, category=category, folder=str(folder), record=record))
            elif event == 'error':
                failures.append(dict(category=category, section=job['section'],
                                     recordings=job['recordings'], message=values['message']))
                send('message', task=task, message=f'{job["section"].upper()} 失败，继续生成其他部分')
            else:
                progress(event, **values)

        part = dict(request, jobs=[job])
        part.pop('report', None)
        run_jobs(part, receive)
    folder = assemble_report(report, entries, failures, request.get('font_percent', DEFAULT_FONT_PERCENT), progress)
    if failures:
        send('error', task=task, message=f'报告有 {len(failures)} 项失败，已保存可用 PNG 到 {folder}',
             folder=str(folder))
    else:
        send('complete', task=task, folder=str(folder), cached=False, seconds=0, skipped=len(report['skipped']))
    send('done', failures=len(failures))
    return int(bool(failures))
