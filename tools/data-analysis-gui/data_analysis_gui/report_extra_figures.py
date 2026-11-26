"""Supplementary report figures derived from the selected sections' exported tables."""
from collections import defaultdict
import html
import json
import math
import os
from pathlib import Path

import numpy as np
from matplotlib.cbook import boxplot_stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from . import storage
from .core.channels import ELECTRODE_SITE_TICKS, read_layout
from .core.crossday import recording_ticks
from .report_bga import SHANK_COLORS
from .sections.base import new_figure
from .sections.s13_online_behavior_v1 import rolling_bps


def _rows(entry, key):
    table = entry['record'].get('tables', {}).get(key)
    return storage.read_table(Path(entry['folder'])/table, limit=1000000) if table else []


def _finite(value):
    if value is None or value == '':
        return np.nan
    return float(value)


def tuning_by_day(entry, out, font_percent, save):
    by_day = defaultdict(list)
    for row in _rows(entry, 'direction_tuning_by_session'):
        value = _finite(row.get('direction_tuned'))
        if math.isfinite(value):
            by_day[row['date']].append(value)
    if not by_day:
        return []
    dates = sorted(by_day)
    stats = boxplot_stats([by_day[date] for date in dates], whis=1.5)
    fig = new_figure(max(9, len(dates)*1.15), 6.1)
    ax = fig.subplots()
    edge, fill, median = '#21618a', '#d4e7f4', '#163d56'
    ax.bxp(stats, showfliers=False, patch_artist=True, widths=.46,
           boxprops=dict(facecolor=fill, edgecolor=edge, linewidth=1.4),
           medianprops=dict(color=median, linewidth=2),
           whiskerprops=dict(color=edge, linewidth=1.3), capprops=dict(color=edge, linewidth=1.3))
    whisker = ax.errorbar([], [], yerr=[], fmt='none', ecolor=edge, capsize=5)
    ax.legend([Patch(facecolor=fill,edgecolor=edge), Line2D([],[],color=median,lw=2), whisker],
              ['Box: Q1–Q3', 'Median', 'Whiskers: 1.5 × IQR'], loc='upper right', ncol=3, frameon=False)
    ax.set_xticks(range(1,len(dates)+1), [date[5:] for date in dates])
    ax.set(xlabel='Date', ylabel='Direction-tuned channels',
           ylim=(0, max(1, max(s['whishi'] for s in stats)*1.14)))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', alpha=.23)
    fig.suptitle('Direction-tuned channel counts by day')
    path = out/'X01_Direction_Tuned_Channel_Counts_By_Day.png'
    save(fig, path, font_percent)
    table = path.with_suffix('.csv')
    storage.write_csv(table, [
        dict(date=date, n_recordings=len(by_day[date]), q1=s['q1'], median=s['med'], q3=s['q3'],
             whisker_low=s['whislo'], whisker_high=s['whishi'], outliers=', '.join(f'{x:g}' for x in s['fliers']))
        for date,s in zip(dates,stats)])
    return [path, table]


def shank_mapping(entry, report, out, copied, font_percent, save):
    mapping = next((f for f in entry['record']['figures']
                    if 'mapping' in f.get('data_files', {}) and f['title']=='BGA Chip Layout Mapping'), None)
    if mapping is None:
        return []
    layout = read_layout(report['layout_path'])
    xy, bga = layout['position'], layout['bga_position']
    order = sorted(range(8), key=lambda s: np.median(bga[xy[:,1]==s,1]))
    x_coords = {s:i for i,s in enumerate(order,1)}
    fig = new_figure(9, 24)
    ax = fig.subplots()
    ax.set(xlim=(.15,8.45), ylim=(128.7,-2), ylabel='Electrode #')
    ax.set_xticks(range(1,9), [f'Shank {s+1}' for s in order])
    ax.xaxis.tick_top()
    ax.set_yticks(ELECTRODE_SITE_TICKS)
    ax.tick_params(axis='both', length=0)
    ax.spines[:].set_visible(False)
    ax.grid(axis='y', color='#e7edf2', linewidth=.7)
    ax.set_axisbelow(True)
    ax.scatter([x_coords[s]+.03 for s in xy[:,1]], xy[:,0]+1, s=15,
               c=[SHANK_COLORS[s] for s in xy[:,1]], edgecolors='none')
    for ch,(site,shank) in enumerate(xy):
        ax.text(x_coords[shank]-.08, site+1, str(ch), ha='right', va='center', fontsize=7)
    fig.suptitle('Shank electrode sites · BGA left-to-right order')
    fig.supxlabel(Path(layout['source']).name)
    image = out/'X01_Shank_Electrode_Site_Mapping_Final.png'
    save(fig, image, font_percent)
    interactive = out/'X01_BGA_Shank_Interactive_Mapping.html'
    template = Path(__file__).with_name('report_bga_interactive.html').read_text(encoding='utf-8')
    rows = [[int(ch), int(shank)+1, int(site)+1, int(row), int(col)]
            for ch, ((site,shank),(row,col)) in enumerate(zip(xy,bga))]
    source = next((p for p in copied if p.name.endswith(Path(mapping['files']['png']).name)), None)
    if source is None:
        return [image]
    template = (template.replace('__ROWS__',json.dumps(rows,separators=(',',':')))
                .replace('__COLORS__',json.dumps(SHANK_COLORS))
                .replace('__ORDER__',json.dumps([int(s)+1 for s in order]))
                .replace('__SOURCE__',html.escape(Path(os.path.relpath(source, out)).as_posix(), quote=True))
                .replace('__DIGEST__',storage.file_digest(layout['source'])))
    interactive.write_text(template,encoding='utf-8')
    return [image, interactive]


_MODELS = ('ridge', 'velocity_kf')
_VELOCITY_FIELDS = tuple(f'{model}_{axis}' for model in _MODELS for axis in ('x','y'))


def _run_velocity_rows(entries, selected):
    groups = defaultdict(list)
    for entry in entries:
        path, = entry['job']['recordings']
        rec = selected[str(Path(path))]
        summary = entry['record']['summary']
        values = {f'{model}_{axis}': _finite(summary.get(f'{model}_r2_{axis}'))
                  for model in _MODELS for axis in ('x', 'y')}
        if any(math.isfinite(v) for v in values.values()):
            group = rec.get('run_group') or rec['path']
            groups[group].append((rec, values))
    rows = []
    for group, results in groups.items():
        first = results[0][0]
        label = f'{first["date"][5:]} · {Path(group).name}' if Path(group).is_dir() else first['label']
        row = dict(date=first['date'], run_group=group, label=label, recordings=len(results))
        for key in _VELOCITY_FIELDS:
            values = [v[key] for _,v in results if math.isfinite(v[key])]
            row[key] = float(np.mean(values)) if values else np.nan
        rows.append(row)
    return rows


def _daily_velocity_rows(runs):
    days = defaultdict(list)
    for row in runs:
        days[row['date']].append(row)
    result = []
    for day, rows in sorted(days.items()):
        record = dict(date=day, n_runs=len(rows))
        for key in _VELOCITY_FIELDS:
            values = [row[key] for row in rows if math.isfinite(row[key])]
            stats = np.quantile(values, [.25,.5,.75]) if values else (np.nan,)*3
            record.update({f'{key}_{q}':float(v) for q,v in zip(('q1','median','q3'),stats)})
        result.append(record)
    return result


def _velocity_trend(rows, series, *, daily, title, narrow=False):
    width = (6 if narrow else 12) if daily else max(12, len(rows)*.59)
    fig = new_figure(width, 5.8 if daily else 8.7)
    ax = fig.subplots()
    x = np.arange(len(rows))
    ax.axhline(0, color='#89929a', lw=.8)
    for (field,label), color, marker in zip(series, ('#2176ae','#ba3e31'), ('o','s')):
        ax.plot(x, [r[field+('_median' if daily else '')] for r in rows],
                marker=marker, color=color, markersize=4, lw=1.4, label=label)
        if daily:
            ax.fill_between(x, [r[field+'_q1'] for r in rows], [r[field+'_q3'] for r in rows], color=color, alpha=.15)
    if daily:
        ax.set_xticks(x, [f'{r["date"][5:]} · n={r["n_runs"]}' for r in rows],
                      rotation=35 if narrow else 0, ha='right' if narrow else 'center')
        fig.supxlabel('Daily median across runs; shading shows Q1–Q3.')
    else:
        recording_ticks(ax, x, [r['label'] for r in rows])
        for i in range(1,len(rows)):
            if rows[i-1]['date'] != rows[i]['date']:
                ax.axvline(i-.5, color='#bbb', ls='--', lw=.7)
    ax.set(xlabel='Date' if daily else 'Run', ylabel='Held-out R²')
    ax.grid(axis='y',alpha=.2)
    ax.legend(frameon=False, ncol=2)
    fig.suptitle(title)
    return fig


def velocity_trends(entries, selected, out, font_percent, save):
    runs = _run_velocity_rows(entries, selected)
    if not runs:
        return []
    paths = []
    for grain,rows in [('Run',runs),('Day',_daily_velocity_rows(runs))]:
        daily = grain == 'Day'
        charts = [(f'Velocity_R2_{axis.upper()}_By_{grain}', f'{axis.upper()} velocity R²',
                   [(f'ridge_{axis}','Ridge'),(f'velocity_kf_{axis}','Vel-KF')]) for axis in ('x','y')]
        charts.append((f'Vel_KF_R2_By_{grain}', 'Vel-KF velocity R²',
                       [('velocity_kf_x','X R²'),('velocity_kf_y','Y R²')]))
        for name,heading,series in charts:
            plot = out/f'{name}.png'
            fig = _velocity_trend(rows,series,daily=daily,title=f'{heading} by {grain.lower()}')
            if daily and name.startswith('Vel_KF'):
                _save_daily_kf(fig, plot, font_percent)
            else:
                save(fig, plot, font_percent)
            paths.append(plot)
        if daily:
            plot = out/'Vel_KF_R2_By_Day_Half_Width.png'
            _,heading,series = charts[-1]
            fig = _velocity_trend(rows,series,daily=True,title=f'{heading} by day',narrow=True)
            _save_daily_kf(fig, plot, font_percent)
            paths.append(plot)
        for prefix, subset in [('Velocity', rows),
                               ('Vel_KF',[{k:v for k,v in row.items() if not k.startswith('ridge_')} for row in rows])]:
            table = out/f'{prefix}_R2_By_{grain}.csv'
            storage.write_csv(table,subset)
            paths.append(table)
    return paths


def grid_assist_off(entry, out, font_percent, save):
    rows = [r for r in _rows(entry,'grid_run_rolling_bps') if rolling_bps._grid_config_cells(r)[-1] == 'Off']
    if not rows:
        return []
    fields = ('bps_q1','bps_median','bps_q3','bps_peak','rt_median_s','rt_q1_s','rt_q3_s','grid_cells')
    rows = [dict(r, **{k:_finite(r.get(k)) for k in fields}) for r in rows]
    _, fig = rolling_bps.grid_run_figure(rows, online=True, show_assist=False)
    fig.suptitle('Online Grid · BPS and Time to Target by Run · Assist Off')
    path = out/'X08_Online_Grid_BPS_TTT_Assist_Off.png'
    save(fig,path,font_percent)
    return [path]


def _save_daily_kf(fig, path, font_percent):
    from .core.style import finish_figure

    finish_figure(fig, font_percent)
    fig.savefig(path, dpi=storage.PNG_DPI, facecolor='white')
    fig.clear()


def render_report_extras(work, report, entries, sources, font_percent, send, failures):
    from .reporting import CATEGORIES, _save_png, velocity_summary_entries

    def destination(category):
        folder = work/category/'summary_all_runs'
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    labels = {str(Path(r['path'])): r for r in report['recordings']}
    offline = [entry for entry in entries if entry['job']['section'] == 's10']
    x01 = next((entry for entry in entries if entry['job']['section'] == 'x01'), None)
    x08 = next((entry for entry in entries if entry['job']['section'] == 'x08'
                and 'grid_run_rolling_bps' in entry['record'].get('tables', {})), None)
    steps = []
    if x01:
        copied = [work/path for source in sources if source['section'] == 'x01' for path in source['pngs']]
        steps.extend([
            ('每日方向调谐箱线图', lambda: tuning_by_day(x01, destination(CATEGORIES[1]), font_percent, _save_png)),
            ('Shank / BGA 映射', lambda: shank_mapping(
                x01, report, destination(CATEGORIES[1]), copied, font_percent, _save_png)),
        ])
    if offline:
        steps.append(('速度 R² 趋势图', lambda: velocity_trends(
            velocity_summary_entries(offline), labels, destination(CATEGORIES[3]), font_percent, _save_png)))
    if x08:
        steps.append(('Assist Off 在线 Grid 图', lambda: grid_assist_off(
            x08, destination(CATEGORIES[5]), font_percent, _save_png)))
    files = []
    for title, export in steps:
        try:
            send('message', message=f'生成{title}')
            files.extend(path.relative_to(work).as_posix() for path in export())
        except Exception as exc:
            failures.append(dict(stage=title, message=str(exc)))
    return files
