"""Replay saved report statistics with the common 18/14/11 point font policy."""
from pathlib import Path
import collections
import functools
import json
import math
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from data_analysis_gui.core.channels import read_layout
from data_analysis_gui.sections.s14_decoder_weights_v1.depth_distribution import load_depths, figure as active_figure
from data_analysis_gui import config as C, storage
from data_analysis_gui.report_spatial import sites_figure, profile_figure
from data_analysis_gui.sections.s13_online_behavior_v1 import rolling_bps
from data_analysis_gui.sections.s13_online_behavior_v1.section import behavior_figure
from update_report_font_sizes import ROOT, AUDIT, LOG, backup, record, read, typed, save

LAYOUT = read_layout()
DEPTHS = load_depths(C.DEFAULT_DEPTH_CSV)
OLD = ROOT/'2026-10-26_2026-11-11_45runs_V1'


@functools.lru_cache(maxsize=256)
def data(path):
    return typed(read(Path(path)))


@functools.lru_cache(maxsize=64)
def meta(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def s05_values(rows, field=None):
    values = np.full(1024, np.nan)
    for r in rows:
        ch = int(r.get('channel_0based', int(r.get('electrode', 1))-1))
        if field:
            try:values[ch] = float(r[field])
            except (ValueError, TypeError):pass
        else:
            values[ch] = 2 if str(r.get('tested', True)).lower() == 'false' else float(str(r['significant']).lower() == 'true')
    return values


def ttt(rows, title):
    fig=Figure(figsize=(max(10,len(rows)*.6),5),layout='constrained');ax=fig.subplots();x=np.arange(len(rows))
    ax.errorbar(x,[r['rt_median_s'] for r in rows],
        yerr=[[r['rt_median_s']-r['rt_q1_s'] for r in rows],[r['rt_q3_s']-r['rt_median_s'] for r in rows]],
        fmt='o-',capsize=3,color='#c25e00')
    ax.set_xticks(x,[r['run'] for r in rows],rotation=60,ha='right');ax.set(ylabel='Time to target (s)',title=title)
    return fig


def statistical_figure(p):
    name=p.stem;parent=p.parent
    seven='7shank' in name or '7shank' in str(parent)
    if 'S05_tuning_' in name:
        source=parent/'channels_with_depth.csv';rows=data(str(source))
        fig=sites_figure(LAYOUT,DEPTHS,s05_values(rows),'S05 direction tuning\n'+parent.name,
                         depth='electrode' not in name,seven=seven,tuning='run')
        return fig,source
    if re.fullmatch(r'S05_(mean|median)_(depth|electrode)(_7shank)?',name):
        source=parent/'channel_tuning_statistics.csv';rows=data(str(source));stat=name.split('_')[1]
        fig=sites_figure(LAYOUT,DEPTHS,s05_values(rows,'tuning_'+stat),f'S05 direction tuning - {stat}',
            depth='electrode' not in name,seven=seven,tuning='median' if stat=='median' else None,
            unit='Fraction of tested runs with tuning')
        return fig,source
    if name.startswith('S14_depth_profiles'):
        suffix=name.removeprefix('S14_depth_profiles');source=parent/('depth_profiles'+suffix+'.csv')
        return profile_figure(data(str(source)),parent.name+' '+suffix),source
    m=re.fullmatch(r'S14_(vx|vy|norm)(?:_(mean|median))?_(electrode|depth)(_7shank)?',name)
    if m:
        component,stat,coordinate,_=m.groups();source=parent/('channel_weight_statistics.csv' if stat else 'channels_with_depth.csv')
        rows=data(str(source));values=np.full(1024,np.nan);field=component+('_'+stat if stat else '')
        for r in rows:
            if str(r.get('transmitted',True)).lower()=='false':continue
            try:values[int(r['channel_0based'])]=float(r[field])
            except (ValueError,TypeError):pass
        upper=25
        if stat:
            all_values=np.array([float(r[k]) for r in rows for k in r if k.endswith(('_mean','_median')) and k.startswith(('vx_','vy_','norm_')) and r[k]!=''])
            finite=all_values[np.isfinite(all_values)]
            upper=float(MaxNLocator(nbins=6).tick_values(0,float(finite.max()))[-1]) if len(finite) and finite.max()>0 else 1
        fig=sites_figure(LAYOUT,DEPTHS,values,f'{component} weighting'+(' - '+stat if stat else '')+'\n'+parent.name,
            depth=coordinate=='depth',seven=seven,limits=(0,upper),unit='velocity / input SD')
        return fig,source
    # Behavior-only exports already contain the exact windows used in each version.
    if name=='Behavior_by_block':
        source=parent/'block_metrics.csv';mode='centerout' if 'CO8' in str(parent) else 'grid'
        return behavior_figure(data(str(source)),mode,parent.name+'\nOffline behavior'),source
    if name=='Rolling_BPS_60s':
        source=parent/'rolling_bps_windows.csv';mode='centerout' if 'CO8' in str(parent) else 'grid'
        return rolling_bps.rolling_figure(data(str(source)),mode,parent.name+'\nOffline BPS'),source
    if name=='TTT_distribution':
        source=parent/'trials.csv';rows=data(str(source));v=[r['rt_s'] for r in rows if r['outcome']=='acquired' and math.isfinite(r['rt_s'])]
        fig=Figure(figsize=(9,5),layout='constrained');ax=fig.subplots()
        if v:ax.hist(v,bins=min(40,max(5,int(np.sqrt(len(v))))),color='#c25e00');ax.axvline(np.median(v),ls='--',color='#333')
        ax.set(xlabel='Time to target (s; includes dwell)',ylabel='Successful trials',title=parent.name+'\nOffline TTT distribution')
        return fig,source
    if name.startswith(('Run_BPS_TTT_summary','Run_TTT_summary')):
        suffix='_exclude1116' if 'exclude1116' in name else '';source=parent/('run_summary'+suffix+'.csv')
        rows=data(str(source));mode='centerout' if 'CO8' in str(parent) else 'grid'
        title=f'Offline {"CO8" if mode=="centerout" else "Grid"} | Run BPS and TTT'+('\nExcluding 2026-11-16' if suffix else '')
        if name.startswith('Run_TTT'):return ttt(rows,title),source
        _,fig=rolling_bps.run_figure(rows,mode);fig._suptitle.set_text(title);return fig,source
    return None


@functools.lru_cache(maxsize=1)
def collection():
    entries=meta(str(OLD/'S14/collection_manifest.json'))['recordings']
    return {r['filename']:r for r in entries}


def historical_45(p):
    if not p.is_relative_to(OLD):return None
    name=p.stem;seven='7shank' in str(p);s14=OLD/'S14';s05=OLD/'S05'
    if p.parent.name.startswith('S14_0') or p.parent.name in ('S14_05_norm_weighting_heatmap','S14_05_norm_weighting_heatmap_origin_lower'):
        if '_active_' not in p.parent.name:
            from data_analysis_gui.sections.s14_decoder_weights_v1.section import weight_figure
            entry=collection()[p.name];folder=Path(entry['source']);m=meta(str(folder/'result.json'))
            source=folder/'tables/channel_weights.csv';rows=[r for r in data(str(source)) if str(r['transmitted']).lower()=='true']
            rows.sort(key=lambda r:int(r['stored_column_0based']));component='vx' if '_vx_' in p.parent.name else 'vy' if '_vy_' in p.parent.name else 'norm'
            s=m['summary'];banner=f'{s["recording"]}\n{s["decoder"]}; {s["weight_basis"]}; {s["n_bins"]} bins'
            return weight_figure(LAYOUT,np.array([r['electrode_id'] for r in rows],int),np.array([r[component] for r in rows]),component,'values' in p.parent.name,banner,'velocity / input SD'),source
    if p.parent.name.startswith('tuning_layout'):
        source=s05/'tables_tuning_layout'/(name.removesuffix('_7shank')+'.csv');rows=data(str(source))
        return sites_figure(LAYOUT,DEPTHS,s05_values(rows),'S05 direction tuning\n'+name,
            depth='electrode_number' not in str(p),seven=seven,tuning='run'),source
    if p.parent.name=='summary' and p.name.startswith('S05_direction'):
        stat='median' if 'median' in name else 'mean'
        source=p.parent/'channel_tuning_summary_7shank.csv' if seven else p.parent/f'S05_direction_tuning_{stat}_depth.csv'
        rows=data(str(source))
        return sites_figure(LAYOUT,DEPTHS,s05_values(rows,'tuning_'+stat),f'S05 direction tuning - {stat}\n45 recordings',seven=seven,
            tuning='median' if stat=='median' else None,unit='Fraction of tested runs with tuning'),source
    if 'figures_depth' in str(p.parent):
        from export_s14_depth_summary import depth_figure
        source=p.parent/'channel_statistics_with_depth.csv';rows=data(str(source));m=re.search(r'_(vx|vy|norm)_weighting_(median|average)',name)
        if not m:return None
        fig=depth_figure(rows,*m.groups(),[0,12]);fig.axes[0].set_xlim(1.5 if seven else .5,8.5);fig.axes[0].set_xticks(range(2 if seven else 1,9))
        return fig,source
    if p.parent.name.startswith('S14_07_active'):
        source=p.parent/'active_depth_bins.csv';rows=data(str(source));group=[r for r in rows if r['recording']==name.removesuffix('_7shank')]
        ymax=max(1,int(np.ceil(max(r['single_unit_count'] for r in rows)*1.15)))
        return active_figure(group,name,y_max=ymax),source
    if 'depth_weighting_and_firing' in str(p):
        from export_depth_weights_and_firing import fig_profiles
        base=p.parent.parent
        if p.parent.name=='per_run':
            source=base/'per_run_depth_bins.csv';rows=[r for r in data(str(source)) if r['recording']==name.removesuffix('_7shank')]
            manifest=base/'manifest.json';limits=meta(str(manifest)).get('limits') if manifest.exists() else None
        else:
            source=p.parent/'summary_depth_bins_7shank.csv' if seven and (p.parent/'summary_depth_bins_7shank.csv').exists() else base/'summary_depth_bins.csv'
            stat='median' if 'median' in name else 'average';rows=[r for r in data(str(source)) if r['statistic']==stat];limits=None
        return fig_profiles(rows,name,limits),source
    return None


def supplementary(p):
    report=next((q for q in p.parents if (q/'report.json').exists()),None)
    if report is None:return None
    m=meta(str(report/'report.json'))
    if 'Electrode_Activity_and_Direction_Tuning' in p.name and m.get('activity_success_overlay',{}).get('data'):
        from data_analysis_gui.sections.x01_channel_stability_v1.section import ChannelStability
        return ChannelStability.activity(m['activity_success_overlay']['data']),report/'report.json'
    if p.name.startswith('X00_'):
        from data_analysis_gui.core.catalog import Recording, describe
        from data_analysis_gui.sections.session_overview import session_overview,PAGES
        for source in m['sources']:
            if source['section']!='x00' or str(p.relative_to(report)) not in source['pngs']:continue
            folder=Path(source['folder']);r=meta(str(folder/'result.json'))
            names={x['name'] for x in r['inputs']['recordings']};records=[]
            for x in m['selection']:
                path=Path(x['path'])
                if path.name not in names:continue
                if path.exists():rec=describe(path,C.DEFAULT_DATA_ROOT)
                else:
                    run=x.get('run') or x['label'].split('run')[-1].split()[0]
                    rec=Recording(path,x.get('key',path.name),x['date'],x['task_mode'],x['online'],run)
                records.append(rec)
            match=re.search(r'_X00_(\d+)_',p.name)
            if match:
                fig,_=session_overview(records,PAGES[int(match[1])-1][0]);return fig,folder/'result.json'
    if p.name=='Velocity.png':
        from data_analysis_gui.reporting import _velocity_figure
        entries=[]
        for source in m['sources']:
            if source['section']!='s10':continue
            folder=Path(source['folder']);r=meta(str(folder/'result.json'));original=r['inputs']['recordings'][0]['name']
            selected=next(x for x in m['selection'] if Path(x['path']).name==original)
            entries.append(dict(record=r,label=selected['label'],date=selected['date']))
        # Draw through the existing helper, intercepting only its final save.
        import data_analysis_gui.reporting as reporting
        original=reporting._save_png;captured=[]
        reporting._save_png=lambda fig,*args:captured.append(fig)
        try:_velocity_figure(entries,100,p.parent)
        finally:reporting._save_png=original
        return captured[0],report/'report.json'
    if p.name.endswith('Grid_Behavior.png'):
        from data_analysis_gui.reporting import _grid_behavior
        import data_analysis_gui.reporting as reporting
        for source in m['sources']:
            if source['section']!='s13':continue
            folder=Path(source['folder']);r=meta(str(folder/'result.json'));recording=r['inputs']['recordings'][0]['name']
            selected=next(x for x in m['selection'] if Path(x['path']).name==recording)
            identity=storage.fingerprint([selected['path']])[:8]
            if identity not in p.name and not any(p.name.startswith(Path(rel).name.split('_S13')[0]) for rel in []):
                label=storage.slug(selected['label'])
                if label not in p.name:continue
            original=reporting._save_png;captured=[];reporting._save_png=lambda fig,*args:captured.append(fig)
            try:_grid_behavior(dict(record=r,folder=str(folder),label=selected['label']),100,p)
            finally:reporting._save_png=original
            if captured:return captured[0],folder/'result.json'
    # Saved backfill windows and plotted run summaries preserve each historical version.
    if 'Run_Rolling_BPS' in p.name:
        mode='centerout' if 'CO8' in str(p.parent) else 'grid';tables=p.parent/'统计数据'
        source=tables/'Grid_BPS_Summary_Plotted_Runs.csv' if mode=='grid' and (tables/'Grid_BPS_Summary_Plotted_Runs.csv').exists() else tables/'Run_Rolling_BPS_Summary.csv'
        if source.exists():
            rows=data(str(source));_,fig=rolling_bps.run_figure(rows,mode,peak='Peak' in p.name and 'Median' not in p.name);return fig,source
    if p.name.endswith('_Rolling_BPS.png'):
        back=report/'rolling_bps_backfill.json'
        if back.exists():
            sources=meta(str(back))['sources']
            item=next((x for x in sources if storage.fingerprint([x['path']])[:8] in p.name),None)
            if item:
                table=p.parent/'统计数据/Rolling_BPS_Windows.csv'
                if table.exists():
                    rows=[r for r in data(str(table)) if r['session']==item['run']]
                    if rows or not item.get('windows'):
                        return rolling_bps.rolling_figure(rows,'centerout' if 'CO8' in str(p.parent) else 'grid',item['run']+' · Rolling BPS'),table
    return None


def main():
    pending=list(map(Path,json.loads((AUDIT/'pending_figures.json').read_text(encoding='utf-8'))))
    completed=set()
    if LOG.exists():
        for line in LOG.read_text(encoding='utf-8').splitlines():
            x=json.loads(line)
            if x['method']=='saved statistics redraw' and x.get('layout_version')=='wrapped-v2':completed.add(x['path'])
    errors=[];unknown=[];count=0
    for i,p in enumerate(pending,1):
        if str(p) in completed:continue
        try:
            result=historical_45(p) or statistical_figure(p) or supplementary(p)
            if result:
                fig,source=result;save(fig,p,source);count+=1
            else:unknown.append(str(p))
        except Exception as exc:errors.append(dict(path=str(p),error=str(exc)))
        if i%20==0:print(f'statistics {i}/{len(pending)} redrawn={count} unknown={len(unknown)} errors={len(errors)}',flush=True)
    (AUDIT/'statistics_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
    (AUDIT/'unknown_figures.json').write_text(json.dumps(unknown,ensure_ascii=False,indent=2),encoding='utf-8')
    print('statistics DONE',count,'unknown',len(unknown),'errors',len(errors),flush=True)


if __name__=='__main__':main()
