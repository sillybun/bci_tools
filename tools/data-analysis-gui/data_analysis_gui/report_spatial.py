"""Per-recording and selected-recording spatial exports for one-click reports."""
from collections import defaultdict
from pathlib import Path
import csv,json
import numpy as np
from matplotlib.figure import Figure
from matplotlib import colormaps
from matplotlib.colors import Normalize,ListedColormap,BoundaryNorm
from matplotlib.ticker import MaxNLocator
from . import config as C,storage
from .core.channels import ELECTRODE_SITE_TICKS, add_missing_sites_key, read_layout
from .sections.s14_decoder_weights_v1.depth_distribution import load_depths


def is_true(value):
    return value is True or str(value).lower()=='true'

def number(value):
    try:return float(value)
    except (ValueError,TypeError):return np.nan

def _channel_cube(runs, fields):
    cube=np.full((len(runs),1024,len(fields)),np.nan)
    for i,rows in enumerate(runs):
        seen=set()
        for r in rows:
            ch=int(r['channel_0based'])
            if ch in seen:raise ValueError('Duplicate channel within one recording')
            seen.add(ch)
            if not is_true(r.get('transmitted',True)) or not is_true(r.get('tested',True)):continue
            cube[i,ch]=[number(r.get(k)) for k in fields]
    return cube


def channel_statistics(runs, fields):
    return _summarize_cube(_channel_cube(runs, fields))


def _summarize_cube(cube):
    count=np.isfinite(cube).sum(0);stats={}
    for name,fn in [('mean',np.nanmean),('median',np.nanmedian)]:
        matrix=np.full(cube.shape[1:],np.nan)
        for j in range(cube.shape[2]):
            valid=count[:,j]>0
            matrix[valid,j]=fn(cube[:,valid,j],axis=0)
        stats[name]=matrix
    return stats,count


def daily_channel_statistics(records_by_run, fields, *, trim_weights):
    """Reduce recordings within a run first, then give each eligible run equal weight."""
    run_names = sorted(records_by_run)
    cube = np.array([channel_statistics(records_by_run[run], fields)[0]['mean'] for run in run_names])
    excluded = []
    if trim_weights:
        for ch, j in zip(*np.nonzero(np.isfinite(cube).sum(axis=0) >= 4)):
            values = cube[:,ch,j]
            q1, q3 = np.percentile(values[np.isfinite(values)], [25,75])
            iqr = q3 - q1
            if iqr <= 0:
                continue
            low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
            for i in np.flatnonzero((values < low) | (values > high)):
                excluded.append(dict(run=run_names[i], channel_0based=int(ch), metric=fields[j],
                                     value=float(values[i]), lower=float(low), upper=float(high)))
                values[i] = np.nan
    stats, count = _summarize_cube(cube)
    return stats, count, excluded


def sites_figure(layout,depths,values,title,*,depth=True,seven=False,limits=(0,1),tuning=None,unit=''):
    axial = 'axial_distance_um' in layout
    max_distance = float(np.max(layout['axial_distance_um'])) if axial else 8500
    fig=Figure(figsize=(max(8.27, .13*max(map(len,title.splitlines()))),11.69));ax=fig.add_axes([.12,.10,.68,.79]);ax.set_facecolor('#d6d4cf')
    cmap=colormaps['hot_r'].with_extremes(bad='#d6d4cf');norm=Normalize(*limits)
    if tuning=='run':cmap=ListedColormap(['white','#c92929','#badfea']).with_extremes(bad='#d6d4cf');norm=BoundaryNorm([-.5,.5,1.5,2.5],3)
    if tuning=='median':cmap=ListedColormap(['white','#efaa32','#c92929']).with_extremes(bad='#d6d4cf');norm=BoundaryNorm([-.25,.25,.75,1.25],3)
    inv={(int(r),int(c)):ch for ch,(r,c) in enumerate(layout['position'])}
    for col in range(1 if seven else 0,8):
        channels=[inv[(r,col)] for r in range(128)];centres=np.array([depths[ch]['depth_um'] for ch in channels]) if depth else np.arange(1,129)
        order=np.argsort(centres);centres=centres[order];half=(2 if axial else 25) if depth else .5
        edges=np.r_[centres[0]-half,(centres[:-1]+centres[1:])/2,centres[-1]+half]
        image=ax.pcolormesh([col+.5,col+1.5],edges,np.ma.masked_invalid(np.asarray(values)[channels][order,None]),cmap=cmap,norm=norm,shading='flat')
    ax.set(xlim=(1.5 if seven else .5,8.5),ylim=((-2,max_distance+2) if axial else (-25,8500)) if depth else (.5,128.5),xticks=range(2 if seven else 1,9),xlabel='Shank',ylabel=('Axial distance from site 1 (µm)' if axial else 'Depth (um)') if depth else 'Electrode #')
    ax.set_yticks((range(0,501,100) if axial else range(0,8500,1000)) if depth else ELECTRODE_SITE_TICKS);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(title+(' | 7shank' if seven else ''),fontsize=12,y=.905,va="bottom")
    cax=fig.add_axes([.84,.12,.025,.77])
    if tuning in ('run','median'):
        ticks=[0,1,2] if tuning=='run' else [0,.5,1];bar=fig.colorbar(image,cax=cax,ticks=ticks)
        bar.ax.set_yticklabels(['Not significant','Tuned','Not tested'] if tuning=='run' else ['0','0.5','1'],fontsize=8)
    else:
        fig.colorbar(image,cax=cax,label=unit)
    add_missing_sites_key(cax, '#d6d4cf')
    return fig

def depth_bins(rows,fields,seven=False,*,stop=8300):
    out=[]
    for left in range(0,stop,100):
        g=[r for r in rows if (not seven or int(r['shank'])!=1) and is_true(r.get('transmitted',True)) and left<=number(r['depth_um'])<left+100]
        d=dict(depth_center_um=left+50,n_channels=len(g))
        for k in fields:
            v=[number(r.get(k)) for r in g];v=[x for x in v if np.isfinite(x)];d[k]=float(np.mean(v)) if v else np.nan
        out.append(d)
    return out

def profile_figure(bins,title,*,axial=False):
    fig=Figure(figsize=(12,8),layout='constrained');axes=fig.subplots(2,2)
    for ax,k,color in zip(axes.flat,('vx','vy','norm','event_hz'),('#0072bd','#d95319','#7e2f8e','#388443')):
        ax.plot([r['depth_center_um'] for r in bins],[r[k] for r in bins],color=color,lw=1.7)
        ax.set(xlim=(0,bins[-1]['depth_center_um']+50),ylim=(0,None),xlabel='Axial distance from site 1 (µm)' if axial else 'Depth (um)',ylabel={'vx':'Mean |wx| / channel','vy':'Mean |wy| / channel','norm':'Mean norm / channel','event_hz':'Binary events / s / channel'}[k]);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(title,fontsize=12)
    return fig

def render_report_spatial(work,entries,report,font_percent,send,*,pool=None):
    from .reporting import SPATIAL_CATEGORIES
    relevant=[e for e in entries if e['job']['section'] in ('s05','s14') and any(t in e['record'].get('tables',{}) for t in ('anova_channels','channel_weights'))]
    if not relevant:return dict(status='no spatial tables')
    layout = read_layout(report.get('layout_path') or C.DEFAULT_LAYOUT_XLSX)
    axial = 'axial_distance_um' in layout
    depth_path = '' if axial else C.DEFAULT_DEPTH_CSV
    depths = load_depths(depth_path, layout=layout)
    depth_source = layout['source'] if axial else str(depth_path)
    depth_sha256 = storage.file_digest(depth_source)
    profile_stop = (int(np.max(layout['axial_distance_um'])) // 100 + 1) * 100 if axial else 8300
    exported=[];runs={'s05':[],'s14':[]}
    daily = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    by_path = {str(Path(r['path'])): r for r in report['recordings']}
    weight_entries=[e for e in relevant if e['job']['section']=='s14']
    settings=[e['record']['summary'] for e in weight_entries]
    if settings:
        first=settings[0]
        if any((s.get('decoder'),s.get('weight_basis'),s.get('source_section')) != (first.get('decoder'),first.get('weight_basis'),first.get('source_section')) for s in settings[1:]):
            raise ValueError('S14 summary requires identical models, units and S10 settings across recordings')
        weight_unit='velocity / input SD' if first.get('weight_basis','per_SD')=='per_SD' else ('velocity / sqrt(Hz)' if first.get('source_section',{}).get('params',{}).get('sqrt') else 'velocity / Hz')
    else:weight_unit='velocity / input SD'
    def write(path,rows):
        if not rows:return
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    def save(fig,folder,name):
        folder.mkdir(parents=True,exist_ok=True)
        for ext in ('png','pdf'):fig.savefig(folder/f'{name}.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
        fig.clear()
        if folder.is_relative_to(work):exported.append(str((folder/f'{name}.png').relative_to(work)))
    def render_run(folder,sid,rows,label):
        folder.mkdir(parents=True,exist_ok=True)
        write(folder/'channels_with_depth.csv',rows)
        if sid=='s05':
            values=np.full(1024,np.nan)
            for r in rows:values[r['channel_0based']]=r['tuning'] if is_true(r.get('tested',True)) else 2
            for depth,seven,tag in [(False,False,'electrode'),(True,False,'depth'),(True,True,'depth_7shank')]:
                save(sites_figure(layout,depths,values,'S05 direction tuning · '+label,depth=depth,seven=seven,tuning='run'),folder,'S05_tuning_'+tag)
        else:
            for k in ('vx','vy','norm'):
                values=np.full(1024,np.nan)
                for r in rows:
                    if is_true(r.get('transmitted',True)):values[r['channel_0based']]=r[k]
                for seven in (False,True):save(sites_figure(layout,depths,values,f'{k} weighting · {label}',seven=seven,limits=(0,25),unit=weight_unit),folder,f'S14_{k}_depth'+('_7shank' if seven else ''))
            for seven in (False,True):
                bins=depth_bins(rows,('vx','vy','norm','event_hz'),seven,stop=profile_stop);tag='_7shank' if seven else '';write(folder/f'depth_profiles{tag}.csv',bins)
                save(profile_figure(bins,label+(' | 7shank' if seven else ''),axial=axial),folder,'S14_depth_profiles'+tag)
        return {}
    for entry in relevant:
        sid=entry['job']['section'];table='anova_channels' if sid=='s05' else 'channel_weights';rows=storage.read_table(Path(entry['folder'])/entry['record']['tables'][table],limit=2000)
        if not rows:continue
        label=entry['record']['summary'].get('recording') or Path(entry['job']['recordings'][0]).stem
        folder=work/SPATIAL_CATEGORIES[0 if sid=='s05' else 1]/'runs'/storage.slug(label)
        for r in rows:
            ch=int(r.get('channel_0based',int(r.get('electrode',1))-1));a,b=map(int,layout['position'][ch]);r.update(channel_0based=ch,shank=b+1,elec_index=a+1,depth_um=depths[ch]['depth_um'])
            if sid=='s05':r['tuning']=float(is_true(r['significant']))
            else:
                for k in ('vx','vy','norm','event_hz'):r[k]=abs(number(r.get(k)))
        runs[sid].append(rows)
        rec = by_path[str(Path(entry['job']['recordings'][0]))]
        daily[sid][rec['date']][rec.get('run_group') or rec['path']].append(rows)
        if not report.get('include_run_exports',False):continue
        if pool:
            from .report_run_pool import dependencies
            inputs=dict(table_sha256=storage.file_digest(Path(entry['folder'])/entry['record']['tables'][table]),
                layout_sha256=storage.file_digest(layout['source']),depth_sha256=depth_sha256,label=label,unit=weight_unit,
                code=dependencies(render_run,sites_figure,profile_figure,depth_bins,number,is_true,Path(__file__).parent/'core/style.py'))
            _,bundle,hit=pool.get(sid.upper()+'_spatial',entry['job']['recordings'][0],inputs,lambda p:render_run(p,sid,rows,label))
            exported.extend(str(p.relative_to(work)) for p in pool.publish(bundle,folder) if p.suffix=='.png')
            send('message',message=f'{sid.upper()} 深度图 {"缓存复用" if hit else "首次生成"}: {label}')
        else:render_run(folder,sid,rows,label)
    # Statistics use per-run observations, never a mean/median of already-rendered pixels.
    if runs['s05']:
        stats,count=channel_statistics(runs['s05'],('tuning',));folder=work/SPATIAL_CATEGORIES[0]/'summary_all_runs'
        data=[dict(channel_0based=ch,shank=int(layout['position'][ch,1])+1,depth_um=depths[ch]['depth_um'],n_runs_selected=len(report['recordings']),n_runs_tested=int(count[ch,0]),tuning_mean=stats['mean'][ch,0],tuning_median=stats['median'][ch,0]) for ch in range(1024)]
        for stat in ('mean','median'):
            for depth,seven,tag in [(False,False,'electrode'),(True,False,'depth'),(True,True,'depth_7shank')]:save(sites_figure(layout,depths,stats[stat][:,0],f'S05 direction tuning - {stat}\n{len(report["recordings"])} selected recordings; valid tests only',depth=depth,seven=seven,tuning=stat if stat=='median' else None,unit='Fraction of tested runs with tuning'),folder,f'S05_{stat}_{tag}')
        write(folder/'channel_tuning_statistics.csv',data)
    if runs['s14']:
        stats,count=channel_statistics(runs['s14'],('vx','vy','norm'));valid=np.r_[stats['mean'].ravel(),stats['median'].ravel()];valid=valid[np.isfinite(valid)];maximum=float(valid.max()) if len(valid) else 0.;upper=float(MaxNLocator(nbins=6).tick_values(0,maximum)[-1]) if maximum>0 else 1.
        folder=work/SPATIAL_CATEGORIES[1]/'summary_all_runs';data=[]
        for ch in range(1024):
            d=dict(channel_0based=ch,shank=int(layout['position'][ch,1])+1,depth_um=depths[ch]['depth_um'],n_runs_observed=int(count[ch,0]))
            for stat in ('mean','median'):
                for j,k in enumerate(('vx','vy','norm')):d[f'{k}_{stat}']=stats[stat][ch,j]
            data.append(d)
        for stat in ('mean','median'):
            for j,k in enumerate(('vx','vy','norm')):
                for depth,seven,tag in [(False,False,'electrode'),(True,False,'depth'),(True,True,'depth_7shank')]:save(sites_figure(layout,depths,stats[stat][:,j],f'{k} weighting - {stat}\n{len(runs["s14"])} recordings; equal recording weight',depth=depth,seven=seven,limits=(0,upper),unit=weight_unit),folder,f'S14_{k}_{stat}_{tag}')
        write(folder/'channel_weight_statistics.csv',data)
        for seven in (False,True):
            profiles=[depth_bins(r,('vx','vy','norm','event_hz'),seven,stop=profile_stop) for r in runs['s14']]
            for stat,fn in [('mean',np.mean),('median',np.median)]:
                bins=[]
                for i in range(len(profiles[0])):
                    d=dict(depth_center_um=profiles[0][i]['depth_center_um'])
                    for k in ('vx','vy','norm','event_hz'):
                        vals=[p[i][k] for p in profiles if np.isfinite(p[i][k])];d[k]=float(fn(vals)) if vals else np.nan
                    bins.append(d)
                tag='_7shank' if seven else '';write(folder/f'depth_profiles_{stat}{tag}.csv',bins);save(profile_figure(bins,f'All selected runs - {stat}'+(' | 7shank' if seven else ''),axial=axial),folder,f'S14_depth_profiles_{stat}{tag}')
    daily_manifest = []
    for sid, dates in sorted(daily.items()):
        tuning = sid == 's05'
        fields = ('tuning',) if tuning else ('vx', 'vy', 'norm')
        category = SPATIAL_CATEGORIES[0 if tuning else 1]
        for date, records_by_run in sorted(dates.items()):
            folder = work/category/'summary_by_day'/date
            folder.mkdir(parents=True, exist_ok=True)
            stats, count, excluded = daily_channel_statistics(records_by_run, fields, trim_weights=not tuning)
            rows = []
            for ch, (site, shank) in enumerate(layout['position']):
                row = dict(channel_0based=ch, shank=int(shank)+1, elec_index=int(site)+1)
                for j, field in enumerate(fields):
                    row.update({f'{field}_n_runs': int(count[ch,j]),
                                f'{field}_mean': stats['mean'][ch,j], f'{field}_median': stats['median'][ch,j]})
                rows.append(row)
            write(folder/'channel_statistics.csv', rows)
            outliers_path = folder/'excluded_outliers.csv'
            with outliers_path.open('w', encoding='utf-8-sig', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['run','channel_0based','metric','value','lower','upper'])
                writer.writeheader()
                writer.writerows(excluded)
            for j, field in enumerate(fields):
                values = stats['mean'][:,j]
                finite = values[np.isfinite(values)]
                upper = max(float(np.max(finite)), np.finfo(float).eps) if finite.size else 1.
                title = f'{date} · {field} · mean over {len(records_by_run)} runs'
                save(sites_figure(layout,depths,values,title,depth=False,
                                  limits=(0,1 if tuning else upper),
                                  unit='Fraction of tested runs with tuning' if tuning else weight_unit),
                     folder,f'{sid.upper()}_{field}_daily_mean')
            daily_manifest.append(dict(section=sid, date=date, runs=sorted(records_by_run), excluded=len(excluded),
                                       folder=str(folder.relative_to(work))))
    meta=dict(daily=daily_manifest, daily_outlier_policy='S14 per date/channel/metric: 1.5 IQR fences with four or more finite run values and positive IQR. S05 uses valid tested binary outcomes. Run means carry equal weight.', electrode_position_range=[1,128],layout_path=layout['source'],layout_sha256=storage.file_digest(layout['source']),depth_path=depth_source,depth_sha256=depth_sha256,shank1_offset_um=None if axial else 106850,depth_coordinate="workbook Y distance from site 1" if axial else "calibrated depth CSV",included_shanks_7shank=list(range(2,9)),s05_successful_recordings=len(runs['s05']),s14_successful_recordings=len(runs['s14']),selected_recordings=len(report['recordings']),pngs=exported,missing_policy='Untransmitted and untested observations excluded; transmitted constant weights retain zero',tuning='Per-run active-channel BH-FDR significance; summary mean/median of 0/1',profiles='100um bins; per-run mean over transmitted channels; binary firing = count(code>0)/(received frames * configured frame duration); cross-run equal weight')
    (work/'spatial_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    send('message',message=f'S05/S14: exported {len(exported)} spatial PNG/PDF figures')
    return meta
