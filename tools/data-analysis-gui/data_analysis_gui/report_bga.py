"""Daily chip status from the first run with usable XDF neural data."""
from pathlib import Path
import json,re
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from . import config as C,storage
from .core.catalog import day_folder,RUN_PREFIX
from .core.channels import read_layout
from .core.xdf_io import load_streams,electrode_ids
from .core.style import finish_figure

SHANK_COLORS=('#FF5722','#8BC34A','#00BCD4','#F44336','#9C27B0','#FF9800','#4CAF50','#2196F3')

def first_run(day):
    candidates=[p for p in Path(day).iterdir() if p.is_dir() and RUN_PREFIX.match(p.name)]
    return min(candidates,key=lambda p:(int(RUN_PREFIX.match(p.name)['number']),RUN_PREFIX.match(p.name)['part'] or '',p.name)) if candidates else None


def first_usable_run(day,pool=None):
    """Skip missing/broken recordings, retaining actual run identity and cache reuse."""
    candidates=sorted((p for p in Path(day).iterdir() if p.is_dir() and RUN_PREFIX.match(p.name)),
        key=lambda p:(int(RUN_PREFIX.match(p.name)['number']),RUN_PREFIX.match(p.name)['part'] or '',p.name))
    skipped=[];first_unavailable=None
    def load(folder):
        if pool:
            def build(staged):
                values,info=first_run_status(folder)
                return dict(values=values.tolist(),info=info)
            data,bundle,_=pool.get('BGA_data',folder or day,bga_data_inputs(folder),build)
            return np.asarray(data['values'],float),data['info'],bundle
        values,info=first_run_status(folder)
        return values,info,None
    for folder in candidates:
        if not any(folder.rglob('*.xdf')):
            skipped.append(dict(run_folder=str(folder),status='no_xdf'));continue
        values,info,bundle=load(folder)
        if np.isfinite(values).any():return folder,values,info,bundle,skipped
        skipped.append(dict(run_folder=str(folder),status=info['status'],notes=info['notes']))
        if first_unavailable is None:first_unavailable=(folder,values,info,bundle)
    if first_unavailable is not None:return (*first_unavailable,skipped)
    folder=candidates[0] if candidates else None
    values,info,bundle=load(folder)
    return folder,values,info,bundle,skipped

def first_run_status(folder):
    values=np.full(1024,np.nan);sources=[]
    files=sorted(Path(folder).rglob('*.xdf')) if folder else []
    status='no_xdf' if not files else 'ok';notes=[]
    for path in files:
        try:
            streams,_=load_streams(path)
            if 'spike_counts' not in streams:
                notes.append(f'{path.name}: no spike_counts');continue
            stream=streams['spike_counts'];counts=np.asarray(stream['time_series']);ids=electrode_ids(stream)
            if counts.ndim!=2 or counts.shape[1]!=len(ids) or not len(counts) or not np.isfinite(counts).all():raise ValueError('invalid spike matrix')
            if ids.max()>1024 or counts.min()<0 or counts.max()>3 or not np.array_equal(counts,np.rint(counts)):raise ValueError('invalid channel IDs or non-2-bit codes')
            values[ids-1]=np.fmax(values[ids-1],counts.max(axis=0))
            sources.append(dict(path=str(path),sha256=storage.file_digest(path),frames=len(counts),channels=len(ids)))
        except Exception as exc:notes.append(f'{path.name}: {exc}')
    if files and not sources:status='unavailable'
    elif sources and notes:status='partial_xdf_coverage'
    return values,dict(status=status,run_folder=str(folder) if folder else None,sources=sources,notes=notes)

def bga_figure(layout,values,date,run,status):
    # Exact typography/layout from bga_shanks/restyle_all.py. This requested
    # reference style takes precedence over the general 18/14/11 export policy.
    fig=Figure(figsize=(16,13.6));ax=fig.subplots()
    fig._preserve_schematic_fonts=True
    fig._schematic_legacy_finished=True
    coords=layout['bga_position'];x=coords[:,1]+(coords[:,1]>=20);y=coords[:,0]+(coords[:,0]>=16)
    shanks=layout['position'][:,1];observed=np.isfinite(values)
    face=['#BFC4CA' if not observed[ch] else '#000000' if values[ch]==0 else SHANK_COLORS[shanks[ch]] for ch in range(1024)]
    ax.scatter(x,y,s=245,c=face,edgecolors='none',zorder=3)
    for ch in range(1024):
        dark=face[ch]=='#000000' or (face[ch]!='#BFC4CA' and shanks[ch] in (3,4))
        ax.text(x[ch],y[ch],str(ch),ha='center',va='center',fontsize=5.8,
                color='white' if dark else '#17212B',zorder=4)
    ax.set(xlim=(.3,39.7),ylim=(.25,31.75),xlabel='BGA Col',ylabel='BGA Row',aspect='equal')
    ax.set_xticks(np.arange(1,39)+(np.arange(1,39)>=20),np.arange(1,39))
    ax.set_yticks(np.arange(1,31)+(np.arange(1,31)>=16),np.arange(1,31))
    ax.grid(color='#ECECEC',linewidth=.65);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_linewidth(.6)
    ax.xaxis.label.set_fontsize(20);ax.yaxis.label.set_fontsize(20)
    ax.xaxis.label.set_color('black');ax.yaxis.label.set_color('black');ax.tick_params(labelsize=9,colors='black')
    match=RUN_PREFIX.match(Path(run).name)
    run_id=(match['number']+('-'+match['part'] if match['part'] else '')) if match else run
    fig.suptitle(f'Tianshu 3.0 - BGA Package - Run {run_id}',fontsize=24,fontweight='bold',y=.97,color='#17212B')
    handles=[Line2D([],[],ls='',marker='o',markersize=5,color=color,label=f'Shank {i+1}') for i,color in enumerate(SHANK_COLORS)]
    missing_label='Not transmitted' if status in ('ok','partial_xdf_coverage') else 'Data unavailable'
    handles += [Line2D([],[],ls='',marker='o',markersize=5,color=color,label=label) for color,label in (
        ('#000000',f'All codes = 0 ({int((observed&(values==0)).sum())})'),
        ('#BFC4CA',f'{missing_label} ({int((~observed).sum())})'))]
    fig.legend(handles=handles,ncol=5,frameon=False,loc='lower center',bbox_to_anchor=(.54,.015),handletextpad=.4,columnspacing=1.4,fontsize=10,labelcolor='#17212B')
    if status not in ('ok','partial_xdf_coverage'):
        fig.text(.5,.003,'No usable XDF neural data found in any run of this day.',ha='center',fontsize=8,color='#17212B')
    fig.subplots_adjust(left=.075,right=.985,top=.91,bottom=.11)
    from matplotlib.text import Text
    for text in fig.findobj(Text):text.set_fontfamily('DejaVu Sans')
    return fig

def bga_data_inputs(folder):
    from .report_run_pool import dependencies,file_inputs
    inventory={str(p.relative_to(folder)):p for p in sorted(folder.rglob('*.xdf'))} if folder else {}
    return dict(files=file_inputs(inventory),code=dependencies(first_run_status,Path(__file__).parent/'core/xdf_io.py'))


def render_daily_bga(work,report,font_percent=100,send=lambda *a,**k:None,*,pool=None):
    layout=read_layout(report.get('layout_path') or C.DEFAULT_LAYOUT_XLSX)
    if layout is None or layout.get('bga_position') is None:return dict(days=[],skipped='No BGA coordinates in workbook')
    excluded={d.replace('-','') for d in report.get('analysis_excluded_dates',[])}
    paths=[r['path'] for r in report.get('recordings',report.get('selection',[]))]
    paths+=report.get('offline_behavior_folders',[])+report.get('online_behavior_folders',[])
    paths += [r['folder'] for key in ('offline_behavior','online_behavior') for r in report.get(key,{}).get('runs',[])]
    days={day_folder(Path(p)) for p in paths};days={p for p in days if p and p.name.replace('-','') not in excluded}
    out=Path(work)/'02_通道稳定性分析'/'BGA_daily_first_run';out.mkdir(parents=True,exist_ok=True)
    result=dict(layout_path=layout['source'],layout_sha256=storage.file_digest(layout['source']),style_reference=str(C.BGA_SHANK_PROJECTION.parent/'bga_shanks/BGA_20261111_run006_status.png'),color_policy='Any nonzero code: solid Shank color; observed all-zero: black; untransmitted: #BFC4CA. No colored outlines.',policy='First run in run-ID order with usable XDF spike data in each selected day. Skip runs without XDF or readable spike data; record selected run and skipped candidates. Entire received spike stream, independent of task-marker availability.',excluded_dates=sorted(excluded),days=[])
    for day in sorted(days):
        folder,values,info,data_folder,skipped=first_usable_run(day,pool)
        if pool:from .report_run_pool import dependencies
        date=day.name;run=folder.name if folder else 'No run folder';stem=out/f'BGA_{date}_first_run'
        def draw(staged):
            target=staged/stem.name
            fig=bga_figure(layout,values,date,run,info['status']);finish_figure(fig,font_percent)
            for ext in ('png','pdf','svg'):fig.savefig(target.with_suffix('.'+ext),dpi=220 if ext=='png' else None)
            fig.clear()
            data=[dict(channel_0based=ch,shank=int(layout['position'][ch,1])+1,bga_row=int(layout['bga_position'][ch,0]),bga_col=int(layout['bga_position'][ch,1]),observed=bool(np.isfinite(values[ch])),max_code=int(values[ch]) if np.isfinite(values[ch]) else '') for ch in range(1024)]
            storage.write_csv(target.with_suffix('.csv'),data,[result['policy'],f'first run: {folder}',f'status: {info["status"]}'])
            return {}
        if pool:
            _,bundle,hit=pool.get('BGA_figures',folder or day,dict(data=data_folder.name,layout_sha256=result['layout_sha256'],date=date,run=run,colors=SHANK_COLORS,code=dependencies(bga_figure)),draw)
            pool.publish(bundle,out)
            send('message',message=f'BGA {"缓存复用" if hit else "首次生成"}: {date} {run}')
        else:draw(out)
        result['days'].append(dict(date=date,**info,skipped_candidates=skipped,transmitted_channels=int(np.isfinite(values).sum()),files={ext:str(stem.with_suffix('.'+ext).relative_to(work)) for ext in ('png','pdf','svg','csv')}))
        send('message',message=f'BGA first run: {date} {run}, {info["status"]}')
    (Path(work)/'bga_daily_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
