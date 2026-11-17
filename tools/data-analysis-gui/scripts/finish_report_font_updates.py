"""Resolve legacy montage/annex figures and preserve report-specific overlays."""
from pathlib import Path
import inspect
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from matplotlib.figure import Figure
from update_report_font_sizes import ROOT,AUDIT,read,typed,save,is_schematic
from redraw_report_statistics_fonts import data,meta,statistical_figure,supplementary
from data_analysis_gui.sections.s13_online_behavior_v1 import rolling_bps


def montages(p):
    report=next(q for q in p.parents if (q/'report.json').exists());m=meta(str(report/'report.json'))
    sources=[]
    for item in m['sources']:
        if item['section']!='s13' or 'CO8' not in item['category']:continue
        folder=Path(item['folder']);r=meta(str(folder/'result.json'))
        sources.append((folder,r))
    page=int(p.stem.rsplit('_',1)[1])-1;group=sources[page*4:(page+1)*4]
    assert group
    fig=Figure(figsize=(24,13.44),layout='constrained');grid=fig.add_gridspec(4,2)
    for i,(folder,r) in enumerate(group):
        rows=data(str(folder/r['tables']['block_metrics']));x=np.arange(len(rows));row,col=divmod(i,2)
        upper=fig.add_subplot(grid[row*2,col]);lower=fig.add_subplot(grid[row*2+1,col],sharex=upper)
        def number(value):
            try:return float(value)
            except (ValueError,TypeError):return np.nan
        upper.plot(x,[number(r['success_pct']) for r in rows],'o-',color='#286e9e');upper.set(ylim=(0,105),ylabel='Success rate (%)')
        upper.set_title(r['summary'].get('recording',folder.name)+' · CO8 Behavior');upper.tick_params(labelbottom=False)
        med=np.array([number(r['rt_median_s']) for r in rows]);q1=np.array([number(r['rt_q1_s']) for r in rows]);q3=np.array([number(r['rt_q3_s']) for r in rows])
        lower.errorbar(x,med,yerr=[med-q1,q3-med],fmt='o-',capsize=3,color='#a65b24',label='Median and IQR')
        lower.set(ylabel='Time to target (s)',xlabel='Task block',ylim=(0,None));lower.legend(frameon=False)
        indices=sorted(set(range(0,len(rows),max(1,len(rows)//12)))|{len(rows)-1})
        lower.set_xticks(indices,[str(rows[j]['block']) for j in indices])
        for ax in [upper,lower]:ax.grid(axis='y',alpha=.23)
    return fig,report/'report.json'


def comparison(p):
    script=Path(__file__).with_name('compare_success_distribution_displays.py');code=script.read_text(encoding='utf-8')
    stage=AUDIT/'comparison';stage.mkdir(exist_ok=True)
    a=code.index('OUT=Path(');b=code.index('\n',a)
    code=code[:a]+f'OUT=Path({str(stage)!r})'+code[b:]
    source=p.parent/'Electrode_Activity_and_Direction_Tuning_Average_Full_Distribution.csv'
    a=code.index('source=OUT/');b=code.index('\n',a)
    code=code[:a]+f'source=Path({str(source)!r})'+code[b:]
    namespace={'__name__':'_font_redraw'};exec(compile(code,str(script),'exec'),namespace)
    return namespace['fig'],source


def remainders():
    paths=json.loads((AUDIT/'unknown_figures.json').read_text(encoding='utf-8'))
    paths += [r['path'] for r in json.loads((AUDIT/'statistics_errors.json').read_text(encoding='utf-8'))]
    errors=[]
    for value in dict.fromkeys(paths):
        p=Path(value)
        try:
            if p.name.startswith('CO8_Behavior_Overview_'):result=montages(p)
            elif p.name=='Success_Rate_Distribution_Display_Comparison.png':result=comparison(p)
            elif p.name=='Run004_Run005_BPS_TTT_summary.png':
                source=p.with_suffix('.csv');rows=typed(read(source));_,fig=rolling_bps.run_figure(rows,'grid');result=fig,source
            elif p.name=='Rolling_BPS_60s.png' and not (p.parent/'rolling_bps_windows.csv').exists():
                fig=rolling_bps.rolling_figure([],'centerout' if 'CO8' in str(p) else 'grid',p.parent.name+' · Rolling BPS')
                for text in fig.axes[0].texts:
                    if 'N/A' in text.get_text():text.set_text('N/A · less than 60 verified active seconds in this historical version')
                result=fig,p.parent/'trials.csv'
            elif p.name=='TTT_distribution.png' and not (p.parent/'trials.csv').exists():
                source=p.parent/'trials_analyzed.csv';rows=typed(read(source));v=[r['rt_s'] for r in rows if r['outcome']=='acquired' and np.isfinite(r['rt_s'])]
                fig=Figure(figsize=(9,5),layout='constrained');ax=fig.subplots();ax.hist(v,bins=min(40,max(5,int(np.sqrt(len(v))))),color='#c25e00');ax.axvline(np.median(v),color='#333',ls='--')
                ax.set(xlabel='Time to target (s; includes dwell)',ylabel='Successful trials',title=p.parent.name+'\nTTT distribution');result=fig,source
            else:result=statistical_figure(p) or supplementary(p)
            if result:save(result[0],p,result[1]);print('DONE',p.name,flush=True)
            else:errors.append(dict(path=str(p),error='No saved-data renderer'))
        except Exception as exc:errors.append(dict(path=str(p),error=repr(exc)))
    (AUDIT/'remainder_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Remainders complete; errors',len(errors),flush=True)


def overlays():
    paths=json.loads((AUDIT/'customized_vector_targets.json').read_text(encoding='utf-8'));errors=[]
    for value in paths:
        p=Path(value)
        if is_schematic(p):continue
        try:
            result=supplementary(p)
            if result:save(result[0],p,result[1]);print('OVERLAY',p.name,flush=True)
            else:errors.append(dict(path=str(p),error='No saved overlay renderer'))
        except Exception as exc:errors.append(dict(path=str(p),error=repr(exc)))
    (AUDIT/'overlay_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Overlays complete; errors',len(errors),flush=True)


if __name__=='__main__':
    overlays() if '--overlays' in sys.argv else remainders()
