"""Compare all Excel-listed runs using the shared, verified 60s/1s behavior cache."""
from pathlib import Path
import csv,json,math,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_analysis_gui import storage
from data_analysis_gui.report_run_pool import RunPool
from data_analysis_gui.report_behavior import bundle_files,behavior_run_inputs,_compute_run_behavior
from data_analysis_gui.sections.s13_online_behavior_v1 import rolling_bps
from data_analysis_gui.core import style

REPORT=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report/2026-11-11_2026-11-20_52runs_v1')
OUT=REPORT/'11_1117_1120_脑控与离线摇杆_BPS_TTT'
COLORS={'BCI':'#0072B2','Joystick':'#D55E00','Unknown (startup)':'#969696'}
MARKERS={'BCI':'o','Joystick':'s'}

def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def draw(rows,stem,title):
    width=24 if len(rows)>20 else 15
    fig,axes=plt.subplots(2,1,figsize=(width,10),sharex=True)
    fig.subplots_adjust(left=.065,right=.985,bottom=.16,top=.84,hspace=.28)
    fig.suptitle(title,fontsize=18,y=.98)
    legend=[Line2D([],[],color=COLORS['BCI'],marker='o',linestyle='none',markersize=8,label='脑控 BCI'),
            Line2D([],[],color=COLORS['Joystick'],marker='s',linestyle='none',markersize=8,label='离线摇杆 Joystick'),
            Line2D([],[],color='#666666',marker='*',linestyle='none',markersize=10,label='BPS 窗口峰值')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.52,.94),ncol=3,frameon=False,fontsize=11)
    for ax,key,q1,q3,ylabel in [
        (axes[0],'bps_median','bps_q1','bps_q3','BPS（bits/s）'),
        (axes[1],'rt_median_s','rt_q1_s','rt_q3_s','TTT（秒，含 dwell）')]:
        for i,r in enumerate(rows):
            color=COLORS[r['control']]
            if r['status']!='ok':
                ax.axvspan(i-.45,i+.45,color='#EEEEEE',zorder=0)
                ax.text(i,.5,'无数据\n空启动',transform=ax.get_xaxis_transform(),ha='center',va='center',color='#777777',fontsize=9)
                continue
            value=r[key]
            if math.isfinite(value):
                ax.errorbar(i,value,yerr=[[value-r[q1]],[r[q3]-value]],fmt=MARKERS[r['control']],
                            color=color,ecolor=color,capsize=3,markersize=6,elinewidth=1.4,zorder=3)
                ax.annotate(f'{value:.2f}',(i,value),(6,2 if value>0.15 else 10),textcoords='offset points',
                            ha='left' if value>0.15 else 'center',va='bottom',color=color,fontsize=9,
                            bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=.4))
            if key=='bps_median' and math.isfinite(r['bps_peak']):
                ax.scatter(i,r['bps_peak'],marker='*',s=68,color=color,alpha=.65,zorder=2)
        for i in range(1,len(rows)):
            if rows[i]['date']!=rows[i-1]['date']:ax.axvline(i-.5,color='#777777',lw=1,ls='--',alpha=.6)
        ax.set_ylabel(ylabel,fontsize=14);ax.set_xlim(-.7,len(rows)-.3);ax.set_ylim(bottom=0)
        low,high=ax.get_ylim();ax.set_ylim(0,high*1.13)
        ax.grid(axis='y',alpha=.22);ax.spines[['top','right']].set_visible(False)
    axes[0].set_title('60 s 滚动 BPS：中位数与 IQR；星号为窗口峰值',fontsize=18,pad=12)
    axes[1].set_title('成功 trial 的 TTT：中位数与 IQR',fontsize=18,pad=12)
    axes[1].set_xticks(range(len(rows)),[r['date'][5:]+'\nrun'+r['run'] for r in rows])
    axes[1].set_xlabel('日期 / run（顺序与 config Excel 一致）',fontsize=14,labelpad=10)
    for label,r in zip(axes[1].get_xticklabels(),rows):label.set_color(COLORS[r['control']])
    fig.text(.5,.03,'BPS：60 s 有效尝试时长窗口，step 1 s；长 run 从 60 s 开始，仅整 run 不足 60 s 时按实际时长折算。\n'
             'TTT：每个成功 trial 仅统计一次。误差线为 IQR，非置信区间；不同 Grid、dwell、辅助或模型设置仍需分组比较。',
             ha='center',va='bottom',fontsize=11)
    for ext in ('png','pdf','svg'):fig.savefig(OUT/(stem+'.'+ext),dpi=220,facecolor='white')
    plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    config_path=REPORT/'00_Config_1117_1120/config_1117_1120.json'
    source=json.loads(config_path.read_text(encoding='utf-8'));pool=RunPool(REPORT.parent/'individual_runs')
    summaries=[];all_windows=[];all_trials=[];sources=[]
    for excel_row,r in enumerate(source['runs'],2):
        folder=Path(r['folder']);cfg_path=folder/'task_config.json'
        row=dict(date=r['date'],run=r['run'],excel_task_sheet_row=excel_row,control=r['control'],status='ok',
                 run_folder=str(folder),grid=r['grid'],dwell_s=r['dwell_s'],nearby_assist=r['nearby_assist'],
                 configured_model=r['configured_model'],model_ref=r['model_ref'])
        files=bundle_files(folder,'grid')
        if not cfg_path.is_file() or not files:
            row.update(status='missing_config_or_trials',**{k:math.nan for k in ['bps_median','bps_q1','bps_q3','bps_peak','rt_median_s','rt_q1_s','rt_q3_s']})
            summaries.append(row);continue
        cfg=json.loads(cfg_path.read_text(encoding='utf-8-sig'));control=cfg['CONTROL']
        row['control']='BCI' if control['BCI_ENABLED'] else 'Joystick'
        assert row['control']==r['control']
        row.update(preview_next_target=cfg['GRID_TASK']['PREVIEW_NEXT_TARGET'],
                   bci_velocity_scale=control['BCI_VELOCITY_SCALE'],bci_smooth_alpha=control['BCI_VELOCITY_SMOOTH_ALPHA'],
                   bci_velocity_deadzone=control['BCI_VELOCITY_DEADZONE'],bci_max_dt_sec=control['BCI_MAX_DT_SEC'])
        files['task_config']=cfg_path;date=r['date'].replace('-','');label=f'{date[4:6]}-{date[6:]} {folder.name}'
        inputs=behavior_run_inputs(files,'grid',label)
        data,bundle,hit=pool.get('BPS_TTT_data',str(folder),inputs,lambda p:_compute_run_behavior(files,'grid',label,date,folder))
        entry=dict(run_key=str(folder),date=date,session=label)
        stats,=rolling_bps.run_summary(data['windows'],'grid',runs=[entry],trials_by_run={str(folder):data['trials']})
        row.update({k:v for k,v in stats.items() if k not in ('date','run','run_key')})
        row.update(trials=len(data['trials']),successes=sum(t['outcome']=='acquired' for t in data['trials']),cache_hit=hit)
        for target,items in ((all_windows,data['windows']),(all_trials,data['trials'])):
            target.extend(dict(v,config_date=r['date'],config_run=r['run'],control=row['control']) for v in items)
        sources.append(dict(date=r['date'],run=r['run'],pool_folder=str(bundle),inputs=inputs))
        summaries.append(row)
        print(r['date'],r['run'],row['control'],'cache' if hit else 'computed',f'BPS={row["bps_median"]:.3g}, TTT={row["rt_median_s"]:.3g}',flush=True)
    assert len(summaries)==35 and sum(r['status']=='ok' for r in summaries)==34
    write_csv(OUT/'run_summary_with_config.csv',summaries)
    write_csv(OUT/'rolling_bps_windows_with_config.csv',all_windows)
    write_csv(OUT/'trials_with_config.csv',all_trials)
    style.apply();plt.rcParams['font.family']=['Microsoft YaHei','Arial','DejaVu Sans']
    draw(summaries,'1117_1120_BCI_Joystick_BPS_TTT','1117–1120 Grid：脑控与离线摇杆逐 run 对照')
    draw([r for r in summaries if r['date']=='2026-11-20'],'1120_BCI_Joystick_BPS_TTT','1120 Grid：脑控与离线摇杆逐 run 对照')
    manifest=dict(config=str(config_path),excel=str(config_path.with_name('config_1117_1120_full.xlsx')),
                  method=rolling_bps.METHOD,sources=sources,run_count=35,valid_runs=34,
                  counts={k:sum(r['control']==k and r['status']=='ok' for r in summaries) for k in ('BCI','Joystick')},
                  cache=pool.manifest(),speed_code_reference='E:/AitBCI/CursorTasks_namedpipe/runtime/input_control.py')
    (OUT/'analysis_manifest.json').write_text(json.dumps(storage.jsonable(manifest),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(OUT),counts=manifest['counts'],cache_hits=pool.manifest()['hits'],cache_misses=pool.manifest()['misses']),ensure_ascii=False))

if __name__=='__main__':main()
