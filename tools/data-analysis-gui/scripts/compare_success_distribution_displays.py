"""Compare four displays using the same 589 block rates from 25 runs."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

OUT=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report/2026-11-04_2026-11-11_25runs_v1/additional_figures')
source=OUT/'Electrode_Activity_and_Direction_Tuning_Average_Full_Distribution.csv'
rows=list(csv.DictReader(source.open(encoding='utf-8-sig')))
groups=[np.array([float(r['success_pct']) for r in rows if int(r['session_index'])==i]) for i in range(25)]
assert sum(map(len,groups))==589 and all(len(g)>0 for g in groups)
plt.rcParams.update({'font.family':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,
    'font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(19,10.5),layout='constrained')
fig.suptitle('25 个 run 的成功率分布：四种展示方式对比',fontsize=21)
x=np.arange(1,26);mean=np.array([g.mean() for g in groups]);sd=np.array([g.std(ddof=0) for g in groups])
orange='#bc5026';blue='#2677a7';green='#187d6d'
for ax in axs.flat:
    ax.set(xlim=(.3,25.7),xticks=x,xlabel='Run 编号（与原图从左到右顺序一致）',ylabel='Block 成功率（%）')
    ax.tick_params(axis='x',labelsize=8)
    ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
    ax.spines['left'].set_color('#aaaaaa');ax.spines['bottom'].set_color('#aaaaaa')
ax=axs[0,0]
ax.errorbar(x,mean,yerr=sd,fmt='D',color=orange,ms=4,capsize=3,lw=1)
ax.set(title='① 高斯拟合：均值 ± 1 个标准差',ylim=(-4,124),yticks=np.arange(0,121,20))
ax.axhline(100,color='#888888',lw=.8,ls='--')
ax=axs[0,1]
bp=ax.boxplot(groups,positions=x,widths=.55,whis=(0,100),showfliers=False,patch_artist=True,
             boxprops=dict(facecolor='#d0e4ee',edgecolor=blue,linewidth=1),
             medianprops=dict(color=blue,linewidth=1.7),whiskerprops=dict(color=blue),capprops=dict(color=blue))
ax.plot(x,mean,'D',color=orange,ms=3.5)
ax.set(title='② 箱线图：中位数 / 25–75%区间 / 最小–最大',ylim=(-4,108),yticks=np.arange(0,101,20))
ax.legend(handles=[Line2D([],[],marker='D',color=orange,ls='',label='Block 均值')],loc='lower right',frameon=False)
ax=axs[1,0]
grid=np.linspace(0,100,401)
for i,g in enumerate(groups,1):
    if np.ptp(g)<1e-10:
        ax.hlines(g[0],i-.32,i+.32,color=green,lw=2)
    else:
        density=gaussian_kde(g,bw_method=.3)(grid)
        half=.38*density/density.max()
        ax.fill_betweenx(grid,i-half,i+half,facecolor='#a8d4c9',edgecolor=green,lw=.6)
    ax.hlines(np.median(g),i-.16,i+.16,color=green,lw=1.3)
ax.set(title='③ 小提琴图：平滑密度（限制在 0–100%）',ylim=(-4,108),yticks=np.arange(0,101,20))
ax=axs[1,1]
# Eleven bins: ten 10-percentage-point intervals and a separate exact-100% bin.
hist=np.zeros((11,25))
for j,g in enumerate(groups):
    bins=np.where(g==100,10,np.floor(g/10).astype(int))
    hist[:,j]=np.bincount(bins,minlength=11)/len(g)*100
np.testing.assert_allclose(hist.sum(axis=0),100)
im=ax.imshow(hist,origin='lower',aspect='auto',extent=(.5,25.5,-.5,10.5),vmin=0,vmax=100,cmap='Blues',interpolation='nearest')
ax.set(title='④ 分布热图：颜色表示该成功率区间的 block 占比',ylabel='Block 成功率区间',
       yticks=[0,2,4,6,8,10],yticklabels=['0–<10%','20–<30%','40–<50%','60–<70%','80–<90%','100%'])
ax.grid(False)
fig.colorbar(im,ax=ax,pad=.015,fraction=.035,label='Block 占比（%）')
target=OUT/'Success_Rate_Distribution_Display_Comparison'
fig.savefig(target.with_suffix('.png'),dpi=220,facecolor='white')
fig.savefig(target.with_suffix('.pdf'),facecolor='white')
stats=[]
for i,g in enumerate(groups):
    session=next(r['session'] for r in rows if int(r['session_index'])==i)
    q=np.quantile(g,[.1,.25,.5,.75,.9])
    stats.append(dict(run_number=i+1,session=session,n_blocks=len(g),mean=float(g.mean()),sigma=float(g.std(ddof=0)),
                      minimum=float(g.min()),p10=q[0],q1=q[1],median=q[2],q3=q[3],p90=q[4],maximum=float(g.max()),
                      perfect_block_pct=float(np.mean(g==100)*100)))
with target.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
target.with_suffix('.json').write_text(json.dumps(dict(source=str(source),blocks=589,runs=25,
    constant_runs=[i+1 for i,g in enumerate(groups) if np.ptp(g)==0],
    notes=['All four panels use exactly the same block-level success percentages; no individual block points are plotted.',
           'Gaussian sigma is MLE with ddof=0; box whiskers show actual minimum and maximum, not 1.5 IQR.',
           'Violin density uses scipy gaussian_kde bandwidth factor 0.3, evaluated only on 0..100, with equal peak width per run. Constant runs shown as a horizontal line. Smoothing can suggest unobserved intermediate values and is not a distributional test.',
           'Heatmap uses ten half-open bins of 10 percentage points plus an exact-100 bin. Each column totals 100%; sample sizes differ across runs.']
),ensure_ascii=False,indent=2),encoding='utf-8')
print(target.with_suffix('.png'))
print('Constant runs:',[i+1 for i,g in enumerate(groups) if np.ptp(g)==0])
