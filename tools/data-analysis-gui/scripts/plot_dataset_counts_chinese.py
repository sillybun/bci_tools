"""Chinese session-count figure, using counts explicitly supplied by the user."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

OUT = Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report/2026-11-04_2026-11-11_25runs_v1/additional_figures')
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':['Microsoft YaHei','SimHei','DejaVu Sans'],
                     'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none',
                     'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
counts=[7,6,2,2,8]
assert sum(counts)==25 and sum(counts[:3])==15 and sum(counts[3:])==10
labels=['离线采集','在线脑控','手部束缚\n在线脑控','离线采集','在线脑控']
colors=['#858585','#c44e22','#823497','#2668ad','#13816b']
x=[0,1,2,3.6,4.6]
fig,ax=plt.subplots(figsize=(10.8,6.2))
fig.subplots_adjust(left=.09,right=.97,bottom=.26,top=.78)
ax.set_axisbelow(True)
ax.grid(axis='y',color='#e8e8e8',lw=.8)
ax.bar(x,counts,width=.64,color=colors,zorder=3)
for pos,value in zip(x,counts):
    ax.text(pos,value+.18,f'{value} 组',ha='center',va='bottom',fontsize=14,fontweight='bold')
ax.set(xticks=x,xticklabels=labels,ylabel='数据组数（组）',ylim=(0,9.3),xlim=(-.65,5.25))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.spines['left'].set_color('#777777');ax.spines['bottom'].set_color('#777777')
ax.tick_params(axis='x',length=0,pad=10)
ax.axvline(2.8,color='#cccccc',ls='--',lw=.8)
for start,end,center,text in [(-.32,2.32,1,'八方向中心向外任务：15 组（60%）'),
                              (3.28,4.92,4.1,'网格任务：10 组（40%）')]:
    ax.plot([start,end],[-.24,-.24],transform=ax.get_xaxis_transform(),clip_on=False,color='#777777',lw=1)
    ax.text(center,-.31,text,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=12)
fig.text(.5,.94,'25 组数据采集条件统计',ha='center',fontsize=21,fontweight='bold')
fig.text(.5,.865,'离线采集 9 组  ·  在线脑控 16 组（含手部束缚 2 组）',ha='center',color='#555555',fontsize=12)
target=OUT/'Dataset_Counts_Chinese'
fig.savefig(target.with_suffix('.png'),dpi=300,facecolor='white')
fig.savefig(target.with_suffix('.pdf'),facecolor='white')
with target.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['任务','采集条件','组数','占全部比例'])
    for task,condition,value in zip(['八方向']*3+['网格']*2,labels,counts):
        w.writerow([task,condition.replace('\n',''),value,f'{value/25:.0%}'])
print(target.with_suffix('.png'))
