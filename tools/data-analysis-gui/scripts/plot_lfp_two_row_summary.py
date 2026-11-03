"""Three analysis panels in the first row; eight trajectory examples below.

Uses cached evidence exactly; single-direction AUC and eight-way permutation
significance are explicitly kept as different analyses.
"""
from pathlib import Path
import json
import hashlib
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'output/lfp_20261111_run009'
DEST=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report/2026-11-04_2026-11-11_25runs_v1/additional_figures')


def main():
    global DATA
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=DATA)
    parser.add_argument('--signal',choices=['LFP','Spike'],default='LFP')
    args=parser.parse_args();DATA=args.data_root
    files=[DATA/'one_vs_rest/one_vs_rest_results.json',DATA/'analysis_results.json',
           DATA/'trajectory_r2/trajectory_predictions.npz',DATA/'trajectory_r2/trajectory_results.json',
           DATA/'trajectory_r2/example_selection.json']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    binary=json.loads(files[0].read_text(encoding='utf-8'))
    direction=json.loads(files[1].read_text(encoding='utf-8'))
    trajectory=np.load(files[2]);trajectory_metrics=json.loads(files[3].read_text(encoding='utf-8'))
    examples=json.loads(files[4].read_text(encoding='utf-8'))['trial_ids']
    eightway=direction.get('multiclass',direction.get('nested_band_selection'))
    cm=np.array(eightway['confusion_matrix']);fraction=cm/cm.sum(1,keepdims=True)
    assert cm.sum()==binary['trials']
    np.testing.assert_allclose(np.diag(fraction).mean(),eightway['balanced_accuracy'])
    if 'multiclass' in direction:
        observed=eightway['balanced_accuracy']*100;null=np.array(eightway['null_scores'])*100;p_value=eightway['p_value']
    else:
        observed=max(direction['band_accuracy'])*100;null=np.array(direction['band_comparison']['max_band_null'])*100;p_value=direction['band_comparison']['max_statistic_p_value']
    assert (1+sum(null>=observed))/(len(null)+1)==p_value
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':12,'axes.spines.top':False,
                          'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(16,11.5),layout='constrained')
    fig.get_layout_engine().set(w_pad=.025,h_pad=.025,wspace=.015,hspace=.015)
    rows=fig.add_gridspec(2,1,height_ratios=[1,1.56],hspace=.025)
    top=rows[0].subgridspec(1,3,width_ratios=[1.1,1,.3],wspace=.025)
    fig.suptitle('1111 CO8 run-001 Spike' if args.signal=='Spike' else '1111 CO8 run-009 LFP',fontsize=23)

    # A: eight one-vs-rest binary tests, not eight-class accuracy.
    ax=fig.add_subplot(top[0,0]);result=binary['results'];x=np.arange(8)
    means=np.array([r['ROC_AUC'] for r in result]);lo=np.array([r['AUC_CI95_low'] for r in result]);hi=np.array([r['AUC_CI95_high'] for r in result])
    ax.bar(x,means,color=['#e07b19' if r['significant_FWER'] else '#2176ae' for r in result],width=.78)
    ax.errorbar(x,means,yerr=np.array([means-lo,hi-means]),fmt='none',color='#333',capsize=4,lw=1.5)
    ax.axhline(.5,color='#555',ls='--',lw=1.5,label='Chance AUC = 0.5')
    for i,r in enumerate(result):
        ax.text(i,hi[i]+.025,f'{r["ROC_AUC"]:.2f}'+('*' if r['significant_FWER'] else ''),ha='center',va='bottom',fontsize=13)
    ax.set(xticks=x,xticklabels=np.arange(8)*45,ylim=(0,1),xlabel='Target direction (deg)',ylabel='Held-out ROC AUC',
           title='A  Single-direction Decoding')
    ax.legend(loc='upper left',frameon=False,fontsize=11)
    if args.signal=='Spike':
        ax.set_ylim(0,1.10)
        ax.set_yticks(np.linspace(0,1,6))

    # B: preserve the requested existing eight-way confusion matrix.
    ax=fig.add_subplot(top[0,1]);ax.imshow(fraction,vmin=0,vmax=max(.4,float(fraction.max())),cmap='Blues',aspect='equal')
    for i in range(8):
        for j in range(8):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=11,color='white' if fraction[i,j]>.22 else '#222')
    ax.set(xticks=range(8),xticklabels=np.arange(8)*45,yticks=range(8),yticklabels=np.arange(8)*45,
        xlabel='Predicted direction (deg)',ylabel='Target direction (deg)',
        title='B  Confusion Matrix')
    ax.tick_params(labelsize=10)

    # C: preserve the original vertical histogram; narrow its allotted width only.
    ax=fig.add_subplot(top[0,2])
    ax.hist(null,bins=20,color='#aaa',edgecolor='white',linewidth=.4)
    ax.axvline(observed,color='#c44e22',lw=1.8)
    ax.text(.97,.98,f'{observed:.1f}%',transform=ax.transAxes,ha='right',va='top',color='#c44e22',fontsize=11)
    ax.set(xlim=(9,29),xticks=[10,20,27],yticks=[0,20,40,60],xlabel='Balanced\naccuracy (%)',ylabel='Permutations',
        title='C  Significance')
    ax.text(.03,.98,f'p = {p_value:.3f}',transform=ax.transAxes,ha='left',va='top',fontsize=10)
    if args.signal=='Spike':
        from matplotlib.ticker import MaxNLocator
        ax.set_xlim(min(9,float(null.min())-1),max(29,float(observed)+2,float(null.max())+1))
        ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=10)
    ax.title.set_fontsize(12);ax.xaxis.label.set_fontsize(10);ax.yaxis.label.set_fontsize(10)

    # Second composite row, preserving the original 2-by-4 trajectory arrangement.
    bottom=rows[1].subgridspec(4,4,height_ratios=[.10,1,1,.08],wspace=.02,hspace=.025)
    header=fig.add_subplot(bottom[0,:]);header.axis('off')
    r2=trajectory_metrics['scores']['forecast_ridge']['position_r2_mean_xy']
    header.text(.5,.82,'D  Trajectory Prediction',ha='center',va='top',fontsize=17)
    axes=[]
    for d,tid in enumerate(examples):
        ax=fig.add_subplot(bottom[1+d//4,d%4]);axes.append(ax)
        idx=np.flatnonzero(trajectory['trial']==tid)
        actual=np.vstack([trajectory['initial_anchor'][idx[0]],trajectory['truth'][idx]])
        pred=np.vstack([trajectory['initial_anchor'][idx[0]],trajectory['forecast_ridge'][idx]])
        ax.plot(actual[:,0],actual[:,1],'-o',color='#333',lw=1.5,ms=3,label='True trajectory')
        ax.plot(pred[:,0],pred[:,1],'-o',color='#2176ae',lw=1.5,ms=3,label=f'{args.signal} prediction')
        ax.scatter(*trajectory['initial_anchor'][idx[0]],s=30,marker='s',color='#169582',label='Known start',zorder=4)
        ax.axhline(0,color='#ddd',lw=.6);ax.axvline(0,color='#ddd',lw=.6);ax.grid(alpha=.12)
        ax.set(title=f'{d*45}° | trial {tid}',xlim=(-450,450),ylim=(-450,450),xticks=[-400,-200,0,200,400],yticks=[-400,-200,0,200,400])
        ax.set_aspect('equal');ax.tick_params(labelsize=10)
        if d%4==0:ax.set_ylabel('Cursor Y (px)')
        if d//4==1:ax.set_xlabel('Cursor X (px)')
    legend=fig.add_subplot(bottom[3,:]);legend.axis('off')
    handles,labels=axes[0].get_legend_handles_labels();legend.legend(handles,labels,loc='center',ncol=3,frameon=False,fontsize=12)
    fig.canvas.draw()
    stem='1111_CO8_run-001_Spike_Same_Six_Channels_Summary' if args.signal=='Spike' else '1111_CO8_run-009_LFP_Two_Row_Summary'
    DEST.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf','svg'):
        p=DATA/(stem+'.'+ext)
        fig.savefig(p,dpi=300 if ext=='png' else None,facecolor='white',bbox_inches='tight',pad_inches=.2)
        shutil.copy2(p,DEST/p.name)
    plt.close(fig)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    metadata=dict(source_sha256=hashes,layout='First row: narrower one-vs-rest AUC / eight-way confusion / narrow vertical eight-way permutation histogram with original axis orientation. Second row: 2x4 trajectory examples.',
        statistical_scope=f'A asterisk is max-AUC-corrected p across eight binary tasks. C p={p_value:.3f} is a separate EIGHT-WAY decoding test, not significance of A. '+('C repeats nested spike-feature selection under label permutations.' if args.signal=='Spike' else 'C corrects the six-band search.'),
        note=f'{binary["trials"]} direction trials; {trajectory_metrics["trials"]} complete-trajectory trials. All numbers and chronological-middle trajectory examples unchanged.')
    p=DATA/(stem+'.json');p.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(p,DEST/p.name)
    print(DEST/(stem+'.png'))

if __name__=='__main__':main()
