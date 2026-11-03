"""Four-panel LFP summary from reviewed results; does not recompute analyses."""
from pathlib import Path
import json
import hashlib
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'output/lfp_20261111_run009'
REPORT=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report/2026-11-04_2026-11-11_25runs_v1/additional_figures')


def main():
    files=[DATA/'unpacked_lfp.npz',DATA/'analysis_results.json',DATA/'trajectory_r2/trajectory_predictions.npz',
           DATA/'trajectory_r2/trajectory_results.json',DATA/'trajectory_r2/example_selection.json']
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    raw=np.load(files[0]);direction=json.loads(files[1].read_text(encoding='utf-8'))
    trajectory=np.load(files[2]);trajectory_result=json.loads(files[3].read_text(encoding='utf-8'))
    examples=json.loads(files[4].read_text(encoding='utf-8'))['trial_ids']
    cm=np.array(direction['nested_band_selection']['confusion_matrix'])
    assert cm.shape==(8,8) and cm.sum()==188
    np.testing.assert_allclose(np.mean(np.diag(cm)/cm.sum(1)),direction['nested_band_selection']['balanced_accuracy'])
    null=np.array(direction['band_comparison']['max_band_null'])*100
    observed=max(direction['band_accuracy'])*100
    assert len(null)==499 and (1+np.sum(null>=observed))/500==direction['band_comparison']['max_statistic_p_value']
    assert len(np.unique(trajectory['trial']))==150 and len(trajectory['truth'])==1777
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':12,'axes.titlesize':15,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(21,13),layout='constrained')
    grid=fig.add_gridspec(2,2,width_ratios=[1,1.32],height_ratios=[1,1.08],wspace=.09,hspace=.13)
    fig.suptitle('1111 CO8 run-009 | Six-channel LFP',fontsize=23)
    ax=fig.add_subplot(grid[0,0])
    fs=float(raw['effective_sample_rate_hz']);i=int(round(30*fs));j=i+int(round(2*fs))
    t=raw['sample_times_lsl_estimated'][i:j];t=t-t[0]
    values=raw['values'][i:j].astype(float);ids=raw['channel_ids'];offsets=np.arange(5,-1,-1)*100
    for c in range(6):
        wave=values[:,c]-np.median(values[:,c])
        ax.plot(t,wave+offsets[c],color='#2176ae',lw=.65)
    ax.plot([2.1,2.1],[540,580],color='#444',lw=1.7)
    ax.text(2.08,590,'40 ADC',ha='center',va='bottom',fontsize=10)
    ax.set(xlim=(0,2.22),ylim=(-65,625),xticks=[0,.5,1,1.5,2],yticks=offsets,yticklabels=ids,
        xlabel='Time (s)',ylabel='Wire channel ID (0-based)',
        title='A  LFP visualization\nRaw signals, median-centered and vertically offset')
    ax.grid(axis='x',alpha=.16)

    ax=fig.add_subplot(grid[0,1]);fraction=cm/cm.sum(1,keepdims=True)
    image=ax.imshow(fraction,vmin=0,vmax=.4,cmap='Blues',aspect='equal')
    for r in range(8):
        for c in range(8):
            ax.text(c,r,str(cm[r,c]),ha='center',va='center',fontsize=12,color='white' if fraction[r,c]>.22 else '#222')
    ax.set(xticks=range(8),xticklabels=np.arange(8)*45,yticks=range(8),yticklabels=np.arange(8)*45,
        xlabel='Predicted direction (deg)',ylabel='Target direction (deg)',
        title='B  Direction confusion matrix\n188 trials | held-out-block balanced accuracy = 26.8%')
    fig.colorbar(image,ax=ax,fraction=.032,pad=.025,format=PercentFormatter(1),label='Fraction within target direction')

    ax=fig.add_subplot(grid[1,0])
    ax.hist(null,bins=20,color='#aaa',edgecolor='white',linewidth=.5,label='Null: best of 6 bands (499 shuffles)')
    ax.axvline(observed,color='#c44e22',lw=2,label=f'Observed {observed:.1f}%')
    ax.set(xlim=(9,28.5),xlabel='Best-of-six-band balanced accuracy (%)',ylabel='Permutations',
        title='C  Direction decoding significance\nWithin-block permutation | six-band corrected p = 0.002')
    ax.legend(loc='upper right',fontsize=10,frameon=False)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)

    sub=grid[1,1].subgridspec(4,4,height_ratios=[.26,1,1,.15],wspace=.12,hspace=.12)
    heading=fig.add_subplot(sub[0,:]);heading.axis('off')
    score=trajectory_result['scores']['forecast_ridge']['position_r2_mean_xy']
    heading.text(.5,.9,'D  Eight-direction trajectories',ha='center',va='top',fontsize=15)
    heading.text(.5,.3,f'Ridge | 150-trial evaluation | position R² = {score:.3f}',ha='center',va='top',fontsize=11)
    trial=trajectory['trial'];truth=trajectory['truth'];prediction=trajectory['forecast_ridge'];anchor=trajectory['initial_anchor']
    axes=[]
    for d,tid in enumerate(examples):
        ax=fig.add_subplot(sub[1+d//4,d%4]);axes.append(ax);idx=np.flatnonzero(trial==tid)
        actual=np.vstack([anchor[idx[0]],truth[idx]]);pred=np.vstack([anchor[idx[0]],prediction[idx]])
        ax.plot(actual[:,0],actual[:,1],'-o',color='#333',ms=2.5,lw=1.3,label='True trajectory')
        ax.plot(pred[:,0],pred[:,1],'-o',color='#2176ae',ms=2.5,lw=1.3,label='LFP prediction')
        ax.scatter(*anchor[idx[0]],s=25,marker='s',color='#169582',label='Known start',zorder=4)
        ax.axhline(0,color='#ddd',lw=.5);ax.axvline(0,color='#ddd',lw=.5)
        ax.set(title=f'{d*45}° | trial {tid}',xlim=(-450,450),ylim=(-450,450),xticks=[-400,0,400],yticks=[-400,0,400])
        ax.set_aspect('equal');ax.tick_params(labelsize=9);ax.title.set_fontsize(11)
        if d%4==0:ax.set_ylabel('Y (px)',fontsize=10)
        if d//4==1:ax.set_xlabel('X (px)',fontsize=10)
    legend=fig.add_subplot(sub[3,:]);legend.axis('off')
    handles,names=axes[0].get_legend_handles_labels()
    legend.legend(handles,names,loc='center',ncol=3,frameon=False,fontsize=10)
    stem='1111_CO8_run-009_LFP_Four_Panel_Summary'
    for ext in ('png','pdf','svg'):
        destination=DATA/(stem+'.'+ext)
        fig.savefig(destination,dpi=300 if ext=='png' else None,facecolor='white',bbox_inches='tight',pad_inches=.2)
        REPORT.mkdir(parents=True,exist_ok=True)
        shutil.copy2(destination,REPORT/destination.name)
        assert hashlib.sha256(destination.read_bytes()).digest()==hashlib.sha256((REPORT/destination.name).read_bytes()).digest()
    plt.close(fig)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
    metadata=dict(source_sha256=before,panels=dict(
        A='Two-second raw ADC excerpt at ~30 s; median-centered per channel, vertical separation 100 ADC units; all channels share amplitude scale.',
        B='188 direction trials; nested band-selection outer-held-out confusion matrix. Cell numbers are counts; color is row-normalized fraction.',
        C='499 within-block shuffled-label maxima across six frequency bands; observed best band 105-180 Hz; six-band corrected p=0.002.',
        D='Forecast Ridge trajectories: 150 trials / 1777 points overall, mean position R2=0.094836. One chronological-middle trial per direction; known initial point only, no within-trial correction.'),
        note='Direction and trajectory cohorts differ and are explicitly labelled. Original analysis values, selections and predictions are unchanged.',
        example_trial_ids=examples,report_directory=str(REPORT))
    (DATA/(stem+'.json')).write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(DATA/(stem+'.json'),REPORT/(stem+'.json'))
    print(REPORT/(stem+'.png'))


if __name__=='__main__':main()
