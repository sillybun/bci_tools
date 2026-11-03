"""Open-loop integration of outer-held-out LFP velocity predictions, one anchor/trial.

First run decode_lfp_velocity_1111.py --full-trial. Start locations are known;
no later true locations, target coordinates or endpoints correct the predictions.
"""
from pathlib import Path
import csv
import json
import sys
import hashlib
import numpy as np
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parent))
from decode_lfp_velocity_1111 import interval

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/lfp_20261111_run009/trajectory_r2'
SOURCE=OUT/'velocity_model'

def r2(y,p):return r2_score(y,p,multioutput='raw_values')

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item()),encoding='utf-8')

def main():
    source_files=[SOURCE/'velocity_bins.csv',SOURCE/'heldout_predictions.npz']
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    raw=list(csv.DictReader((SOURCE/'velocity_bins.csv').open(encoding='utf-8-sig')))
    trial=np.array([int(r['trial']) for r in raw]);blocks=np.array([int(r['block']) for r in raw]);direction=np.array([int(r['direction']) for r in raw])
    truth=np.array([[float(r['p1_x']),float(r['p1_y'])] for r in raw])
    starts=np.array([[float(r['p0_x']),float(r['p0_y'])] for r in raw])
    left=np.array([float(r['left']) for r in raw]);right=np.array([float(r['right']) for r in raw])
    velocity=np.load(SOURCE/'heldout_predictions.npz')
    anchor=np.zeros_like(truth);elapsed=np.zeros(len(trial));end_indices=[]
    for tid in np.unique(trial):
        idx=np.flatnonzero(trial==tid)
        assert int(raw[idx[0]]['bin'])==0, 'Trajectory must begin at first active cursor sample, not after a gap.'
        assert len(np.unique([raw[i]['sequence'] for i in idx]))==1,'Do not use true positions to reset after within-trial gaps.'
        np.testing.assert_allclose(left[idx][1:],right[idx][:-1],atol=1e-8,rtol=0)
        np.testing.assert_allclose(right[idx]-left[idx],.1,atol=1e-8,rtol=0)
        np.testing.assert_allclose(starts[idx][1:],truth[idx][:-1],atol=1e-6)
        anchor[idx]=starts[idx[0]]
        elapsed[idx]=right[idx]-left[idx[0]]
        end_indices.append(idx[-1])
        # Independently reconcile actual coordinates with integrated target velocities.
        reconstructed=anchor[idx]+.1*np.cumsum(velocity['truth'][idx],axis=0)
        np.testing.assert_allclose(reconstructed,truth[idx],atol=1e-6,rtol=0)
    end_indices=np.array(end_indices)
    def integrate(pred_v,indices=None):
        indices=np.arange(len(trial)) if indices is None else np.asarray(indices)
        result=np.full_like(truth,np.nan)
        for tid in np.unique(trial[indices]):
            idx=indices[trial[indices]==tid]
            assert np.array_equal(idx,np.flatnonzero(trial==tid)), 'Forward tests must cover each test trial completely.'
            result[idx]=anchor[idx]+.1*np.cumsum(pred_v[idx],axis=0)
        return result
    paths={'stationary_start':anchor.copy()}
    for name in velocity.files:
        if name!='truth':paths[name]=integrate(velocity[name])
    displacement=truth-anchor;scores={}
    for name,p in paths.items():
        absolute=r2(truth,p);relative=r2(displacement,p-anchor)
        scores[name]=dict(position_r2_x=absolute[0],position_r2_y=absolute[1],position_r2_mean_xy=absolute.mean(),
            displacement_r2_x=relative[0],displacement_r2_y=relative[1],displacement_r2_mean_xy=relative.mean(),
            position_bootstrap_95=interval(truth,p,blocks),displacement_bootstrap_95=interval(displacement,p-anchor,blocks),
            trajectory_rmse_xy=float(np.sqrt(np.mean(np.sum((p-truth)**2,axis=1)))),
            mean_endpoint_error=float(np.linalg.norm(p[end_indices]-truth[end_indices],axis=1).mean()),
            median_endpoint_error=float(np.median(np.linalg.norm(p[end_indices]-truth[end_indices],axis=1))))
    forward=[]
    for cutoff in (10,15,20):
        for kind in ('ridge','kf'):
            d=np.load(SOURCE/f'forward_{cutoff}_{kind}.npz');idx=d['indices']
            values=np.full_like(truth,np.nan);values[idx]=d['prediction']
            p=integrate(values,idx)
            a=r2(truth[idx],p[idx]);b=r2(displacement[idx],p[idx]-anchor[idx])
            forward.append(dict(model=kind,test_blocks=np.unique(blocks[idx]),n_samples=len(idx),trials=len(np.unique(trial[idx])),
                position_r2_x=a[0],position_r2_y=a[1],position_r2_mean_xy=a.mean(),displacement_r2_mean_xy=b.mean()))
    np.savez_compressed(OUT/'trajectory_predictions.npz',truth=truth,initial_anchor=anchor,trial=trial,blocks=blocks,elapsed=elapsed,**paths)
    export=[]
    for i,r in enumerate(raw):
        item=dict(trial=int(r['trial']),block=int(r['block']),direction=int(r['direction']),elapsed=elapsed[i],true_x=truth[i,0],true_y=truth[i,1],
                  initial_x=anchor[i,0],initial_y=anchor[i,1])
        for name,p in paths.items():item[name+'_x']=p[i,0];item[name+'_y']=p[i,1]
        export.append(item)
    with (OUT/'trajectory_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(export[0]));writer.writeheader();writer.writerows(export)
    velocity_summary=json.loads((SOURCE/'velocity_results.json').read_text(encoding='utf-8'))
    summary=dict(trials=len(np.unique(trial)),trajectory_points=len(truth),blocks=np.unique(blocks),excluded=velocity_summary['excluded'],
        start_policy='Each trial starts at its first measured active cursor position. This one initial position is known. No teacher forcing, subsequent position resets, endpoint correction, target assistance, rotation, rescaling or drift fitting.',
        duration='All complete 100 ms outbound bins from first active cursor sample to acquisition; final <100 ms fractional bin is omitted. Initial exactly-matched anchors are excluded from R2.',
        model_training='Velocity models retrained and nested-selected on full outbound bins using earlier established +/-1-block-purged CV. Earlier 0.5..1.2 s speed-window model results are preserved separately.',
        scores=scores,forward=forward,
        initial_position_nonzero_trials=int(sum(np.linalg.norm(anchor[np.flatnonzero(trial==tid)[0]])>1e-8 for tid in np.unique(trial))),
        position_score='Pooled held-out cursor positions across all evaluated trial timepoints; R2x and R2y computed separately then averaged.',
        displacement_score='Subtract each trial initial anchor from true and predicted coordinates before scoring; controls for known-start information.',
        validation='Inherited nested outer block CV, test trial never in training, +/-1 adjacent block purge. Bootstrap intervals resample complete blocks of fixed held-out paths, not full training pipelines.',
        limitations=['Known initial cursor position is supplied once per trial. This is not uninitialized absolute localization.',
                     'Pooled position R2 weights longer trials more; not an equal-trial mean of per-trial R2.',
                     'Only complete successful manual-control trials, one recording; no independent-session online verification.',
                     'Acquisition clock and device/task latency remain uncertain; timing uses reconstructed sample clock.',
                     'Integration can accumulate velocity bias. No endpoint or true-position correction was applied.'])
    dump(OUT/'trajectory_results.json',summary)
    figures(trial,blocks,direction,truth,anchor,paths,scores,forward)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in before.items())
    print(json.dumps(summary,ensure_ascii=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item()))

def figures(trial,blocks,direction,truth,anchor,paths,scores,forward):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    names=['forecast_ridge','forecast_kf','nowcast_ridge','nowcast_kf']
    labels=['Forecast\nRidge','Forecast\nVelocity KF','Nowcast\nRidge','Nowcast\nVelocity KF']
    fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    values=np.array([scores[k]['position_r2_mean_xy'] for k in names]);ci=np.array([scores[k]['position_bootstrap_95']['mean_xy'] for k in names])
    axs[0].bar(range(4),values,color=['#2176ae','#2176ae','#169582','#169582'])
    axs[0].errorbar(range(4),values,yerr=np.maximum(np.array([values-ci[:,0],ci[:,1]-values]),0),fmt='none',capsize=4,color='#444')
    axs[0].axhline(scores['stationary_start']['position_r2_mean_xy'],color='#777',ls=':',label='Stay at known start')
    axs[0].set(xticks=range(4),xticklabels=labels,ylabel='Held-out Position R² (mean of x / y)',title='Integrated trajectories | nested block CV')
    axs[0].legend(fontsize=9)
    for i,v in enumerate(values):axs[0].annotate(f'{v:.3f}',(i,v),xytext=(0,8),textcoords='offset points',ha='center')
    for kind,offset,color in [('ridge',-.17,'#2176ae'),('kf',.17,'#169582')]:
        v=[r['position_r2_mean_xy'] for r in forward if r['model']==kind]
        axs[1].bar(np.arange(3)+offset,v,width=.34,label=kind,color=color)
    axs[1].axhline(0,color='#777',lw=.8)
    axs[1].set(xticks=range(3),xticklabels=['Blocks 11–15','Blocks 16–20','Blocks 21–25'],ylabel='Held-out Position R²',title='Forward trajectories | earlier blocks train only')
    axs[1].legend()
    fig.suptitle('1111 CO8 run-009 | LFP-only Trajectory Prediction',fontsize=16)
    fig.savefig(OUT/'lfp_trajectory_r2.png',dpi=180);fig.savefig(OUT/'lfp_trajectory_r2.pdf');plt.close(fig)
    fig,axs=plt.subplots(2,4,figsize=(14,8),layout='constrained')
    prediction=paths['forecast_ridge'];chosen=[]
    lim=max(380,float(np.max(np.abs(truth)))*1.05)
    for d,ax in enumerate(axs.flat):
        candidates=np.unique(trial[direction==d]);tid=candidates[len(candidates)//2];chosen.append(int(tid));idx=np.flatnonzero(trial==tid)
        actual=np.vstack([anchor[idx[0]],truth[idx]]);p=np.vstack([anchor[idx[0]],prediction[idx]])
        ax.plot(actual[:,0],actual[:,1],'-o',color='#333333',ms=3,label='True')
        ax.plot(p[:,0],p[:,1],'-o',color='#2176ae',ms=3,label='LFP prediction')
        ax.scatter(*anchor[idx[0]],marker='s',s=28,color='#169582',label='Known start',zorder=4)
        ax.axhline(0,color='#ddd',lw=.5);ax.axvline(0,color='#ddd',lw=.5)
        ax.set(title=f'{d*45}° | trial {tid}',xlabel='Cursor X (px)',ylabel='Cursor Y (px)',xlim=(-lim,lim),ylim=(-lim,lim));ax.set_aspect('equal');ax.grid(alpha=.1)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.025),frameon=False)
    fig.suptitle('Forecast Ridge | one chronological-middle trial per direction\nKnown initial position only; no correction during each trajectory',fontsize=14)
    fig.savefig(OUT/'lfp_trajectory_examples.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'lfp_trajectory_examples.pdf',bbox_inches='tight');plt.close(fig)
    dump(OUT/'example_selection.json',dict(method='Chronological-middle trial per target direction, independent of prediction errors. Forecast Ridge fixed for display.',trial_ids=chosen))

if __name__=='__main__':main()
