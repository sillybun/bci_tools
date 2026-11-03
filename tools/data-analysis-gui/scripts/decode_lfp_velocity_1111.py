"""LFP-only velocity decoding and 100-ms-ahead prediction with nested block CV.

Uses the protocol-verified samples and estimated sample times from run 009.
Continuous preprocessing is causal (sosfilt); test velocities are never decoder inputs.
"""
from pathlib import Path
import sys
import csv
import argparse
import json
from collections import Counter
from types import SimpleNamespace
import numpy as np
from scipy import signal
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_analysis_gui.core.xdf_io import load_streams
from data_analysis_gui.core.features import _interp
from data_analysis_gui.core.analysis import block_folds
from data_analysis_gui.decoders import velocity_kf as KF

BASE=ROOT/'output/lfp_20261111_run009'
OUT=BASE/'velocity_r2'
SOURCE=Path('E:/AitBCI/data/keaiduo/20261111/run-009-CO8-offline-LFP')
BANDS=[(4,8),(8,13),(13,30),(30,45),(65,95),(105,180)]

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,
        default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item()),encoding='utf-8')

def r2(y,p):
    return 1-np.square(y-p).sum(0)/np.square(y-y.mean(0)).sum(0)

def table(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def dataset(full_trial=False):
    raw=np.load(BASE/'unpacked_lfp.npz')
    times=raw['sample_times_lsl_estimated']; x=raw['values'].astype(float); fs=float(raw['effective_sample_rate_hz'])
    # Filter choices fixed before model scoring. No filtfilt or centered smoothing.
    power=[]
    for lo,hi in BANDS:
        filtered=signal.sosfilt(signal.butter(3,[lo,hi],btype='bandpass',fs=fs,output='sos'),x,axis=0)
        power.append(filtered**2)
    power=np.stack(power,axis=2).reshape(len(x),36) # channel-major, then frequency band
    low=signal.sosfilt(signal.butter(3,8,btype='lowpass',fs=fs,output='sos'),x,axis=0)
    p_integral=np.vstack([np.zeros((1,36)),np.cumsum(power,axis=0)])
    l_integral=np.vstack([np.zeros((1,6)),np.cumsum(low,axis=0)])
    def avg(integral,a,b):
        i,j=np.searchsorted(times,[a,b],side='left')
        assert i>=0 and j>i and j<=len(times)
        return (integral[j]-integral[i])/(j-i)
    streams,_=load_streams(next((SOURCE/'xdf').glob('*.xdf')))
    cursor=streams['cursortasks_cursor_state'];ct=cursor['time_stamps'];cv=cursor['time_series']
    trials=list(csv.DictReader((BASE/'reconstructed_trials.csv').open(encoding='utf-8-sig')))
    eligible=[r for r in trials if r['complete']=='True' and r['outcome']=='success' and not r['incomplete_reason']]
    output=[]; features={}; bad=Counter();sequence=0
    for row in eligible:
        start=float(row['start']);end=float(row['end']);acq=float(row['acquire_time']);trial=int(row['trial_id'])
        active=(ct>=start)&(ct<=end)&(cv[:,11]==1)&(cv[:,10]==trial)
        t=ct[active];pos=cv[active,2:4]
        if not len(t):
            bad['trial_without_active_cursor']+=1;continue
        anchor=float(t[0]) if full_trial else start+.5
        stop=start+(acq if full_trial else min(acq,1.2))
        count=int(np.floor((stop-anchor)/.1+1e-8))
        if full_trial:
            candidate_edges=anchor+np.arange(count+1)*.1
            if any(_interp(t,pos,float(edge),.1) is None for edge in candidate_edges):
                bad['trial_with_cursor_gap']+=1
                bad['potential_bins_removed_for_cursor_gap']+=count
                continue
        previous_k=None
        for k in range(max(count,0)):
            left=anchor+k*.1;right=left+.1
            earliest=times[0]+2 if full_trial else max(times[0]+2,start)
            if left-.5<earliest-1e-8:
                bad['neural_window_or_filter_warmup']+=1;previous_k=None;continue
            p0=_interp(t,pos,left,.1);p1=_interp(t,pos,right,.1)
            if p0 is None or p1 is None:
                bad['cursor_gap_or_missing_brackets']+=1;previous_k=None;continue
            if previous_k is None or k!=previous_k+1:sequence+=1
            previous_k=k
            output.append(dict(trial=trial,block=int(row['block']),direction=int(row['direction']),bin=k,sequence=sequence,
                               left=left,right=right,elapsed=left-start,vx=(p1[0]-p0[0])/.1,vy=(p1[1]-p0[1])/.1,
                               p0_x=float(p0[0]),p0_y=float(p0[1]),p1_x=float(p1[0]),p1_y=float(p1[1])))
            for horizon,cutoff in [('forecast',left),('nowcast',right)]:
                slow=np.concatenate([avg(l_integral,cutoff-lag-.1,cutoff-lag) for lag in (0,.1,.2)])
                for window in (.2,.3):
                    band=np.concatenate([np.log10(np.maximum(avg(p_integral,cutoff-lag-window,cutoff-lag),1e-12)) for lag in (0,.1,.2)])
                    high=band[np.arange(len(band))%6==5]
                    for name,z in [('power',band),('highgamma',high),('combined',np.r_[band,slow])]:
                        features.setdefault(f'{horizon}_{name}_{window:.1f}',[]).append(z)
                features.setdefault(horizon+'_lowfreq',[]).append(slow)
    features={k:np.array(v) for k,v in features.items()}
    y=np.array([[r['vx'],r['vy']] for r in output]);block=np.array([r['block'] for r in output]);seq=np.array([r['sequence'] for r in output])
    assert np.isfinite(y).all() and all(np.isfinite(z).all() and len(z)==len(y) for z in features.values())
    return output,features,y,block,seq,dict(bad),fs

def fit_predict(z,y,seq,tr,te,kind,alpha):
    if kind=='ridge':
        mu=z[tr].mean(0);sd=z[tr].std(0);sd[sd<1e-12]=1
        model=Ridge(alpha=alpha*len(tr)).fit((z[tr]-mu)/sd,y[tr])
        return model.predict((z[te]-mu)/sd)
    train=SimpleNamespace(features=z[tr],velocity=y[tr],sequence=seq[tr])
    test=SimpleNamespace(features=z[te],sequence=seq[te])
    return KF.predict(KF.fit(train,dict(lam_kf=1e-4,q_shrinkage=alpha)),test)

def candidates(features,horizon,kind):
    keys=[k for k in features if k.startswith(horizon+'_')]
    parameters=(.001,.01,.1,1.,10.) if kind=='ridge' else (.5,.95)
    return [(key,a) for key in keys for a in parameters]

def select(features,y,block,seq,tr,horizon,kind):
    inner=list(block_folds(block[tr],3,purge=1))
    best=None;best_score=-np.inf
    for key,alpha in candidates(features,horizon,kind):
        prediction=np.full((len(tr),2),np.nan)
        for _,itr,ite in inner:
            train_blocks=np.unique(block[tr[itr]]);test_blocks=np.unique(block[tr[ite]])
            assert np.all(np.abs(train_blocks[:,None]-test_blocks[None,:])>1)
            prediction[ite]=fit_predict(features[key],y,seq,tr[itr],tr[ite],kind,alpha)
        valid=np.isfinite(prediction).all(1)
        score=float(r2(y[tr][valid],prediction[valid]).mean())
        if score>best_score:
            best=(key,alpha);best_score=score
    return best,best_score

def interval(y,p,blocks):
    unique=np.unique(blocks)
    n=np.array([np.sum(blocks==b) for b in unique])
    s=np.array([y[blocks==b].sum(0) for b in unique]);s2=np.array([(y[blocks==b]**2).sum(0) for b in unique])
    error=np.array([np.square(y[blocks==b]-p[blocks==b]).sum(0) for b in unique])
    rng=np.random.default_rng(20261113);draw=rng.integers(0,len(unique),size=(2000,len(unique)))
    denom=s2[draw].sum(1)-s[draw].sum(1)**2/n[draw].sum(1)[:,None]
    scores=1-error[draw].sum(1)/denom
    return dict(x=np.quantile(scores[:,0],[.025,.975]),y=np.quantile(scores[:,1],[.025,.975]),
                mean_xy=np.quantile(scores.mean(1),[.025,.975]))

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-trial',action='store_true',help='Train/evaluate all complete outbound bins for independent trajectory integration.')
    full_trial=parser.parse_args().full_trial
    if full_trial:OUT=BASE/'trajectory_r2'/'velocity_model'
    OUT.mkdir(parents=True,exist_ok=True)
    rows,features,y,blocks,seq,excluded,fs=dataset(full_trial=full_trial)
    print('Dataset',len(y),'bins',len(set(r['trial'] for r in rows)),'trials','blocks',np.unique(blocks),'excluded',excluded,flush=True)
    table('velocity_bins.csv',rows)
    np.savez_compressed(OUT/'features_and_targets.npz',velocity=y,blocks=blocks,sequence=seq,trial_ids=[r['trial'] for r in rows],**features)
    folds=list(block_folds(blocks,5,purge=1)); predictions={};scores={};selection=[];foldrows=[]
    trial_ids=np.array([r['trial'] for r in rows])
    for _,tr,te in folds:
        assert not set(trial_ids[tr])&set(trial_ids[te])
        assert np.all(np.abs(np.unique(blocks[tr])[:,None]-np.unique(blocks[te])[None,:])>1)
    base=np.zeros_like(y)
    for f,tr,te in folds:base[te]=y[tr].mean(0)
    predictions['train_mean_baseline']=base
    with threadpool_limits(limits=1):
        for horizon in ('forecast','nowcast'):
            for kind in ('ridge','kf'):
                pred=np.full_like(y,np.nan)
                for f,tr,te in folds:
                    (key,alpha),inner=select(features,y,blocks,seq,tr,horizon,kind)
                    pred[te]=fit_predict(features[key],y,seq,tr,te,kind,alpha)
                    selection.append(dict(horizon=horizon,model=kind,outer_fold=f+1,features=key,alpha_or_q_shrinkage=alpha,inner_r2=inner,
                        train_blocks=np.unique(blocks[tr]).tolist(),test_blocks=np.unique(blocks[te]).tolist(),train_trials=len(set(rows[i]['trial'] for i in tr)),test_trials=len(set(rows[i]['trial'] for i in te))))
                    foldrows.append(dict(horizon=horizon,model=kind,fold=f+1,n_test=len(te),r2_x=float(r2(y[te],pred[te])[0]),r2_y=float(r2(y[te],pred[te])[1])))
                    print(horizon,kind,'fold',f+1,key,alpha,'test R2',r2(y[te],pred[te]),flush=True)
                assert np.isfinite(pred).all()
                name=horizon+'_'+kind;predictions[name]=pred
        for name,p in predictions.items():
            r=r2(y,p)
            np.testing.assert_allclose(r,r2_score(y,p,multioutput='raw_values'),atol=1e-12)
            scores[name]=dict(r2_x=float(r[0]),r2_y=float(r[1]),r2_mean_xy=float(r.mean()),bootstrap_95=interval(y,p,blocks))
        # Prefix invariance: adding later test bins cannot alter earlier KF outputs.
        first=next(s for s in selection if s['horizon']=='forecast' and s['model']=='kf')
        _,tr,te=folds[0]
        key=first['features'];alpha=first['alpha_or_q_shrinkage']
        for sequence_id in np.unique(seq[te])[:5]:
            idx=te[seq[te]==sequence_id]
            for end in range(1,len(idx)+1):
                prefix=fit_predict(features[key],y,seq,tr,idx[:end],'kf',alpha)
                np.testing.assert_allclose(prefix,predictions['forecast_kf'][idx[:end]],atol=1e-8,rtol=1e-8)
        # Forward tests use only earlier blocks, excluding the immediately preceding block.
        forward=[]
        for cutoff in (10,15,20):
            tr=np.flatnonzero(blocks<cutoff);te=np.flatnonzero((blocks>cutoff)&(blocks<=cutoff+5))
            for kind in ('ridge','kf'):
                (key,alpha),inner=select(features,y,blocks,seq,tr,'forecast',kind)
                p=fit_predict(features[key],y,seq,tr,te,kind,alpha);r=r2(y[te],p)
                name=f'forward_{cutoff}_{kind}'
                np.savez_compressed(OUT/(name+'.npz'),indices=te,prediction=p,truth=y[te])
                forward.append(dict(model=kind,train_blocks=np.unique(blocks[tr]),test_blocks=np.unique(blocks[te]),n_test=len(te),features=key,alpha_or_q_shrinkage=alpha,r2_x=r[0],r2_y=r[1],r2_mean_xy=r.mean()))
                print('Forward',cutoff,kind,r,flush=True)
    np.savez_compressed(OUT/'heldout_predictions.npz',truth=y,**predictions)
    prediction_rows=[]
    for i,row in enumerate(rows):
        item=dict(row)
        for name,p in predictions.items():
            item[name+'_vx']=float(p[i,0]);item[name+'_vy']=float(p[i,1])
        prediction_rows.append(item)
    table('heldout_predictions.csv',prediction_rows)
    table('outer_fold_scores.csv',foldrows)
    summary=dict(recording=str(SOURCE),samples=len(y),trials=len(set(r['trial'] for r in rows)),blocks=np.unique(blocks),excluded=excluded,
        channels_zero_based=[302,350,388,393,409,414],estimated_sample_rate_hz=fs,
        target='Cursor position displacement / 0.1 s within one trial, linear interpolation of bracketing active cursor samples; bracketing gaps <=0.1 s.',
        trial_selection=('Complete successful event-defined trials; bins start at first active cursor sample and stop at the last complete 100 ms bin before acquisition. Entire trial excluded if any bin edge cannot be interpolated with <=100 ms bracketing gap. Causal feature history may include pre-trial LFP. No target direction or cursor history enters any neural model.' if full_trial else 'Complete successful event-defined trials; bins start 0.5 s after trial start and stop at min(acquisition,1.2 s). No target direction or cursor history enters any neural model.'),
        horizons=dict(forecast='LFP features end at left edge of target velocity bin: predicts next 100 ms.',nowcast='LFP features end at right edge of target velocity bin: estimates velocity of the preceding 100 ms.'),
        features='Causal third-order Butterworth band power in 4-8,8-13,13-30,30-45,65-95,105-180 Hz. Trailing 200/300 ms average powers with 0/100/200 ms lags; log10. Low-frequency features use causal lowpass 8 Hz then trailing 100 ms means at same lags.',
        validation='5 contiguous outer block folds, purge +/-1 adjacent block. All feature choices and regularization/shrinkage selected only in 3 inner block folds with +/-1 purge. Standardization fits training samples only. KF resets to zero at each sequence and never receives true test velocity.',
        checks='No shared trials or adjacent blocks between train/test in outer and inner folds; R2 independently reconciled with sklearn; KF forecast prefix-invariance verified for first five test sequences.',
        scores=scores,outer_selections=selection,forward=forward,
        limitations=['Original acquisition clock and absolute device-to-task latency remain uncalibrated; nominal Hz bands depend on estimated sample rate.',
                     'Signal identity is LFP per protocol, but frontend filters and voltage gain are unknown; motor/visual/muscle artifacts not separated.',
                     ('Successful manual-control trials only, complete 100 ms outbound bins; not a direct comparison with earlier 0.5..1.2 s speed-window results.' if full_trial else 'Successful manual-control trials only, with 0.5..1.2 s windows; not a direct numerical comparison with other spike runs or their trial cohorts.'),
                     'Bootstrap resamples blocks of fixed outer-held-out predictions; does not rerun selection/training.',
                     'Single exploratory session. Current-bin nowcast is not future prediction; forecast is only relative to reconstructed sample timing.'])
    dump('velocity_results.json',summary)
    plots(rows,y,predictions,scores,forward)
    print('SCORES',json.dumps({k:dict(x=v['r2_x'],y=v['r2_y'],mean=v['r2_mean_xy']) for k,v in scores.items()}),flush=True)

def plots(rows,y,predictions,scores,forward):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    labels=['Forecast\nRidge','Forecast\nVelocity KF','Nowcast\nRidge','Nowcast\nVelocity KF']
    names=['forecast_ridge','forecast_kf','nowcast_ridge','nowcast_kf']
    values=np.array([scores[k]['r2_mean_xy'] for k in names]);ci=np.array([scores[k]['bootstrap_95']['mean_xy'] for k in names])
    ax=axs[0,0];ax.bar(np.arange(4),values,color=['#2176ae','#2176ae','#159783','#159783'])
    ax.errorbar(np.arange(4),values,yerr=np.maximum(np.array([values-ci[:,0],ci[:,1]-values]),0),fmt='none',color='#333',capsize=4)
    ax.axhline(0,color='#555',lw=.8);ax.axhline(scores['train_mean_baseline']['r2_mean_xy'],ls=':',color='#555',label='Training-mean baseline')
    ax.set(xticks=np.arange(4),xticklabels=labels,ylabel='Held-out Velocity R² (mean of x / y)',title='6-channel LFP | nested 5-fold block CV');ax.legend(fontsize=9)
    for i,v in enumerate(values):ax.annotate(f'{v:.3f}',(i,v),xytext=(0,8 if v>=0 else -15),textcoords='offset points',ha='center')
    ax=axs[0,1]
    for kind,offset,color in [('ridge',-.17,'#2176ae'),('kf',.17,'#159783')]:
        vals=[f['r2_mean_xy'] for f in forward if f['model']==kind]
        ax.bar(np.arange(3)+offset,vals,width=.34,color=color,label=kind)
    ax.axhline(0,color='#555',lw=.8);ax.set(xticks=np.arange(3),xticklabels=['Blocks 11–15','Blocks 16–20','Blocks 21–25'],ylabel='100-ms-ahead velocity R²',title='Forward tests | earlier blocks train only');ax.legend()
    best=max(['forecast_ridge','forecast_kf'],key=lambda k:scores[k]['r2_mean_xy'])
    for d,ax in enumerate(axs[1]):
        p=predictions[best][:,d]
        density=ax.hexbin(y[:,d],p,gridsize=35,mincnt=1,cmap='Blues')
        fig.colorbar(density,ax=ax,fraction=.035,pad=.02,label='Bins per hexagon')
        lo=min(y[:,d].min(),p.min());hi=max(y[:,d].max(),p.max());ax.plot([lo,hi],[lo,hi],ls='--',color='#555',lw=.8)
        ax.set(xlabel=f'True V{"xy"[d]} (task units / s)',ylabel=f'Predicted V{"xy"[d]} (task units / s)',title=f'{best} | R²{"xy"[d]} = {r2(y,predictions[best])[d]:.3f}')
    fig.suptitle('1111 CO8 run-009 | LFP-only Velocity Prediction',fontsize=16)
    fig.savefig(OUT/'lfp_velocity_r2.png',dpi=180);fig.savefig(OUT/'lfp_velocity_r2.pdf');plt.close(fig)

if __name__=='__main__':main()
