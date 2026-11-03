"""1110 channel budget, exactly matched to the GUI S10 Vel-KF baseline."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]))
import csv
import json
import hashlib
import pickle
import shutil
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor,cho_solve
from threadpoolctl import threadpool_limits
from data_analysis_gui.session_cache import _unpack
from data_analysis_gui.core.analysis import block_folds,r2_score
from data_analysis_gui.decoders import velocity_kf as KF

ROOT=Path('E:/AitBCI/data_analysis_gui_output')
OUT=ROOT/'channel_budget_1110_velkf_matched'
ORIGINAL=ROOT/'keaiduo/single/2026-11-10_CO8_run-001_run-001-CO8-offline-20261110-rec-001-main-spike/s10_v1/c672db424160/result.json'
CACHE=ROOT/'keaiduo/.session-cache/234da2a6b7ccbe24a931/decoder_data/2ea32d6023df8a0c8330e9153fd9ad41fca1416ee831cc535246bd3ad9ba4376.npz'
PARAMS=dict(lam_kf=1e-4,q_shrinkage=.5)
RUN_LABEL='1110 CO8 run-001'
PLOT_TRIALS=193

def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def table(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load():
    with np.load(CACHE,allow_pickle=False) as z:d=_unpack(json.loads(z['metadata'].item()),z)
    reference=json.loads(ORIGINAL.read_text(encoding='utf-8'))
    assert d.features.shape==(reference['summary']['n_bins'],512)
    assert len(np.unique(d.trial))==reference['summary']['n_trials']
    assert d.bin_seconds==.1 and d.neural_window==.2
    return d
def fit(d):
    m=KF.fit(d,PARAMS)
    return dict(par=m['par'],mu=m['mu'],sd=m['sd'],scale=m['scale'],cols=np.flatnonzero(m['keep']))
def subset(m,cols):
    # A requested physical input may be constant in this training fold. The
    # original estimator drops it as well; include it in the requested budget
    # while keeping only its fitted, nonconstant observation dimensions here.
    cols=np.intersect1d(np.array(cols,int),m['cols']);indices=np.searchsorted(m['cols'],cols)
    assert np.array_equal(m['cols'][indices],cols)
    p=m['par'];q=dict(p,C=p['C'][indices],bias=p['bias'][indices],Q=p['Q'][np.ix_(indices,indices)])
    return dict(par=q,mu=m['mu'][indices],sd=m['sd'][indices],scale=m['scale'],cols=cols)
def predict(m,d):
    z=(d.features[:,m['cols']]-m['mu'])/m['sd']
    return KF.kf_predict(m['par'],z,d.sequence)*m['scale']
def terms(m,length):
    p=m['par'];a,c,w,q=p['A'],p['C'],p['W'],p['Q']
    cr=cho_solve(cho_factor(q),c).T;info=cr@c
    cov=np.zeros((2,2));fs=[];gs=[]
    for _ in range(length):
        prior=np.linalg.inv(a@cov@a.T+w)
        cov=np.linalg.inv(prior+info)
        fs.append(cov@prior@a);gs.append(cov@cr)
    return np.array(fs),np.array(gs)
def contributions(m,d,center=False):
    fs,gs=terms(m,10)
    out=np.zeros((d.n,2,len(m['cols'])))
    for seq in np.unique(d.sequence):
        rows=np.flatnonzero(d.sequence==seq);assert len(rows)==10
        state=np.zeros((2,len(m['cols'])))
        for step,row in enumerate(rows):
            x=d.features[row,m['cols']]
            if center:x=x-m['mu']
            state=fs[step]@state+gs[step]*(x/m['sd'])[None,:]
            out[row]=state*m['scale']
    return out
def importance(m,d):
    return np.sqrt(np.mean(np.square(contributions(m,d,True)).sum(1),axis=0))
def impulse_importance(m):
    fs,gs=terms(m,10);cov=np.zeros((len(m['cols']),2,2));energy=np.zeros(len(cov))
    for f,g in zip(fs,gs):
        cov=f[None]@cov@f.T[None]+g.T[:,:,None]*g.T[:,None,:]
        energy+=cov[:,0,0]+cov[:,1,1]
    return np.sqrt(energy/10)*m['scale']
def path_sse(response,residual,permutations):
    gram=np.array([response[:,a].T@response[:,a] for a in range(2)])
    cross=np.array([response[:,a].T@residual[:,a] for a in range(2)])
    base=np.square(residual).sum(0);values=[]
    for permutation in permutations:
        axes=[]
        for a in range(2):
            g=gram[a][np.ix_(permutation,permutation)]
            delta=np.diag(g)+2*np.tril(g,-1).sum(1)-2*cross[a,permutation]
            axes.append(np.r_[base[a],base[a]+np.cumsum(delta)])
        values.append(np.array(axes).T)
    return np.array(values)

def plot_dropout(random,ranked,most,scale='log'):
    quantiles=np.quantile(random.mean(2),[.05,.5,.95],axis=0)
    fig,ax=plt.subplots(figsize=(10,5.6),layout='constrained')
    end=511 if scale=='log' else 512
    retained=np.arange(512-end,513)
    ax.fill_between(retained,quantiles[0,end::-1],quantiles[2,end::-1],color='#2176ae',alpha=.16,label='Random 5-95% (200 paths)')
    ax.plot(retained,quantiles[1,end::-1],color='#2176ae',label='Random median',lw=2)
    ax.plot(retained,ranked.mean(1)[end::-1],color='#c16e14',label='Drop least-important first',lw=2)
    ax.plot(retained,most.mean(1)[end::-1],color='#7b3f96',ls='-.',label='Drop most-important first',lw=2)
    ax.axhline(.6,color='#228b22',ls='--',lw=1.5,label='R² = 0.6')
    ax.axhline(0,ls=':',color='#444444',lw=.8);ax.grid(alpha=.18)
    ax.set(xlabel=f'Channels retained ({"log" if scale=="log" else "linear"} scale)',ylabel='Held-out velocity R²: mean of x / y',xscale=scale,xlim=(512-end,512))
    ticks=[1,2,4,8,16,32,64,128,256,512] if scale=='log' else [0,64,128,192,256,320,384,448,512]
    ax.set_xticks(ticks,labels=[str(t) for t in ticks])
    ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=9,loc='lower left')
    fig.suptitle(f'{RUN_LABEL} | Vel-KF frozen per fold | zero-fill lost channels\n{PLOT_TRIALS} trials; 100 ms bins / 200 ms neural windows; 5-fold block CV, ±1 block purge',fontsize=11)
    name='dropout' if scale=='log' else 'dropout_linear'
    fig.savefig(OUT/f'{name}.png',dpi=180);fig.savefig(OUT/f'{name}.svg');plt.close(fig)

def dropout(d):
    rng=np.random.default_rng(20261110)
    permutations=np.array([rng.permutation(512) for _ in range(200)])
    random_sse=np.zeros((200,513,2));ranked_sse=np.zeros((513,2));most_sse=np.zeros((513,2));baseline=np.full((d.n,2),np.nan)
    rankings=[];checks=[];folds=[]
    for f,tr,te in block_folds(d.block):
        train,test=d.subset(tr),d.subset(te);m=fit(train);p=predict(m,test);baseline[te]=p
        # Fold-specific ordering uses only that fold's training data.
        weights=np.zeros(512);weights[m['cols']]=importance(m,train)
        order=np.argsort(weights,kind='stable')
        response=np.zeros((len(te),2,512));response[:,:,m['cols']]=contributions(m,test)
        residual=p-test.velocity
        ranked_sse+=path_sse(response,residual,[order])[0]
        most_order=order[::-1]
        most_sse+=path_sse(response,residual,[most_order])[0]
        random_sse+=path_sse(response,residual,permutations)
        for count in (0,64,256,512):
            dropped=test.features.copy();dropped[:,order[:count]]=0
            direct=predict(m,test.with_features(dropped))
            additive=p-response[:,:,order[:count]].sum(2)
            np.testing.assert_allclose(direct,additive,atol=1e-7,rtol=1e-8)
            checks.append(dict(fold=f,dropped=count,max_prediction_error=float(abs(direct-additive).max())))
        for count in (1,64,256,512):
            dropped=test.features.copy();dropped[:,most_order[:count]]=0
            direct=predict(m,test.with_features(dropped))
            additive=p-response[:,:,most_order[:count]].sum(2)
            np.testing.assert_allclose(direct,additive,atol=1e-7,rtol=1e-8)
            checks.append(dict(fold=f,method='most-important first',dropped=count,max_prediction_error=float(abs(direct-additive).max())))
        rankings.extend(dict(fold=f,removal_rank=i+1,column_0based=int(c),physical_channel_id=int(d.electrode_ids[c]),
                             training_importance=float(weights[c])) for i,c in enumerate(order))
        folds.append((f,tr,te,m))
        print('Dropout fold',f,'active',len(m['cols']),flush=True)
    original=json.loads(ORIGINAL.read_text(encoding='utf-8'))
    observed=r2_score(d.velocity,baseline)
    np.testing.assert_allclose(observed,[original['summary']['velocity_kf_r2_x'],original['summary']['velocity_kf_r2_y']],atol=1e-10)
    sst=np.square(d.velocity-d.velocity.mean(0)).sum(0)
    ranked=1-ranked_sse/sst;random=1-random_sse/sst;most=1-most_sse/sst
    np.testing.assert_allclose(most[0],ranked[0],atol=1e-12)
    np.testing.assert_allclose(most[-1],ranked[-1],atol=1e-12)
    quantiles=np.quantile(random.mean(2),[.05,.5,.95],axis=0)
    rows=[dict(dropped=i,retained=512-i,ranked_r2_mean=float(ranked[i].mean()),ranked_r2_x=float(ranked[i,0]),
               ranked_r2_y=float(ranked[i,1]),most_first_r2_mean=float(most[i].mean()),most_first_r2_x=float(most[i,0]),
               most_first_r2_y=float(most[i,1]),random_r2_p05=float(quantiles[0,i]),random_r2_median=float(quantiles[1,i]),
               random_r2_p95=float(quantiles[2,i])) for i in range(513)]
    table(OUT/'dropout_curve.csv',rows);table(OUT/'fold_channel_importance.csv',rankings)
    np.savez_compressed(OUT/'dropout_predictions.npz',baseline=baseline,truth=d.velocity,random_r2=random,ranked_r2=ranked,most_first_r2=most,permutations=permutations)
    plot_dropout(random,ranked,most)
    plot_dropout(random,ranked,most,scale='linear')
    dump(OUT/'dropout_summary.json',dict(baseline_r2_x=float(observed[0]),baseline_r2_y=float(observed[1]),
         baseline_r2_mean=float(observed.mean()),original_result=str(ORIGINAL),original_summary=original['summary'],
         criteria=original['inputs']['criteria'],params=original['inputs']['params'],
         cache_sha256=hashlib.sha256(CACHE.read_bytes()).hexdigest(),random_paths=200,
         dropout='zero-fill input Hz features BEFORE frozen training mean/SD; complete trained KF unchanged',
         importance='RMS centered training-channel contribution through complete frozen KF; independent ordering per outer fold',
         direct_checks=checks,selected_points=[rows[i] for i in (0,64,128,256,384,448,480,496,504,508,512)]))
    return folds,baseline

def select(d,folds,baseline):
    """Fully nested feature selection; outer-fold data never ranks channels."""
    out=OUT/'selection';out.mkdir(exist_ok=True)
    lower_targets=[256,192,128,96,80,64,56,48,40,32,24,16,12,8,4,2,1]
    targets=[512,448,384,320]+lower_targets
    predictions={k:np.full((d.n,2),np.nan) for k in targets}
    history=[];chosen_rows=[];checks=[];start=time.time()
    for fold,tr,te,outer in folds:
        train,test=d.subset(tr),d.subset(te)
        inner=[]
        for _,itr,ite in block_folds(train.block,3):
            td,vd=train.subset(itr),train.subset(ite)
            inner.append((fit(td),vd))
        # Eligible channels must vary in every inner training fold.
        pool=outer['cols'].copy()
        for m,_ in inner:pool=np.intersect1d(pool,m['cols'])
        ranks=np.mean([impulse_importance(subset(m,pool)) for m,_ in inner],axis=0)
        initial=pool[np.argsort(-ranks,kind='stable')]
        previous=pool.copy();memo={};rng=np.random.default_rng(1110+fold)
        def score(cols):
            key=tuple(sorted(map(int,cols)))
            if key not in memo:
                values=[float(r2_score(v.velocity,predict(subset(m,key),v)).mean()) for m,v in inner]
                memo[key]=float(np.mean(values))
                history.append(dict(outer_fold=fold,channels=len(key),inner_r2=memo[key],
                                    inner_fold_r2=json.dumps(values),columns=json.dumps(key)))
            return memo[key]
        # Complete the previously omitted upper range. Use only inner training
        # fits for ranking, including silent input slots when k exceeds the
        # number of varying channels. Preserve the original lower-k search.
        full_weight=np.zeros(512)
        for m,_ in inner:full_weight[m['cols']]+=impulse_importance(m)/len(inner)
        full_rank=np.argsort(-full_weight,kind='stable')
        for k in (512,448,384,320):
            best=np.sort(full_rank[:k]);bs=score(best)
            for _ in range(2):
                outside=[c for c in full_rank if c not in set(best)]
                weak=sorted(best,key=lambda c:full_weight[c])[:2]
                next_best=best;next_score=bs
                for remove in weak:
                    for add in outside[:4]:
                        candidate=np.sort(np.r_[best[best!=remove],add]);value=score(candidate)
                        if value>next_score+1e-6:next_best,next_score=candidate,value
                if next_score<=bs+1e-6:break
                best,bs=next_best,next_score
            predictions[k][te]=predict(subset(outer,best),test)
            chosen_rows.append(dict(outer_fold=fold,channels=k,inner_score=bs,
                 columns_0based=json.dumps(best.tolist()),physical_channel_ids=json.dumps(d.electrode_ids[best].tolist())))
            independent=fit(train.columns(best))
            direct=predict(independent,test.columns(best))
            np.testing.assert_allclose(direct,predictions[k][te],atol=2e-7,rtol=1e-8)
            checks.append(dict(fold=fold,channels=k,independent_refit_prediction_error=float(abs(direct-predictions[k][te]).max())))
        for k in lower_targets:
            w=impulse_importance(subset(inner[-1][0],previous))
            warm=previous[np.argsort(-w,kind='stable')[:k]]
            starts=[warm,initial[:k]]
            if k<128:
                extra=initial[:k].copy();n=max(1,min(3,k//4))
                extra[-n:]=rng.choice(initial[k:min(len(pool),k+32)],n,replace=False);starts.append(extra)
            best=None;bs=-np.inf
            for seed in starts:
                current=np.sort(seed);value=score(current)
                for _ in range(2):
                    w=impulse_importance(subset(inner[-1][0],current))
                    weak=current[np.argsort(w)[:min(k,2)]]
                    outside=np.setdiff1d(pool,current)
                    incoming=[c for c in initial if c in set(outside)][:3]+list(rng.choice(outside,min(2,len(outside)),replace=False))
                    nxt=current;nv=value
                    for remove in weak:
                        for add in set(incoming):
                            candidate=np.sort(np.r_[current[current!=remove],add]);s=score(candidate)
                            if s>nv+1e-6:nxt,nv=candidate,s
                    if nv<=value+1e-6:break
                    current,value=nxt,nv
                if value>bs:best,bs=current,value
            previous=best
            # Set selected from inner data; score stored only after selection.
            predictions[k][te]=predict(subset(outer,best),test)
            chosen_rows.append(dict(outer_fold=fold,channels=k,inner_score=bs,
                  columns_0based=json.dumps(best.tolist()),physical_channel_ids=json.dumps(d.electrode_ids[best].tolist())))
            if k in (64,8,1):
                independent=fit(train.columns(best))
                direct=predict(independent,test.columns(best))
                np.testing.assert_allclose(direct,predictions[k][te],atol=2e-7,rtol=1e-8)
                checks.append(dict(fold=fold,channels=k,independent_refit_prediction_error=float(abs(direct-predictions[k][te]).max())))
        table(out/'chosen_channels_by_outer_fold.csv',chosen_rows);table(out/'search_history.csv',history)
        print('Selection outer fold',fold,'complete; candidates',len(memo),'elapsed',round(time.time()-start,1),flush=True)
    # Aggregate held-out points and paired block bootstrap uncertainty.
    rng=np.random.default_rng(20261110);blocks=np.unique(d.block)
    count=np.array([(d.block==b).sum() for b in blocks])
    ys=np.array([d.velocity[d.block==b].sum(0) for b in blocks]);y2=np.array([np.square(d.velocity[d.block==b]).sum(0) for b in blocks])
    draws=rng.integers(len(blocks),size=(2000,len(blocks)))
    sst=y2[draws].sum(1)-np.square(ys[draws].sum(1))/count[draws].sum(1)[:,None]
    def boots(p):
        sse=np.array([np.square(d.velocity[d.block==b]-p[d.block==b]).sum(0) for b in blocks])
        return (1-sse[draws].sum(1)/sst).mean(1)
    base=boots(baseline);rows=[]
    for k in targets:
        p=predictions[k];assert np.isfinite(p).all();v=r2_score(d.velocity,p);draw=boots(p)
        rows.append(dict(channels=k,heldout_r2_mean=float(v.mean()),heldout_r2_x=float(v[0]),heldout_r2_y=float(v[1]),
             ci_low=float(np.quantile(draw,.025)),ci_high=float(np.quantile(draw,.975)),
             paired_delta_ci_low=float(np.quantile(draw-base,.025)),paired_delta_ci_high=float(np.quantile(draw-base,.975)),
             mean_inner_score=float(np.mean([r['inner_score'] for r in chosen_rows if r['channels']==k]))))
    table(out/'selection_curve.csv',rows);np.savez_compressed(out/'heldout_predictions.npz',truth=d.velocity,**{f'prediction_{k}':v for k,v in predictions.items()})
    # Full effective-channel model: training-fold constants are dropped by the
    # same estimator, so its honest OOF score is the full-budget baseline.
    effective_rows=[]
    for f,tr,te,m in folds:
        effective_rows.append(dict(outer_fold=f,effective_channels=len(m['cols']),
            heldout_r2_mean=float(r2_score(d.velocity[te],baseline[te]).mean()),
            columns_0based=json.dumps(m['cols'].tolist()),physical_channel_ids=json.dumps(d.electrode_ids[m['cols']].tolist())))
    table(out/'all_effective_channels_cv.csv',effective_rows)
    final=KF.fit(d,PARAMS)
    active=np.flatnonzero(final['keep'])
    artifact=dict(model=final,decoder='velocity_kf',params=PARAMS,
        input_electrode_ids=d.electrode_ids, effective_electrode_ids=d.electrode_ids[active],
        effective_columns_0based=active,training_trials=193,training_bins=d.n,
        note='Final refit on all 193 trials. Reported OOF score comes from separate training-fold fits, not this all-data model.')
    with (out/'velkf_all_effective_channels.pkl').open('wb') as stream:pickle.dump(artifact,stream)
    effective_counts=np.array([r['effective_channels'] for r in effective_rows])
    effective_info=dict(final_all_data_effective_channels=len(active),outer_fold_effective_counts=effective_counts.tolist(),
        plot_x_mean_effective_channels=float(effective_counts.mean()),
        outer_oof_r2_mean=float(r2_score(d.velocity,baseline).mean()),
        ci_low=float(np.quantile(base,.025)),ci_high=float(np.quantile(base,.975)),
        final_model_path=str(out/'velkf_all_effective_channels.pkl'),
        qualification='Active count determined from training rows only in each CV fold. Final all-data model has no independent test score.')
    dump(out/'all_effective_channels_summary.json',effective_info)
    dump(out/'summary.json',dict(search='backward pruning; 2-3 starts, up to 2 swap rounds per start; no global optimum guarantee',
        outer_folds=5,inner_folds=3,purge_blocks=1,evaluated_candidates=len(history),independent_refit_checks=checks,
        selection_uses_only_outer_training_data=True,confidence='2000 block-bootstrap draws of OOF predictions; fixed selected sets; no selection-uncertainty correction',
        results=rows,chosen_channels=chosen_rows))
    fig,ax=plt.subplots(figsize=(10,5.6),layout='constrained');rs=sorted(rows,key=lambda r:r['channels'])
    xs=[r['channels'] for r in rs]
    ax.fill_between(xs,[r['ci_low'] for r in rs],[r['ci_high'] for r in rs],color='#2176ae',alpha=.15,label='95% block-bootstrap interval')
    ax.plot(xs,[r['heldout_r2_mean'] for r in rs],'-o',color='#2176ae',label='Selected / retrained, outer held-out')
    ax.plot(xs,[r['mean_inner_score'] for r in rs],'--',color='#c16e14',label='Inner validation score')
    ax.axhline(float(r2_score(d.velocity,baseline).mean()),ls=':',color='#444444',label='512-channel baseline = 0.720')
    ax.axhline(.6,color='#228b22',ls='--',lw=1.5,label='R² = 0.6')
    center=effective_counts.mean();point=float(r2_score(d.velocity,baseline).mean())
    ax.errorbar(center,point,xerr=np.array([[center-effective_counts.min()],[effective_counts.max()-center]]),
        fmt='D',markersize=6,color='#7b3f96',capsize=4,
        label=f'All effective: {effective_counts.min()}–{effective_counts.max()} / fold')
    ax.set(xlabel='Channels retained (log scale)',ylabel='Velocity R²: mean of x / y',xscale='log',
        title='1110 CO8 run-001 | GUI-matched Vel-KF | nested channel selection\n193 trials; outer 5-fold / inner 3-fold block CV, ±1 block purge')
    ticks=[1,2,4,8,16,32,64,128,256,512];ax.set_xticks(ticks,labels=[str(t) for t in ticks]);ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8,loc='lower right')
    fig.savefig(out/'selection.png',dpi=180);fig.savefig(out/'selection.svg');plt.close(fig)
    return rows

def main():
    OUT.mkdir(exist_ok=True);d=load()
    print('Matched data loaded',d.features.shape,flush=True)
    folds,baseline=dropout(d)
    print('BASELINE REPRODUCED',r2_score(d.velocity,baseline),flush=True)
    rows=select(d,folds,baseline)
    drop=list(csv.DictReader((OUT/'dropout_curve.csv').open(encoding='utf-8-sig')))
    combined=[]
    for r in rows:
        t=drop[512-r['channels']]
        combined.append(dict(retained=r['channels'],random_median=float(t['random_r2_median']),
                             ranked_frozen=float(t['ranked_r2_mean']),retrained=r['heldout_r2_mean']))
    table(OUT/'comparison.csv',combined)
    shutil.copy2(__file__,OUT/Path(__file__).name)
    dump(OUT/'provenance.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         source_recording_sha256='234da2a6b7ccbe24a9311a25142865db461fa863c16365e14945c3ddc7a32aa6',original_result=str(ORIGINAL),
         note='GUI-matched rerun replaces prior PVKF analysis for this comparison. No original data or GUI results modified.'))
    print('COMPLETE',json.dumps(combined),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
