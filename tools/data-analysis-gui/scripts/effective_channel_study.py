"""Single-run channel studies, budgets count training-effective channels only.

Common integer budgets use exactly the same five outer test folds. The separate
all-effective endpoint has fold-varying counts and is labelled as such.
"""
import json
import pickle
import shutil
import hashlib
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import channel_budget_1110_velkf as S

GRID=[320,288,256,224,192,160,128,96,80,64,56,48,40,32,24,16,12,8,4,2,1]

def weights(cols,models):
    result={int(c):0. for c in cols}
    for model in models:
        sub=S.subset(model,cols)
        if len(sub['cols']):
            values=S.impulse_importance(sub)
            for c,v in zip(sub['cols'],values):result[int(c)]+=float(v)/len(models)
    return np.array([result[int(c)] for c in cols])

def bootstrap(data,predictions):
    blocks=np.unique(data.block)
    count=np.array([(data.block==b).sum() for b in blocks])
    sums=np.array([data.velocity[data.block==b].sum(0) for b in blocks])
    squares=np.array([np.square(data.velocity[data.block==b]).sum(0) for b in blocks])
    rng=np.random.default_rng(20261110);draws=rng.integers(len(blocks),size=(2000,len(blocks)))
    denom=squares[draws].sum(1)-np.square(sums[draws].sum(1))/count[draws].sum(1)[:,None]
    result={}
    for k,p in predictions.items():
        assert np.isfinite(p).all()
        sse=np.array([np.square(data.velocity[data.block==b]-p[data.block==b]).sum(0) for b in blocks])
        result[k]=(1-sse[draws].sum(1)/denom).mean(1)
    return result

def axis(ax,counts,scale):
    maximum=int(max(counts))
    ax.set(xscale=scale,xlim=(1 if scale=='log' else 0,maximum*1.025),
           xlabel=f'Channel Retained ({"Log" if scale=="log" else "Linear"} scale)',
           ylabel='Held-out Velocity R²')
    ticks=([1,2,4,8,16,32,64,128,256] if scale=='log' else list(range(0,maximum+1,32)))
    ticks=[v for v in ticks if v<maximum*.96]+[maximum]
    ax.set_xticks(ticks,labels=[str(v) for v in ticks]);ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
    ax.axhline(.6,color='#228b22',ls='--',lw=1.5,label='R² = 0.6')
    ax.axhline(0,color='#555555',ls=':',lw=.7)

def endpoint(ax,counts,score):
    center=float(np.mean(counts));lo=min(counts);hi=max(counts)
    ax.errorbar(center,score,xerr=np.array([[center-lo],[hi-center]]),fmt='D',markersize=6,
                color='#444444',capsize=3,label=f'All effective: {lo}–{hi} / fold')

def annotate_selection_scores(ax,rows):
    """Label the exact held-out scores, without interpolation or model refits."""
    offsets={8:(-30,25),16:(-12,-43),32:(22,-35)}
    for k,offset in offsets.items():
        row,=[r for r in rows if int(r['channels'])==k]
        score=float(row['heldout_r2_mean'])
        ax.annotate(f'{k} ch\nR² = {score:.3f}',xy=(k,score),xytext=offset,
                    textcoords='offset points',ha='center',va='center',fontsize=9,
                    color='#185781',bbox=dict(boxstyle='round,pad=.25',fc='white',ec='none',alpha=.9),
                    arrowprops=dict(arrowstyle='-',color='#2176ae',lw=.8),zorder=8)

def presentation_font_scale(fig,factor=2):
    """Scale all plot typography once; leave the numerical curves unchanged."""
    from matplotlib.text import Text, Annotation
    fig.canvas.draw()
    for text in fig.findobj(Text):
        text.set_fontsize(text.get_fontsize()*factor)
        if isinstance(text,Annotation) and text.get_text().startswith(('8 ch','16 ch','32 ch')):
            offsets={'8 ch':(-60,32),'16 ch':(-24,-65),'32 ch':(35,-60)}
            text.set_position(offsets[text.get_text().split('\n')[0]])
    fig.canvas.draw()

def plot_dropout(out,label,trials,counts,random,least,most,baseline,scales=('log','linear')):
    q=np.quantile(random.mean(2),[.05,.5,.95],axis=0)
    for scale in scales:
        fig,ax=plt.subplots(figsize=(10,5.8),layout='constrained')
        start=1 if scale=='log' else 0;x=np.arange(start,len(least))
        ax.fill_between(x,q[0,start:],q[2,start:],color='#2176ae',alpha=.16,label='Random 5-95% (200 paths)')
        ax.plot(x,q[1,start:],color='#2176ae',lw=2,label='Random median')
        ax.plot(x,most.mean(1)[start:],color='#e07b19',ls='-.',lw=2,label='Drop most-important first')
        axis(ax,counts,scale);endpoint(ax,counts,baseline)
        ax.legend(fontsize=8,loc='upper left')
        ax.set_title(f'{label} Dropout',fontsize=11)
        presentation_font_scale(fig)
        name='dropout' if scale=='log' else 'dropout_linear'
        fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight',pad_inches=.15);fig.savefig(out/f'{name}.svg',bbox_inches='tight',pad_inches=.15);plt.close(fig)

def dropout(data,out,label):
    folds=[];baseline=np.full((data.n,2),np.nan);fold_rows=[]
    for f,tr,te in S.block_folds(data.block):
        model=S.fit(data.subset(tr));baseline[te]=S.predict(model,data.subset(te));folds.append((f,tr,te,model))
        fold_rows.append(dict(fold=f+1,effective_channels=len(model['cols']),train_bins=len(tr),eval_bins=len(te),
             train_trials=len(np.unique(data.trial[tr])),eval_trials=len(np.unique(data.trial[te])),
             train_blocks=json.dumps(np.unique(data.block[tr]).tolist()),eval_blocks=json.dumps(np.unique(data.block[te]).tolist())))
    reference=json.loads(S.ORIGINAL.read_text(encoding='utf-8'));observed=S.r2_score(data.velocity,baseline)
    np.testing.assert_allclose(observed,[reference['summary']['velocity_kf_r2_x'],reference['summary']['velocity_kf_r2_y']],atol=1e-10)
    counts=[len(m['cols']) for _,_,_,m in folds];common=min(counts)
    rand_sse=np.zeros((200,common+1,2));least_sse=np.zeros((common+1,2));most_sse=least_sse.copy()
    checks=[];ranks=[]
    for f,tr,te,model in folds:
        train,test=data.subset(tr),data.subset(te);n=len(model['cols'])
        contribution=S.contributions(model,test);residual=baseline[te]-test.velocity
        importance=S.importance(model,train);order=np.argsort(importance,kind='stable')
        rng=np.random.default_rng(20261110+f)
        random_orders=np.array([rng.permutation(n) for _ in range(200)])
        keep_indices=n-np.arange(common+1)
        least_sse+=S.path_sse(contribution,residual,[order])[0,keep_indices]
        most_sse+=S.path_sse(contribution,residual,[order[::-1]])[0,keep_indices]
        rand_sse+=S.path_sse(contribution,residual,random_orders)[:,keep_indices]
        for method,permutation in [('least',order),('most',order[::-1]),('random',random_orders[0])]:
            for retained in (0,8,64,common):
                removed=model['cols'][permutation[:n-retained]]
                x=test.features.copy();x[:,removed]=0
                direct=S.predict(model,test.with_features(x))
                additive=baseline[te]-contribution[:,:,permutation[:n-retained]].sum(2)
                np.testing.assert_allclose(direct,additive,atol=2e-7,rtol=1e-8)
                checks.append(dict(fold=f+1,method=method,retained=retained,max_error=float(abs(direct-additive).max())))
        ranks.extend(dict(fold=f+1,removal_rank=j+1,column_0based=int(model['cols'][c]),
            physical_channel_id=int(data.electrode_ids[model['cols'][c]]),importance=float(importance[c])) for j,c in enumerate(order))
    sst=np.square(data.velocity-data.velocity.mean(0)).sum(0)
    random=1-rand_sse/sst;least=1-least_sse/sst;most=1-most_sse/sst
    np.testing.assert_allclose(least[0],most[0],atol=1e-12)
    quantile=np.quantile(random.mean(2),[.05,.5,.95],axis=0)
    rows=[dict(retained=k,least_first_r2_mean=float(least[k].mean()),most_first_r2_mean=float(most[k].mean()),
               random_p05=float(quantile[0,k]),random_median=float(quantile[1,k]),random_p95=float(quantile[2,k])) for k in range(common+1)]
    S.table(out/'dropout_curve.csv',rows);S.table(out/'folds.csv',fold_rows);S.table(out/'channel_importance.csv',ranks)
    np.savez_compressed(out/'dropout_details.npz',random_r2=random,least_r2=least,most_r2=most,baseline=baseline,truth=data.velocity)
    S.dump(out/'dropout_summary.json',dict(run=label,baseline_r2_mean=float(observed.mean()),effective_counts_by_training_fold=counts,
        all_data_effective_count=int((data.features.std(0)>1e-12).sum()),common_integer_budget_max=common,
        all_effective_endpoint_x=float(np.mean(counts)),endpoint_note='Separate full endpoint uses fold-varying effective counts; horizontal bar is count range. Integer curves use identical complete OOF samples at every k.',
        random_policy='200 random permutations of ONLY training-effective channels per fold; inactive input slots excluded',direct_checks=checks))
    plot_dropout(out,label,len(np.unique(data.trial)),counts,random,least,most,float(observed.mean()))
    print(label,'DROPOUT COMPLETE; effective counts',counts,'baseline',observed.mean(),flush=True)
    return folds,baseline,counts

def selection(data,out,label,folds,baseline,counts):
    targets=sorted(set([min(counts)]+[k for k in GRID if k<min(counts)]),reverse=True)
    prediction={k:np.full((data.n,2),np.nan) for k in targets};history=[];chosen=[];checks=[]
    for f,tr,te,outer in folds:
        train,test=data.subset(tr),data.subset(te);pool=outer['cols'];inner=[]
        for _,itr,ite in S.block_folds(train.block,3):inner.append((S.fit(train.subset(itr)),train.subset(ite)))
        models=[m for m,_ in inner];initial=pool[np.argsort(-weights(pool,models),kind='stable')]
        previous=pool.copy();memo={};rng=np.random.default_rng(1111+f)
        def score(cols):
            key=tuple(sorted(map(int,cols)))
            if key not in memo:
                values=[float(S.r2_score(v.velocity,S.predict(S.subset(m,key),v)).mean()) for m,v in inner]
                memo[key]=float(np.mean(values));history.append(dict(outer_fold=f+1,channels=len(key),inner_r2=memo[key],
                            inner_fold_r2=json.dumps(values),columns=json.dumps(key)))
            return memo[key]
        for k in targets:
            warm=previous[np.argsort(-weights(previous,models),kind='stable')[:k]]
            starts=[warm,initial[:k]]
            if k<128 and k<len(pool):
                extra=initial[:k].copy();n=min(max(1,min(3,k//4)),len(pool)-k)
                extra[-n:]=rng.choice(initial[k:min(len(pool),k+32)],n,replace=False);starts.append(extra)
            best=None;best_score=-np.inf
            for seed in starts:
                current=np.sort(seed);value=score(current)
                for _ in range(2):
                    outside=np.setdiff1d(pool,current)
                    if not len(outside):break
                    weak=current[np.argsort(weights(current,models),kind='stable')[:min(2,k)]]
                    eligible=set(outside);incoming=[c for c in initial if c in eligible][:3]
                    incoming+=list(rng.choice(outside,min(2,len(outside)),replace=False))
                    nxt=current;nv=value
                    for remove in weak:
                        for add in sorted(set(incoming)):
                            candidate=np.sort(np.r_[current[current!=remove],add]);v=score(candidate)
                            if v>nv+1e-6:nxt,nv=candidate,v
                    if nv<=value+1e-6:break
                    current,value=nxt,nv
                if value>best_score:best,best_score=current,value
            assert len(best)==k and set(best).issubset(pool)
            previous=best;prediction[k][te]=S.predict(S.subset(outer,best),test)
            chosen.append(dict(outer_fold=f+1,channels=k,inner_score=best_score,columns_0based=json.dumps(best.tolist()),
                               physical_channel_ids=json.dumps(data.electrode_ids[best].tolist())))
            if k in (64,8,1,targets[0]):
                independent=S.fit(train.columns(best));direct=S.predict(independent,test.columns(best))
                np.testing.assert_allclose(direct,prediction[k][te],atol=2e-7,rtol=1e-8)
                checks.append(dict(fold=f+1,channels=k,max_prediction_error=float(abs(direct-prediction[k][te]).max())))
        print(label,'selection fold',f+1,'candidates',len(memo),flush=True)
    boots=bootstrap(data,dict(prediction,all_effective=baseline));base=boots['all_effective'];rows=[]
    for k,p in prediction.items():
        r=S.r2_score(data.velocity,p);b=boots[k]
        rows.append(dict(channels=k,heldout_r2_mean=float(r.mean()),heldout_r2_x=float(r[0]),heldout_r2_y=float(r[1]),
             ci_low=float(np.quantile(b,.025)),ci_high=float(np.quantile(b,.975)),
             mean_inner_score=float(np.mean([a['inner_score'] for a in chosen if a['channels']==k])),
             paired_delta_ci_low=float(np.quantile(b-base,.025)),paired_delta_ci_high=float(np.quantile(b-base,.975))))
    S.table(out/'selection_curve.csv',rows);S.table(out/'selection_chosen_channels.csv',chosen);S.table(out/'selection_search_history.csv',history)
    np.savez_compressed(out/'selection_predictions.npz',**{f'prediction_{k}':p for k,p in prediction.items()},all_effective=baseline,truth=data.velocity)
    S.dump(out/'selection_summary.json',dict(run=label,targets=targets,candidates_evaluated=len(history),outer_folds=5,inner_folds=3,purge_blocks=1,
         pool='outer-training-effective channels only; inner estimator drops its own training constants',search='backward importance pruning, 2–3 starts, 2 local swap rounds; no global optimum guarantee',
         effective_counts_by_fold=counts,all_effective_r2=float(S.r2_score(data.velocity,baseline).mean()),independent_refit_checks=checks,
         confidence='2000 block-bootstrap draws of fixed OOF predictions; does not rerun feature selection',results=rows))
    for scale in ('log','linear'):
        fig,ax=plt.subplots(figsize=(10,5.8),layout='constrained');rs=sorted(rows,key=lambda r:r['channels']);x=[r['channels'] for r in rs]
        ax.fill_between(x,[r['ci_low'] for r in rs],[r['ci_high'] for r in rs],color='#2176ae',alpha=.16,label='95% block-bootstrap interval')
        ax.plot(x,[r['heldout_r2_mean'] for r in rs],'-o',color='#2176ae',label='Selected / retrained, outer held-out')
        b=float(S.r2_score(data.velocity,baseline).mean())
        ax.axhline(b,color='#555555',ls=':',label=f'All-effective baseline = {b:.3f}')
        axis(ax,counts,scale);endpoint(ax,counts,b);ax.legend(fontsize=8,loc='lower right')
        annotate_selection_scores(ax,rs)
        ax.set_title(f'{label} Best Channel Retrain',fontsize=11)
        presentation_font_scale(fig)
        name='selection' if scale=='log' else 'selection_linear'
        fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight',pad_inches=.15);fig.savefig(out/f'{name}.svg',bbox_inches='tight',pad_inches=.15);plt.close(fig)
    final=S.KF.fit(data,S.PARAMS)
    with (out/'all_effective_final_model.pkl').open('wb') as f:
        pickle.dump(dict(model=final,params=S.PARAMS,input_electrode_ids=data.electrode_ids,
              effective_electrode_ids=data.electrode_ids[final['keep']],training_trials=len(np.unique(data.trial)),
              note='Full-data final refit; plotted OOF score belongs to separate CV models.'),f)
    print(label,'SELECTION COMPLETE',json.dumps([{k:r[k] for k in ('channels','heldout_r2_mean')} for r in rows]),flush=True)

def run_job(date,task,run,label,retrain=True):
    root=S.ROOT/'keaiduo';sources=list((root/'single').glob(f'2026-11-{date[-2:]}_{task}_{run}_*/s10_v1/*/result.json'))
    expected={('1111','run-001'):'55fce96bf13e',('1111','run-006'):'a852bba83f05',('1110','run-001'):'c672db424160'}
    sources=[p for p in sources if p.parent.name==expected[(date,run)]]
    assert len(sources)==1,sources
    S.ORIGINAL=sources[0];reference=json.loads(S.ORIGINAL.read_text(encoding='utf-8'))
    sha=reference['inputs']['recordings'][0]['sha256'];caches=list((root/'.session-cache'/sha[:20]/'decoder_data').glob('*.npz'))
    expected_cache={('1111','run-001'):'ee048596a8b5890576c00096548df9fc2e58524e4e7bab1b5f333872593f3285',
                    ('1111','run-006'):'d3a2c8a997c0552d98ba862df11d5a80bb853cc98095a87741b6c84857c5bd53',
                    ('1110','run-001'):'2ea32d6023df8a0c8330e9153fd9ad41fca1416ee831cc535246bd3ad9ba4376'}
    caches=[p for p in caches if p.stem==expected_cache[(date,run)]]
    assert len(caches)==1,caches
    S.CACHE=caches[0];S.PARAMS={k:reference['inputs']['params'][k] for k in ('lam_kf','q_shrinkage')}
    data=S.load();out=S.ROOT/f'channel_budget_{date}_{run.replace("-","")}_effective';out.mkdir(exist_ok=True)
    folds,baseline,counts=dropout(data,out,label)
    if retrain:selection(data,out,label,folds,baseline,counts)
    S.dump(out/'provenance.json',dict(run=label,original_result=str(S.ORIGINAL),source_recording_sha256=sha,
        cache_sha256=hashlib.sha256(S.CACHE.read_bytes()).hexdigest(),independent_single_run=True,
        scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(S.__file__))}))
    for p in (Path(__file__),Path(S.__file__)):shutil.copy2(p,out/p.name)
    note=f'''# {label}: effective-channel-only analysis

Single run; {len(np.unique(data.trial))} trials / {data.n} bins. Same S10 filtering and Vel-KF parameters. Training-effective counts by outer fold: {counts}; all-data count: {int((data.features.std(0)>1e-12).sum())}. No inactive input slots included in the dropout pool or candidate budget.

Integer dropout curves use k=0..{min(counts)} so every point is evaluated on identical complete OOF samples. The separate all-effective marker uses each fold's full model, drawn at mean effective count with horizontal range min..max; no interpolation or changed test coverage. Randomly losing active channels is recomputed, not a cropped 512-slot curve.

All figures have green R²=0.6 lines. Log plots omit k=0; linear plots include it. dropout.png / dropout_linear.png freeze one complete trained model per fold; selection.png / selection_linear.png (when produced) use nested training-only channel selection. The final all-data model has no separate independent test score. Bootstrap uncertainty conditions on selected sets and does not repeat the search. Candidate search is heuristic, not a global optimum.

Original result: {S.ORIGINAL}. Numeric results, channel IDs, actual block splits and independent direct-replay/refit checks are saved alongside the figures. Previous 512-slot output folders are retained as historical versions.
'''
    (out/'METHOD.md').write_text(note,encoding='utf-8')

if __name__=='__main__':
    with S.threadpool_limits(limits=2):
        run_job('1111','CO8','run-001','1111 CO8 run-001')
        run_job('1111','Grid','run-006','1111 Grid 12x12 run-006')
        run_job('1110','CO8','run-001','1110 CO8 run-001',retrain=False)
