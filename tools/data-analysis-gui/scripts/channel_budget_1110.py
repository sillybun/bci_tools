"""Reproducible channel dropout / retraining study of the archived 1110 PVKF.

No source data or archived weights are modified. Dropout is raw-count zero fill;
the complete KF observation model and covariance stay frozen.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse
import csv
import hashlib
import json
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

SOURCE = Path('E:/AitBCI/data/keaiduo_1109_1110_1111_整理包/1110/training_and_deployment/run001_pvkf_raw_lag6_chronological80_20_final')
OUTPUT = Path('E:/AitBCI/data_analysis_gui_output/channel_budget_1110_run001')

def sym(a):
    return (a + a.T) / 2

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def table(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def load():
    with (SOURCE / 'position_velocity_kf_raw_lag6.pkl').open('rb') as f:
        model = pickle.load(f)
    metrics = json.loads((SOURCE / 'metrics.json').read_text())
    trials = []
    with np.load(SOURCE / 'run001_raw_processed_velocity_trials.npz', allow_pickle=True) as z:
        neural, velocity, position, times = [z[k] for k in ('neural', 'velocity', 'recon_position', 'lsl_time')]
        for i in range(len(neural)):
            trials.append(dict(index=i, x=np.asarray(neural[i][:-6], float),
                               y=np.asarray(velocity[i], np.float32)[6:].astype(float),
                               p=np.asarray(position[i], np.float32)[6:].astype(float),
                               time=np.asarray(times[i], float)[6:]))
    train = [trials[i] for i in metrics['train_trial_indices']]
    test = [trials[i] for i in metrics['validation_trial_indices']]
    assert max(t['time'][-1] for t in train) < min(t['time'][0] for t in test)
    assert sum(len(t['x']) for t in test) == metrics['validation_samples']
    return model, train, test, metrics

def filter_terms(model, length):
    """Exactly the archived information-form update, factored into F*x+G*y+b."""
    a, q = model['A'], sym(model['Q'])
    cr = model['C'].T @ np.linalg.pinv(sym(model['R']))
    info = sym(cr @ model['C'])
    p = sym(model['P0'])
    fs, gs, bs = [], [], []
    for _ in range(length):
        prior = np.linalg.pinv(sym(a @ p @ a.T + q))
        p = sym(np.linalg.pinv(sym(prior + info)))
        fs.append(p @ prior @ a)
        gs.append(p @ cr)
        bs.append(p @ (prior @ model['state_offset'] - cr @ model['observation_bias']))
    return np.array(fs), np.array(gs), np.array(bs)

def predict(model, trials, terms=None, cols=None):
    if terms is None:
        terms = filter_terms(model, max(len(t['x']) for t in trials))
    fs, gs, bs = terms
    out = []
    for t in trials:
        state = np.r_[t['p'][0], 0., 0.]
        pred = []
        for i, x in enumerate(t['x']):
            if cols is not None:
                x = x.copy(); x[cols] = 0
            state = fs[i] @ state + gs[i] @ x + bs[i]
            pred.append(state[2:])
        out.append(np.array(pred))
    return np.vstack(out)

def contributions(trials, terms, center=None):
    """Exact per-channel linear responses; resets match archived trial resets."""
    fs, gs, _ = terms
    out = []
    for t in trials:
        response = np.zeros((4, t['x'].shape[1]))
        v = np.empty((len(t['x']), 2, t['x'].shape[1]))
        for i, x in enumerate(t['x']):
            if center is not None:
                x = x - center
            response = fs[i] @ response + gs[i] * x[None, :]
            v[i] = response[2:]
        out.append(v)
    return np.concatenate(out)

def r2(y, p):
    return 1 - np.square(y - p).sum(0) / np.square(y - y.mean(0)).sum(0)

def fit_full(trials, dt):
    """Archived PVKF estimator; all observation rows fitted independently.

    Selecting C rows / R principal submatrices is algebraically identical to
    fitting that subset again, because A/Q/P0 depend only on behavior, and each
    C row is its own least-squares fit. Gains are recomputed for every subset.
    """
    x = np.vstack([t['x'] for t in trials]).T
    state = np.vstack([np.column_stack([t['p'], t['y']]) for t in trials]).T
    v1 = np.vstack([t['y'][:-1] for t in trials]).T
    v2 = np.vstack([t['y'][1:] for t in trials]).T
    av = (v2 @ v1.T) @ np.linalg.pinv(v1 @ v1.T)
    a = np.eye(4); a[:2,2:] = np.eye(2)*dt; a[2:,2:] = av
    residual_v = v2-av@v1
    q = np.zeros((4,4)); q[2:,2:] = residual_v@residual_v.T/(residual_v.shape[1]-1) + 1e-8*np.eye(2)
    design = np.vstack([state, np.ones(state.shape[1])])
    coef = (x @ design.T) @ np.linalg.pinv(design @ design.T)
    residual = x - coef @ design
    return dict(A=a, Q=q, C=coef[:,:4], observation_bias=coef[:,4], R=np.cov(residual)+1e-6*np.eye(x.shape[0]),
                P0=np.cov(state)+np.eye(4), state_offset=np.zeros(4), state0=np.r_[trials[0]['p'][0],0.,0.],
                format_version=1, dt=dt)

def subset_model(model, cols):
    cols = np.array(cols, int)
    return dict(model, C=model['C'][cols], R=model['R'][np.ix_(cols,cols)], observation_bias=model['observation_bias'][cols])

def fast_terms(model, length):
    a, q = model['A'], sym(model['Q'])
    cr = cho_solve(cho_factor(sym(model['R']), lower=True), model['C']).T
    info = sym(cr @ model['C'])
    p = sym(model['P0'])
    fs, ps, bs = [], [], []
    for _ in range(length):
        prior = np.linalg.pinv(sym(a @ p @ a.T + q))
        p = sym(np.linalg.pinv(sym(prior + info)))
        fs.append(p @ prior @ a); ps.append(p.copy())
        bs.append(p @ (prior @ model['state_offset'] - cr @ model['observation_bias']))
    return np.array(fs), np.array(ps), np.array(bs), cr

def predict_fast(model, trials, cols):
    """Batch equal within-trial steps, never carrying state between trials."""
    lengths = np.array([len(t['x']) for t in trials])
    fs, ps, bs, cr = fast_terms(model, int(lengths.max()))
    x = np.vstack([t['x'][:,cols] for t in trials])
    drive = x @ cr.T
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    state = np.array([np.r_[t['p'][0], 0., 0.] for t in trials])
    result = np.empty((len(x),2))
    for step in range(int(lengths.max())):
        active = np.flatnonzero(lengths > step)
        rows = starts[active] + step
        state[active] = state[active] @ fs[step].T + drive[rows] @ ps[step].T + bs[step]
        result[rows] = state[active,2:]
    return result

def gain_importance(model, std):
    """Training-scaled finite-horizon impulse energy, includes filter feedback."""
    fs, ps, _, cr = fast_terms(model, 120)
    cov = np.zeros((len(std),4,4))
    energy = np.zeros(len(std))
    for f,p in zip(fs,ps):
        g = (p @ cr).T * std[:,None]
        cov = f[None,:,:] @ cov @ f.T[None,:,:] + g[:,:,None]*g[:,None,:]
        energy += cov[:,2,2] + cov[:,3,3]
    return np.sqrt(energy/len(fs))

def selection():
    started = time.time()
    out = OUTPUT / 'selection'; out.mkdir(parents=True, exist_ok=True)
    archived, train, test, metadata = load()
    dt = archived['dt']
    full = fit_full(train, dt)
    refit_errors = {key:float(np.max(np.abs(full[key]-archived[key]))) for key in ('A','Q','C','R','P0')}
    print('Full-estimator reproduction errors', refit_errors, flush=True)
    base_retrained = predict_fast(full,test,list(range(512)))
    reference = predict(archived,test)
    # Historical trainer accumulated float32 behavior products; this refit uses
    # float64, so validate small numerical differences rather than bit identity.
    np.testing.assert_allclose(base_retrained, reference, atol=.002, rtol=1e-5)
    np.testing.assert_allclose(r2(np.vstack([t['y'] for t in test]),base_retrained),
                               r2(np.vstack([t['y'] for t in test]),reference),atol=1e-6)
    x = np.vstack([t['x'] for t in train]); std = x.std(0)
    pool = np.flatnonzero(std > 0)
    # Three expanding-window validations, all strictly before the outer test.
    folds = []
    for split,stop in ((78,104),(104,130),(130,156)):
        folds.append(dict(model=fit_full(train[:split],dt), validation=train[split:stop],
                          truth=np.vstack([t['y'] for t in train[split:stop]])))
    rank = np.mean([gain_importance(subset_model(f['model'],pool),
                  np.vstack([t['x'] for t in train[:split]])[:,pool].std(0))
                  for f,split in zip(folds,(78,104,130))], axis=0)
    initial_rank = pool[np.argsort(-rank,kind='stable')]
    rng = np.random.default_rng(1110)
    cache = {}; history=[]
    def score(cols):
        key = tuple(sorted(map(int,cols)))
        if key not in cache:
            scores=[]
            for f in folds:
                p = predict_fast(subset_model(f['model'],key), f['validation'], list(key))
                scores.append(float(r2(f['truth'],p).mean()))
            cache[key] = float(np.mean(scores))
            history.append(dict(evaluation=len(history)+1, channels=len(key), inner_mean_r2=cache[key],
                                inner_fold_r2=json.dumps(scores), columns=json.dumps(key)))
        return cache[key]
    targets=[341,256,192,128,96,80,64,56,48,40,32,24,16,12,8,4,2,1]
    best_sets={}; previous=np.array(pool)
    for k in targets:
        # Recompute importance as the model shrinks, then try separate starts.
        weights=gain_importance(subset_model(folds[-1]['model'],previous),std[previous])
        warm=previous[np.argsort(-weights,kind='stable')[:k]]
        starts=[warm,initial_rank[:k]]
        if k < 128:
            exploratory=initial_rank[:k].copy()
            nreplace=max(1,min(3,k//4))
            exploratory[-nreplace:]=rng.choice(initial_rank[k:min(len(pool),k+32)],nreplace,replace=False)
            starts.append(exploratory)
        best=None; best_score=-np.inf
        for seed_cols in starts:
            chosen=np.array(sorted(seed_cols)); current=score(chosen)
            for iteration in range(2):
                outside=np.setdiff1d(pool,chosen)
                if not len(outside): break
                w=gain_importance(subset_model(folds[-1]['model'],chosen),std[chosen])
                weak=chosen[np.argsort(w)[:min(2,k)]]
                outside_order=[c for c in initial_rank if c in set(outside)]
                incoming=list(outside_order[:3])
                incoming+=list(rng.choice(outside,min(2,len(outside)),replace=False))
                next_cols=chosen; next_score=current
                for remove in weak:
                    for add in sorted(set(incoming)):
                        candidate=np.sort(np.r_[chosen[chosen!=remove],add])
                        value=score(candidate)
                        if value > next_score+1e-6: next_score=value; next_cols=candidate
                if next_score <= current+1e-6: break
                chosen,current=next_cols,next_score
            if current>best_score: best,best_score=chosen,current
        best_sets[k]=best; previous=best
        dump(out/'chosen_sets_checkpoint.json',{str(k):v.tolist() for k,v in best_sets.items()})
        table(out/'search_history.csv',history)
        print('Selected',k,'channels, inner R2',round(best_score,4),'evaluations',len(cache),
              'seconds',round(time.time()-started,1),flush=True)
    # Lock all channel sets BEFORE inspecting any outer held-out subset score.
    rows=[]; prediction_arrays={}; ids=np.array(archived['metadata']['physical_channel_ids'])
    for k,cols in best_sets.items():
        p=predict_fast(subset_model(full,cols),test,list(cols))
        values=r2(np.vstack([t['y'] for t in test]),p)
        prediction_arrays[f'prediction_{k}']=p
        rows.append(dict(channels=k,inner_validation_r2=score(cols),heldout_r2_mean=float(values.mean()),
                         heldout_r2_x=float(values[0]),heldout_r2_y=float(values[1]),
                         physical_channel_ids=json.dumps(ids[cols].tolist()), columns_0based=json.dumps(cols.tolist())))
        with (out/f'pvkf_{k}ch.pkl').open('wb') as f:
            pickle.dump(dict(subset_model(full,cols),model_type='position_velocity',
                metadata=dict(archived['metadata'],physical_channel_ids=ids[cols].tolist(),
                              selected_channel_indices=cols.tolist(),channels=k,input_channels=512,
                              channel_selection='training-only backward pruning with local swaps')),f)
    table(out/'selection_curve.csv',rows)
    np.savez_compressed(out/'heldout_predictions.npz',truth=np.vstack([t['y'] for t in test]),**prediction_arrays)
    fig,ax=plt.subplots(figsize=(9,5.4),layout='constrained')
    plotted=sorted(rows,key=lambda r:r['channels'])
    ax.plot([r['channels'] for r in plotted],[r['heldout_r2_mean'] for r in plotted],'-o',color='#2176ae',label='Retrained selected subset (held-out)')
    ax.plot([r['channels'] for r in plotted],[r['inner_validation_r2'] for r in plotted],'--',color='#c16e14',label='Inner validation (selection score)')
    ax.axhline(float(r2(np.vstack([t['y'] for t in test]),base_retrained).mean()),ls=':',color='#444444',label='512-channel baseline')
    ax.set(xlabel='Channels retained',ylabel='Velocity R²: mean of x / y',title='1110 CO8 run-001 | PVKF retrained channel selection',xscale='log',ylim=(-.3,.85))
    ticks=[1,2,4,8,16,32,64,128,256,341]
    ax.set_xticks(ticks,labels=[str(v) for v in ticks]); ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=9)
    fig.savefig(out/'selection.png',dpi=180);fig.savefig(out/'selection.svg');plt.close(fig)
    summary=dict(available_recorded_channels=512,training_variable_channels=len(pool),training_zero_channels=int((std==0).sum()),
                 outer_train_trials=156,outer_test_trials=39,inner_splits=[[78,104],[104,130],[130,156]],
                 inner_score='equal-weight mean of 3 expanding-window mean-axis R2 scores',
                 search='adaptive backward pruning, 2-3 starts per k, up to 2 local one-for-one swap rounds per start; no global optimum guarantee',
                 refitting='exact cached sufficient statistics: training-fit C/R rows and principal submatrices; subset-specific KF gain/covariance recomputed',
                 source_estimator_reproduction_max_errors=refit_errors,candidate_sets_evaluated=len(cache),
                 heldout_used_only_after_all_sets_locked=True,elapsed_seconds=time.time()-started,results=rows)
    dump(out/'selection_summary.json',summary)
    print('SELECTION COMPLETE',json.dumps([{k:r[k] for k in ('channels','heldout_r2_mean','heldout_r2_x','heldout_r2_y')} for r in rows]),flush=True)

def verify():
    out=OUTPUT/'selection'
    archived,train,test,_=load()
    full=fit_full(train,archived['dt'])
    sets=json.loads((out/'chosen_sets_checkpoint.json').read_text())
    checks=[]
    for k in (341,64,8,1):
        cols=np.array(sets[str(k)])
        selected=subset_model(full,cols)
        independent=fit_full([dict(t,x=t['x'][:,cols]) for t in train],archived['dt'])
        errors={name:float(abs(independent[name]-selected[name]).max()) for name in ('A','Q','C','R','P0')}
        for name in errors:
            np.testing.assert_allclose(independent[name],selected[name],atol=1e-9,rtol=1e-8)
        reduced_test=[dict(t,x=t['x'][:,cols]) for t in test]
        direct=predict(independent,reduced_test)
        fast=predict_fast(selected,test,list(cols))
        np.testing.assert_allclose(direct,fast,atol=2e-7,rtol=1e-8)
        checks.append(dict(channels=k,refit_array_max_errors=errors,prediction_max_error=float(abs(direct-fast).max())))
    # Repair metadata on generated offline model artifacts, including initial state.
    for k,cols in sets.items():
        path=out/f'pvkf_{k}ch.pkl'
        with path.open('rb') as f: model=pickle.load(f)
        model.update(state0=full['state0'],format_version=archived['format_version'])
        with path.open('wb') as f:pickle.dump(model,f)
    with np.load(SOURCE/'run001_raw_processed_velocity_trials.npz',allow_pickle=True) as z:
        trial_metadata=[json.loads(str(t)) for t in z['trial_metadata_json']]
    blocks=np.concatenate([np.repeat(trial_metadata[t['index']]['block_num'],len(t['x'])) for t in test])
    unique=np.unique(blocks)
    truth=np.vstack([t['y'] for t in test])
    counts=np.array([(blocks==b).sum() for b in unique])
    sums=np.array([truth[blocks==b].sum(0) for b in unique])
    squares=np.array([np.square(truth[blocks==b]).sum(0) for b in unique])
    rng=np.random.default_rng(20261110)
    draws=rng.integers(len(unique),size=(2000,len(unique)))
    denominator=squares[draws].sum(1)-np.square(sums[draws].sum(1))/counts[draws].sum(1)[:,None]
    base=predict(archived,test)
    base_sse=np.array([np.square(truth[blocks==b]-base[blocks==b]).sum(0) for b in unique])
    base_draw=(1-base_sse[draws].sum(1)/denominator).mean(1)
    rows=list(csv.DictReader((out/'selection_curve.csv').open(encoding='utf-8-sig')))
    with np.load(out/'heldout_predictions.npz') as predictions:
        for row in rows:
            k=int(row['channels']); p=predictions[f'prediction_{k}']
            sse=np.array([np.square(truth[blocks==b]-p[blocks==b]).sum(0) for b in unique])
            boot=(1-sse[draws].sum(1)/denominator).mean(1)
            row.update(heldout_r2_block_bootstrap_ci_low=float(np.quantile(boot,.025)),
                       heldout_r2_block_bootstrap_ci_high=float(np.quantile(boot,.975)),
                       delta_vs_512=float(float(row['heldout_r2_mean'])-r2(truth,base).mean()),
                       paired_delta_ci_low=float(np.quantile(boot-base_draw,.025)),
                       paired_delta_ci_high=float(np.quantile(boot-base_draw,.975)))
    table(out/'selection_curve_with_ci.csv',rows)
    fig,ax=plt.subplots(figsize=(10,5.6),layout='constrained')
    plotted=sorted(rows,key=lambda r:int(r['channels']))
    k=np.array([int(r['channels']) for r in plotted]); v=np.array([float(r['heldout_r2_mean']) for r in plotted])
    lo=np.array([r['heldout_r2_block_bootstrap_ci_low'] for r in plotted]);hi=np.array([r['heldout_r2_block_bootstrap_ci_high'] for r in plotted])
    ax.fill_between(k,lo,hi,color='#2176ae',alpha=.15,label='95% block-bootstrap interval (5 blocks)')
    ax.plot(k,v,'-o',color='#2176ae',label='Selected / retrained, held-out')
    ax.plot(k,[float(r['inner_validation_r2']) for r in plotted],'--',color='#c16e14',label='Inner validation, selection score')
    ax.axhline(float(r2(truth,base).mean()),ls=':',color='#444444',label='512-channel baseline')
    ax.set(xlabel='Channels retained (log scale)',ylabel='Velocity R²: mean of x / y',
           title='1110 CO8 run-001 | retrained channel subsets\n156 training trials; 39 held-out trials; 18 channel counts searched',xscale='log')
    ticks=[1,2,4,8,16,32,64,128,256,341]; ax.set_xticks(ticks,labels=[str(v) for v in ticks])
    ax.grid(alpha=.18); ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=9,loc='lower right')
    fig.savefig(out/'selection.png',dpi=180);fig.savefig(out/'selection.svg');plt.close(fig)
    validation=dict(independent_subset_refit_checks=checks,heldout_blocks=unique.tolist(),bootstrap_replicates=2000,
                    bootstrap_unit='original task block; resample same blocks for paired delta vs baseline',
                    uncertainty_limit='Only five held-out blocks; percentile intervals are exploratory.',
                    baseline_block_bootstrap_ci=np.quantile(base_draw,[.025,.975]).tolist())
    dump(out/'verification.json',validation)
    print(json.dumps(validation,indent=2),flush=True)
    print('Selected CI points',json.dumps([{k:r[k] for k in ('channels','heldout_r2_mean','heldout_r2_block_bootstrap_ci_low','heldout_r2_block_bootstrap_ci_high')} for r in rows if int(r['channels']) in (64,32,12,8,4)]),flush=True)

def dropout():
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model, train, test, metrics = load()
    print('Loaded', len(train), 'train /', len(test), 'held-out trials', flush=True)
    terms = filter_terms(model, max(len(t['x']) for t in train + test))
    y = np.vstack([t['y'] for t in test])
    baseline = predict(model, test, terms)
    with np.load(SOURCE / 'position_velocity_kf_validation_predictions.npz') as z:
        np.testing.assert_array_equal(y, z['true_velocity'])
        np.testing.assert_allclose(baseline, z['pred_velocity'], atol=2e-7, rtol=1e-8)
        reproduction_error = float(np.max(np.abs(baseline - z['pred_velocity'])))
    print('Archived baseline reproduced', r2(y, baseline), 'max error', reproduction_error, flush=True)
    train_mean = np.vstack([t['x'] for t in train]).mean(0)
    response_train = contributions(train, terms, center=train_mean)
    importance = np.sqrt(np.mean(np.square(response_train).sum(1), axis=0))
    del response_train
    order = np.argsort(importance, kind='stable')
    response = contributions(test, terms)
    residual = baseline - y
    sst = np.square(y - y.mean(0)).sum(0)
    sse = np.square(residual).sum(0)
    gram = np.array([response[:, axis, :].T @ response[:, axis, :] for axis in range(2)])
    cross = np.array([response[:, axis, :].T @ residual[:, axis] for axis in range(2)])

    def curve(permutation):
        values = []
        for axis in range(2):
            g = gram[axis][np.ix_(permutation, permutation)]
            delta = np.diag(g) + 2 * np.tril(g, -1).sum(1) - 2 * cross[axis, permutation]
            values.append(1 - np.r_[sse[axis], sse[axis] + np.cumsum(delta)] / sst[axis])
        return np.array(values).T

    ranked = curve(order)
    rng = np.random.default_rng(20261110)
    permutations = np.array([rng.permutation(512) for _ in range(200)])
    random_curves = np.array([curve(p) for p in permutations])
    # Independent direct replay checks verify zero-fill and quadratic acceleration.
    checks = []
    for method, permutation in [('ranked', order), ('random', permutations[0])]:
        for count in (0, 64, 256, 512):
            direct = predict(model, test, terms, permutation[:count])
            expected = curve(permutation)[count]
            np.testing.assert_allclose(r2(y, direct), expected, atol=2e-8, rtol=1e-8)
            checks.append(dict(method=method, dropped=count, max_r2_error=float(np.max(np.abs(r2(y, direct)-expected)))))
    random_mean = random_curves.mean(2)
    ranked_mean = ranked.mean(1)
    quantiles = np.quantile(random_mean, [.05, .5, .95], axis=0)
    rows = []
    for dropped in range(513):
        rows.append(dict(dropped=dropped, retained=512-dropped, ranked_r2_mean=float(ranked_mean[dropped]),
                         ranked_r2_x=float(ranked[dropped,0]), ranked_r2_y=float(ranked[dropped,1]),
                         random_r2_p05=float(quantiles[0,dropped]), random_r2_median=float(quantiles[1,dropped]),
                         random_r2_p95=float(quantiles[2,dropped])))
    table(OUTPUT / 'dropout_curve.csv', rows)
    ids = model['metadata']['physical_channel_ids']
    table(OUTPUT / 'channel_importance.csv', [dict(removal_rank=i+1, column_0based=int(c), physical_channel_id=ids[c],
             importance_train_centered_velocity_rms=float(importance[c]), train_mean_raw_count=float(train_mean[c])) for i,c in enumerate(order)])
    np.savez_compressed(OUTPUT / 'dropout_details.npz', random_r2=random_curves, ranked_r2=ranked,
                        permutations=permutations, removal_order=order, importance=importance,
                        baseline_prediction=baseline, truth=y)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), layout='constrained')
    for ax in axes:
        ax.fill_between(np.arange(513), quantiles[0], quantiles[2], color='#2176ae', alpha=.16, label='Random 5-95% (200 paths)')
        ax.plot(quantiles[1], color='#2176ae', lw=2, label='Random median')
        ax.plot(ranked_mean, color='#c16e14', lw=2, label='Least-important first')
        ax.axhline(0, color='#555555', lw=.8, ls=':')
        ax.set(xlabel='Channels lost (of 512)', ylabel='Held-out velocity R²: mean of x / y', xlim=(0,512))
        ax.grid(alpha=.18); ax.spines[['top','right']].set_visible(False)
    axes[0].set_title('Complete result (full vertical range)')
    axes[1].set(title='Detail around useful decoding range', ylim=(-.15,.75))
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('1110 CO8 run-001 | saved PVKF frozen | raw-count zero-fill\n156 train / 39 held-out trials; lag +6; importance uses training data only', fontsize=12)
    fig.savefig(OUTPUT / 'dropout.png', dpi=180)
    fig.savefig(OUTPUT / 'dropout.svg')
    plt.close(fig)
    summary = dict(source=str(SOURCE), training_trials=len(train), heldout_trials=len(test), heldout_samples=len(y),
                   baseline_r2_x=float(r2(y, baseline)[0]), baseline_r2_y=float(r2(y, baseline)[1]),
                   baseline_r2_mean=float(r2(y,baseline).mean()), archived_prediction_max_abs_error=reproduction_error,
                   random_paths=200, seed=20261110, dropout='raw count zero-fill, fixed full C/R and filter parameters',
                   importance='RMS velocity contribution of each centered training-channel signal under the frozen full KF, including temporal dynamics',
                   primary_metric='unweighted mean of x and y velocity R2; not pooled R2', direct_replay_checks=checks,
                   weights_sha256=hashlib.sha256((SOURCE/'position_velocity_kf_raw_lag6.pkl').read_bytes()).hexdigest(),
                   processed_data_sha256=hashlib.sha256((SOURCE/'run001_raw_processed_velocity_trials.npz').read_bytes()).hexdigest(),
                   limitation='Previously used historical validation split; offline movement-only processed trials, not prospective closed-loop validation.',
                   elapsed_seconds=time.time()-started, selected_points=[rows[i] for i in (0,32,64,128,256,384,448,480,496,504,512)])
    dump(OUTPUT / 'dropout_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['dropout','selection','verify'])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        {'dropout':dropout,'selection':selection,'verify':verify}[args.mode]()
