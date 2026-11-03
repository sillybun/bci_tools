"""Run-001 spike analysis on exactly the six run-009 LFP physical channels.

Not a simultaneous modality comparison. Explicit XDF wire/one-based ID mapping,
nested purged block validation, and independent held-out trajectory integration.
"""
from pathlib import Path
import sys
import json
import csv
import hashlib
import time
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, r2_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import lfp_one_vs_rest_1111 as OVR
import decode_lfp_velocity_1111 as V
from data_analysis_gui.core.session import Session
from data_analysis_gui.core.xdf_io import read_headers
from data_analysis_gui.core.features import _interp
from data_analysis_gui.core.analysis import block_folds

SOURCE=Path('E:/AitBCI/data/keaiduo/20261111/run-001-CO8-offline')
OUT=ROOT/'output/spike_20261111_run001_same_six_channels'
WIRE=np.array([302,350,388,393,409,414]);ONE=WIRE+1

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item()),encoding='utf-8')

def table(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def load_data():
    path=next((SOURCE/'xdf').glob('*.xdf'))
    header=next(h for h in read_headers(path) if h.findtext('name') in ('spike_counts','spike_counts_pipe'))
    assert header.findtext('desc/channel_mapping')=='explicit_one_based'
    assert header.findtext('desc/wire_channel_index_base')=='0'
    wire=np.array([int(v) for v in header.findtext('desc/wire_channel_ids').split(',')])
    one=np.array([int(v) for v in header.findtext('desc/channel_ids').split(',')])
    np.testing.assert_array_equal(wire+1,one)
    columns=np.array([int(np.flatnonzero(wire==ch)[0]) for ch in WIRE])
    session=Session.load(path);np.testing.assert_array_equal(session.electrode_ids[columns],ONE)
    nt=session.neural_times;counts=session.counts[:,columns]
    cs=np.vstack([np.zeros((1,6)),np.cumsum(counts,axis=0,dtype=float)])
    period=session.effective_frame_period
    def mean_rate(a,b):
        lo,hi=np.searchsorted(nt,[a,b],side='left');n=hi-lo
        gap=np.max(np.diff(np.r_[a,nt[lo:hi],b]))
        if n<max(1,int(np.floor(.8*(b-a)/period))) or gap>.15+1e-9:return None
        return (cs[hi]-cs[lo])/n/session.frame_seconds
    mapping=[dict(physical_zero_based=int(ch),spike_electrode_one_based=int(ch+1),stored_column_zero_based=int(c),
                  total_recorded_count=int(counts[:,i].sum()),maximum_frame_count=int(counts[:,i].max())) for i,(ch,c) in enumerate(zip(WIRE,columns))]
    provenance=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mapping=mapping,
        physical_mapping_basis='XDF explicit_one_based channel_ids and zero-based wire_channel_ids independently reconcile for all 512 stored columns.',
        profile_id=header.findtext('desc/profile_id'),profile_sha256=header.findtext('desc/profile_sha256'),
        nominal_frame_seconds=session.frame_seconds,effective_frame_period=period,
        rate_definition='Mean observed count per frame / configured 0.02 s. Approximate Hz; ~19.2 ms actual mean frame interval differs from nominal. Training standardization absorbs constant rate scaling.',
        limitations='Run-001 spike and run-009 LFP were recorded at different times, with dwell 0.2 vs 0.01 s and joystick gain 0.012 vs 0.018. This is not a controlled simultaneous modality comparison.')
    dump(OUT/'channel_mapping_and_source.json',provenance)
    return session,mean_rate,provenance

def direction_data(session,rate):
    selected=[];features={'mean_rate':[],'temporal_rates':[],'mean_temporal':[]};excluded={}
    for r in session.trials:
        reason=None
        if not r['complete'] or r['incomplete_reason']:reason='incomplete_or_ambiguous'
        elif r['outcome']!='success':reason='not_success'
        elif r['direction']<0:reason='invalid_direction'
        elif r['acquire_time']<.8:reason='acquisition_before_0.8s'
        if reason:excluded[reason]=excluded.get(reason,0)+1;continue
        start=r['start'];mean=rate(start+.1,start+.7)
        bins=[rate(start+.1+k*.1,start+.2+k*.1) for k in range(6)]
        if mean is None or any(v is None for v in bins):
            excluded['neural_coverage']=excluded.get('neural_coverage',0)+1;continue
        temporal=np.array(bins).T.ravel()
        features['mean_rate'].append(mean);features['temporal_rates'].append(temporal);features['mean_temporal'].append(np.r_[mean,temporal])
        selected.append(r)
    features={k:np.array(v) for k,v in features.items()}
    labels=np.array([r['direction'] for r in selected]);blocks=np.array([r['block'] for r in selected]);trial=np.array([r['trial_id'] for r in selected])
    np.savez_compressed(OUT/'direction_features.npz',labels=labels,blocks=blocks,trial_ids=trial,**features)
    table(OUT/'direction_trials.csv',[{k:r[k] for k in ('trial_id','block','direction','start','end','acquire_time')}for r in selected])
    return features,labels,blocks,trial,excluded

def lda_predict(z,y,tr,te):
    model=make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage=.5))
    return model.fit(z[tr],y[tr]).predict(z[te])

def multiclass(features,y,blocks,permutations=499):
    folds=list(block_folds(blocks,5,purge=1))
    def pipeline(labels,record=False):
        predicted=np.full(len(labels),-1,dtype=int);selected=[]
        for f,tr,te in folds:
            scores={}
            for key,z in features.items():
                p=np.full(len(tr),-1,dtype=int)
                for _,itr,ite in block_folds(blocks[tr],3,purge=1):p[ite]=lda_predict(z,labels,tr[itr],tr[ite])
                scores[key]=balanced_accuracy_score(labels[tr],p)
            best=max(scores,key=scores.get);predicted[te]=lda_predict(features[best],labels,tr,te)
            if record:selected.append(dict(fold=f+1,feature_set=best,inner_balanced_accuracy=scores[best]))
        return predicted,selected
    prediction,choices=pipeline(y,True);accuracy=float(balanced_accuracy_score(y,prediction));null=[];rng=np.random.default_rng(20261114)
    for k in range(permutations):
        yp=y.copy()
        for b in np.unique(blocks):
            idx=np.flatnonzero(blocks==b);yp[idx]=rng.permutation(yp[idx])
        perm,_=pipeline(yp);null.append(balanced_accuracy_score(yp,perm))
        if (k+1)%100==0:print('Eight-way permutation',k+1,'/',permutations,flush=True)
    p=(1+sum(v>=accuracy for v in null))/(permutations+1)
    result=dict(confusion_matrix=confusion_matrix(y,prediction,labels=np.arange(8)),balanced_accuracy=accuracy,
        predictions=prediction,null_scores=null,p_value=p,fold_choices=choices,
        method='8-way StandardScaler + LDA shrinkage 0.5; inner 3-fold selects among 3 spike feature sets; outer 5-fold chronological block CV, +/-1 adjacent block purge. Every permutation reruns nested feature selection; labels shuffled within block. No LFP frequency-band correction applies.')
    dump(OUT/'analysis_results.json',dict(multiclass=result,trials=len(y),direction_counts=np.bincount(y,minlength=8)))
    print('EIGHT WAY',accuracy,'p',p,flush=True)

def binary(features,y,blocks,trial,permutations=199):
    target=OUT/'one_vs_rest';target.mkdir(exist_ok=True)
    plans=OVR.prepared_folds(features,blocks);score,pred,choices=OVR.nested(plans,y)
    rows=OVR.measure(y,score,pred);ci=OVR.bootstrap(y,score,pred,blocks)
    null=[];rng=np.random.default_rng(20261113);start=time.perf_counter()
    for k in range(permutations):
        yp=y.copy()
        for b in np.unique(blocks):
            idx=np.flatnonzero(blocks==b);yp[idx]=rng.permutation(yp[idx])
        s,_,_=OVR.nested(plans,yp,with_threshold=False);null.append(OVR.auc_columns(yp,s))
        if (k+1)%20==0:print('Binary permutation',k+1,'/',permutations,'seconds',time.perf_counter()-start,flush=True)
    null=np.array(null);observed=OVR.auc_columns(y,score)
    p=(1+np.sum(null>=observed,axis=0))/(permutations+1)
    adjusted=(1+np.sum(null.max(1)[:,None]>=observed,axis=0))/(permutations+1)
    for d,r in enumerate(rows):r.update(AUC_CI95_low=ci[0,d,0],AUC_CI95_high=ci[1,d,0],BA_CI95_low=ci[0,d,1],BA_CI95_high=ci[1,d,1],
        permutation_p=p[d],maxAUC_FWER_p_8=adjusted[d],significant_FWER=bool(adjusted[d]<.05))
    dump(target/'one_vs_rest_results.json',dict(trials=len(y),channels_zero_based=WIRE,results=rows,outer_choices=choices,
        method='Same class-balanced ridge, nested purged block CV and inner-only threshold selection as LFP binary analysis. Spike candidates: 6 mean rates, 36 temporal rates, or combined 42 rates.',
        significance=f'{permutations} whole-pipeline restricted label permutations; max AUC across eight tasks corrects familywise p.'))
    table(target/'one_vs_rest_results.csv',rows);table(target/'outer_fold_choices.csv',choices)
    np.savez_compressed(target/'heldout_binary_scores.npz',labels=y,blocks=blocks,trial_ids=trial,scores=score,predictions=pred)
    np.savez_compressed(target/'permutation_null.npz',auc=null)
    print('BINARY',[(r['direction_deg'],r['ROC_AUC'],r['maxAUC_FWER_p_8'])for r in rows],flush=True)

def trajectories(session,rate):
    target=OUT/'trajectory_r2';target.mkdir(exist_ok=True);records=[];excluded={};feature_rows=[]
    ct,cv=session.cursor_times,session.cursor
    for r in session.trials:
        if not r['complete'] or r['incomplete_reason'] or r['outcome']!='success' or r['direction']<0:continue
        idx=session.active_cursor(r);idx=idx[cv[idx,10]==r['trial_id']];t=ct[idx];pos=cv[idx,2:4]
        if not len(t):excluded['no_cursor']=excluded.get('no_cursor',0)+1;continue
        anchor=float(t[0]);count=int(np.floor((r['start']+r['acquire_time']-anchor)/.1+1e-8))
        if count<1:excluded['too_short']=excluded.get('too_short',0)+1;continue
        edges=anchor+np.arange(count+1)*.1;positions=[_interp(t,pos,float(edge),.1) for edge in edges]
        if any(p is None for p in positions):excluded['cursor_gap']=excluded.get('cursor_gap',0)+1;continue
        trial_feats=[]
        for left in edges[:-1]:
            w2=[rate(left-lag-.2,left-lag) for lag in (0,.1,.2)]
            w3=[rate(left-lag-.3,left-lag) for lag in (0,.1,.2)]
            if any(a is None for a in w2+w3):break
            trial_feats.append(dict(forecast_mean_0p2=w2[0],forecast_history_0p2=np.concatenate(w2),forecast_history_0p3=np.concatenate(w3)))
        if len(trial_feats)!=count:excluded['neural_coverage']=excluded.get('neural_coverage',0)+1;continue
        for k,feat in enumerate(trial_feats):
            feature_rows.append(feat)
            records.append(dict(trial=r['trial_id'],block=r['block'],direction=r['direction'],bin=k,sequence=r['trial_id'],left=edges[k],right=edges[k+1],
                p0_x=positions[k][0],p0_y=positions[k][1],p1_x=positions[k+1][0],p1_y=positions[k+1][1]))
    features={k:np.array([r[k] for r in feature_rows]) for k in feature_rows[0]}
    y=np.array([[(r['p1_x']-r['p0_x'])/.1,(r['p1_y']-r['p0_y'])/.1]for r in records])
    trial=np.array([r['trial']for r in records]);blocks=np.array([r['block']for r in records]);seq=trial.copy()
    truth=np.array([[r['p1_x'],r['p1_y']]for r in records]);initial=np.zeros_like(truth);paths=np.zeros_like(truth);pred=np.full_like(y,np.nan);choices=[]
    for f,tr,te in block_folds(blocks,5,purge=1):
        (key,alpha),inner=V.select(features,y,blocks,seq,tr,'forecast','ridge')
        pred[te]=V.fit_predict(features[key],y,seq,tr,te,'ridge',alpha)
        choices.append(dict(fold=f+1,feature_set=key,ridge_alpha=alpha,inner_velocity_r2=inner))
    assert np.isfinite(pred).all()
    endpoints=[]
    for tid in np.unique(trial):
        idx=np.flatnonzero(trial==tid);assert records[idx[0]]['bin']==0
        initial[idx]=[records[idx[0]]['p0_x'],records[idx[0]]['p0_y']]
        paths[idx]=initial[idx]+.1*np.cumsum(pred[idx],axis=0)
        np.testing.assert_allclose(initial[idx]+.1*np.cumsum(y[idx],axis=0),truth[idx],atol=1e-6)
        endpoints.append(idx[-1])
    actual_r2=r2_score(truth,paths,multioutput='raw_values');vel_r2=r2_score(y,pred,multioutput='raw_values')
    scores=dict(position_r2_x=actual_r2[0],position_r2_y=actual_r2[1],position_r2_mean_xy=actual_r2.mean(),
        position_bootstrap_95=V.interval(truth,paths,blocks),velocity_r2_x=vel_r2[0],velocity_r2_y=vel_r2[1],
        trajectory_rmse_xy=float(np.sqrt(np.mean(np.sum((paths-truth)**2,axis=1)))),mean_endpoint_error=float(np.linalg.norm(paths[endpoints]-truth[endpoints],axis=1).mean()))
    examples=[]
    for d in range(8):
        candidates=np.unique(trial[np.array([r['direction'] for r in records])==d]);examples.append(int(candidates[len(candidates)//2]))
    dump(target/'example_selection.json',dict(trial_ids=examples,method='Chronological-middle retained trial per direction; no prediction-error-based choice.'))
    dump(target/'trajectory_results.json',dict(trials=len(np.unique(trial)),trajectory_points=len(y),excluded=excluded,
        scores=dict(forecast_ridge=scores),outer_choices=choices,method='Full outbound complete 100 ms bins; entire trial dropped for cursor/neural gaps. Known initial position once only, integrate outer-held-out forecast velocity without correction. Nested block CV with +/-1 block purge; train-only scaling, feature and alpha selection.'))
    np.savez_compressed(target/'trajectory_predictions.npz',truth=truth,initial_anchor=initial,forecast_ridge=paths,trial=trial,blocks=blocks,velocity_truth=y,velocity_prediction=pred)
    table(target/'trajectory_bins.csv',records)
    print('TRAJECTORY',len(np.unique(trial)),len(y),excluded,scores,flush=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True);session,rate,source=load_data()
    features,y,blocks,trial,excluded=direction_data(session,rate)
    dump(OUT/'cohort.json',dict(direction_trials=len(y),direction_counts=np.bincount(y,minlength=8),excluded=excluded))
    print('MAPPING',source['mapping'],flush=True)
    print('DIRECTION',len(y),np.bincount(y,minlength=8),excluded,flush=True)
    with threadpool_limits(limits=1):
        binary(features,y,blocks,trial)
        multiclass(features,y,blocks)
        trajectories(session,rate)
    print('OUTPUT',OUT,flush=True)

if __name__=='__main__':main()
