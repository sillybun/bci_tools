"""Eight independent one-direction-vs-seven classifiers on the six LFP channels.

Class-balanced ridge classifiers. Every feature, regularization and threshold
choice is nested inside chronological purged block CV. Complete restricted
direction-label permutations rerun the whole nested selection pipeline.
"""
from pathlib import Path
import sys
import json
import csv
import argparse
import time
import hashlib
import shutil
import numpy as np
from scipy import linalg
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, balanced_accuracy_score
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_analysis_gui.core.analysis import block_folds
from data_analysis_gui.core.style import DIRECTION_COLORS

BASE=ROOT/'output/lfp_20261111_run009'
OUT=BASE/'one_vs_rest'
DELIVERY=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report/2026-11-04_2026-11-11_25runs_v1/additional_figures/LFP_One_vs_Rest')
ALPHAS=(.01,.1,1.,10.)

def dump(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item()),encoding='utf-8')

def table(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def prepare(features,tr,te):
    result={}
    for name,z in features.items():
        mu=z[tr].mean(0);sd=z[tr].std(0);sd[sd<1e-12]=1
        result[name]=(np.c_[(z[tr]-mu)/sd,np.ones(len(tr))],np.c_[(z[te]-mu)/sd,np.ones(len(te))])
    return dict(tr=tr,te=te,features=result)

def fit_all(prepared,labels):
    target=(labels[prepared['tr'],None]==np.arange(8)).astype(float)
    n=len(target);positives=target.sum(0);assert np.all((positives>0)&(positives<n))
    weight=np.where(target==1,n/(2*positives),n/(2*(n-positives)))
    results={}
    for name,(train,test) in prepared['features'].items():
        gram=np.array([train.T@(train*weight[:,d,None]) for d in range(8)])
        rhs=np.array([train.T@(weight[:,d]*target[:,d]) for d in range(8)])
        regularization=np.eye(train.shape[1])*n;regularization[-1,-1]=0
        for alpha in ALPHAS:
            coef=np.array([linalg.cho_solve(linalg.cho_factor(gram[d]+alpha*regularization,lower=True,check_finite=False),rhs[d],check_finite=False) for d in range(8)])
            results[(name,alpha)]=test@coef.T
    return results

def auc_columns(labels,scores):
    return np.array([roc_auc_score(labels==d,scores[:,d]) for d in range(8)])

def choose_threshold(y,score):
    values=np.unique(np.r_[.5,np.quantile(score,np.linspace(.05,.95,19))])
    balanced=np.array([balanced_accuracy_score(y,score>=t) for t in values])
    candidates=np.flatnonzero(balanced>=balanced.max()-1e-12)
    best=candidates[np.argmin(np.abs(values[candidates]-.5))]
    return float(values[best])

def prepared_folds(features,blocks):
    plans=[]
    for f,tr,te in block_folds(blocks,5,purge=1):
        assert np.all(np.abs(np.unique(blocks[tr])[:,None]-np.unique(blocks[te])[None,:])>1)
        inner=[prepare(features,tr[itr],tr[ite]) for _,itr,ite in block_folds(blocks[tr],3,purge=1)]
        plans.append(dict(fold=f+1,tr=tr,te=te,outer=prepare(features,tr,te),inner=inner))
    return plans

def fit_selection(plan,labels,with_threshold=True):
    tr=plan['tr'];inner_scores={}
    for inner in plan['inner']:
        result=fit_all(inner,labels)
        # Cache by global row number; all training rows appear once in inner validation.
        for key,values in result.items():
            if key not in inner_scores:inner_scores[key]=np.full((len(labels),8),np.nan)
            inner_scores[key][inner['te']]=values
    keys=list(inner_scores)
    aucs=np.array([auc_columns(labels[tr],inner_scores[key][tr]) for key in keys])
    chosen=np.argmax(aucs,axis=0);outer=fit_all(plan['outer'],labels)
    score=np.zeros((len(plan['te']),8));pred=np.zeros_like(score,dtype=bool);selected=[]
    for d in range(8):
        key=keys[chosen[d]];score[:,d]=outer[key][:,d]
        threshold=choose_threshold(labels[tr]==d,inner_scores[key][tr,d]) if with_threshold else .5
        pred[:,d]=score[:,d]>=threshold
        selected.append(dict(direction_deg=d*45,fold=plan.get('fold',0),feature_set=key[0],ridge_alpha=key[1],inner_AUC=float(aucs[chosen[d],d]),threshold=threshold))
    return score,pred,selected

def nested(plans,labels,with_threshold=True):
    score=np.full((len(labels),8),np.nan);pred=np.zeros_like(score,dtype=bool);selected=[]
    for plan in plans:
        s,p,choice=fit_selection(plan,labels,with_threshold)
        score[plan['te']]=s;pred[plan['te']]=p;selected+=choice
    assert np.isfinite(score).all()
    return score,pred,selected

def measure(labels,score,pred):
    rows=[]
    for d in range(8):
        y=labels==d;p=pred[:,d];tp=int(np.sum(y&p));tn=int(np.sum(~y&~p));fp=int(np.sum(~y&p));fn=int(np.sum(y&~p))
        rows.append(dict(direction_deg=d*45,positive_trials=int(sum(y)),negative_trials=int(sum(~y)),
            ROC_AUC=float(roc_auc_score(y,score[:,d])),average_precision=float(average_precision_score(y,score[:,d])),
            balanced_accuracy=(tp/(tp+fn)+tn/(tn+fp))/2,recall=tp/(tp+fn),specificity=tn/(tn+fp),false_positive_rate=fp/(tn+fp),
            precision=tp/(tp+fp) if tp+fp else 0,accuracy=(tp+tn)/len(y),TP=tp,TN=tn,FP=fp,FN=fn))
    return rows

def bootstrap(labels,score,pred,blocks,n=1000):
    rng=np.random.default_rng(131111);unique=np.unique(blocks);indices=[np.flatnonzero(blocks==b) for b in unique]
    results=[]
    for _ in range(n):
        idx=np.concatenate([indices[i] for i in rng.integers(len(unique),size=len(unique))])
        auc=auc_columns(labels[idx],score[idx]);ba=[]
        for d in range(8):
            y=labels[idx]==d;p=pred[idx,d];ba.append((p[y].mean()+(~p[~y]).mean())/2)
        results.append(np.c_[auc,ba])
    array=np.array(results)
    return np.quantile(array,[.025,.975],axis=0)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--permutations',type=int,default=199)
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    source=BASE/'decoding_features.npz';source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    data=np.load(source);labels=data['labels'];blocks=data['blocks'];trial=data['trial_ids']
    features=dict(all_bands=data['early_power'],high_gamma=data['early_power'][:,5::6],power_waveform=data['early_combined'])
    pre_features=dict(all_bands=data['pre_power'],high_gamma=data['pre_power'][:,5::6],power_waveform=np.c_[data['pre_power'],data['pre_wave']])
    plans=prepared_folds(features,blocks);pre_plans=prepared_folds(pre_features,blocks)
    with threadpool_limits(limits=1):
        start=time.perf_counter();score,pred,choices=nested(plans,labels);elapsed=time.perf_counter()-start
        print('Observed nested pipeline seconds',elapsed,flush=True)
        rows=measure(labels,score,pred);print('Observed metrics',rows,flush=True)
        pre_score,pre_pred,pre_choices=nested(pre_plans,labels)
        intervals=bootstrap(labels,score,pred,blocks)
        np.savez_compressed(OUT/'heldout_binary_scores.npz',scores=score,predictions=pred,pre_scores=pre_score,pre_predictions=pre_pred,labels=labels,blocks=blocks,trial_ids=trial)
        # Every shuffled data set repeats inner feature/alpha selection; thresholds do not affect ROC AUC.
        null=[];rng=np.random.default_rng(20261113)
        for perm in range(args.permutations):
            yp=labels.copy()
            for b in np.unique(blocks):
                idx=np.flatnonzero(blocks==b);yp[idx]=rng.permutation(yp[idx])
            perm_score,_,_=nested(plans,yp,with_threshold=False)
            null.append(auc_columns(yp,perm_score))
            if (perm+1)%10==0:
                np.save(OUT/'permutation_auc_progress.npy',np.array(null))
                print('Permutation',perm+1,'/',args.permutations,'seconds',time.perf_counter()-start,flush=True)
        null=np.array(null);observed=auc_columns(labels,score)
        raw=(1+np.sum(null>=observed,axis=0))/(len(null)+1)
        corrected=(1+np.sum(null.max(1)[:,None]>=observed,axis=0))/(len(null)+1)
        pre_metrics=measure(labels,pre_score,pre_pred)
        for d,row in enumerate(rows):
            row.update(AUC_CI95_low=float(intervals[0,d,0]),AUC_CI95_high=float(intervals[1,d,0]),
                BA_CI95_low=float(intervals[0,d,1]),BA_CI95_high=float(intervals[1,d,1]),
                permutation_p=float(raw[d]),maxAUC_FWER_p_8=float(corrected[d]),significant_FWER=bool(corrected[d]<.05),
                pre_event_AUC=pre_metrics[d]['ROC_AUC'])
        # One forward test: earlier blocks train, block 20 purged, test 21..25.
        tr=np.flatnonzero(blocks<20);te=np.flatnonzero(blocks>20)
        forward_plan=dict(tr=tr,te=te,fold=0,outer=prepare(features,tr,te),
            inner=[prepare(features,tr[itr],tr[ite]) for _,itr,ite in block_folds(blocks[tr],3,purge=1)])
        forward_score,forward_pred,forward_choice=fit_selection(forward_plan,labels)
        forward=measure(labels[te],forward_score,forward_pred)
    table('one_vs_rest_results.csv',rows);table('forward_test_results.csv',forward);table('outer_fold_choices.csv',choices)
    export=[]
    for i,tid in enumerate(trial):
        row=dict(trial=int(tid),block=int(blocks[i]),true_direction_deg=int(labels[i]*45))
        for d in range(8):row[f'score_{d*45}']=float(score[i,d]);row[f'positive_{d*45}']=bool(pred[i,d])
        export.append(row)
    table('trial_binary_predictions.csv',export)
    np.savez_compressed(OUT/'permutation_auc_null.npz',auc=null)
    dump('one_vs_rest_results.json',dict(trials=len(labels),channels_zero_based=[302,350,388,393,409,414],direction_counts=np.bincount(labels,minlength=8),
        model='Eight independent class-balanced ridge classifiers. Targets 0/1; unpenalized intercept; weights give positive and negative classes equal total weight. Scores are not calibrated probabilities.',
        inputs='Only six-channel LFP trial features from 0.1..0.7 s after target/trial onset. Candidate all-six-band power, high-gamma-only power, or power+waveform. Ridge alphas 0.01/0.1/1/10 scaled by n_train.',
        validation='Five chronological outer block folds with +/-1-block purge. Three purged inner block folds choose feature set and alpha by ROC AUC separately for each direction. Threshold maximizes inner-OOF balanced accuracy over score quantiles; never uses outer labels.',
        significance=f'{len(null)} within-block multiclass direction-label permutations; whole nested feature/alpha selection rerun for every permutation. Maximum ROC AUC across the eight binary tasks corrects familywise p. Threshold need not be rerun for threshold-independent AUC.',
        bootstrap='1000 whole-block resamples of fixed outer-OOF predictions; intervals are pointwise, do not repeat training/selection and are not simultaneous across eight tasks.',
        results=rows,outer_choices=choices,pre_event_control=pre_metrics,forward_test=forward,forward_choices=forward_choice,
        forward_split=dict(train_blocks=np.unique(blocks[tr]),purge_block=20,test_blocks=np.unique(blocks[te]),n_test=len(te)),
        multi_detector_positives=dict(no_direction=float(np.mean(pred.sum(1)==0)),exactly_one_direction=float(np.mean(pred.sum(1)==1)),multiple_directions=float(np.mean(pred.sum(1)>1))),
        limitations=['One-vs-rest ROC AUC and binary balanced accuracy have chance 0.5; cannot compare their percentages directly with eight-way chance 12.5% or earlier 26.8% accuracy.',
            'Positive prevalence is only 22..24 /188. Ordinary accuracy has a misleading approximately 87.5% all-negative baseline; report recall, false-positive rate and precision.',
            'Eight independent detectors may fire together or all stay silent; this is not a complete eight-direction controller.',
            'Same exploratory recording; prior evidence motivated feature families. No independent session, causal-intention or online-control validation.',
            'Pre-event reference may contain previous-trial activity because ITI=0. Its AUC is a descriptive negative control, not resting-state evidence.',
            'Small held-out positive counts per fold and small forward test; sampling clock and absolute latency remain uncertain.'],
        source_sha256=source_hash))
    plots(rows,labels,score,pred,forward,null)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    DELIVERY.mkdir(parents=True,exist_ok=True)
    for p in OUT.iterdir():
        if p.suffix in ('.png','.pdf','.csv','.json'):shutil.copy2(p,DELIVERY/p.name)
    print('FINAL',rows,flush=True);print('DELIVERY',DELIVERY,flush=True)

def plots(rows,labels,score,pred,forward,null):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    degrees=np.arange(8)*45;x=np.arange(8)
    fig,axs=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    auc=np.array([r['ROC_AUC'] for r in rows]);lo=np.array([r['AUC_CI95_low'] for r in rows]);hi=np.array([r['AUC_CI95_high'] for r in rows])
    ax=axs[0,0];ax.bar(x,auc,color=['#e07b19' if r['significant_FWER'] else '#2176ae' for r in rows])
    ax.errorbar(x,auc,yerr=np.maximum(np.array([auc-lo,hi-auc]),0),fmt='none',color='#333',capsize=3)
    ax.axhline(.5,color='#555',ls='--',label='Chance AUC = 0.5')
    ax.set(xticks=x,xticklabels=degrees,ylim=(0,1),ylabel='Held-out ROC AUC',xlabel='Target direction (deg)',title='One direction vs the other seven | 6 LFP channels')
    for i,r in enumerate(rows):ax.text(i,min(.98,r['AUC_CI95_high']+.025),f'{r["ROC_AUC"]:.2f}'+('*' if r['significant_FWER'] else ''),ha='center',fontsize=10)
    ax.legend(loc='lower right',fontsize=9)
    ax=axs[0,1]
    for key,offset,color,name in [('recall',-.25,'#2176ae','Recall'),('false_positive_rate',0,'#d16a28','False-positive rate'),('precision',.25,'#168a75','Precision')]:
        ax.bar(x+offset,[r[key] for r in rows],width=.24,color=color,label=name)
    ax.set(xticks=x,xticklabels=degrees,ylim=(0,1),ylabel='Fraction',xlabel='Target direction (deg)',title='Training-selected decision thresholds');ax.legend(ncol=3,fontsize=9)
    for d in range(8):
        fpr,tpr,_=roc_curve(labels==d,score[:,d]);axs[1,0].plot(fpr,tpr,color=DIRECTION_COLORS[d],label=f'{d*45}° ({auc[d]:.2f})')
        precision,recall,_=precision_recall_curve(labels==d,score[:,d]);axs[1,1].plot(recall,precision,color=DIRECTION_COLORS[d],label=f'{d*45}°')
    axs[1,0].plot([0,1],[0,1],ls='--',color='#999');axs[1,0].set(xlabel='False-positive rate',ylabel='Recall',title='Outer-held-out ROC curves',xlim=(0,1),ylim=(0,1));axs[1,0].legend(ncol=2,fontsize=9)
    axs[1,1].axhline(1/8,color='#999',ls='--',label='Prevalence ≈ 0.125');axs[1,1].set(xlabel='Recall',ylabel='Precision',title='Outer-held-out precision–recall curves',xlim=(0,1),ylim=(0,1));axs[1,1].legend(ncol=3,fontsize=9)
    fig.suptitle('1111 CO8 run-009 | Eight Independent Direction Detectors\n* Whole-pipeline permutation max-AUC corrected p < 0.05 across 8 directions',fontsize=16)
    fig.savefig(OUT/'LFP_One_vs_Rest_Direction_Decoding.png',dpi=220);fig.savefig(OUT/'LFP_One_vs_Rest_Direction_Decoding.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    values=[r['ROC_AUC'] for r in forward];ax.bar(x,values,color='#2176ae');ax.axhline(.5,color='#555',ls='--')
    ax.set(xticks=x,xticklabels=degrees,ylim=(0,1),xlabel='Target direction (deg)',ylabel='ROC AUC',title='Forward binary detectors | train blocks 2–19, test 21–25\nBlock 20 purged; few positive test trials per direction')
    for i,r in enumerate(forward):ax.text(i,r['ROC_AUC']+.025,f'{r["ROC_AUC"]:.2f}\nn+={r["positive_trials"]}',ha='center',fontsize=10)
    fig.savefig(OUT/'LFP_One_vs_Rest_Forward_Test.png',dpi=220);fig.savefig(OUT/'LFP_One_vs_Rest_Forward_Test.pdf');plt.close(fig)

if __name__=='__main__':main()
