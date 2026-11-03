"""Train-only total-rate residualization and PCA, displayed separately by fold."""
import numpy as np

from ...core import analysis as A,style
from ..base import new_figure


def fit_residual_pca(x,n_components):
    channels=np.flatnonzero(x.std(axis=0)>1e-12)
    if len(channels)<3:
        raise ValueError('Fewer than 3 training-variable channels')
    train=x[:,channels]
    mean=train.mean(axis=0)
    total=train.sum(axis=1);total_mean=total.mean()
    centered=total-total_mean
    beta=centered@(train-mean)/max(float(centered@centered),1e-12)
    residual=train-mean-centered[:,None]*beta
    surviving=residual.std(axis=0)>max(float(train.std(axis=0).max())*1e-10,1e-12)
    if surviving.sum()<3 or len(train)<4:
        raise ValueError('Insufficient training residual dimensions or trials')
    pca=A.pca(residual[:,surviving],min(n_components,int(surviving.sum()),len(train)-1))
    # Deterministic orientation using TRAINING loadings only; never align to a
    # full-cohort reference whose fit would include these test trials.
    peak=np.argmax(abs(pca['components']),axis=1)
    sign=np.where(pca['components'][np.arange(len(peak)),peak]<0,-1.,1.)
    pca['components']*=sign[:,None];pca['scores']*=sign
    return dict(channels=channels,mean=mean,total_mean=total_mean,beta=beta,
                surviving=surviving,pca=pca)


def project_residual_pca(model,x):
    values=x[:,model['channels']]
    residual=values-model['mean']-(values.sum(axis=1)-model['total_mean'])[:,None]*model['beta']
    fit=model['pca']
    z,_,_=A.zscore(residual[:,model['surviving']],fit['mu'],fit['sd'])
    return z@fit['components'].T


def cross_validate(x,blocks,n_components,n_folds):
    folds=[]
    for fold,tr,te in A.block_folds(blocks,n_folds):
        record=dict(fold=int(fold),train=tr,test=te)
        try:
            model=fit_residual_pca(x[tr],n_components)
            record.update(model=model,scores=project_residual_pca(model,x[te]))
        except (ValueError,np.linalg.LinAlgError) as exc:
            record['error']=str(exc)
        folds.append(record)
    return folds


def append_heldout(result,x,dirs,blocks,trial_ids,electrodes,p,scatter,banner):
    folds=cross_validate(x,blocks,p['n_components'],p['n_folds'])
    count=max(len(folds)+1,1)
    fig=new_figure(13,3.6*int(np.ceil(count/3))+.6)
    axes=fig.subplots(int(np.ceil(count/3)),3,squeeze=False).ravel()
    trials=[];models=[];fold_rows=[]
    for ax,fold in zip(axes,folds):
        tr,te=fold['train'],fold['test']
        row=dict(fold=fold['fold'],n_train=len(tr),n_test=len(te),
                 test_block_min=int(blocks[te].min()),test_block_max=int(blocks[te].max()),
                 status=fold.get('error','computed'))
        fold_rows.append(row)
        if 'error' in fold:
            style.text_panel(ax,[f'Fold {fold["fold"]+1} unavailable',fold['error']],fontsize=8)
            continue
        scores=fold['scores'];model=fold['model'];fit=model['pca']
        scatter(ax,scores[:,:2],dirs[te],
                f'Residual PC1 ({100*fit["explained"][0]:.1f}% train residual var.)',
                f'Residual PC2 ({100*fit["explained"][1]:.1f}% train residual var.)',
                f'Fold {fold["fold"]+1} · held-out {len(te)} trials / train {len(tr)}',legend=fold['fold']==0)
        for local,i in enumerate(te):
            trials.append(dict(trial_id=int(trial_ids[i]),block=int(blocks[i]),fold=fold['fold'],direction_deg=int(dirs[i]*45),
                               **{f'residual_PC{k+1}':float(value) for k,value in enumerate(scores[local])}))
        rank=np.full(len(model['channels']),-1,int)
        rank[model['surviving']]=np.arange(int(model['surviving'].sum()))
        for local,c in enumerate(model['channels']):
            j=rank[local]
            record=dict(fold=fold['fold'],electrode=int(electrodes[c]),stored_column=int(c+1),
                        training_rate_mean=float(model['mean'][local]),training_total_mean=float(model['total_mean']),
                        beta=float(model['beta'][local]),residual_used=bool(j>=0))
            if j>=0:
                record.update(residual_training_mean=float(fit['mu'][j]),residual_training_sd=float(fit['sd'][j]),
                              PC1_loading=float(fit['components'][0,j]),PC2_loading=float(fit['components'][1,j]))
            models.append(record)
    info_ax=axes[len(folds)]
    style.text_panel(info_ax,['Held-out residual PCA',
        'Training only: variable-channel selection,',
        'total-rate regression (mean, beta),',
        'residual scaling and PCA loadings.',
        'Test trials: transform and project only.',
        'Direction labels are display-only.',
        'Each fold has its own PC basis:',
        'read panels separately; do not pool axes.',
        'Axis variance % uses TRAINING residuals.',
        'No held-out direction-decoding claim.'],fontsize=8)
    for ax in axes[count:]:ax.axis('off')
    fig.suptitle(banner+'\nResidual PCA · held-out projections with training-only total-rate adjustment',fontsize=9)
    result.add_figure('Supplementary Residual PCA held-out projections',fig,
        method=('Chronological block CV with +/-1 block purge. On EACH training fold only, select nonconstant '
                'channels, compute total rate as their sum; estimate channel intercept, centered-total slope '
                'beta, residual channel SD and PCA loadings. On test trials, compute total rate over those '
                'training-selected channels; subtract training channel mean and beta*(test total-training '
                'mean total), use training residual mean/SD and PCA loadings. No test-trial statistics or '
                'direction labels enter fitting. Loading signs are chosen by largest absolute TRAINING '
                'loading, not full-cohort reference. Fold panels have different PCA bases and are not pooled. '
                'Axis percentages are training residual explained variance, not held-out explained variance '
                'or the original rate variance. Unfilled 1 SD ellipses show TEST trial dispersion. This '
                'is structural stability visualization, not an independently evaluated decoder.'),
        data=dict(test_projections=trials,training_parameters=models,folds=fold_rows))
    result.tables['residual_pca_heldout_trials']=trials
    result.tables['residual_pca_training_parameters']=models
    result.tables['residual_pca_folds']=fold_rows
