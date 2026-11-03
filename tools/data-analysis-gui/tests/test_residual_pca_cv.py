"""Residual PCA fits regressors, feature selection, scaling and axes on training only."""
import numpy as np

from data_analysis_gui.core import analysis as A
from data_analysis_gui.sections.s04_dimensionality_v1.residual_cv import (
    cross_validate,fit_residual_pca,project_residual_pca,
)


def test_training_projection_and_residual_regression_are_consistent():
    rng=np.random.default_rng(5)
    x=rng.normal(size=(80,12))+rng.normal(size=(80,1))*np.arange(1,13)
    model=fit_residual_pca(x,5)
    np.testing.assert_allclose(project_residual_pca(model,x),model['pca']['scores'],atol=1e-12)
    selected=x[:,model['channels']]
    centered=selected.sum(axis=1)-model['total_mean']
    residual=selected-model['mean']-centered[:,None]*model['beta']
    np.testing.assert_allclose(residual.mean(axis=0),0,atol=1e-12)
    np.testing.assert_allclose(centered@residual,0,atol=1e-9)


def test_heldout_features_cannot_change_training_beta_scaler_or_pca():
    rng=np.random.default_rng(7)
    x=rng.normal(size=(96,16))+10
    blocks=np.repeat(np.arange(12),8)
    _,tr,te=next(A.block_folds(blocks,5))
    x[:,0]=3  # not selected even if held-out variance becomes very large
    initial=cross_validate(x,blocks,5,5)[0]
    changed=x.copy();changed[te]+=rng.normal(size=(len(te),16))*1000
    repeated=cross_validate(changed,blocks,5,5)[0]
    np.testing.assert_array_equal(initial['train'],tr)
    for field in ('channels','mean','beta','surviving'):
        np.testing.assert_array_equal(initial['model'][field],repeated['model'][field])
    for field in ('mu','sd','components','explained'):
        np.testing.assert_array_equal(initial['model']['pca'][field],repeated['model']['pca'][field])
    assert 0 not in repeated['model']['channels']
    assert not np.allclose(initial['scores'],repeated['scores'])
