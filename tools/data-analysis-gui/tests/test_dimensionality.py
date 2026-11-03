"""Scientific controls and presentation contract for the three S04 subsections."""
import numpy as np

from data_analysis_gui.core import analysis as A
from data_analysis_gui.sections.s04_dimensionality_v1 import diagnostics as DG


def test_gain_control_removes_intercept_and_total_rate():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 12)) + rng.normal(size=(80, 1)) * np.arange(1, 13)
    residual = DG.gain_residuals(x)
    np.testing.assert_allclose(residual.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose((x.sum(1) - x.sum(1).mean()) @ residual, 0, atol=1e-9)


def test_pca_control_uses_separate_bases_and_block_nulls():
    rng = np.random.default_rng(8)
    dirs = np.tile(np.arange(8), 10)
    blocks = np.repeat(np.arange(10), 8)
    elapsed = np.arange(len(dirs)) * 2.5
    x = 20 + rng.normal(size=(80, 12))
    x[:, :3] += np.cos(dirs[:, None] * np.pi / 4) * 5
    fit = A.pca(x, 5)
    controls = DG.pca_controls(x, fit, dirs, blocks, elapsed, 10, 42, lambda _: None, lambda: None)
    assert {r['condition'] for r in controls['comparison']} == {'original', 'gain_removed'}
    assert controls['nulls'].shape == (10, 2)
    assert np.isfinite(controls['nulls']).all()
    assert all(0 < r['p_max_F'] <= 1 for r in controls['comparison'])
    disabled = DG.pca_controls(x, fit, dirs, blocks, elapsed, 0, 42, lambda _: None, lambda: None)
    assert all(np.isnan(r['p_max_F']) for r in disabled['comparison'])


def test_tdr_test_kinematics_do_not_enter_fold_fitting():
    rng = np.random.default_rng(5)
    dirs = np.tile(np.arange(8), 12)
    blocks = np.repeat(np.arange(12), 8)
    elapsed = np.arange(96, dtype=float)
    x = rng.normal(size=(96, 16))
    x[:, :3] += 4 * np.cos(dirs[:, None] * np.pi / 4)
    x[:, 3:6] += 4 * np.sin(dirs[:, None] * np.pi / 4)
    kin = rng.normal(size=(96, 3))
    first = DG.tdr_fit_cv(x, dirs, blocks, elapsed, 5, kin)
    _, _, test = next(A.block_folds(blocks, 5))
    perturbed = kin.copy()
    perturbed[test] += 1000 * rng.normal(size=(len(test), 3))
    second = DG.tdr_fit_cv(x, dirs, blocks, elapsed, 5, perturbed)
    np.testing.assert_allclose(first['scores'][test], second['scores'][test], atol=1e-12)
    assert np.isfinite(first['scores'][test, 2:4]).all()


def test_neighborhood_quality_uses_same_seeded_subset_and_valid_k():
    x = np.random.default_rng(1).normal(size=(30, 2))
    z = A.zscore(x)[0]
    rows, subset = DG.neighborhood_quality(x, z, z, seed=7, cap=16)
    _, repeated = DG.neighborhood_quality(x, z, z, seed=7, cap=16)
    np.testing.assert_array_equal(subset, repeated)
    assert len(subset) == 16
    assert all(r['neighbors'] < 8 and r['n_total'] == 30 for r in rows)
    np.testing.assert_allclose([r['umap'] for r in rows], 1)
    np.testing.assert_allclose([r['pca2'] for r in rows], 1)
