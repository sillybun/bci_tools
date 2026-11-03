"""Decoder package: interface, synthetic accuracy, CV plumbing and vendor integrity."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from data_analysis_gui import decoders as D
from data_analysis_gui.core import analysis as A
from data_analysis_gui.core.filters import TrialFilter, apply_filter
from data_analysis_gui.core.synthetic import make_session
from data_analysis_gui.decoders.vendor import vendor_tool as V

SDK_REPO = Path('E:/AitBCI/bci-decoder-sdk')


@pytest.fixture(scope='module')
def grids():
    # 20 blocks (~7.5k native 20 ms rows): with the SDK default penalties the 20 ms decoders
    # overfit 460 overlapping-window features on fewer rows (8 blocks: ReFIT R² < 0).
    s = make_session(seed=5, n_blocks=20)
    c = apply_filter(s, TrialFilter())
    spec = D.BinSpec()
    return D.build_data(c, spec), D.build_data(c, spec, 0.02)


@pytest.mark.parametrize('name', D.NAMES)
def test_interface(name):
    dec = D.get(name)
    assert dec.NAME == name and dec.LABEL and dec.METHOD
    assert dec.SOURCE['kind'] in ('local', 'vendored')
    assert dec.BIN_SECONDS in (None, 0.02)
    assert dec.__doc__ and ('Source:' in dec.__doc__) and ('Bins:' in dec.__doc__)
    if dec.SOURCE['kind'] == 'vendored':
        assert len(dec.SOURCE['commit']) == 40 and dec.SOURCE['commit'] in dec.__doc__


@pytest.mark.parametrize('name', D.NAMES)
def test_decoder_beats_zero_prediction(name, grids):
    data, native = grids
    dec = D.get(name)
    res = D.cv_decode(dec, data, D.decoder_params(dec, {}), native=native if D.needs_native(dec, data) else None)
    ok = np.isfinite(res['pred']).all(1)
    assert ok.mean() > 0.8
    y, p = data.velocity[ok], res['pred'][ok]
    assert np.sum((y - p) ** 2) < np.sum(y ** 2), 'must beat predicting zero velocity'
    assert np.mean(A.r2_score(y, p)) > 0.2
    assert all(r['n_train_bins'] and r['fit_seconds'] >= 0 for r in res['info'])


def test_aggregate_is_exact_for_velocity(grids):
    data, native = grids
    dec = D.get('refit_kf_filter')
    agg = D.aggregate(native.velocity, native, data, D.native_ratio(dec, data.bin_seconds))
    ok = np.isfinite(agg).all(1)
    assert ok.mean() > 0.9
    assert np.allclose(agg[ok], data.velocity[ok], atol=1e-9)


def test_cv_velocity_unchanged_api(grids):
    data, _ = grids
    pred, folds, info = A.cv_velocity(data.features, data.velocity, data.trial, data.sequence, data.block)
    assert set(pred) == {'ridge', 'kf'} and (folds >= 0).any()
    assert 'kf_spectral_radius' in info[0]


def test_align_features_matches_crossday_transform():
    from data_analysis_gui.core import crossday as X
    rng = np.random.default_rng(0)
    xi, xj = rng.gamma(2, 3, (200, 6)), rng.gamma(3, 2, (150, 6))
    scaler = X.fit_scaler(xi)
    for mode in X.STANDARDIZE:
        got = X.transform(scaler, D.align_features(xi, xj, mode))
        assert np.allclose(got, X.transform(scaler, xj, mode))


def test_selection_parsing():
    assert D.parse_selection('ridge, kf,ridge') == ('ridge', 'velocity_kf')
    assert D.parse_selection('refit,madduri,ecog')[0] == 'refit_kf_filter'
    with pytest.raises(ValueError):
        D.parse_selection('nope')


def test_ecog_trainer_rejects_spike_features():
    msg = D.get('ecog_linear_velocity_decoder').trainer_contract_error()
    assert 'unexpected ECoG feature schema' in msg


# ---------------------------------------------------------------- vendor integrity
def _vendored():
    for sha in V.SNAPSHOTS:
        meta = json.loads((V.snapshot_dir(sha) / 'VENDOR.json').read_text(encoding='utf-8'))
        for rec in meta['files']:
            yield sha, meta, rec


VENDORED = list(_vendored())


def test_vendor_metadata():
    for sha in V.SNAPSHOTS:
        meta = json.loads((V.snapshot_dir(sha) / 'VENDOR.json').read_text(encoding='utf-8'))
        assert meta['commit'] == sha and meta['pip_spec'].endswith('@' + sha)
        on_disk = {p.relative_to(V.snapshot_dir(sha)).as_posix() for p in V.snapshot_dir(sha).rglob('*')
                   if p.is_file() and '__pycache__' not in p.parts}
        listed = {r['path'] for r in meta['files']}
        assert on_disk - listed == {'VENDOR.json', '__init__.py'}
        assert listed <= on_disk


@pytest.mark.parametrize('sha,meta,rec', VENDORED, ids=lambda v: v['path'] if isinstance(v, dict) and 'path' in v
                         else None)
def test_vendored_file_reconstructs_original_blob(sha, meta, rec):
    """Header names the source; header-stripped body with imports un-rewritten hashes to the git blob."""
    path = V.snapshot_dir(sha) / rec['path']
    head, body = V.split_header(path.read_bytes())
    text = head.decode()
    assert sha in text and rec['original_path'] in text and rec['original_blob_sha'] in text
    original = V.unrewrite(rec['path'], body)
    if rec['eol'] == 'lf':
        original = original.replace(b'\r\n', b'\n')     # tolerate a CRLF checkout of this repo
    assert V.blob_sha(original) == rec['original_blob_sha']
    changed = [(a, b) for a, b in zip(original.split(b'\n'), body.split(b'\n')) if a != b]
    assert all(V.is_import_line(a) and V.is_import_line(b) for a, b in changed)


@pytest.mark.skipif(not (SDK_REPO / '.git').exists(), reason='bci-decoder-sdk checkout not present')
@pytest.mark.parametrize('sha,meta,rec', VENDORED, ids=lambda v: v['path'] if isinstance(v, dict) and 'path' in v
                         else None)
def test_vendored_file_matches_git(sha, meta, rec):
    original = subprocess.run(['git', '-C', str(SDK_REPO), 'show', f'{sha}:{rec["original_path"]}'],
                              check=True, capture_output=True).stdout
    blob = subprocess.run(['git', '-C', str(SDK_REPO), 'rev-parse', f'{sha}:{rec["original_path"]}'],
                          check=True, capture_output=True, text=True).stdout.strip()
    assert blob == rec['original_blob_sha']
    body = V.body_of(V.snapshot_dir(sha) / rec['path']).replace(b'\r\n', b'\n')
    expected = V.rewrite(rec['path'], original).replace(b'\r\n', b'\n')
    assert body == expected
    for a, b in zip(original.split(b'\n'), body.split(b'\n')):
        if a != b:
            assert V.is_import_line(a) and V.is_import_line(b)


# ---------------------------------------------------------------- sections with every decoder
@pytest.mark.parametrize('section_id', ['s10', 's11', 'x05', 'x06'])
def test_sections_with_all_decoders(section_id, tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    from data_analysis_gui.runner import build_context, run_section
    from data_analysis_gui.sections import sections
    section = next(s for mode in ('single', 'cross') for s in sections(mode) if s.id == section_id)
    sessions = [make_session(seed=i, n_blocks=6, date=f'2026-01-0{i + 1}', online=(i == 0)) for i in range(2)]
    use = sessions if section.mode == 'cross' else sessions[:1]
    params = {'decoders': ','.join(D.NAMES), 'n_boot': 20, 'repro': False}
    params = {k: v for k, v in params.items() if k in {p.name for p in section.params}}
    ctx = build_context(use, TrialFilter(), params, section)
    result = run_section(section, ctx)
    assert result.figures and all(m for m in result.methods), 'every figure needs a method text'
    assert all(d for d in result.figure_data), 'every figure needs plotted-data tables'


def test_build_data_follows_trial_filter():
    s = make_session(seed=3, n_blocks=4)
    crit = TrialFilter(velocity_neural_window=0.1, velocity_min_frames=2, velocity_max_gap=0.12,
                       velocity_cursor_gap=0.08, velocity_bin_policy='drop_bin')
    c = apply_filter(s, crit)
    spec = D.BinSpec()
    data, native = D.build_data(c, spec), D.build_data(c, spec, 0.02)
    assert (data.extra['neural_window'], data.extra['min_frames'], data.extra['max_gap'], data.extra['cursor_gap'],
            data.extra['policy']) == (0.1, 2, 0.12, 0.08, 'drop_bin')
    assert native.extra['neural_window'] == spec.native_window and native.extra['policy'] == 'drop_bin'
    assert native.extra['cursor_gap'] == 0.08 and native.extra['max_gap'] == 0.12
    assert 'drop_bin' in D.cohort_rule_text(c, spec)


@pytest.mark.parametrize('section_id', ['s10', 'x05', 'x06'])
def test_comparison_figure_with_one_decoder(section_id):
    import matplotlib
    matplotlib.use('Agg')
    from data_analysis_gui.runner import build_context, run_section
    from data_analysis_gui.sections import sections
    section = next(s for mode in ('single', 'cross') for s in sections(mode) if s.id == section_id)
    sessions = [make_session(seed=i, n_blocks=6, date=f'2026-01-0{i + 1}') for i in range(2)]
    use = sessions if section.mode == 'cross' else sessions[:1]
    result = run_section(section, build_context(use, TrialFilter(), {'decoders': 'ridge'}, section))
    titles = [t for t, _ in result.figures]
    assert 'Decoder comparison' in titles
    i = titles.index('Decoder comparison')
    assert result.methods[i] and result.figure_data[i]
