"""Velocity decoders: one module per decoder, a shared data container and CV helper.

Decoder module interface (every module named in `NAMES`):

    NAME: str                 registry key (== module name)
    LABEL: str                short display label
    SOURCE: dict              {'kind': 'local'} or {'kind': 'vendored', 'repo', 'commit', 'path', ...}
    BIN_SECONDS: float|None   bin period the decoder's contract requires (None = any)
    DEFAULTS: dict            decoder parameters; keys are namespaced (e.g. 'lam_ridge')
    METHOD: str               one paragraph used in figure method texts
    fit(train: DecoderData, params: dict) -> model      (model['info'] = diagnostics dict)
    predict(model, test: DecoderData) -> ndarray (n, 2) velocity, task units / s

Every decoder standardises / normalises features itself from the rows it is
fitted on (train fold or training day) — never from test rows. Decoders whose
contract fixes the bin period (`BIN_SECONDS`, e.g. the SDK's 20 ms state update)
are fitted on their own "native" bins covering the same trial windows; their
predictions are averaged back onto the evaluation bins. Because bin velocity is
a displacement divided by the bin width, the mean of r consecutive native
velocities equals the velocity of the enclosing evaluation bin exactly, so the
held-out comparison against GT is on identical targets.

`cv_decode` runs the shared protocol: `core.analysis.block_folds` (five
chronological block groups, training drops blocks within ±1 of the test range).
"""
from __future__ import annotations

import hashlib
import importlib
import time
from dataclasses import dataclass, field, replace

import numpy as np

from ..core.analysis import block_folds

NAMES = ('ridge', 'velocity_kf', 'refit_kf_filter', 'madduri_coadaptive_decoder', 'ecog_linear_velocity_decoder')
DEFAULT_SELECTION = 'ridge,velocity_kf'
ALIASES = {'kf': 'velocity_kf', 'refit': 'refit_kf_filter', 'refit_kf': 'refit_kf_filter',
           'standalone_refit_kf': 'refit_kf_filter', 'madduri': 'madduri_coadaptive_decoder',
           'ecog': 'ecog_linear_velocity_decoder', 'ecog_linear': 'ecog_linear_velocity_decoder'}
# One place for decoder colours (sections add GT / online / their own extras).
COLORS = {'ridge': '#2176ae', 'velocity_kf': '#e07b39', 'refit_kf_filter': '#d62728',
          'madduri_coadaptive_decoder': '#17becf', 'ecog_linear_velocity_decoder': '#9a8c1c'}
SELECTION_HELP = ('逗号分隔：' + ', '.join(NAMES) + '（别名 kf/refit/madduri/ecog）。'
                  'SDK decoder（后三个）按其合同使用 20 ms 原生 bin 或 SDK Runtime 逐帧推理，较慢。')


def get(name: str):
    name = ALIASES.get(name, name)
    if name not in NAMES:
        raise ValueError(f'unknown decoder {name!r}; choose from {", ".join(NAMES)}')
    return importlib.import_module(f'{__name__}.{name}')


def parse_selection(text) -> tuple[str, ...]:
    """'ridge, kf' -> ('ridge', 'velocity_kf'); order kept, duplicates dropped."""
    raw = text if isinstance(text, (list, tuple)) else str(text or '').replace(';', ',').split(',')
    out = []
    for item in raw:
        item = item.strip().lower()
        if not item:
            continue
        name = ALIASES.get(item, item)
        if name not in NAMES:
            raise ValueError(f'未知 decoder "{item}"；可选 {", ".join(NAMES)}')
        if name not in out:
            out.append(name)
    if not out:
        raise ValueError('至少选择一个 decoder')
    return tuple(out)


def label(name: str) -> str:
    return get(name).LABEL


def source_text(name: str) -> str:
    s = get(name).SOURCE
    if s.get('kind') == 'local':
        return f'local numpy implementation (data_analysis_gui/decoders/{name}.py)'
    return f'{s["repo"]}@{s["commit"]}:{s["path"]} (vendored, data_analysis_gui/decoders/vendor/sdk_{s["commit"][:7]})'


# ---------------------------------------------------------------- data
@dataclass
class DecoderData:
    features: np.ndarray            # (n, ch) feature values (Hz, or sqrt Hz — see feature_unit)
    velocity: np.ndarray            # (n, 2) cursor-displacement velocity over the bin, task units / s
    position: np.ndarray            # (n, 2) cursor position at the bin's left edge
    sequence: np.ndarray            # (n,) contiguous-run id: state resets here, never spans gaps/trials
    trial: np.ndarray               # (n,) cohort row index
    bin_seconds: float | None       # bin period (None only for legacy cv_velocity callers)
    electrode_ids: np.ndarray       # (ch,) electrode ids of the feature columns
    block: np.ndarray | None = None
    bin: np.ndarray | None = None   # bin index within the trial
    time: np.ndarray | None = None  # right-edge receive time / s
    direction: np.ndarray | None = None
    feature_unit: str = 'Hz'
    neural_window: float | None = None
    extra: dict = field(default_factory=dict)

    ROW_FIELDS = ('features', 'velocity', 'position', 'sequence', 'trial', 'block', 'bin', 'time', 'direction')

    @property
    def n(self) -> int:
        return len(self.velocity)

    @property
    def end_position(self) -> np.ndarray:
        """Cursor position at the bin's right edge (= left edge + velocity · bin)."""
        return self.position + self.velocity * float(self.bin_seconds)

    def subset(self, rows) -> 'DecoderData':
        rows = np.asarray(rows)
        values = {k: (None if getattr(self, k) is None else getattr(self, k)[rows]) for k in self.ROW_FIELDS}
        return replace(self, **values)

    def columns(self, cols) -> 'DecoderData':
        cols = np.asarray(cols)
        return replace(self, features=self.features[:, cols], electrode_ids=np.asarray(self.electrode_ids)[cols])

    def with_features(self, x) -> 'DecoderData':
        return replace(self, features=x)

    def sequence_bounds(self):
        """[(start, stop)] row ranges of contiguous sequences, in row order."""
        if not self.n:
            return []
        cut = np.flatnonzero(np.diff(self.sequence) != 0) + 1
        edges = np.r_[0, cut, self.n]
        return list(zip(edges[:-1], edges[1:]))

    def timestamps_ns(self) -> np.ndarray:
        """int64 right-edge times; exactly bin-spaced inside each sequence (no float jitter)."""
        period = int(round(float(self.bin_seconds) * 1e9))
        out = np.empty(self.n, np.int64)
        t = self.time if self.time is not None else np.arange(self.n) * float(self.bin_seconds) + 1.0
        b = self.bin if self.bin is not None else np.arange(self.n)
        for a, z in self.sequence_bounds():
            out[a:z] = int(round(float(t[a]) * 1e9)) + (b[a:z] - b[a]).astype(np.int64) * period
        return out

    @classmethod
    def from_bins(cls, vb, bin_seconds, electrode_ids, sqrt=False, neural_window=None) -> 'DecoderData':
        x = np.sqrt(np.clip(vb.features, 0, None)) if sqrt else vb.features
        return cls(features=x, velocity=vb.velocity, position=vb.position, sequence=vb.sequence, trial=vb.trial,
                   bin_seconds=float(bin_seconds), electrode_ids=np.asarray(electrode_ids), block=vb.block,
                   bin=vb.bin, time=vb.time, direction=vb.direction, feature_unit='sqrt(Hz)' if sqrt else 'Hz',
                   neural_window=neural_window, extra=dict(exclusions=vb.exclusions))


def electrode_layout_id(prefix: str, electrode_ids) -> str:
    digest = hashlib.sha1(','.join(map(str, np.asarray(electrode_ids).tolist())).encode()).hexdigest()[:16]
    return f'{prefix}.{len(electrode_ids)}ch.{digest}'


# ---------------------------------------------------------------- native bins
def needs_native(decoder, data: DecoderData) -> bool:
    want = decoder.BIN_SECONDS
    return want is not None and (data.bin_seconds is None or abs(want - data.bin_seconds) > 1e-9)


def native_ratio(decoder, bin_seconds: float) -> int:
    r = float(bin_seconds) / decoder.BIN_SECONDS
    if abs(r - round(r)) > 1e-6 or round(r) < 1:
        raise ValueError(f'{decoder.NAME} 需要 {decoder.BIN_SECONDS * 1000:g} ms bin：评估 bin 宽 '
                         f'{bin_seconds * 1000:g} ms 必须是其整数倍')
    return int(round(r))


def native_min_frames(neural_window: float, min_frames: int, frame_s: float = 0.02) -> int:
    """A window of w seconds holds about w/20 ms frames; never demand more than that."""
    return int(max(1, min(int(min_frames), round(neural_window / frame_s))))


def aggregate(native_pred, native: DecoderData, data: DecoderData, ratio: int) -> np.ndarray:
    """Mean of the `ratio` native predictions inside each evaluation bin (NaN if any is missing)."""
    out = np.full((data.n, 2), np.nan)
    if not native.n or not data.n:
        return out
    m = int(max(native.bin.max() // ratio, data.bin.max())) + 2
    key_eval = data.trial.astype(np.int64) * m + data.bin
    key_nat = native.trial.astype(np.int64) * m + native.bin // ratio
    order = np.argsort(key_eval)
    pos = np.searchsorted(key_eval[order], key_nat)
    pos = np.clip(pos, 0, len(order) - 1)
    hit = key_eval[order][pos] == key_nat
    target = order[pos[hit]]
    ok = np.isfinite(native_pred[hit]).all(1)
    sums = np.zeros((data.n, 2))
    counts = np.zeros(data.n)
    np.add.at(sums, target[ok], native_pred[hit][ok])
    np.add.at(counts, target[ok], 1)
    full = counts == ratio
    out[full] = sums[full] / ratio
    return out


# ---------------------------------------------------------------- alignment across recordings
def align_features(train_x, test_x, mode='train'):
    """Make a train-normalising decoder see the requested test-day standardisation.

    Every decoder here normalises with its training mean μᵢ and SD σᵢ. Feeding
      'train'    x                       → (x − μᵢ)/σᵢ   (fixed model)
      'recentre' x − μⱼ + μᵢ             → (x − μⱼ)/σᵢ   (test-day mean removed)
      'test_own' (x − μⱼ)/σⱼ · σᵢ + μᵢ   → (x − μⱼ)/σⱼ   (test day re-standardised)
    reproduces `core.crossday.transform` for any of them without refitting.
    """
    if mode == 'train':
        return test_x
    mu_i, sd_i = train_x.mean(0), train_x.std(0)
    mu_j, sd_j = test_x.mean(0), test_x.std(0)
    if mode == 'recentre':
        return test_x - mu_j + mu_i
    if mode == 'test_own':
        return (test_x - mu_j) / np.where(sd_j > 1e-12, sd_j, 1.0) * sd_i + mu_i
    raise ValueError(f'unknown standardisation mode {mode!r}')


# ---------------------------------------------------------------- fit / predict / CV
def fit_predict(decoder, train: DecoderData, test: DecoderData, params: dict, *, train_native=None,
                test_native=None, test_mode='train'):
    """Fit on `train` (or its native bins) and predict `test` on the evaluation grid.

    Returns (pred (test.n, 2), info dict, native_pred or None).
    """
    fitted = fit_model(decoder, train, params, train_native=train_native)
    pred, info, raw = predict_grid(fitted, test, test_native=test_native, test_mode=test_mode)
    return pred, info, raw


def fit_model(decoder, train: DecoderData, params: dict, *, train_native=None) -> dict:
    """Fit once (on native bins when the decoder's contract requires them)."""
    native = needs_native(decoder, train)
    fit_on = train_native if native else train
    if native and fit_on is None:
        raise ValueError(f'{decoder.NAME}: needs {decoder.BIN_SECONDS * 1000:g} ms native bins')
    t0 = time.perf_counter()
    model = decoder.fit(fit_on, params)
    return dict(decoder=decoder, model=model, fit_on=fit_on, native=native,
                fit_seconds=round(time.perf_counter() - t0, 3))


def predict_grid(fitted: dict, test: DecoderData, *, test_native=None, test_mode='train'):
    """Predict `test` on its evaluation grid with a `fit_model` result. Returns (pred, info, native_pred|None)."""
    decoder, model, fit_on = fitted['decoder'], fitted['model'], fitted['fit_on']
    pred_on = test_native if fitted['native'] else test
    if pred_on is None:
        raise ValueError(f'{decoder.NAME}: needs {decoder.BIN_SECONDS * 1000:g} ms native bins')
    if test_mode != 'train':
        pred_on = pred_on.with_features(align_features(fit_on.features, pred_on.features, test_mode))
    t1 = time.perf_counter()
    raw = np.asarray(decoder.predict(model, pred_on), float)
    info = dict(model.get('info', {}), fit_seconds=fitted['fit_seconds'],
                predict_seconds=round(time.perf_counter() - t1, 3), n_fit_rows=int(fit_on.n),
                n_predict_rows=int(pred_on.n))
    if fitted['native']:
        return aggregate(raw, pred_on, test, native_ratio(decoder, test.bin_seconds)), info, raw
    return raw, info, None


def cv_decode(decoder, bins: DecoderData, params: dict, n_folds: int = 5, native: DecoderData | None = None,
              progress=lambda message: None) -> dict:
    """Held-out predictions over chronological block folds (±1 block purge).

    `bins` is the evaluation grid; `native` holds the decoder's own bins when
    `decoder.BIN_SECONDS` differs (fold membership is by block, identical for both).
    Returns dict(pred (n, 2; NaN = untested), folds (n,; -1 = untested), info [per fold],
    native_pred, seconds).
    """
    if isinstance(decoder, str):
        decoder = get(decoder)
    use_native = needs_native(decoder, bins)
    if use_native and native is None:
        raise ValueError(f'{decoder.NAME}: needs {decoder.BIN_SECONDS * 1000:g} ms native bins')
    pred = np.full((bins.n, 2), np.nan)
    folds = np.full(bins.n, -1)
    native_pred = np.full((native.n, 2), np.nan) if use_native else None
    info = []
    start = time.perf_counter()
    for f, tr, te in block_folds(bins.block, n_folds):
        progress(f'{decoder.LABEL} fold {f + 1}')
        tr_blocks, te_blocks = np.unique(bins.block[tr]), np.unique(bins.block[te])
        kw = {}
        if use_native:
            ntr = np.flatnonzero(np.isin(native.block, tr_blocks))
            nte = np.flatnonzero(np.isin(native.block, te_blocks))
            kw = dict(train_native=native.subset(ntr), test_native=native.subset(nte))
        p, row, raw = fit_predict(decoder, bins.subset(tr), bins.subset(te), params, **kw)
        pred[te] = p
        folds[te] = f
        if raw is not None:
            native_pred[nte] = raw
        info.append(dict(fold=int(f), n_train_bins=int(len(tr)), n_test_bins=int(len(te)),
                         test_blocks=f'{te_blocks.min()}-{te_blocks.max()}', **row))
    return dict(pred=pred, folds=folds, info=info, native_pred=native_pred,
                seconds=round(time.perf_counter() - start, 3), decoder=decoder.NAME)


# ---------------------------------------------------------------- section helper
@dataclass(frozen=True)
class BinSpec:
    """How evaluation bins and (when required) native bins are built for one cohort.

    Velocity-bin quality thresholds left as None come from the cohort's TrialFilter
    (`velocity_neural_window`, `velocity_min_frames`, `velocity_max_gap`,
    `velocity_cursor_gap`, `velocity_bin_policy`). Native 20 ms bins keep their own
    trailing window (`native_window`) but apply the same cursor-gap rule, max gap and
    drop_trial/drop_bin policy; their minimum frame count is the criteria value capped
    at the number of 20 ms frames the window can hold.
    """
    offset: float = 0.2
    bin_s: float = 0.1
    n_bins: int | None = 10            # None with until='end'/'acquire'
    until: str = 'window'
    neural_window: float | None = None  # evaluation-bin trailing window (None = criteria)
    native_window: float = 0.2         # trailing window of 20 ms native bins (0.02 = non-overlapping frames)
    min_frames: int | None = None
    max_gap: float | None = None
    cursor_gap: float | None = None
    policy: str | None = None          # drop_trial | drop_bin (None = criteria)
    sqrt: bool = False

    def resolved(self, criteria) -> dict:
        def pick(value, name):
            return getattr(criteria, name) if value is None else value
        return dict(neural_window=float(pick(self.neural_window, 'velocity_neural_window')),
                    min_frames=int(pick(self.min_frames, 'velocity_min_frames')),
                    max_gap=float(pick(self.max_gap, 'velocity_max_gap')),
                    cursor_gap=float(pick(self.cursor_gap, 'velocity_cursor_gap')),
                    policy=str(pick(self.policy, 'velocity_bin_policy')))

    def key(self, criteria=None):
        base = (self.offset, self.bin_s, self.n_bins, self.until, self.native_window, self.sqrt)
        return base + (tuple(sorted(self.resolved(criteria).items())) if criteria is not None else
                       (self.neural_window, self.min_frames, self.max_gap, self.cursor_gap, self.policy))


def build_data(cohort, spec: BinSpec, bin_s: float | None = None, cache=None) -> DecoderData:
    """DecoderData from `core.features.velocity_bins` (native when bin_s != spec.bin_s)."""
    from ..core.features import velocity_bins
    bin_s = spec.bin_s if bin_s is None else bin_s
    native = abs(bin_s - spec.bin_s) > 1e-9
    rule = spec.resolved(cohort.criteria)
    window = spec.native_window if native else rule['neural_window']
    n_bins = spec.n_bins
    if native and n_bins is not None:
        n_bins = int(round(n_bins * spec.bin_s / bin_s))
    min_frames = native_min_frames(window, rule['min_frames']) if native else rule['min_frames']

    def compute():
        vb = velocity_bins(cohort, offset=spec.offset, bin_s=bin_s, n_bins=n_bins, until=spec.until,
                           neural_window=window, min_frames=min_frames, max_gap=rule['max_gap'],
                           cursor_gap=rule['cursor_gap'], policy=rule['policy'])
        data = DecoderData.from_bins(vb, bin_s, cohort.session.electrode_ids, spec.sqrt, window)
        data.extra.update(rule, neural_window=window, min_frames=min_frames, n_bins=n_bins,
                          n_trials=int(len(np.unique(vb.trial))) if vb.n else 0)
        return data
    if cache is None:
        return compute()
    return cache(('decoder_data', id(cohort), spec.key(cohort.criteria), round(bin_s, 9)), compute)


def cohort_rule_text(cohort, spec: BinSpec) -> str:
    """Velocity cohort rule actually applied (for method texts)."""
    r = spec.resolved(cohort.criteria)
    policy = ('drop_trial: a trial with any bad bin in the evaluated range is removed entirely'
              if r['policy'] == 'drop_trial' else 'drop_bin: only the bad bins are removed, the trial keeps its good bins')
    return (f'Velocity cohort (TrialFilter criterion 8): policy {policy}; a bin is bad if it extends beyond T_TRIAL_END, '
            f'the active same-trial cursor has a sampling gap > {r["cursor_gap"]:g} s around an edge, or its trailing '
            f'{r["neural_window"] * 1000:g} ms neural window has < {r["min_frames"]} frames or a boundary-inclusive '
            f'gap > {r["max_gap"]:g} s. Native 20 ms bins (SDK decoders) use the same cursor/gap/policy rules with a '
            f'{spec.native_window * 1000:g} ms window and ≥ {native_min_frames(spec.native_window, r["min_frames"])} '
            'frames.')


def decoder_params(decoder, params: dict) -> dict:
    return {k: params.get(k, v) for k, v in decoder.DEFAULTS.items()}


def section_cv(ctx, cohort, names, spec: BinSpec, params: dict, *, columns=None, data=None, n_folds=5):
    """Cached block-CV for several decoders on one cohort.

    Returns dict(data=eval DecoderData, results={name: cv_decode result}, native={name: DecoderData}).
    `columns` restricts every grid to those stored columns (e.g. electrodes shared with another day).
    """
    cache = ctx.cached
    col_key = None if columns is None else hashlib.sha1(np.asarray(columns).tobytes()).hexdigest()
    data = data if data is not None else build_data(cohort, spec, cache=cache)
    if columns is not None:
        data = data.columns(columns)
    results, natives = {}, {}
    for name in names:
        dec = get(name)
        dp = decoder_params(dec, params)
        native = None
        if needs_native(dec, data):
            native_ratio(dec, spec.bin_s)
            native = build_data(cohort, spec, dec.BIN_SECONDS, cache=cache)
            if columns is not None:
                native = native.columns(columns)
            natives[name] = native
        key = ('decoder_cv', id(cohort), name, spec.key(cohort.criteria), col_key, tuple(sorted(dp.items())), n_folds)
        ctx.check_cancel()
        results[name] = cache(key, lambda dec=dec, dp=dp, native=native: cv_decode(
            dec, data, dp, n_folds=n_folds, native=native, progress=ctx.progress))
    return dict(data=data, results=results, native=natives)


def describe(names, params: dict, spec: BinSpec | None = None) -> str:
    """Method sentences for a set of decoders (source, contract, parameters)."""
    parts = []
    for name in names:
        dec = get(name)
        dp = decoder_params(dec, params)
        grid = ''
        if dec.BIN_SECONDS is not None and spec is not None and abs(dec.BIN_SECONDS - spec.bin_s) > 1e-9:
            grid = (f' Fitted/predicted on native {dec.BIN_SECONDS * 1000:g} ms bins (trailing '
                    f'{spec.native_window * 1000:g} ms neural window) over the same trial windows; native '
                    f'predictions averaged onto each {spec.bin_s * 1000:g} ms evaluation bin (NaN unless all '
                    f'{native_ratio(dec, spec.bin_s)} native bins exist).')
        parts.append(f'[{dec.LABEL}] {dec.METHOD} Source: {source_text(name)}. Parameters: '
                     + ', '.join(f'{k}={v}' for k, v in dp.items()) + '.' + grid)
    return ' '.join(parts)
