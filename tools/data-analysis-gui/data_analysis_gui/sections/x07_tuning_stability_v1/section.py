"""Compare day-level tuning profiles on the same electrode IDs."""
from __future__ import annotations

from dataclasses import replace
from itertools import groupby
import re
import warnings

import numpy as np
from matplotlib import colormaps
from matplotlib.lines import Line2D

from ... import config as C
from ...core import analysis as A
from ...core.crossday import recording_labels
from ..base import Param, Section, SectionResult, new_figure
from ..s05_anova_v1.section import channel_anova
from ..s06_psth_v1.section import temporal_statistics


def direction_profile(ctx):
    stats, rates, _ = channel_anova(ctx)
    if np.any(stats['counts'] < 2):
        raise ValueError('方向比较需要每个方向至少 2 个有效 trial')
    active = rates.std(0) > 1e-12
    q = A.bh_fdr(stats['p'], np.flatnonzero(active))
    return stats['mean'], active, active & (q < ctx.params['alpha']), None


def psth_profile(ctx):
    info = temporal_statistics(ctx)
    post = info['post']
    covered = np.stack([np.sum(info['valid'][:, None] & np.isfinite(info['binned'][:, b]), axis=0) >= 3
                        for b in post])
    profile = np.where(covered, info['dmean'], np.nan)
    active = info['active'] & (covered.sum(0) >= 3)
    return profile, active, active & info['sig'], info['sign']


def daily_profiles(runs):
    """Average testable runs with equal weight, retaining run-level significance."""
    profiles, tested, tuned, signs = map(np.stack, zip(*runs))
    available = tested.any(0)
    significant = available & np.all(~tested | tuned, axis=0)
    sign = np.max(np.where(tested, signs, -np.inf), axis=0)
    significant &= np.all(~tested | (signs == sign), axis=0)
    valid = np.isfinite(profiles)
    count = valid.sum(0)
    mean = np.divide(np.nansum(profiles, axis=0), count,
                     out=np.full(count.shape, np.nan), where=count > 0)
    return mean, available, significant, sign


def profile_correlation(a, b):
    common = np.isfinite(a) & np.isfinite(b)
    n = common.sum(0)
    a0, b0 = np.where(common, a, 0), np.where(common, b, 0)
    a0 = np.where(common, a0 - a0.sum(0) / np.maximum(n, 1), 0)
    b0 = np.where(common, b0 - b0.sum(0) / np.maximum(n, 1), 0)
    norm = np.sqrt(np.sum(a0 ** 2, axis=0) * np.sum(b0 ** 2, axis=0))
    r = np.divide(np.sum(a0 * b0, axis=0), norm, out=np.full(n.shape, np.nan),
                  where=(n >= 3) & (norm > 1e-12))
    return np.clip(r, -1, 1)


def stability(days, kind, min_r, max_pd):
    profiles, tested, tuned, signs = map(np.stack, zip(*days))
    n_dates, n_electrodes = tested.shape
    pairs = np.zeros(n_electrodes, int)
    stable_pairs = pairs.copy()
    corr_count = pairs.copy()
    corr_sum = np.zeros(n_electrodes)
    compatible = np.zeros((n_dates, n_dates, n_electrodes), bool)
    pd = np.argmax(np.where(np.isfinite(profiles), profiles, -np.inf), axis=1) * 45 if kind == 'direction' else None
    for j in range(n_dates):
        for i in range(j):
            eligible = tested[i] & tested[j]
            r = profile_correlation(profiles[i], profiles[j])
            good = tuned[i] & tuned[j] & (r >= min_r)
            if pd is None:
                good &= signs[i] == signs[j]
            else:
                good &= np.abs((pd[i] - pd[j] + 180) % 360 - 180) <= max_pd
            pairs += eligible
            stable_pairs += good
            corr_count += np.isfinite(r)
            corr_sum += np.nan_to_num(r)
            compatible[i, j] = good
    average = np.divide(corr_sum, corr_count, out=np.full(n_electrodes, np.nan), where=corr_count > 0)
    longest = longest_span(tuned, compatible, list(range(n_dates)))
    return tested.sum(0), tuned.sum(0), longest, pairs, stable_pairs, average, compatible


def longest_span(tuned, compatible, indices):
    longest = np.zeros(tuned.shape[1], int)
    start = longest.copy()
    for j, day in enumerate(indices):
        for i, previous in enumerate(indices[:j]):
            # An incompatible earlier day rules out spans that still include it.
            start = np.where(compatible[previous, day], start, np.maximum(start, i + 1))
        longest = np.maximum(longest, np.where(tuned[day], j - start + 1, 0))
    return longest


def one_day_sensitivity(tuned, compatible, longest, eligible, dates, kind):
    best = longest.copy()
    skipped = np.full(len(best), '', dtype='U10')
    diagnostics = []
    if len(dates) >= 3:
        for i, date in enumerate(dates):
            alternative = longest_span(tuned, compatible, [j for j in range(len(dates)) if j != i])
            improves = alternative > best
            best[improves], skipped[improves] = alternative[improves], date
            strict, after = longest[eligible], alternative[eligible]
            diagnostics.append(dict(analysis=kind, skipped_date=date,
                                    electrodes_with_longer_span=int(np.sum(after > strict)),
                                    newly_stable_2plus=int(np.sum((strict < 2) & (after >= 2))),
                                    no_longer_stable_2plus=int(np.sum((strict >= 2) & (after < 2)))))
    return best, skipped, diagnostics


def excluded_dates(raw, dates):
    excluded = set()
    for token in re.split(r'[,，\s]+', raw):
        if not token:
            continue
        matches = [date for date in dates if token in (date, date[5:], date[5:].replace('-', ''))]
        if len(matches) > 1:
            raise ValueError(f'排除日期 {token} 跨多个年份，请输入 YYYY-MM-DD')
        if not matches:
            raise ValueError(f'排除日期 {token} 不在所选记录中')
        excluded.add(matches[0])
    return excluded


class TuningStability(Section):
    id = 'x07'
    mode = 'cross'
    title = '方向调谐与 PSTH 跨天稳定性'
    description = ('按电极 ID 比较 S05 方向曲线和 S06 时间响应。按不同日期统计稳定天数，并展示从稳定到不稳定的示例。'
                   '使用单日 S05 / S06 的显著性与时间设置。')
    source_sections = ('s05', 's06')
    params = (
        Param('min_r', '曲线相关 r 下限', 'float', 0.7, minimum=0, maximum=1),
        Param('max_pd', '偏好方向差上限 / °', 'float', 45, minimum=0, maximum=180),
        Param('examples', '每端示例电极数', 'int', 3, minimum=1, maximum=8),
        Param('exclude_dates', '排除日期（如 1026,1027）', 'text', ''),
        Param('rank_allow_one_skip', '示例按容许跳过一天排序', 'bool', True),
    )

    def run(self, ctx):
        dates = sorted({s.recording.date for s in ctx.sessions})
        excluded = excluded_dates(ctx.params['exclude_dates'], dates)
        dates = [date for date in dates if date not in excluded]
        if not all(dates) or len(dates) < 2:
            raise ValueError('排除指定日期后，X07 需要至少两个已知日期的记录')
        sources = ctx.params['_source_sections']
        result = SectionResult()
        status = []
        inputs = sorted(((s, c) for s, c in zip(ctx.sessions, ctx.cohorts) if s.recording.date not in excluded),
                        key=lambda pair: pair[0].recording.date)
        labels = dict(zip((s.recording.key for s, _ in inputs), recording_labels(s.recording for s, _ in inputs)))
        histogram = []
        date_diagnostics = []
        results = []
        for kind, source, compute in (('direction', 's05', direction_profile), ('psth', 's06', psth_profile)):
            settings = sources[source]['params']
            if kind == 'direction':
                x = np.arange(8) * 45
            else:
                edges = np.arange(settings['tmin'], settings['tmax'] + 1e-9, settings['bin'])
                x = (edges[:-1] + edges[1:]) / 2
                x = x[x > 0]
            blank = (np.full((len(x), C.N_ELECTRODES), np.nan), np.zeros(C.N_ELECTRODES, bool),
                     np.zeros(C.N_ELECTRODES, bool), np.zeros(C.N_ELECTRODES))
            days = []
            for _, runs in groupby(inputs, key=lambda pair: pair[0].recording.date):
                daily = []
                for session, cohort in runs:
                    ctx.check_cancel()
                    label = labels[session.recording.key]
                    ctx.progress(f'{kind} · {label}')
                    single = replace(ctx, sessions=[session], cohorts=[cohort], params=settings)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', RuntimeWarning)
                            key = ('xday_tuning_profile', id(cohort), kind, tuple(sorted(settings.items())))
                            curve, tested, tuned, sign = ctx.cached(key, lambda: compute(single))
                    except ValueError as exc:
                        status.append(dict(analysis=kind, recording=label, status=str(exc)))
                        continue
                    ids = np.asarray(session.electrode_ids) - 1
                    full, tested_all, tuned_all, signs = (a.copy() for a in blank)
                    full[:, ids[tested]] = curve[:, tested]
                    tested_all[ids], tuned_all[ids] = tested, tuned
                    if sign is not None:
                        signs[ids] = sign
                    daily.append((full, tested_all, tuned_all, signs))
                    status.append(dict(analysis=kind, recording=label, status='computed',
                                       testable_electrodes=int(tested.sum())))
                days.append(daily_profiles(daily) if daily else blank)
            tested, tuned, longest, pairs, stable, corr, compatible = stability(
                days, kind, ctx.params['min_r'], ctx.params['max_pd'])
            eligible = np.flatnonzero(tested >= 2)
            robust, skipped, diagnostics = one_day_sensitivity(
                np.stack([day[2] for day in days]), compatible, longest, eligible, dates, kind)
            date_diagnostics.extend(diagnostics)
            rows = [dict(electrode=int(i + 1), tested_days=int(tested[i]), tuned_days=int(tuned[i]),
                         stable_days=int(longest[i]), stable_days_allow_one_skip=int(robust[i]),
                         skipped_date=skipped[i], tested_pairs=int(pairs[i]), stable_pairs=int(stable[i]),
                         stable_pair_fraction=float(stable[i] / pairs[i]), mean_profile_r=float(corr[i]))
                    for i in eligible]
            rank_key = 'stable_days_allow_one_skip' if ctx.params['rank_allow_one_skip'] else 'stable_days'
            rows.sort(key=lambda r: (-r[rank_key], -r['stable_days'], -r['stable_pair_fraction'],
                                     -np.nan_to_num(r['mean_profile_r'], nan=-2), -r['tested_days'], r['electrode']))
            for rank, row in enumerate(rows, 1):
                row['rank'] = rank
            result.tables[f'{kind}_electrodes'] = rows
            result.tables[f'{kind}_day_status'] = [
                dict(date=date, electrode=int(i + 1), testable=bool(day[1][i]), significant=bool(day[2][i]))
                for date, day in zip(dates, days) for i in eligible]
            counts = np.bincount(longest[eligible], minlength=len(dates) + 1)
            robust_counts = np.bincount(robust[eligible], minlength=len(dates) + 1)
            histogram.extend(dict(analysis=kind, stable_days=k, channels=int(n), allow_one_skip=int(r))
                             for k, (n, r) in enumerate(zip(counts, robust_counts)))
            result.summary.update({f'{kind}_eligible': len(eligible), f'{kind}_stable': int(counts[2:].sum()),
                                   f'{kind}_stable_allow_one_skip': int(robust_counts[2:].sum())})
            results.append((kind, x, days, rows, counts, robust_counts))

        method = (f'Electrodes are matched by ID. Within each selected date, testable runs contribute equally to the '
                  f'mean profile, and every testable run must pass its source BH q threshold for that date to count '
                  f'as tuned (S05 q < {sources["s05"]["params"]["alpha"]:g}; '
                  f'S06 q < {sources["s06"]["params"]["alpha"]:g}). S06 also requires consistent response sign. '
                  f'Dates in a stable span are consecutive in the selected date list and every pair in the span '
                  f'must be tuned with Pearson r ≥ {ctx.params["min_r"]:g} over at least three common profile values. '
                  f'S05 also requires peak direction separation ≤ {ctx.params["max_pd"]:g}°; S06 requires equal '
                  'response sign. Missing or untestable selected dates break a span. Unselected calendar days are '
                  'not evaluated. Direction profiles are S05 mean rate per direction; temporal profiles are raw S06 '
                  'paired post-event bin rates minus pre-event baseline. Sensitivity takes the maximum of the '
                  'strict span and all single-date deletions, separately per electrode. The omitted date does '
                  'not count toward the span. Date diagnostics use the same original eligible electrode set.')
        fig = new_figure(10, 4)
        axes = fig.subplots(1, 2)
        for ax, (kind, _, _, rows, counts, robust_counts) in zip(axes, results):
            bins = np.arange(len(counts))
            ax.bar(bins - .2, counts, width=.4, color='#286e9e', label='Strict')
            ax.bar(bins + .2, robust_counts, width=.4, color='#b95a2b', label='Allow one date skipped')
            ax.legend(frameon=False, fontsize=8)
            ax.set(xlabel='Longest stable span (selected days)', ylabel='Electrodes',
                   title=f'{kind.upper()} · {len(rows)} electrodes tested on ≥2 dates')
            ax.set_xticks(range(len(counts)))
            ax.grid(axis='y', alpha=.2)
        result.add_figure('Tuning stability by number of days', fig, method=method,
                          data={'stable_day_counts': histogram})
        result.tables['stable_day_counts'] = histogram
        if date_diagnostics:
            result.tables['leave_one_date_out'] = date_diagnostics
        result.tables['recording_status'] = status
        for kind, x, days, rows, _, _ in results:
            fig, curves = example_figure(kind, x, dates, days, rows, ctx.params['examples'])
            result.add_figure(f'{kind.upper()} tuning examples', fig, method=method +
                              f' Examples include only electrodes tuned on at least one date. Rank first by '
                              f'{rank_key}, then strict span, stable-pair fraction, mean profile r, coverage, '
                              'and electrode ID.',
                              data={'profiles': curves})
        result.summary['selected_days'] = len(dates)
        result.summary['excluded_dates'] = sorted(excluded)
        return result


def example_figure(kind, x, dates, days, rows, limit):
    candidates = [row for row in rows if row['tuned_days']]
    count = min(limit, len(candidates))
    shown = candidates[:count] + candidates[max(count, len(candidates) - limit):]
    if not shown:
        fig = new_figure(9, 3)
        fig.text(.5, .5, 'No electrode tested on ≥2 dates and tuned on at least one', ha='center')
        return fig, []
    fig = new_figure(11, 2.8 * ((len(shown) + 1) // 2) + .8)
    axes = fig.subplots((len(shown) + 1) // 2, 2, squeeze=False).flat
    colors = colormaps['tab20'](np.arange(len(dates)) % 20)
    curves = []
    date_colors = {}
    for ax, row in zip(axes, shown):
        i = row['electrode'] - 1
        for date, color, (profile, tested, significant, _) in zip(dates, colors, days):
            if not tested[i]:
                continue
            y = profile[:, i]
            ax.plot(x, y, color=color, ls='-' if significant[i] else '--',
                    marker='o' if kind == 'direction' else None, ms=3, lw=1.1)
            date_colors[date] = color
            curves.extend(dict(rank=row['rank'], electrode=i + 1, date=date, x=float(t), value_hz=float(v),
                               significant=bool(significant[i])) for t, v in zip(x, y))
        title = f'#{row["rank"]} · E{i + 1} · {row["stable_days"]} strict / {row["tested_days"]} tested days'
        if row['skipped_date']:
            title += f'\n{row["stable_days_allow_one_skip"]} days if {row["skipped_date"]} omitted'
        ax.set(title=title,
               xlabel='Target direction (°)' if kind == 'direction' else 'Time from alignment (s)',
               ylabel='Rate (Hz)' if kind == 'direction' else 'Change from baseline (Hz)')
        if kind == 'direction':
            ax.set_xticks(x)
        else:
            ax.axhline(0, color='#777777', lw=.6)
        ax.grid(alpha=.18)
    for ax in fig.axes[len(shown):]:
        ax.set_axis_off()
    # Use a separate legend so the same date keeps its color across panels.
    date_labels = [date for date in dates if date in date_colors]
    if date_labels:
        fig.legend([Line2D([], [], color=date_colors[d]) for d in date_labels], date_labels,
                   loc='outside lower center', ncol=min(7, len(date_labels)), fontsize=7, frameon=False)
    fig.suptitle('Ranked examples · solid = significant date · dashed = non-significant date', fontsize=10)
    return fig, curves


SECTION = TuningStability()
