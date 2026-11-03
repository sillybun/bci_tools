"""Compose the selected source figures into the single-day overview."""
from dataclasses import replace

import numpy as np

from ...core import style
from .. import versions
from ..base import SectionResult, new_figure
from ..s07_baseline_change_v1.overlap import PARTITIONS, TIERS, append_overlap, draw_composition

TITLES = ('Task success rate', 'Channel saturation zones', 'Transmission overview',
          'Dimensionality overview', 'Mutually exclusive channel composition')


def overview(ctx, result, summary):
    fig = new_figure(16, 37)
    panels = fig.subfigures(5, 1, height_ratios=[4, 10, 9, 9, 5], hspace=.02)
    fig.suptitle(f'{ctx.session.label} · Single-recording overview', fontsize=14)
    summary['overview_modules'] = list(TITLES)
    methods, data = [], {}

    def source(section_id):
        settings = ctx.params['_source_sections'][section_id]
        section = next(s for s in versions(section_id) if s.version == settings['version'])
        return section, replace(ctx, params=settings['params'])

    def add_source(index, section_id, update_summary):
        ctx.check_cancel()
        ctx.progress(f'overview · {section_id.upper()}')
        section, source_ctx = source(section_id)
        title = TITLES[index]
        try:
            output = section.run(source_ctx, overview_target=panels[index])
        except ValueError as exc:
            style.text_panel(panels[index].subplots(), [title, f'Unavailable: {exc}'])
            result.notes.append(f'{section_id.upper()} · {exc}')
            methods.append(f'{section_id.upper()} {title}: unavailable ({exc}).')
            return
        selected = next(i for i, (name, _) in enumerate(output.figures) if name == title)
        methods.append(f'{section_id.upper()} {title}: {output.methods[selected]}')
        data.update({f'{section_id}_{key}': rows for key, rows in output.figure_data[selected].items()})
        result.notes.extend(f'{section_id.upper()} · {note}' for note in output.notes)
        if update_summary is not None:
            update_summary(output.summary)
        for _, other in output.figures:
            if other is not panels[index]:
                other.clear()

    section, _ = source('s01')
    _, success = section.task_success(ctx.cohort, figure=panels[0])
    data['s01_task_success'] = success
    methods.append('S01 Task success rate: trailing 40 complete attempts with known outcome before '
                   'the analysis trial filter; directional curves use their subset of the same window.')

    def saturation(stats):
        summary.update(overview_saturation_mean=stats['saturation_mean'],
                       overview_often_saturated=stats['channels_by_zone']['often saturated'],
                       overview_transmitted=stats['transmitted_electrodes'])

    def dimensions(stats):
        summary['overview_pca_pc12_variance'] = sum(stats['pc_explained_pct'][:2]) / 100

    add_source(1, 's02', saturation)
    add_source(2, 's03', None)
    add_source(3, 's04', dimensions)

    ctx.check_cancel()
    ctx.progress('overview · S07')
    tuning = SectionResult()
    groups = append_overlap(ctx, tuning)
    axis = panels[4].subplots()
    if groups is None:
        style.text_panel(axis, [TITLES[4], 'Unavailable: S05 / S06 tests unavailable'])
        methods.append('S07 composition unavailable because the source tests could not be computed.')
    else:
        rows = tuning.tables.get('tuning_config_composition')
        if rows is not None:
            tiers = TIERS
            panels[4].suptitle('S07 · Channel composition by config tier', fontsize=10)
        else:
            tiers = ('all transmitted',)
            total = len(groups['categories'])
            rows = []
            for name in PARTITIONS:
                count = int(np.count_nonzero(groups['categories'] == name))
                rows.append(dict(tier=tiers[0], category=name, channels=count, tier_channels=total,
                                 percent=100 * count / total if total else None))
            panels[4].suptitle('S07 · All transmitted channels (no usable config tiers)', fontsize=10)
        draw_composition(axis, rows, tiers)
        if len(tiers) == 1:
            axis.set_xlim(-1.5, 1.5)
        data['s07_disjoint_composition'] = rows
        result.tables['overview_tuned_channels'] = groups['rows']
        summary.update(overview_direction_tuned=int(groups['direction'].sum()),
                       overview_time_tuned=int(groups['temporal'].sum()),
                       overview_both_tuned=int(groups['both'].sum()))
        methods.append('S07 panel A: transmitted channels split into Inactive, Active not significant, '
                       'Direction only, Time only and Both from S05/S06. Counts sum to the tier total. '
                       'Without usable config tiers, all transmitted channels form one bar.')
    result.notes.extend(f'S07 · {note}' for note in tuning.notes if '跳过配置层级' not in note)
    for _, other in tuning.figures:
        other.clear()
    return fig, '\n'.join(methods), data
