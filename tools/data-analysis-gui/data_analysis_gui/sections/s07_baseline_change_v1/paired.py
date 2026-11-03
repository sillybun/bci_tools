"""Paired direction-mean and direction-specific temporal-response channel atlas."""
from dataclasses import replace

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ...core import style
from ..s05_anova_v1.section import channel_anova
from ..s06_psth_v1.section import temporal_statistics, smooth


TITLE = 'ANOVA and PSTH - significant channels'


def append_paired(ctx, result, groups, directional):
    ids = ctx.session.electrode_ids
    st, _, _ = channel_anova(replace(ctx,params=groups['sources']['s05']['params']))
    p = groups['sources']['s06']['params']
    time = temporal_statistics(replace(ctx,params=p))
    # A channel can have no pooled temporal effect yet a directional one. This
    # page's inclusion rule stays the clearly labelled S05/S06 significant union.
    selected = np.flatnonzero(groups['direction'] | groups['temporal'])
    selected = selected[np.argsort(ids[selected],kind='stable')]
    raw = np.full((8,len(time['centres']),len(ids)),np.nan)
    counts = np.zeros(8,int)
    for d in range(8):
        keep = time['valid'] & (time['dirs']==d)
        counts[d] = keep.sum()
        if counts[d]:
            raw[d] = np.nanmean(time['binned'][keep],axis=0)
    plotted = smooth(raw,time['sigma'],axis=1)
    pages = max(1,(len(selected)+31)//32)
    result.summary['paired_tuning_channels'] = int(len(selected))
    result.summary['paired_tuning_pages'] = pages
    for page in range(pages):
        order = selected[page*32:(page+1)*32]
        fig = Figure(figsize=(23,12.5),dpi=100)
        grid = fig.add_gridspec(4,8,left=.035,right=.992,bottom=.08,top=.88,wspace=.5,hspace=.65)
        rows = []
        for cell in range(32):
            inner = grid[cell//8,cell%8].subgridspec(1,2,width_ratios=[.9,1.3],wspace=.48)
            left,right = fig.add_subplot(inner[0]),fig.add_subplot(inner[1])
            if cell>=len(order):
                left.axis('off');right.axis('off')
                if not len(order) and cell==0:
                    left.text(.5,.5,'No significant channels',ha='center',transform=left.transAxes,fontsize=7)
                continue
            j = order[cell]
            qd = groups['rows'][j]['direction_q']
            qt = groups['rows'][j]['time_q']
            color = '#c0392b' if groups['direction'][j] else '#555555'
            left.errorbar(np.arange(8)*45,st['mean'][:,j],yerr=st['sem'][:,j],fmt='o-',
                          color=color,ms=1.8,lw=.7,capsize=1,elinewidth=.5)
            left.set_xticks([0,180,315]);left.set_xlim(-15,330)
            pref = directional['preferred'][j]
            if pref>=0:
                left.axvline(pref*45,color=style.direction_color(pref),ls=':',lw=1)
                left.plot(pref*45,st['mean'][pref,j],marker='*',ms=6,color=style.direction_color(pref))
            left.set_title(f'E{int(ids[j])}  D{"+" if groups["direction"][j] else "−"} '
                           f'T{"+" if groups["temporal"][j] else "−"}\nqD={qd:.1g} qT={qt:.1g}',fontsize=5.5,pad=3)
            left.set_xlabel('Direction / deg',fontsize=5)
            # Draw muted curves first so significant ones remain visible.
            for significant in (False,True):
                for d in range(8):
                    if not counts[d] or bool(directional['sig'][d,j])!=significant:
                        continue
                    right.plot(time['centres'],plotted[d,:,j],color=style.direction_color(d),
                               lw=1.35 if significant else .5,alpha=1 if significant else .25,
                               ls='-' if significant else '--')
            highlighted = np.flatnonzero(directional['sig'][:,j])
            tested = np.flatnonzero(directional['eligible'][:,j])
            label = ','.join(str(int(d*45)) for d in highlighted)
            if not len(highlighted):
                label = 'none' if len(tested) else 'N/A'
            right.set_title(f'Time(d): {label}',fontsize=5.2,pad=3)
            right.axvspan(p['baseline_start'],p['baseline_end'],color='#e8e6df',lw=0,zorder=-2)
            right.axvline(0,color=style.INK,lw=.45,ls=':')
            right.set_xlim(p['tmin'],p['tmax'])
            right.set_xlabel('Time / s',fontsize=5)
            for ax in (left,right):
                ax.tick_params(labelsize=4.5,length=2,pad=1,labelleft=True,labelbottom=True)
                ax.spines[['top','right']].set_visible(False)
                ax.ticklabel_format(axis='y',style='plain',useOffset=False)
                ax.set_ylabel('Hz',fontsize=5,labelpad=1)
            for d in range(8):
                row = dict(electrode=int(ids[j]),stored_column=int(j+1),direction_deg=d*45,
                           direction_anova_significant=bool(groups['direction'][j]),direction_q=qd,
                           pooled_time_significant=bool(groups['temporal'][j]),pooled_time_q=qt,
                           preferred=bool(pref==d),window_mean_hz=float(st['mean'][d,j]),
                           window_sem_hz=float(st['sem'][d,j]),time_trials=int(counts[d]),
                           directional_time_tested=bool(directional['eligible'][d,j]),
                           directional_time_q=float(directional['q'][d,j]) if directional['eligible'][d,j] else None,
                           highlighted=bool(directional['sig'][d,j]))
                row.update({f'raw_bin_{k}_t_{t:+.4f}s':float(v) for k,(t,v) in enumerate(zip(time['centres'],raw[d,:,j]))})
                row.update({f'plotted_bin_{k}_t_{t:+.4f}s':float(v) for k,(t,v) in enumerate(zip(time['centres'],plotted[d,:,j]))})
                rows.append(row)
        handles = [Line2D([],[],color=style.direction_color(d),label=f'{d*45}°',lw=1.2) for d in range(8)]
        fig.legend(handles=handles,ncol=8,loc='upper center',bbox_to_anchor=(.5,.94),frameon=False,fontsize=8)
        fig.suptitle(f'{ctx.session.label} · ANOVA + PSTH · page {page+1}/{pages} · {len(selected)} selected channels\n'
                     f'4×8 channels/page; each cell: direction means | directional PSTH ({p["align"]}, bin {p["bin"]*1000:g} ms)',
                     fontsize=10,y=.988)
        fig.text(.5,.026,'Selection: S05 direction significant OR pooled S06 time significant; electrode order. '
                 'D/T labels = these pooled channel tests. Left star = preferred direction.\n'
                 'Right bold solid = significant time response WITHIN that direction; faint dashed = not significant or untested. '
                 'Highlight is NOT a between-direction comparison. Every axis has its own Hz scale.',ha='center',fontsize=8)
        fig._series_pdf_name = TITLE
        result.add_figure(f'{TITLE} {page+1}',fig,method=(
            'Selection = union of ordinary S05 significant channels and pooled S06 temporal significant '
            'channels, in electrode order; neither direction-specific significant-only nor Both-only selection. '
            '32 cells/page (4 rows, 8 columns); each cell contains independent y-labelled ANOVA and PSTH axes. '
            'Left: S05 mean analysis-window Hz ± SEM, star/dotted line = unique preferred direction among '
            'S05 significant channels, not a pairwise significant-direction test. D/qD = ordinary S05; '
            'T/qT = pooled S06. Right: direction trial-mean absolute Hz, only S06 baseline-eligible trials, '
            'same alignment/bin/baseline; smoothing is display-only. Bold solid curves ONLY when existing '
            'S07 within-direction temporal q < S06 alpha (bin Bonferroni, joint channel×direction BH, '
            'at least 8 baseline/paired trials). Other directions faint dashed; N/A if none testable. '
            'No direction×time interaction or pairwise direction inference. Independent y limits for '
            'every subplot, including the two axes of one channel. Full raw/displayed bins, exact flags '
            'and q values exported. '+str(groups['sources'])),data={'paired_channel_curves':rows})
