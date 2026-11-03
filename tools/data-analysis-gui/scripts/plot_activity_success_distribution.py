"""Additional version of the 25-run activity plot: mean plus every block rate.

Keeps cached neural counts and original report/figures untouched. Outputs actual
block-level numerators/denominators alongside the new PNG and vector PDF.
"""
import csv
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis_gui.core import behavior
from data_analysis_gui.core.style import finish_figure
from data_analysis_gui.sections.x01_channel_stability_v1.section import ChannelStability

REPORT = Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report/2026-11-04_2026-11-11_25runs_v1')
OUT = REPORT / 'additional_figures'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gaussian-std',action='store_true',help='Plot fitted Gaussian mean +/- one sigma, without individual points.')
    gaussian=parser.parse_args().gaussian_std
    manifest_path = REPORT / 'report.json'
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    rows = manifest['activity_success_overlay']['data']
    assert [r['active'] for r in rows] == [360,357,360,335,335,335,321,319,302,277,332,326,339,331,333,336,337,334,328,339,358,328,203,203,204]
    all_blocks = []
    for i, row in enumerate(rows):
        source = Path(row['success_source'])
        # Use the exact original behavioral source, never reconstruct distributions from IQRs.
        raw = behavior._read(source)
        assert len({r['session_timestamp_iso'] for r in raw}) == 1
        assert len({r['trial_num_overall'] for r in raw}) == len(raw)
        mode, = {r['task_mode'] for r in raw}
        trials = behavior._from_csv({'trials': source}, mode)
        summary = behavior.block_success_summary(trials)
        for key in ('success_pct_mean','success_pct_q1','success_pct_q3','success_blocks'):
            np.testing.assert_allclose(summary[key], row[key], atol=1e-10)
        grouped = defaultdict(list)
        known = {'acquired','timeout','miss','incorrect','wrong_target'}
        for trial in trials:
            if trial['block'] is not None and trial['outcome'] in known:
                grouped[trial['block']].append(trial)
        for block, attempts in sorted(grouped.items()):
            successes = sum(t['outcome']=='acquired' for t in attempts)
            all_blocks.append(dict(session_index=i,session=row['session'],block=block,
                successes=successes,attempts=len(attempts),success_pct=100*successes/len(attempts),source=str(source)))
        selected = [b for b in all_blocks if b['session_index']==i]
        assert len(selected) == row['success_blocks']
        weighted = 100*sum(b['successes'] for b in selected)/sum(b['attempts'] for b in selected)
        np.testing.assert_allclose(weighted,row['success_pct_mean'])

    fig = ChannelStability.activity(rows)
    left,right = fig.axes
    # Preserve the bars, annotations, tick labels and run colors; replace only success overlay.
    # Clearing a twinned axis also resets its shared x-axis tick labels.
    for artist in list(right.lines) + list(right.collections):
        artist.remove()
    point_color, mean_color = '#e8884a','#963410'
    plotted = 0
    fits=[]
    for i,row in enumerate(rows):
        blocks = [b for b in all_blocks if b['session_index']==i]
        rates=np.array([b['success_pct'] for b in blocks])
        offsets=np.zeros(len(rates))
        for value in np.unique(rates):
            idx=np.flatnonzero(rates==value)
            # Horizontal spread has no quantitative meaning; every rate remains exact on y.
            offsets[idx]=np.linspace(-.29,.29,len(idx)) if len(idx)>1 else 0
        if gaussian:
            mu,sigma=norm.fit(rates)
            np.testing.assert_allclose([mu,sigma],[rates.mean(),rates.std(ddof=0)],atol=1e-12)
            fits.append(dict(session_index=i,session=row['session'],blocks=len(rates),
                             gaussian_mu_pct=float(mu),gaussian_sigma_pct=float(sigma),
                             lower_mu_minus_sigma=float(mu-sigma),upper_mu_plus_sigma=float(mu+sigma),
                             original_trial_weighted_mean_pct=row['success_pct_mean']))
        else:
            right.scatter(i+offsets,rates,s=8,color=point_color,alpha=.8,linewidths=.25,
                          edgecolors='white',zorder=4,clip_on=False)
        plotted+=len(rates)
    means=np.array([r['success_pct_mean'] for r in rows])
    if gaussian:
        means=np.array([f['gaussian_mu_pct'] for f in fits])
        sigmas=np.array([f['gaussian_sigma_pct'] for f in fits])
        mean_handle=right.errorbar(np.arange(len(rows)),means,yerr=sigmas,fmt='D',color=mean_color,
                        markersize=4.5,markeredgecolor='white',markeredgewidth=.55,
                        elinewidth=1,capsize=3,capthick=1,zorder=6,clip_on=False,
                        label='Gaussian mean ± SD')
    else:
        right.plot(np.arange(len(rows)),means,'D',color=mean_color,markersize=4.5,
                   markeredgecolor='white',markeredgewidth=.55,zorder=6,clip_on=False)
    right.set(ylabel='Success rate (%)',ylim=(0,100),yticks=[0,20,40,60,80,100])
    if gaussian:
        # Show the full fitted SD, even when an unbounded Gaussian extends past 100%.
        lo=min(0,float(np.floor(np.min(means-sigmas)/10)*10))
        hi=max(100,float(np.ceil(np.max(means+sigmas)/10)*10))
        right.set_ylim(lo-2,hi+2)
        right.set_yticks(np.arange(lo,hi+1,20))
    right.tick_params(axis='y',colors=mean_color)
    right.yaxis.label.set_color(mean_color)
    right.spines['right'].set_visible(True)
    right.spines['right'].set_color(mean_color)
    handles,names=left.get_legend_handles_labels()
    names=[name.replace(' (nonzero trial SD)','').replace(' (BH q < 0.05)','') for name in names]
    if gaussian:
        handles.append(mean_handle)
        names.append('Gaussian mean ± SD')
    else:
        handles.extend([Line2D([],[],marker='o',color=point_color,linestyle='none',markersize=3,label='Every block'),
                    Line2D([],[],marker='D',color=mean_color,linestyle='none',markersize=4,label='Run average')])
        names.extend(['Every block','Run average'])
    left.legend(handles,names,loc='lower center',bbox_to_anchor=(.5,1.025),ncol=5,frameon=False)
    fig.suptitle('Channel Activity & Success Rate')
    finish_figure(fig,manifest['font_percent'])
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/('Channel_Activity_Success_Rate_Gaussian_STD' if gaussian else 'Electrode_Activity_and_Direction_Tuning_Average_Full_Distribution')
    fig.savefig(target.with_suffix('.png'),dpi=300,facecolor='white',bbox_inches='tight',pad_inches=.15)
    fig.savefig(target.with_suffix('.pdf'),facecolor='white',bbox_inches='tight',pad_inches=.15)
    export_rows=fits if gaussian else all_blocks
    with target.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(export_rows[0])); writer.writeheader(); writer.writerows(export_rows)
    assert plotted==len(all_blocks)
    assert manifest_path.read_bytes()==original_manifest
    target.with_suffix('.json').write_text(json.dumps(dict(
        source_manifest=str(manifest_path),sessions=len(rows),source_blocks=plotted,individual_block_points_displayed=not gaussian,
        blocks_per_session=[r['success_blocks'] for r in rows],
        mean_definition=('Gaussian MLE mean: unweighted arithmetic mean of block success rates; sigma uses ddof=0, not SEM or a confidence interval.' if gaussian else 'Total successes / total known-outcome attempts per run (trial-weighted).'),
        distribution=('Per-run scipy.stats.norm.fit on all block success percentages. Diamond = mu; bars = mu +/- sigma. Normality is not established. Constant runs have sigma=0; no artificial variance. Error bars are not clipped at 0/100.' if gaussian else 'Every original task block: successes / known attempts; exact percentage, horizontal spread for ties only. No subsampling.'),
        neural_counts='Unchanged cached counts matching the user-provided 25-run figure.',
        checks='All session averages, quartiles and block counts reconcile against original overlay; run means independently reconcile from block successes/attempts; original manifest unchanged.'
    ),ensure_ascii=False,indent=2),encoding='utf-8')
    print(target.with_suffix('.png'))
    print(f'Validated {len(rows)} sessions, {plotted} blocks. Original report and plots preserved.')


if __name__=='__main__':
    main()
