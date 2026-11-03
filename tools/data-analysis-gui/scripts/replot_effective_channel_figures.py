"""Apply requested presentation changes using cached results; no model refits."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import effective_channel_study as E


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-only',action='store_true')
    selection_only=parser.parse_args().selection_only
    for run,label,trials in [('run001','1111 CO8 run-001',133),('run006','1111 Grid 12x12 run-006',179)]:
        out=E.S.ROOT/f'channel_budget_1111_{run}_effective'
        numerical=[out/name for name in ['dropout_details.npz','dropout_summary.json','selection_curve.csv','selection_summary.json']]
        before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in numerical}
        summary=json.loads((out/'dropout_summary.json').read_text(encoding='utf-8'))
        counts=summary['effective_counts_by_training_fold']
        if not selection_only:
            with np.load(out/'dropout_details.npz') as z:
                E.plot_dropout(out,label,trials,counts,z['random_r2'],z['least_r2'],z['most_r2'],summary['baseline_r2_mean'],scales=('linear',))
        rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((out/'selection_curve.csv').open(encoding='utf-8-sig'))]
        rows.sort(key=lambda r:r['channels'])
        baseline=json.loads((out/'selection_summary.json').read_text(encoding='utf-8'))['all_effective_r2']
        fig,ax=plt.subplots(figsize=(10,5.8),layout='constrained')
        x=[r['channels'] for r in rows]
        ax.fill_between(x,[r['ci_low'] for r in rows],[r['ci_high'] for r in rows],color='#2176ae',alpha=.16,label='95% block-bootstrap interval')
        ax.plot(x,[r['heldout_r2_mean'] for r in rows],'-o',color='#2176ae',label='Selected / retrained, outer held-out')
        ax.axhline(baseline,color='#555555',ls=':',label=f'All-effective baseline = {baseline:.3f}')
        E.axis(ax,counts,'log');E.endpoint(ax,counts,baseline);ax.legend(fontsize=8,loc='lower right')
        E.annotate_selection_scores(ax,rows)
        ax.set_title(f'{label} Best Channel Retrain',fontsize=11)
        E.presentation_font_scale(fig)
        fig.savefig(out/'selection.png',dpi=180,bbox_inches='tight',pad_inches=.15);fig.savefig(out/'selection.svg',bbox_inches='tight',pad_inches=.15);plt.close(fig)
        assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in before.items())
        print(out/'selection.png')
        print([(int(r['channels']),r['heldout_r2_mean']) for r in rows if int(r['channels']) in (8,16,32)])


if __name__=='__main__':
    main()
