"""Reproduce GUI direction-F ranking and audit zero-count layout columns."""
from pathlib import Path
import sys
import json
import csv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from matplotlib.figure import Figure

from data_analysis_gui import storage
from data_analysis_gui.core.catalog import describe
from data_analysis_gui.core.session import Session
from data_analysis_gui.core.filters import TrialFilter, apply_filter
from data_analysis_gui.core.analysis import oneway_anova, bh_fdr


def main():
    root = Path('E:/AitBCI/data/keaiduo')
    report = Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report/2026-11-04_2026-11-11_25runs_v1/report.json')
    manifest = json.loads(report.read_text(encoding='utf-8'))
    source = next(s for s in manifest['sources'] if s['section'] == 'x01')
    reference = json.loads((Path(source['folder']) / 'result.json').read_text(encoding='utf-8'))
    criteria = TrialFilter.from_dict(reference['inputs']['criteria'])
    out = Path('E:/AitBCI/data_analysis_gui_output/keaiduo/comparisons/20261111_run001_vs_run006')
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for run in ['run-001-CO8-offline', 'run-006-Grid-offline-12x12']:
        path = next((root / '20261111' / run / 'xdf').glob('*.xdf'))
        session = Session.load(describe(path, root))
        cohort = apply_filter(session, criteria)
        finite = np.isfinite(cohort.window_rates).all(1)
        rates, dirs = cohort.window_rates[finite], cohort.directions[finite]
        st = oneway_anova(rates, dirs)
        active = rates.std(0) > 1e-12
        q = bh_fdr(st['p'], np.flatnonzero(active))
        ids = session.electrode_ids.astype(int)
        assert len(ids) == len(set(ids)) == 512
        order = [int(j) for j in np.argsort(-st['F']) if active[j]]
        rank = {int(ids[j]): i + 1 for i, j in enumerate(order)}
        channels = {int(e): dict(electrode=int(e), layout_column=int((e - 1) // 128 + 1),
                                 stored_column=j + 1, max_code=int(session.counts[:, j].max()),
                                 mean_hz=float(session.counts[:, j].mean() / session.frame_seconds),
                                 active=bool(active[j]), F=float(st['F'][j]), q=float(q[j]),
                                 significant=bool(active[j] and q[j] < .05), rank=rank.get(int(e)))
                    for j, e in enumerate(ids)}
        records.append(dict(run=run, task=session.task_mode, path=str(path), sha256=storage.file_digest(path),
                            frames=len(session.counts), retained_trials=int(finite.sum()),
                            direction_counts=np.bincount(dirs, minlength=8).tolist(),
                            top32=[int(ids[j]) for j in order[:32]], channels=channels))
    a, b = records
    assert set(a['channels']) == set(b['channels'])
    rows = []
    for e in sorted(a['channels']):
        ca, cb = a['channels'][e], b['channels'][e]
        rows.append(dict(electrode=e, layout_column=ca['layout_column'],
                         run001_rank=ca['rank'], run006_rank=cb['rank'],
                         run001_F=ca['F'], run006_F=cb['F'], run001_q=ca['q'], run006_q=cb['q'],
                         run001_max=ca['max_code'], run006_max=cb['max_code'],
                         run001_mean_hz=ca['mean_hz'], run006_mean_hz=cb['mean_hz'],
                         run001_top32=e in a['top32'], run006_top32=e in b['top32'],
                         became_zero=ca['max_code'] > 0 and cb['max_code'] == 0))
    left = [r for r in rows if r['layout_column'] <= 2]
    lost_top = [e for e in a['top32'] if a['channels'][e]['max_code'] > 0 and b['channels'][e]['max_code'] == 0]
    reverse = [dict(run006_rank=i+1, electrode=e, run001_rank=a['channels'][e]['rank'])
               for i, e in enumerate(b['top32'])]
    assert all(r['run001_rank'] is not None for r in reverse)
    summary = dict(
        ranking='Direction ANOVA F descending among active channels, same trial filter and 0.2–1.2 s window as reference report; electrode IDs aligned.',
        column_definition='Report map_layout=default: 128x8 ID grid, column1 E1–E128, column2 E129–E256. Not physical workbook positions.',
        reference_report=str(report), criteria=reference['inputs']['criteria'],
        run001_trials=a['retained_trials'], run006_trials=b['retained_trials'],
        run001_active=sum(c['active'] for c in a['channels'].values()),
        run006_active=sum(c['active'] for c in b['channels'].values()),
        identical_transmitted_ids=True, transmitted=512,
        top32_overlap=sorted(set(a['top32']) & set(b['top32'])),
        run006_top32_within_run001_top={str(k): sum(r['run001_rank'] <= k for r in reverse) for k in [32, 64, 100, 128, 148]},
        run006_top32_max_run001_rank=max(r['run001_rank'] for r in reverse),
        left_columns_transmitted=len(left),
        left_columns_nonzero_run001=sum(r['run001_max'] > 0 for r in left),
        left_columns_nonzero_run006=sum(r['run006_max'] > 0 for r in left),
        left_columns_active_run001=sum(a['channels'][r['electrode']]['active'] for r in left),
        left_columns_significant_run001=sum(a['channels'][r['electrode']]['significant'] for r in left),
        run001_top32_became_zero=lost_top,
        run001_top32_outside_left_still_nonzero=[e for e in a['top32'] if e not in lost_top],
        run006_top32_ranked_in_run001=reverse,
        limitations=['CO8 and Grid differ in task and sampled directions; F rank shifts alone do not establish channel loss.',
                     'Zero counts throughout the retained XDF establish zero observed spike counts, not a verified hardware cause.',
                     'Top32 is a descriptive whole-session ANOVA ranking, not held-out decoder feature importance.'])
    for filename, table in [('all_transmitted_channels.csv', rows),
                             ('run006_top32_rank_in_run001.csv', reverse),
                             ('run001_top32_became_zero.csv', [r for r in rows if r['electrode'] in lost_top])]:
        with (out / filename).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    for rec in records:
        (out / (rec['run'] + '_source.json')).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf-8')
    fig = Figure(figsize=(12, 4.8), layout='constrained')
    ax, bx = fig.subplots(1, 2)
    ax.plot(np.arange(1, 33), [r['run001_rank'] for r in reverse], 'o', color='#2176ae', markersize=4)
    ax.axhline(32, linestyle='--', color='#bd5021', label='run-001 top32 cutoff')
    ax.set(xlabel='Rank in run-006 top32', ylabel='Rank in run-001',
           title=f'Top32 overlap: {len(summary["top32_overlap"])}/32', ylim=(155, 0))
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    bx.bar(['Remain nonzero', 'Became zero'], [32-len(lost_top), len(lost_top)], color=['#2176ae', '#bd5021'])
    for i, n in enumerate([32-len(lost_top), len(lost_top)]):
        bx.text(i, n+.4, str(n), ha='center')
    bx.set(ylabel='Channels from run-001 top32', ylim=(0, 32), title='Observed status in run-006')
    bx.grid(axis='y', alpha=.2)
    fig.suptitle('2026-11-11: CO8 offline run-001 vs Grid offline run-006')
    fig.savefig(out / 'top32_comparison.png', dpi=180)
    fig.clear()
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
