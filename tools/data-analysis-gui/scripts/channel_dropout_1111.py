"""Independent GUI-matched frozen-channel dropout for 1111 run-001/run-006."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import channel_budget_1110_velkf as study


def main():
    jobs=[('CO8','run-001','1111 CO8 run-001'),('Grid','run-006','1111 Grid 12x12 run-006')]
    for task,run,label in jobs:
        sources=list((study.ROOT/'keaiduo/single').glob(f'2026-11-11_{task}_{run}_*/s10_v1/*/result.json'))
        assert len(sources)==1, sources
        source=sources[0];reference=json.loads(source.read_text(encoding='utf-8'))
        sha=reference['inputs']['recordings'][0]['sha256']
        caches=list((study.ROOT/'keaiduo/.session-cache'/sha[:20]/'decoder_data').glob('*.npz'))
        assert len(caches)==1,caches
        study.ORIGINAL=source;study.CACHE=caches[0]
        study.OUT=study.ROOT/f'channel_budget_1111_{run.replace("-", "")}_velkf_matched'
        study.OUT.mkdir(exist_ok=True)
        study.RUN_LABEL=label;study.PLOT_TRIALS=reference['summary']['n_trials']
        study.PARAMS={key:reference['inputs']['params'][key] for key in ('lam_kf','q_shrinkage')}
        data=study.load()
        print('START',label,'trials',study.PLOT_TRIALS,flush=True)
        folds,baseline=study.dropout(data)
        split_rows=[]
        for f,tr,te,m in folds:
            split_rows.append(dict(fold=f+1,train_blocks=json.dumps(np.unique(data.block[tr]).tolist()),
                eval_blocks=json.dumps(np.unique(data.block[te]).tolist()),
                train_trials=len(np.unique(data.trial[tr])),eval_trials=len(np.unique(data.trial[te])),
                train_bins=len(tr),eval_bins=len(te),active_train_channels=len(m['cols'])))
        study.table(study.OUT/'folds.csv',split_rows)
        for script in (Path(__file__),Path(study.__file__)):
            shutil.copy2(script,study.OUT/script.name)
        study.dump(study.OUT/'provenance.json',dict(run=label,original_result=str(source),recording_sha256=sha,
            cache=str(study.CACHE),scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(study.__file__))},
            independent_single_run=True,reference_baseline_r2=reference['summary']['velocity_kf_r2_mean'],
            reproduced_baseline_r2=float(study.r2_score(data.velocity,baseline).mean()),
            effective_channels_all_data=int((data.features.std(0)>1e-12).sum()),
            note='Dropout only: random / least-important / most-important; fixed model per fold; no retraining during channel loss. 200 permutations use same seed as 1110.'))
        print('COMPLETE',label,'baseline',study.r2_score(data.velocity,baseline).mean(),flush=True)


if __name__=='__main__':
    with study.threadpool_limits(limits=2):main()
