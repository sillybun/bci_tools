"""Add requested daily first-run BGA maps to the completed 60-run report."""
from pathlib import Path
import csv,html,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
from data_analysis_gui import storage
from data_analysis_gui.report_bga import render_daily_bga,first_run
from data_analysis_gui.report_run_pool import RunPool,atomic_json

ROOT=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report/2026-11-10_2026-11-23_60runs_v1')
AUDIT=ROOT.parent.parent/'import_20261123/daily_bga_backfill'
def main():
    AUDIT.mkdir(parents=True,exist_ok=True)
    report_path=ROOT/'report.json';report=json.loads(report_path.read_text(encoding='utf-8'))
    if (ROOT/'bga_daily_manifest.json').exists():
        print('Already generated:',ROOT/'bga_daily_manifest.json');return
    original={p.relative_to(ROOT).as_posix():storage.file_digest(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.png','.pdf','.svg','.csv')}
    atomic_json(AUDIT/'report_before.json',report)
    atomic_json(AUDIT/'existing_files.json',original)
    pool=RunPool(ROOT.parent/'individual_runs')
    result=render_daily_bga(ROOT,report,100,lambda *a,**k:print(k.get('message',''),flush=True),pool=pool)
    assert [d['date'] for d in result['days']]==['20261110','20261111','20261117','20261118','20261119','20261120','20261123']
    items=[]
    for d in result['days']:
        folder=Path(d['run_folder'])
        assert folder==first_run(folder.parent)
        assert '20261116' not in str(folder)
        for name in d['files'].values():assert (ROOT/name).is_file()
        p=ROOT/d['files']['csv']
        with p.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(line for line in f if not line.startswith('#')))
        assert len(rows)==1024 and {int(r['channel_0based']) for r in rows}==set(range(1024))
        if not list(folder.rglob('*.xdf')):assert d['status']=='no_xdf' and d['transmitted_channels']==0
        items.append('<article><h2>'+d['date']+' · '+html.escape(folder.name)+'</h2><p>'+html.escape(d['status'])+' · transmitted channels: '+str(d['transmitted_channels'])+'</p><a href="'+Path(d['files']['png']).name+'"><img loading="lazy" src="'+Path(d['files']['png']).name+'"></a></article>')
    index=ROOT/'02_通道稳定性分析/BGA_daily_first_run/index.html'
    index.write_text('<!doctype html><meta charset="utf-8"><title>Daily first-run BGA maps</title><style>body{font:15px Microsoft YaHei,Arial;margin:24px;color:#17212b}img{width:100%;max-width:1100px}article{margin-bottom:36px}</style><h1>每天第一个 run 的 BGA 映射图</h1><p>1110–1123；排除1116。首个run无XDF时显示无数据，不替换为后续run。彩色=非零code（按shank）；黑色=全零；灰色=未传输/无数据。</p>'+''.join(items),encoding='utf-8')
    report['bga_daily']=dict(result,requested_backfill=True,index=str(index.relative_to(ROOT)))
    report['daily_bga_pool']=pool.manifest()
    atomic_json(report_path,report)
    assert all(storage.file_digest(ROOT/name)==digest for name,digest in original.items())
    audit=dict(days=result['days'],existing_artifacts_unchanged=len(original),pool=pool.manifest(),index=str(index))
    atomic_json(AUDIT/'verification.json',audit)
    print(json.dumps(dict(days=len(result['days']),existing_artifacts_unchanged=len(original),index=str(index)),ensure_ascii=False))
if __name__=='__main__':main()
