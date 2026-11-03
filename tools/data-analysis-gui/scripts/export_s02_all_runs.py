"""Compute S02 for the same 45 recordings and copy PNGs into per-type collections."""
from pathlib import Path
import json
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_analysis_gui import config as C,storage
from data_analysis_gui.core.catalog import describe,day_config_path
from data_analysis_gui.core.filters import TrialFilter
from data_analysis_gui.core.session import Session
from data_analysis_gui.runner import build_context,run_section
from data_analysis_gui.sections import versions


def main():
    root=Path('E:/AitBCI/data_analysis_gui_output/keaiduo')
    manifest=json.loads((root/'s14_hot_r_0_25_redraw.json').read_text(encoding='utf-8'))
    section=versions('s02')[-1]
    prefs=C.load_preferences()
    params=section.resolve_params(prefs.get('single',{}).get('params',{}).get('s02_v1',{}))
    criteria=TrialFilter.from_dict(prefs.get('filter',{}))
    font=prefs.get('font_percent',120)
    # Use the default new Tianshu table and the shared lower-origin plotter.
    layout=str(C.DEFAULT_LAYOUT_XLSX)
    if not Path(layout).is_file():raise ValueError(f'布局不存在: {layout}')
    files={p.name:p for p in C.DEFAULT_DATA_ROOT.rglob('*.xdf')}
    out=root/'single/S02'
    out.mkdir(exist_ok=True)
    completed,errors=[],[]
    for index,old in enumerate(manifest['completed'],1):
        result=None
        try:
            meta=json.loads((Path(old['output'])/'result.json').read_text(encoding='utf-8'))
            source=meta['inputs']['recordings'][0]
            path=files[source['name']]
            if storage.recording_sha(path)!=source['sha256']:raise ValueError('XDF 内容已变化')
            rec=describe(path,C.DEFAULT_DATA_ROOT)
            config=day_config_path(path)
            inputs=storage.signature_inputs(section,[rec],criteria,params,layout,channel_config_path=config,
                                            font_percent=font)
            signature=storage.fingerprint(inputs)
            folder=storage.result_folder(root,section,[rec],signature)
            cached=storage.load_cached(folder,signature)
            started=time.time()
            if cached is None:
                session=Session.load(rec)
                ctx=build_context([session],criteria,params,section,layout_path=layout,channel_config_path=config)
                result=run_section(section,ctx,font_percent=font)
                cached=storage.save_result(result,folder,signature,inputs,section,[rec],time.time()-started)
                del ctx,session
            name=rec.date+'_'+path.stem
            copies=[]
            for fig in cached['figures']:
                png=folder/fig['files']['png']
                target=out/png.stem
                target.mkdir(exist_ok=True)
                dest=target/(name+'.png')
                shutil.copy2(png,dest)
                if storage.file_digest(png)!=storage.file_digest(dest):raise ValueError('PNG 复制校验失败')
                copies.append(str(dest))
            completed.append(dict(recording=str(path),label=rec.label,cache=str(folder),pngs=copies,
                                  n_frames=cached['summary']['n_frames'],
                                  transmitted=cached['summary']['transmitted_electrodes']))
            print(f'[{index}/45] {len(copies)} figures: {name}',flush=True)
        except Exception as exc:
            errors.append(dict(source=old['output'],error=str(exc)))
            print(f'ERROR [{index}/45] {exc}',flush=True)
        finally:
            if result is not None:
                for _,fig in result.figures:fig.clear()
    data=dict(completed=completed,errors=errors,params=params,layout=layout,orientation='position 1 bottom, 128 top')
    (out/'manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'DONE: {len(completed)} recordings; {sum(len(r["pngs"]) for r in completed)} PNGs; {len(errors)} errors. {out}',flush=True)
    return int(bool(errors))


if __name__=='__main__':
    raise SystemExit(main())
