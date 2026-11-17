"""Font-only redraws for saved LFP arrays and bespoke report presentation pages."""
from pathlib import Path
import inspect
import json
import runpy
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from data_analysis_gui.core.style import finish_figure
from update_report_font_sizes import ROOT,AUDIT,backup,record,typed,read,save

REPORT=ROOT/'2026-11-04_2026-11-11_25runs_v1'


def keep_save(fig,path,source):
    before=backup(path);finish_figure(fig)
    for ext in ('png','pdf','svg'):
        target=path.with_suffix('.'+ext)
        if ext!='png':backup(target)
        fig.savefig(target,dpi=300 if ext=='png' else None,facecolor='white',bbox_inches='tight',pad_inches=.15)
    record(path,'saved statistics redraw',source,before)


def boolean(rows):
    for r in rows:
        for k,v in r.items():
            if v=='True':r[k]=True
            elif v=='False':r[k]=False
    return rows


def lfp():
    import lfp_direction_spectrogram_1111 as maps
    base=maps.OUT;stage=AUDIT/'lfp_spectrogram';stage.mkdir(exist_ok=True)
    target=REPORT/'additional_figures/LFP_Direction_Spectrogram'
    cache=np.load(base/'trial_spectrograms.npz')
    data=dict(relative=cache['relative_db'],absolute=cache['absolute_db'],post=cache['post_db'],pre=cache['pre_db'],
        labels=cache['directions'],channels=cache['channel_ids'],freq=cache['frequencies_hz'],time=cache['time_s'])
    maps._saved_means=np.load(base/'direction_mean_spectrograms.npz')
    code=inspect.getsource(maps.draw_maps).replace("relative=np.array([data['relative'][labels==d].mean(0) for d in range(8)])","relative=_saved_means['relative_db']").replace("absolute=np.array([data['absolute'][labels==d].mean(0) for d in range(8)])","absolute=_saved_means['absolute_db']")
    exec(code,maps.__dict__)
    maps.OUT=stage
    maps.finish=lambda fig,name,pdf=True:keep_save(fig,target/(name+'.png'),base/'direction_mean_spectrograms.npz')
    maps.draw_maps(data)
    direction=boolean(typed(read(base/'direction_selectivity_tests.csv')))
    response=boolean(typed(read(base/'event_related_power_tests.csv')))
    effect=np.zeros((8,6,6));adjusted=np.ones((8,6,6))
    for r in response:
        d=int(r['direction_deg']/45);c=list(data['channels']).index(r['channel_id'])
        b=[f'{lo}-{hi}' for lo,hi in maps.BANDS].index(r['band_hz'])
        effect[d,c,b]=r['mean_change_db'];adjusted[d,c,b]=r['maxT_FWER_p_288']
    maps.draw_statistics(data,direction,response,effect,adjusted)
    import lfp_one_vs_rest_1111 as binary
    base=binary.OUT;binary.OUT=AUDIT/'lfp_binary';binary.OUT.mkdir(exist_ok=True)
    cached=np.load(base/'heldout_binary_scores.npz');rows=boolean(typed(read(base/'one_vs_rest_results.csv')))
    forward=boolean(typed(read(base/'forward_test_results.csv')))
    close=plt.close;counter=[];names=['LFP_One_vs_Rest_Direction_Decoding','LFP_One_vs_Rest_Forward_Test']
    def capture(fig=None):
        if fig is not None and hasattr(fig,'axes') and len(counter)<2:
            keep_save(fig,REPORT/'additional_figures/LFP_One_vs_Rest'/(names[len(counter)]+'.png'),base/'heldout_binary_scores.npz');counter.append(1)
        close(fig)
    plt.close=capture
    try:binary.plots(rows,cached['labels'],cached['scores'],cached['predictions'],forward,None)
    finally:plt.close=close
    print('LFP saved-array redraws complete',flush=True)


def chinese_counts():
    path=Path(__file__).with_name('plot_dataset_counts_chinese.py')
    stage=AUDIT/'dataset_counts';stage.mkdir(exist_ok=True)
    code=path.read_text(encoding='utf-8')
    start=code.index('OUT = Path(');end=code.index('\n',start)
    code=code[:start]+f'OUT = Path({str(stage)!r})'+code[end:]
    namespace={'__name__':'_font_redraw'};exec(compile(code,str(path),'exec'),namespace)
    fig=namespace['fig'];save(fig,REPORT/'additional_figures/Dataset_Counts_Chinese.png',REPORT/'additional_figures/Dataset_Counts_Chinese.csv')
    print('Chinese counts redrawn',flush=True)


def distributions():
    import plot_activity_success_distribution as module
    blocks=typed(read(REPORT/'additional_figures/Electrode_Activity_and_Direction_Tuning_Average_Full_Distribution.csv'))
    module._saved_blocks=blocks
    code=inspect.getsource(module.main);a=code.index('    all_blocks = []');b=code.index('    fig = ChannelStability.activity')
    code=code[:a]+'    all_blocks = _saved_blocks\n'+code[b:]
    exec(code,module.__dict__)
    module.OUT=AUDIT/'activity_distributions';module.OUT.mkdir(exist_ok=True)
    original=module.ChannelStability.activity;captured=[]
    def capture(rows):
        fig=original(rows);captured.append(fig);return fig
    module.ChannelStability.activity=capture
    args=sys.argv
    try:
        for gaussian in (False,True):
            sys.argv=['font_redraw']+(['--gaussian-std'] if gaussian else [])
            module.main()
            name='Channel_Activity_Success_Rate_Gaussian_STD' if gaussian else 'Electrode_Activity_and_Direction_Tuning_Average_Full_Distribution'
            save(captured[-1],REPORT/'additional_figures'/(name+'.png'),REPORT/'additional_figures/Electrode_Activity_and_Direction_Tuning_Average_Full_Distribution.csv')
    finally:sys.argv=args;module.ChannelStability.activity=original
    print('Success distributions redrawn',flush=True)


if __name__=='__main__':
    for function in (lfp,chinese_counts,distributions):
        try:function()
        except Exception as exc:print('ERROR',function.__name__,repr(exc),flush=True)
