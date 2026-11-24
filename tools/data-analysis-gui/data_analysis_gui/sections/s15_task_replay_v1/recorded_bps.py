"""Read trial-labelled BPS values exactly as saved by the task; never recompute."""
import csv
import math


def read_recorded_bps(files, mode):
    path = files.get('rolling_metrics')
    if path is None:
        return dict(source='',metric='achieved_bitrate_bps',window_basis='unavailable',trials=[])
    trials = {}
    with path.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row.get('task_mode')!=mode or row.get('sample_reason')!='trial_result':
                continue
            try:
                block,trial=int(row['block_num']),int(row['trial_num_overall'])
                value=float(row['achieved_bitrate_bps'])
                seconds=float(row['window_sec'])
            except (KeyError,ValueError,TypeError):
                continue
            if not math.isfinite(value) or value<0 or not math.isfinite(seconds) or seconds<=0:
                continue
            key=(block,trial)
            # Conflicting duplicate trial results are not silently selected.
            if key in trials and trials[key]['value']!=value:
                raise ValueError(f'保存的BPS存在冲突：Block {block} / Trial {trial}')
            trials[key]=dict(block=block,trial=trial,value=value,
                             raw_value=row['achieved_bitrate_bps'],window_seconds=seconds,
                             window_start_elapsed_sec=row.get('window_start_elapsed_sec',''),
                             window_end_elapsed_sec=row.get('window_end_elapsed_sec',''),
                             timestamp_ns=row.get('system_timestamp_ns',''))
    return dict(source=str(path),metric='achieved_bitrate_bps',
                window_basis='原任务保存的滚动60秒（非S13累积active-time）',
                display_policy='按当前trial显示该trial保存的结果值，含片段开始前的窗口历史，不从回放起点累计',
                trials=list(trials.values()))


def bps_for_state(payload, state):
    """Retrospective trial annotation: even the first partial trial keeps its saved BPS."""
    rows = payload.get('recorded_bps',{}).get('trials',[])
    lookup = payload.get('_bps_lookup')
    if lookup is None:
        lookup = {(r['block'],r['trial']):r['value'] for r in rows}
        payload['_bps_lookup']=lookup
    return lookup.get((int(state[7]),int(state[8])))
