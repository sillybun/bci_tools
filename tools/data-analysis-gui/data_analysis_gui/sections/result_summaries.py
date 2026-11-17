"""Measured one-sentence readouts; missing metrics stay unknown, never become zero."""
from __future__ import annotations

import math


def _finite(value):
    try:
        return float(value) if math.isfinite(float(value)) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _fmt(value, decimals=0, scale=1):
    value = _finite(value)
    return '未计算' if value is None else f'{value * scale:.{decimals}f}'


def _range(values, decimals=2, scale=1):
    if values is None:
        return '未计算'
    numbers = [v for v in (_finite(v) for v in values) if v is not None]
    return '未计算' if not numbers else f'{min(numbers)*scale:.{decimals}f}–{max(numbers)*scale:.{decimals}f}'


def _mean(values):
    numbers = [v for v in (_finite(v) for v in (values if values is not None else [])) if v is not None]
    return sum(numbers) / len(numbers) if numbers else None


def result_sentence(section_id, summary):
    """Summarize the actual metrics, without universal good/bad thresholds or causal claims."""
    s = summary
    n = lambda key: _fmt(s.get(key))
    f = lambda key, digits=2: _fmt(s.get(key), digits)
    p = lambda key, digits=1: _fmt(s.get(key), digits, 100) + '%' if _finite(s.get(key)) is not None else '未计算'
    recordings = _fmt(len(s['recordings'])) if isinstance(s.get('recordings'), (list, tuple)) else '未计算'
    if section_id == 's01':
        qualification = ('满足' if s['session_qualified'] else '未满足') + '当前 session 样本标准' if 'session_qualified' in s else 'session 资格未计算'
        text = f'重建 {n("total_trial_rows")} 个 trial，方向/速度队列保留 {n("included_trials")}/{n("velocity_trials")} 个，{qualification}'
    elif section_id == 's02':
        text = (f'当前统计范围内 {n("transmitted_electrodes")} 个传输电极中，{n("channels_max_code_0")} 个无事件，'
                f'平均饱和度 {p("saturation_mean")}，{n("channels_bursty")} 个达到配置的 bursty 标记规则')
    elif section_id == 's03':
        loss = (f'FPGA 推断缺失 {n("fpga_frames_lost")} 帧（{_fmt(s.get("fpga_drop_rate"), 3, 100)}%）'
                if s.get('fpga_clock') else '无 FPGA 时钟，缺帧和相对延迟未知')
        text = f'收到 {n("n_frames")} 帧，存在 {n("n_receive_gaps")} 个超过配置阈值的接收间隔，{loss}'
        if s.get('fpga_nonpositive_steps'):
            text += '，估计不覆盖时钟重置/回绕区间'
    elif section_id == 's04':
        gain = 'PC' + n('gain_correlated_pc') if _finite(s.get('gain_correlated_pc')) is not None else '未标注'
        umap = 'UMAP 为探索性嵌入' if s.get('umap') else 'UMAP 未计算'
        text = (f'{n("trials")} 个 trial 的 PCA 达到 80% 方差需 {n("pcs_for_80pct")} 个成分，'
                f'最强总率相关成分为 {gain}，held-out TDR 圆形相关 {f("tdr_cv_circular_r")}，{umap}')
    elif section_id == 's05':
        text = (f'{n("trials")} 个 trial 中，{n("active_channels")} 个活跃通道有 '
                f'{n("significant_channels")} 个达到当前方向 ANOVA 的 BH-FDR 标准（α={f("alpha")}），显著性不等于解码贡献')
    elif section_id == 's06':
        text = (f'{n("trials_with_baseline")} 个 trial 具有有效基线，{n("active_channels")} 个活跃通道中 '
                f'{n("responsive_channels")} 个检出时间响应（升 {n("responsive_up")}、降 {n("responsive_down")}），'
                f'检出组中位潜伏期 {f("median_latency_s", 3)} s')
    elif section_id == 's07':
        if s.get('tuning_overlap_status') == 'computed':
            text = (f'{n("tuning_transmitted_channels")} 个传输通道中 Active={n("tuning_active_channels")}，'
                    f'Direction/Time/Both 分别 {n("tuning_direction_channels")}/{n("tuning_time_channels")}/'
                    f'{n("tuning_both_channels")} 个（Direction/Time 均含 Both），基线变化另用 {n("paired_trials")} 个配对 trial')
        else:
            text = (f'{n("paired_trials")} 个基线/任务窗口配对 trial 中，至少一个方向显著升高/降低的通道分别为 '
                    f'{n("channels_up_any_direction")}/{n("channels_down_any_direction")} 个，两组可能重叠')
    elif section_id == 's08':
        good = (s.get('transmitted_by_tier') or {}).get('good')
        good_score = (s.get('decoding') or {}).get('good', {}).get('direction_bacc')
        text = (f'实际传输 good 电极 {_fmt(good)} 个，配置中 {n("ini_good_not_transmitted")} 个 good 未传输，'
                f'good 子集方向 BA 为 {(_fmt(good_score, 1, 100) + "%") if _finite(good_score) is not None else "未计算"}，配置标签不等同于实测信号质量')
    elif section_id == 's09':
        if 'grid_n_targets' in s:
            text = (f'Grid 的 {n("grid_n_targets")} 个目标上，held-out 位置 R²={f("grid_target_r2_xy")}，'
                    f'身份 BA={p("grid_identity_balanced_accuracy")}（机会 {p("grid_identity_chance")}），两者是不同任务')
        else:
            permutation = '置换 p=' + f('p_value', 4) if _finite(s.get('p_value')) is not None else '置换显著性未计算'
            text = (f'{n("n_trials")} 个 trial 的 held-out 方向 BA={p("balanced_accuracy")}，'
                    f'±45° 内比例 {p("within_45deg")}，{permutation}')
    elif section_id == 's10':
        models = '；'.join(f'{model} 平均 R²={_fmt(s.get(model + "_r2_mean"), 2)}' for model in s.get('decoders', []))
        text = f'在 {n("n_common_tested_bins")} 个共同 held-out bin 上比较速度解码，{models or "模型指标未计算"}'
    elif section_id == 's11':
        text = (f'{n("n_bins_with_online")} 个在线匹配 bin 的输出/cursor pooled R²={f("online_vs_gt_r2_xy")}，'
                f'帧级神经滞后复现 R²={f("frame_repro_vs_online_r2_xy")}，前者为闭环一致性、后者不等于原在线模型复现')
        replay = s.get('saved_model_replay_metrics') or {}
        if replay:
            text += (f'；另以 {n("saved_model_scored_frames")} 个匹配帧按保存权重切换重放，'
                     f'R²={_fmt(replay.get("r2_xy"), 3)}、RMSE={_fmt(replay.get("rmse"), 2)}，runtime 语义尚未验证')
    elif section_id == 's12':
        if 'grid_target_r2_xy' in s:
            direction = 'Grid 目标位置 R²=' + f('grid_target_r2_xy')
        else:
            direction = '方向 BA=' + p('direction_balanced_accuracy')
        text = (f'方向/速度队列 {n("direction_trials")}/{n("velocity_trials")} 个 trial，'
                f'{n("receive_gaps")} 个超阈值接收间隔，{direction}，各项结论需结合样本资格与对应检验')
        if 'overview_transmitted' in s:
            text = (f'{n("success_trials")}/{n("complete_trials")} 个完整 trial 成功，'
                    f'{n("overview_often_saturated")}/{n("overview_transmitted")} 个通道处于重度截断分区，'
                    f'{n("receive_gaps")} 个长接收间隔，PC1–2 解释 {p("overview_pca_pc12_variance")} 变异，'
                    f'{n("overview_both_tuned")} 个通道同时通过方向与时间检验；补充页{direction}')
    elif section_id == 's13':
        text = (f'已知结局的尝试中成功 {n("successes")}/{n("attempts")}，BPS {f("bps")}，'
                f'成功 trial 到靶时间中位数 {f("rt_median_s")} s')
    elif section_id == 's14':
        text = (f'当前 run 全量拟合使用 {n("n_bins")} 个 bin、{n("n_fitted")} 个非常数通道，'
                f'平均 norm 权重最高的位置组为 Elec# {n("peak_elec_start")}–{n("peak_elec_end")}；'
                '权重表示模型敏感度，不能直接解释为独立解码贡献')
    elif section_id == 's15':
        text = (f'任务回放包含 {n("samples")} 帧 cursor、{n("trials")} 个 trial 起点，'
                f'时长 {f("duration_s")} s，{n("gaps")} 处采样缺口；可在任务回放页以 1× 实时播放')
    elif section_id == 'x09':
        text = (f'对所选 {n("n_recordings")} 份录制逐通道等权计算 median 和 average，'
                f'{n("n_channels_observed")} 个通道至少一次传输，{n("n_channels_all_runs")} 个通道全部录制均传输；缺失不补零')
    elif section_id == 'x08':
        text = f'分别汇总在线 CO8 {n("co8_sessions")} 份和在线 Grid {n("grid_sessions")} 份'
    elif section_id == 'x00':
        text = (f'共选 {n("session_count")} 份记录，Offline CO8 {n("CO8")} 份，Offline Grid {n("Grid")} 份，'
                f'Online CO8 {n("CO8 online")} 份，Online Grid {n("Grid online")} 份')
    elif section_id == 'x01':
        text = (f'{recordings} 份记录中，{n("stable_electrodes")} 个至少两次传输电极状态稳定、'
                f'{n("unstable_electrodes")} 个状态变化，共享电极 log-rate 平均相关 {f("mean_pairwise_log_rate_r")}，状态相同不等于数值相同')
        if s.get('active_fraction_per_recording'):
            text += f'，各记录的活跃通道占比为 {_range(s["active_fraction_per_recording"], 1, 100)}%'
    elif section_id == 'x02':
        text = (f'{n("session_count")} 份记录中，最高 FPGA 缺帧率为 {p("max_fpga_frame_lost_rate", 4)}，'
                f'最高长接收间隔时间占比为 {p("max_long_receive_gap_time_fraction", 4)}')
    elif section_id == 'x03':
        text = (f'各记录日内 CV 圆形相关为 {_range(s.get("cv_circular_r"))}，'
                f'非对角迁移平均圆形相关 {f("mean_transfer_circular_r")}，须区分日内可读性和跨记录平面一致性')
    elif section_id == 'x04':
        text = (f'日内方向 BA 为 {_range(s.get("within_day_accuracy"), 1, 100)}%，'
                f'固定统计/测试日标准化的跨记录平均 BA 为 {p("mean_cross_day_accuracy_train")}/{p("mean_cross_day_accuracy_test_own")}，后者使用无标签测试分布')
    elif section_id == 'x05':
        models = '；'.join(f'{model} 平均迁移 R²={_fmt(s.get("mean_cross_day_"+model+"_r2"), 2)}' for model in s.get('decoders', []))
        text = f'{recordings} 份记录的速度跨记录评估中，{models or "模型指标未计算"}，需同时对照日内结果和运动尺度'
    elif section_id == 'x06':
        models = '；'.join(f'{model} 固定迁移平均 R²={_fmt(_mean(s.get("train_"+model+"_r2_first_ref")), 2)}' for model in s.get('decoders', []))
        text = f'汇总首个参考记录对各测试记录的结果，{models or "模型指标未计算"}，再校准和匹配日内参照需在对应面板分别解读'
    elif section_id == 'x07':
        text = (f'所选 {n("selected_days")} 天中，方向调谐有 {n("direction_stable")}/{n("direction_eligible")} 个可比较电极'
                f'稳定至少两天，PSTH 时间响应有 {n("psth_stable")}/{n("psth_eligible")} 个')
    else:
        return '结果已载入，具体指标及适用条件见各图和说明。'
    online = s.get('online') or 'online' in str(s.get('recording', '')).lower() or 'online' in str(s.get('session_type', '')).lower()
    if online and section_id in ('s10', 's12'):
        text += '，在线 cursor 属于闭环输出'
    return text.rstrip('。') + '。'
