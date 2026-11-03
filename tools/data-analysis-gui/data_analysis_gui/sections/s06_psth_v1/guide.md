# S06 · PSTH

**一句话总结：**以基线配对检验和方向分组 PSTH 描述通道事件响应、升降模式及潜伏期，避免把排序热图本身当作统计证据。

本节回答：**通道在对齐事件之后什么时候、朝哪个方向（升/降）改变发放？群体的时间响应形状如何，是否随目标方向不同？**
数据：Topbar 筛选后的方向队列，按 T_TRIAL_START（或 T_TRIAL_START + 首次 T_TARGET_ACQUIRED 时间）对齐。
「活跃通道」= 全部 trial、全部 bin 平均发放率 > 0 的通道（与 S04/S05 的定义略有不同，所以数目可能差几个）。


阅读约定：以下按 subplot 的位置/标题逐项说明；重复的方向、电极、decoder 或记录面板逐个适用对应条目，双轴/色条与主面板合并解释。质量标准区分配置阈值、统计证据与描述性限制，历史数值不作为通用合格线。

## 图 1：Population PSTH heatmap

### 计算公式
**符号**
- `t`：trial 编号；`n`：参与的 trial 数。
- `c`：通道编号；活跃通道 = 全部 trial、全部 bin 平均发放率 > 0 的通道（m 个；1026 为 266，Grid 为 427）。
- `t₀(t)`：对齐时刻 = T_TRIAL_START（参数 align = target_acquired 时为 T_TRIAL_START + 首次 T_TARGET_ACQUIRED 时间），s。
- `b`：时间 bin 编号；bin 宽 w（参数 bin，默认 0.05 s），覆盖 [tmin, tmax)（默认 −0.5 到 2.0 s）；`τ_b`：bin 中心时间（相对 t₀）。
- `rate(t,b,c)`：第 t 个 trial、bin b、通道 c 的发放率（Hz）= bin 内计数之和 ÷ (bin 内收到的帧数 × 帧时长 0.020 s)；bin 内没有帧则为 NaN（缺失，不参与平均）。
- `PSTH(b,c)`：trial 平均发放率（peri-stimulus time histogram）= 所有非 NaN trial 的 rate(t,b,c) 的平均。
- `σ`：显示用高斯平滑的标准差 = smooth_sd ÷ w 个 bin（smooth_sd 默认 0.05 s；0 = 不平滑）；`G`：高斯核。
- `zPSTH(b,c)`：逐通道 z-score 后的 PSTH = (PSTH(b,c) − 该通道各 bin 的平均) ÷ 该通道各 bin 的标准差，无单位。
- 左侧条带：通道的时间响应检验结果（见图 4）。

**步骤**
1. 分 bin：对每个 trial、bin、通道算 rate(t,b,c)。
   意思是：把每个 trial 切成 50 ms 的小段，算每段的发放率。
2. trial 平均：PSTH(b,c) = 所有方向合并后，各 trial rate(t,b,c) 的平均（忽略 NaN）。
3. 平滑（只用于显示）：smoothed = [G * (x·w)] ÷ [G * w]，x = PSTH 中的值，w = 1（有效）或 0（NaN），* 表示沿时间卷积。
   意思是：用高斯窗把相邻 bin 加权平均，缺失 bin 不会被当成 0；检验全部用未平滑的原始 bin。
4. 逐通道 z-score（参数 zscore，默认开）：zPSTH(b,c) = (smoothed(b,c) − mean_b) ÷ SD_b。
   意思是：每个通道只和自己比，看什么时候相对最高。
5. 排序：peak_latency = 按事件后各 bin 中 |zPSTH| 最大的那个 bin 的先后排序；anova_F = 按分析窗口 ANOVA 的 F 从大到小（与 S05 相同）；electrode = 按电极 ID。
   意思是：把达峰时间相近的通道排在一起，形成对角带。
6. 色标：z-score 时以 0 为中心，上下限 = |zPSTH| 的 99 百分位；不 z-score 时为 Hz，上限 = 99 百分位。

### 结果解读
- **左侧响应类别条带**：逐通道读取显著升高、降低或未显著类别，与右侧相同行对应。
- **右侧群体热图**：每行一个通道，按所选规则排序，读取事件前后相对活动与峰值时序。

### 质量评价标准
- **左侧响应类别条带**：类别来自当前基线/检验规则，灰色不证明无响应，正负类别可能由最显著时间 bin 决定。
- **右侧群体热图**：逐通道 z-score 不能比较绝对幅度；按峰时排序即使噪声也能形成对角带，必须结合左侧类别及检验，整列同步突变需核查采集。

## 图 2：Per-direction PSTH of responsive channels

### 计算公式
**符号**
- `t`：trial 编号；`n`：参与的 trial 数。
- `c`：通道编号；活跃通道 = 全部 trial、全部 bin 平均发放率 > 0 的通道（m 个；1026 为 266，Grid 为 427）。
- `t₀(t)`：对齐时刻 = T_TRIAL_START（参数 align = target_acquired 时为 T_TRIAL_START + 首次 T_TARGET_ACQUIRED 时间），s。
- `b`：时间 bin 编号；bin 宽 w（参数 bin，默认 0.05 s），覆盖 [tmin, tmax)（默认 −0.5 到 2.0 s）；`τ_b`：bin 中心时间（相对 t₀）。
- `rate(t,b,c)`：第 t 个 trial、bin b、通道 c 的发放率（Hz）= bin 内计数之和 ÷ (bin 内收到的帧数 × 帧时长 0.020 s)；bin 内没有帧则为 NaN（缺失，不参与平均）。
- `PSTH(b,c)`：trial 平均发放率（peri-stimulus time histogram）= 所有非 NaN trial 的 rate(t,b,c) 的平均。
- `σ`：显示用高斯平滑的标准差 = smooth_sd ÷ w 个 bin（smooth_sd 默认 0.05 s；0 = 不平滑）；`G`：高斯核。
- `base(t,c)`：基线发放率 = trial t 在 t₀ + [baseline_start, baseline_end)（默认 [−0.3, 0) s）内的平均发放率（Hz）；该窗口没有帧的 trial 不参与检验。
- `diff(t,b,c)`：配对差 = rate(t,b,c) − base(t,c)（Hz）。
- `B`：事件后（τ_b > 0）的 bin 数（默认 40）；`p_b`：bin b 的配对 t 检验 p 值。
- `p_chan`：通道级 p；`q`：BH-FDR 校正后的 q 值（见 S05）；`α`：显著性水平（默认 0.05）。
- Bonferroni 校正：把 p 乘以检验次数（这里为 B），上限 1。
- `peakΔ(c)`：峰值变化 = 事件后各 bin 中 |mean_t diff(t,b,c)| 最大的那个值（保留正负号，Hz）。
- `N`：画出的通道数（参数 top_n，默认 8）。

**步骤**
1. 选通道：在时间响应检验显著的通道中（q < α，见图 4），按 p_chan 从小到大、再按 |peakΔ| 从大到小，取前 N 个；没有显著通道时按未校正的 p_chan 排序展示，并在注释中说明。
   意思是：展示时间响应最可靠、幅度最大的通道。
2. 每个方向 d：该方向 trial 的 rate(t,b,c) 平均，再按图 1 第 3 步平滑。
   意思是：8 条线 = 8 个方向的 PSTH。
3. 虚线 = 潜伏期（见图 4 第 6 步）；灰带 = 基线窗口。

### 结果解读
- **每个响应电极子图**：比较该电极八方向均值曲线的基线、响应起点、持续性和方向差异。

### 质量评价标准
- **每个响应电极子图**：按当前响应强度选择有偏，不能当作独立验证；基线提前分离或所有方向共同断点需核查，升降并存可为真实时序。

## 图 3：Population mean PSTH by direction

### 计算公式
**符号**
- `t`：trial 编号；`n`：参与的 trial 数。
- `c`：通道编号；活跃通道 = 全部 trial、全部 bin 平均发放率 > 0 的通道（m 个；1026 为 266，Grid 为 427）。
- `t₀(t)`：对齐时刻 = T_TRIAL_START（参数 align = target_acquired 时为 T_TRIAL_START + 首次 T_TARGET_ACQUIRED 时间），s。
- `b`：时间 bin 编号；bin 宽 w（参数 bin，默认 0.05 s），覆盖 [tmin, tmax)（默认 −0.5 到 2.0 s）；`τ_b`：bin 中心时间（相对 t₀）。
- `rate(t,b,c)`：第 t 个 trial、bin b、通道 c 的发放率（Hz）= bin 内计数之和 ÷ (bin 内收到的帧数 × 帧时长 0.020 s)；bin 内没有帧则为 NaN（缺失，不参与平均）。
- `PSTH(b,c)`：trial 平均发放率（peri-stimulus time histogram）= 所有非 NaN trial 的 rate(t,b,c) 的平均。
- `σ`：显示用高斯平滑的标准差 = smooth_sd ÷ w 个 bin（smooth_sd 默认 0.05 s；0 = 不平滑）；`G`：高斯核。
- `G_set`：通道组，分别为 (a) 所有活跃通道，(b) 显著升高的通道，(c) 显著降低的通道（升/降的定义见图 4）。
- `pop(t,b)`：第 t 个 trial 在 bin b 上，组内通道 rate 的平均（Hz）。
- `SEM`：标准误 = SD ÷ √(有效 trial 数)。

**步骤**
1. 组内平均：pop(t,b) = mean_{c∈G_set} rate(t,b,c)。
   意思是：先把一组通道合成一条「群体活动」曲线。
2. 方向平均：对方向 d 的 trial 求 mean_t pop(t,b) 和 SEM。
   意思是：8 条线为 8 个方向的群体时间响应，阴影为 ± SEM。
3. 均值和 SEM 都按图 1 第 3 步平滑后显示。

### 结果解读
- **左侧全部活跃通道**：读取方向平均的整体时间变化。
- **中间升高响应组**：查看选为升高组的均值响应及不同方向时序。
- **右侧降低响应组**：查看选为降低组的基线后抑制和恢复。

### 质量评价标准
- **左侧全部活跃通道**：均值平坦不代表通道模式无信息，分离也可能是增益；需看通道组成和 SEM/覆盖。
- **中间升高响应组**：同数据选择后平均响应偏乐观，不与全部组作为独立样本检验；小组数或少量高率通道会影响均值。
- **右侧降低响应组**：降低不代表质量差，需确认基线定义和 trial/ITI 边界，不能把长期低值直接归因为任务抑制。

## 图 4：Temporal response test

### 计算公式
**符号**
- `t`：trial 编号；`n`：参与的 trial 数。
- `c`：通道编号；活跃通道 = 全部 trial、全部 bin 平均发放率 > 0 的通道（m 个；1026 为 266，Grid 为 427）。
- `t₀(t)`：对齐时刻 = T_TRIAL_START（参数 align = target_acquired 时为 T_TRIAL_START + 首次 T_TARGET_ACQUIRED 时间），s。
- `b`：时间 bin 编号；bin 宽 w（参数 bin，默认 0.05 s），覆盖 [tmin, tmax)（默认 −0.5 到 2.0 s）；`τ_b`：bin 中心时间（相对 t₀）。
- `rate(t,b,c)`：第 t 个 trial、bin b、通道 c 的发放率（Hz）= bin 内计数之和 ÷ (bin 内收到的帧数 × 帧时长 0.020 s)；bin 内没有帧则为 NaN（缺失，不参与平均）。
- `PSTH(b,c)`：trial 平均发放率（peri-stimulus time histogram）= 所有非 NaN trial 的 rate(t,b,c) 的平均。
- `σ`：显示用高斯平滑的标准差 = smooth_sd ÷ w 个 bin（smooth_sd 默认 0.05 s；0 = 不平滑）；`G`：高斯核。
- `base(t,c)`：基线发放率 = trial t 在 t₀ + [baseline_start, baseline_end)（默认 [−0.3, 0) s）内的平均发放率（Hz）；该窗口没有帧的 trial 不参与检验。
- `diff(t,b,c)`：配对差 = rate(t,b,c) − base(t,c)（Hz）。
- `B`：事件后（τ_b > 0）的 bin 数（默认 40）；`p_b`：bin b 的配对 t 检验 p 值。
- `p_chan`：通道级 p；`q`：BH-FDR 校正后的 q 值（见 S05）；`α`：显著性水平（默认 0.05）。
- Bonferroni 校正：把 p 乘以检验次数（这里为 B），上限 1。

**步骤**
1. 配对差：diff(t,b,c) = rate(t,b,c) − base(t,c)。
   意思是：同一个 trial 里，这个时刻比自己的基线高多少。
2. 配对 t 检验（每个事件后 bin、每个通道）：t = mean_t(diff) ÷ [SD_t(diff) ÷ √n_valid]，自由度 n_valid − 1，双侧 p_b；n_valid = 该 bin 有效 trial 数。
   意思是：差值的平均是否显著不为 0。
3. 通道级 p：p_chan = min(1, B × min_b p_b)。
   意思是：在 B 个 bin 中挑最显著的一个，再做 Bonferroni 校正，避免「多看几个 bin 总有一个显著」。
4. BH-FDR：在活跃通道内对 p_chan 做 Benjamini–Hochberg，得到 q；显著（有时间响应）= q < α。
5. 方向（升/降）：取 p_b 最小的那个 bin，mean(diff) > 0 为升高，< 0 为降低。
6. 潜伏期：从事件后第一个 bin 起，找第一个满足 p_b < α ÷ B 且符号与第 5 步相同的 bin；只要存在连续两个这样的 bin，就要求连续两个；潜伏期 = 该 bin 的中心 τ_b。
   意思是：响应「可靠地开始」的时刻。
7. (c) 面板：每个事件后 bin 中，p_b < α ÷ B 且 mean(diff) > 0（红）或 < 0（蓝）的活跃通道百分比。
   意思是：每个时刻有多少通道显著偏离基线。

### 结果解读
- **A 响应类别数量**：比较升高、降低与未显著通道数。
- **B 通道 q 分布**：读取时间 bin 校正后再跨通道校正的显著性分布。
- **C 逐时间 bin 比例**：观察相对基线显著升高/降低的通道比例随时间变化。
- **D 潜伏期分布**：比较升/降响应的起始 bin 和中位数。

### 质量评价标准
- **A 响应类别数量**：使用当前多重检验规则和有效通道分母，没有理想响应比例；不显著不等于没有效应。
- **B 通道 q 分布**：q 阈值必须与实际检验族一致，极小 q 不能代替幅度和重复性。
- **C 逐时间 bin 比例**：这是逐 bin 的规则，不等同于通道级 FDR；早期响应需核查对齐，后期响应需核查结束/ITI。
- **D 潜伏期分布**：分辨率由 bin 宽决定、仅含检出的响应通道；不可将更短潜伏期或更整齐分布当作通用质量标准。

## 图 5：PSTH curves - all channels

### 计算公式
**符号**
- r(d,t,c)：方向 d 的有效对齐 trial 在时间 bin t、通道 c 的平均 Hz；q(c)：S06 混合全部方向的时间响应 q。

**步骤**
1. 全部传输通道按电极 ID 排序，每页 8×8；每子图画八方向 trial 平均 PSTH，方向颜色由页顶共享图例解释。
2. NaN-aware Gaussian 平滑仅用于显示，SD 使用本节 smooth_sd；导出原始与显示 bin 均值。阴影为本节基线，事件对齐时刻为点线。
3. 红色标题 * 表示本节原始 bin 配对基线检验经 bin Bonferroni、active 家族 BH 后 q<alpha；不表示方向 ANOVA 显著或方向间时间曲线有差异。
4. 每子图独立 y 范围，并逐个显示 y 刻度；空位隐藏。Prev/Next 翻页，支持整组 PDF 和单页导出。

### 结果解读
- **每个电极八方向时间子图**：按颜色读取方向、按本图 Hz 刻度读取时间响应；标题 q 是混合方向的时间响应检验。

### 质量评价标准
- **每个电极八方向时间子图**：跨面板幅度比较必须读取各自刻度；时间响应可在所有方向共同出现，曲线分离不等于方向×时间交互显著；注意 trial 结束或 ITI 混入。

## 图 6：PSTH curves - significant channels

### 计算公式
**符号**
- S：S06 active 通道中时间响应 q<alpha 的通道；r(d,t,c) 与全部通道分页图使用相同 bin、trial、平滑设置。

**步骤**
1. 使用本节现有 sig 标记，按 q、通道原始 p、电极 ID 排序；不使用 S05 或 S07 的显著通道替代。
2. 每页 8×8，各图独立 y 范围和刻度；八方向时间曲线、颜色、基线和对齐标记与全部通道页一致。
3. 无显著通道时明确提示；支持 Prev/Next 翻页、单页导出和整组 PDF。

### 结果解读
- **每个时间显著电极子图**：检查时间响应通道的八方向过程；一个通道可以对所有方向有共同反应而没有方向偏好。

### 质量评价标准
- **每个时间显著电极子图**：同数据筛选不构成独立验证；不要将 Time-tuned 当作 Direction-tuned 或 Both，或按不同 y 范围下的视觉高度比较强弱。

## 注意事项
- 这是「时间调制」检验，不是方向调谐检验；方向显著性见 S05，基线相对的方向分解见 S07。
- 平滑只影响显示；检验、潜伏期与表格均用原始 bin。
- tmax 超过 trial 时长后，后段 bin 会混入目标获得、trial 结束和 ITI 的活动；默认 2 s 对短 trial 可能已超出 trial 结束。
- 选择 target_acquired 对齐时，基线窗口位于运动过程中，含义与 trial 前基线不同。
- 平行的 t 检验假设配对差近似正态；低发放通道（多数 bin 为 0）的 p 值偏不稳定。
