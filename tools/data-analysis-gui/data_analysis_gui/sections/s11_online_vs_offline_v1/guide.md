# S11 · 在线闭环 decoder 结果复现与离线结果对比

**一句话总结：**按实际权重切换和记录位置反馈重放在线 KF，比较记录输出与离线解码，区分回放一致性、统计替代模型和独立预测验证。

本节回答：**在线闭环 decoder 的输出与光标速度有多一致？离线 decoder 在同一批 bin 上能做到多好、与在线输出有多接近？能否用神经帧线性复现在线输出？**
数据：含 `cursortasks_decoder_velocity` 流的在线记录的速度队列；与 S10 相同的 100 ms trial bin（T_TRIAL_START + 0.2 s 起 10 个）。
离线速度解码固定为 Ridge 与 Vel-KF。在线输出和已保存 KF 权重的回放用于对照。


阅读约定：以下按 subplot 的位置/标题逐项说明；重复的方向、电极、decoder 或记录面板逐个适用对应条目，双轴/色条与主面板合并解释。质量标准区分配置阈值、统计证据与描述性限制，历史数值不作为通用合格线。

## 图 1：Saved online model replay audit

### 计算公式
**符号**
- x=[position_x,position_y,velocity_x,velocity_y]；A/C/Q/R、bias、P0 来自初始/实际 applied 权重；z 是 block 日志原始神经计数。
- s：applied_time 墙钟时刻；o：保存的 cursor 墙钟−LSL 配对差中位数；切换时刻 t=s−o。

**步骤**
1. 在线模型目录为空时，优先使用本次记录的 `xdf` 同级目录 `refit`。旧布局在当天查找唯一的更新 session，多候选要求指定目录。按窗口序号、神经计数、LSL 时间和 decoder 输出逐帧核对日志与 XDF。
2. 初始权重可指定文件或 `initial_weights` 目录。留空时优先使用 `refit` 同级的 `initial_weights/INITIAL_position_velocity_kf.pkl`，旧布局从首个 applied 日志的 old_weights_path 定位。按实际 applied_time 排序切换至 active_weights_path_after_swap。权重、日志和 block NPZ 内容进入缓存签名。
3. 标准 KF：x⁻=Ax+state_offset，P⁻=APAᵀ+Q；用 C/R/bias 对原始 z 更新。仅支持 raw 输入；lag6 是训练标签偏移，不拼接六帧，也不平移输出来提高分数。更新文件已含训练时混合，不再二次 blend。
4. 当前重放提供两种位置政策：默认 recorded 在每帧取时间不晚于神经帧的最新 XDF 光标位置（不插值、不使用未来位置），将位置作为已知量，位置协方差置零，以保存的 C/R 更新速度子状态；位置年龄超过 feedback_max_age（默认 0.25 s）后，当前 trial 剩余帧不评分。decoded 仅在 trial 开始用记录位置初始化，此后由完整 KF 估计位置。两者均以零速度、P0 开始，切换时保持状态/协方差。trial 内神经输入缺口后的预测不评分，直到下一 trial。XDF 开始前用日志中的位置作预热，不借用记录速度作为模型输入。
5. 只对与 XDF 精确匹配且输入连续的帧评分，导出完整逐帧、分权重和切换表。该评分使用全部可匹配的日志 active trial，不按 Topbar 队列筛选，不能与其他图的 trial-bin 指标直接比较。
6. 该文件 model_type=position_velocity，不能冒充 Madduri 权重。本地 SDK Madduri 为无状态 D·u+b 推理，不使用位置反馈；本页按保存 KF 的矩阵进行回放。原在线程序缺失，因此位置反馈是明确可选的回放政策，重置/切换/后处理语义未核对，本页不声称精确复现。applied_time 经墙钟映射也不等于精确到帧的权重切换确认。

### 结果解读
- **A 速度 x 时间线**：对照记录的在线 vx 与实际权重矩阵重放 vx，红线为映射后的切换时刻。
- **B 速度 y 时间线**：独立核对 vy，曲线按 trial 和缺失帧断开；无可评分预测处留空。
- **C 各权重阶段误差**：查看初始模型及每次 applied 权重阶段的向量 RMSE，结合各阶段样本数和切换表。
- **D 逐帧误差分布**：查看完整可评分帧的向量误差长尾，并读取总匹配/评分数量与时钟映射残差。

### 质量评价标准
- **A 速度 x 时间线**：不能把接近曲线当作位级精确复现；需核实原 runtime 状态重置、位置反馈和输出后处理，不用拟合或输出种子掩盖差异。
- **B 速度 y 时间线**：检查与 x 轴是否存在不同误差；缺少输入不能强补为零，起始状态假设需用原程序核对。
- **C 各权重阶段误差**：后续误差增大可能来自权重加载/切换状态/后处理语义差异；这是复现一致性检查，不是闭环性能改善或独立预测验证。
- **D 逐帧误差分布**：必须报告覆盖范围、分母及墙钟映射精度；高 R² 不代表误差为零，不据此声称未知 runtime 已被精确复现。

## 图 2：Velocity timeline (zoom)

### 计算公式
**符号**
- GT：光标位移速度；在线：XDF 中记录的在线输出；离线：Ridge 与 Vel-KF 的留出预测。

**步骤**
1. 按同一时间窗叠画各组的 x、y 速度，阴影标出 trial。

### 结果解读
- **速度时间线**：逐轴比较趋势和误差出现的时段。

### 质量评价标准
- **速度时间线**：跨 trial 不连线，闭环在线结果用于描述性对照。

## 图 3：Scatter: online and offline

### 计算公式
**符号**
- x、y：每个速度轴上的参考值与预测值。

**步骤**
1. 对同一时间 bin 的配对值画对数密度图，并标注逐轴 R² 与 Pearson r。

### 结果解读
- **逐轴密度图**：对照恒等线判断偏差和幅度。

### 质量评价标准
- **逐轴密度图**：按同一速度轴和比较对象解读，注意有效样本数。

## 图 4：Metrics summary

### 计算公式
**符号**
- R²：1 − SSE/SST；r：两轴相关的平均；RMSE：速度向量误差的均方根。

**步骤**
1. 汇总在线输出、Ridge 和 Vel-KF 的比较结果及有效 bin 数。

### 结果解读
- **指标汇总**：结合 R²、相关和 RMSE 比较匹配 bin 的误差。

### 质量评价标准
- **指标汇总**：在线输出参与闭环控制，与 GT 的一致性是描述性的。

## 图 5：Decoder comparison

### 计算公式
**符号**
- `Δ`：bin 宽（默认 0.1 s），每个 trial 从 T_TRIAL_START + 0.2 s 起 10 个 bin。
- `v_k`（GT）：第 k 个 bin 的 cursor 速度 = (bin 右端位置 − 左端位置) ÷ Δ（任务坐标单位/s），位置在同 trial active 采样间线性插值。
- `u_j`：在线 decoder 流（cursortasks_decoder_velocity）的第 j 个样本（二维速度），`τ_j` 为其接收时间。
- `o_k`（online）：第 k 个 bin 的在线 decoder 输出。
- `v̂_k`（offline）：离线 decoder 的 held-out 预测（5 折时间顺序 block 交叉验证，±1 block purge，特征为尾随 200 ms 平均发放率，只用训练行归一化；模型见 S10 guide）。
- `共同 bin`：GT、online 与每个离线 decoder 都有限的 bin（10-30 online 为 3753 个）。
- `R²_xy`：x、y 合并的决定系数（见图 3）；`RMSE`：向量均方根误差；`mean r`：两轴 Pearson r 的平均。
- `vs GT`：以 v_k 为参考；`vs online`：以 o_k 为参考。

**步骤**
1. (a)：每个序列 vs GT 的 R²_xy，以及每个离线 decoder vs online 的 R²_xy（斜线浅色柱），全部只在共同 bin 上。
   意思是：同一批 bin 上比较，柱高差异就是 decoder 差异。
2. (b)：vs GT 的 RMSE；(c)：vs GT 与 vs online 的 mean r。
   意思是：RMSE 看误差大小，r 看形状是否一致。
3. (d, e)：从 zoom_start 之后第一个 bin 开始、长 min(zoom_len, 20 s) 的时间条，GT、online 与所有离线 decoder 叠加（trial 之间断开）。只选一个离线 decoder 时仍出图，标题注明。
   意思是：在一段真实时间上直观比较所有序列。

### 结果解读
- **A pooled R²**：对共同 bin 上各系列相对 GT 的 x/y pooled R² 作比较。
- **B GT RMSE**：读取同单位误差与模型差异。
- **C 平均 Pearson r**：比较协变的时间一致性。
- **D 所有系列 vx**：查看同一时间范围内各系列横向分量重叠与延迟。
- **E 所有系列 vy**：独立检查纵向预测、复现和在线差异。

### 质量评价标准
- **A pooled R²**：与 mean(x/y R²) 不同；在线为闭环描述，离线为训练/测试协议结果。
- **B GT RMSE**：较低不等于部署更优，需结合动态和参考耦合。
- **C 平均 Pearson r**：高 r 可伴随尺度偏差或低 R²，不作为单独质量标准。
- **D 所有系列 vx**：局部展示不替代总体评估，注意 bin 与帧重采样定义。
- **E 所有系列 vy**：不能把某一轴吻合推广到整体，更不能据闭环匹配断言意图重建成功。

## 注意事项
- 闭环记录里 GT（cursor 速度）由在线 decoder 输出积分得到，因此 online vs GT 接近 1 是必然的；离线 decoder 在闭环数据上的高 R² 也部分来自这种耦合，不能直接与开环日比较。
- 速度队列规则来自 Topbar 的 TrialFilter（velocity_bin_policy、cursor 空档、神经窗口、最少帧数、最大空档），方法文本中逐项写明。
- 复现用的是记录到的神经帧到 XDF 中 decoder 输出的线性拟合，不读取在线模型文件；decoder 样本数与神经帧数不同时按接收时间对齐到最近的前一帧（会写在 notes 中）。
