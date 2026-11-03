<!--
Vendored from github.com/sillybun/bci-decoder-sdk
commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
original path: src/decoders/standalone_refit_kf/README.md
original git blob: e7de8618cf0b1a71f4f4c3440869eabebbde4d31
Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
--- end vendor header ---
-->
# ReFIT Kalman Filter

本包提供不绑定信号模态和固定 feature schema 的 ReFIT-KF。公开接口为
`CalibrationBlock`、`RefitKFTrackingBlock`、`RefitKFFitData`、
`RefitKFFitConfig`、`RefitKFAdaptConfig`、`RefitKFTrainer` 和 `RefitKFCore`。

## 代码位置

- `examples/standalone_refit_kf/example.py`：imports、参数设置和四步 `main()` 入口。
- 本 decoder 的 `data_types.py`：`CalibrationBlock`、`RefitKFTrackingBlock`、
  `RefitKFFitData` 和固定时间步常量；`utils.py` 按时间戳与周期、速度与状态转移、
  ReFIT 运动辅助分区，提供时间戳检查、周期提示、速度估计、转移配对、限速和意图速度。
- `examples/standalone_refit_kf/data/blocks.py`：包含校准、训练、测试集和 specs 的 `ExampleData`。
- 示例 `data/generation.py`：合成数据；`data/schemas.py`：feature、向量、action/feedback spec 构造；
  `data/preparation.py`：特征展平与来源记录；`data/validation.py`：数据集和准备流程检查。
- 本 decoder 的 `validation.py`：按生命周期分区，集中提供输入合同和数据块、
  fit/evaluate/infer 输入、数值辅助规则、Trainer 配置、模型兼容性与归一化不变性、
  参数、Artifact、Core 新状态和 checkpoint 校验；只检查，不转换数据或提交状态。
- `examples/standalone_refit_kf/evaluation.py`：集中提供 R²、速度和轨迹对比绘图；
  `evaluate_inference()` 统一计算测试标签、打印 JSON 报告并绘图，
  example 在 `infer()` 后只需调用该入口。
- 本 decoder 目录：`config.py` 保存算法配置，`fitting.py`、`trainer.py`、`runtime.py`
  保存模型数学实现和生命周期接口；批量 `infer()` 定义在 `runtime.py`。
  `RefitKFFitData` 的 Trainer 和 decoder 包导入路径继续可用。
  `ridge()` 和 `velocity_transition()` 的实现位于 `fitting.py`。
- `examples/standalone_refit_kf/workflow.py`：`train_decoder()` 组织示例的默认 fit/adapt 流程。

原 `decoders.refit_kf_decoder.example`、`.data_types`、`.input_validation` 子模块已移除，
调用方需要更新这些子模块路径。当前 decoder 包为 `decoders.standalone_refit_kf`，
其包顶层保留模型公开类名和函数导出；示例 `train_decoder` 改从 `workflow` 导入。
文件拆分保留原有数学公式；合成数据只降低默认运动噪声，不做谱半径筛选或重试。
`examples/refit_kf/` 仍是原来的教学实现，与新入口分开。
原顶层 `data`、`evaluation` 包已删除，不保留旧路径转发；请使用下文的新导入路径。
原顶层校验包已合并为 `decoders.standalone_refit_kf.validation`，旧校验路径不再保留。
模型的正式训练接口是 `RefitKFTrainer.fit/adapt/evaluate`，无需示例目录。
导入 decoder 包仍立即注册 Core，Trainer 的包顶层导出按需加载。
示例不随 wheel 安装，运行示例需从仓库根目录调用。

## 输入边界

- 训练流程：`validation.py` 的输入分区检查数据 → 训练/ReFIT → 参数与 Artifact 分区
  检查产生的模型 → 安装 Runtime。各分区覆盖输入结构、稳定性、协方差、配置和来源信息。
- 每个新帧：SDK Runtime 检查输入合同、时间顺序、反馈时效及 `reject` 策略下的特征有效性 →
  Core 计算 → Core 检查新状态与协方差 → 提交状态并返回结果。失败不推进状态或 update ID。
- 同一次 `fit/adapt/evaluate` 中，输入摘要通过 SDK 生命周期的 `validation` 关键字传给计算 hook，
  复用 trial 数量和有效速度 mask；不再次扫描输入或调用公开速度函数重新校验。
  校验器只返回元数据，不修改或转换原始数据。
- 未混合时，最终 Artifact 校验同时检查候选参数；混合时先检查候选，再检查最终 Artifact 中的
  混合结果。父模型、候选、混合结果是不同对象，均须检查。模型与在线状态复用同一套协方差规则。
- 独立的预检查、后续 `fit/adapt/evaluate`、外部 Artifact 安装仍各自检查边界：输入数组可被调用方
  修改，因此不跨调用缓存“已校验”状态。旧 Trainer 的校验 hook 返回 `None` 时，SDK 保持原调用方式。
- `validation.py` 的 Core 状态分区提供运行时校验规则，Core 在计算后、提交前调用；SDK Runtime 的通用逐帧合同
  检查仍由 SDK 负责，避免 SDK 依赖具体 decoder 的校验模块。
- 常规改动运行相关测试和必要回归；只有明确要求时才运行 seed 1–200 sweep。
- 任务适配器负责原始数据读取、信号预处理、任务语义检查以及构造 SDK specs。
- `validation.py` 的输入分区检查输入是否满足调用方声明的 specs，并检查样本对齐、
  数值、时间戳、trial 分段和最小样本量。Example 在训练前显式调用；Trainer
  也委托该模块检查输入，直接调用 Trainer 时不会跳过检查。
- 特征批次为 `(N, *feature_shape)`，例如 `(N, 96)`、`(N, 16, 6)`、
  `(N, 24, 4)`。不限制频带、轴名或特定 schema ID；同一 Artifact 的输入必须
  匹配其部署时确定的 shape、schema/layout ID 和元素顺序。
- 保留 SDK 的公开张量 `float32` 约定，内部数值计算使用 `float64`。特征按
  NumPy C 顺序展平，归一化参数保留原始单帧 shape。
- 校准数据至少两行；追踪数据至少两个连续分段的 trial，每个 trial 至少三行。
  默认训练要求至少 20 行有效速度、至少两个有效速度转移对。
- 训练/适配位置为 `(N, D)`。action 和 feedback 均为向量且维度一致；Runtime
  每帧需要 D 维位置反馈，输出 D 维速度。状态为 `[position, velocity, 1]`，
  长度为 `2D+1`；二维为 5，三维为 7。
  位置、速度及阈值必须使用一致的单位；内置 ReFIT 阈值沿用 px 和 px/s。

核心不根据信号类型分支，状态维度由 action spec 推导。模型按连续、不重叠的
20 ms 帧设计：检查校准块及同一 tracking trial 内时间戳相邻差是否为
20,000,000 ns。发现丢帧、间隔变长或变短时，通过 logging 输出 ERROR 提示，
但继续处理，每个输入帧仍执行一次固定 20 ms 更新；速度标签与轨迹积分也仍使用
0.020 s，不自动补步、不重采样、不修改时间戳。trial 之间允许间隔并重置状态。
时间戳重复、倒序、负值及其他结构/数值错误仍按原有规则拒绝。

Artifact 的 `required_feature_period_ns=20_000_000` 用于检查在线帧间隔。
SDK Runtime 的间隔不匹配提示不阻断推理：成功时仍返回 OK 和动作，diagnostics
中包含 `feature_period_mismatch=1`、实际/期望间隔及提示信息，日志标识为
`sdk.feature_period_mismatch`；checkpoint 恢复后
继续检查，reset 后重新开始。可通过应用的 logging handler 展示或保存 ERROR 提示。

时间戳表示数据窗口右边界的采集时间，不是函数调用或网络到达时间。调度延迟不应
改变数据时间戳。上游适配器负责将其他采样周期的数据重采样、按 20 ms 分窗，并将
特征、光标位置和目标位置对齐；需要恢复连续采样语义时，由上游按明确策略处理丢帧。
不能仅修改时间戳伪装成连续数据。允许提示后继续不代表缺失数据已恢复，固定步长
计算可能与实际运动时间不同；SDK 不自动插值未知模态的数据。
位置/速度状态结构及移动目标意图重构公式保持不变。

## 训练与推理

`fit()` 使用校准块计算并冻结均值和标准差。有 `initial_tracking` 时，用真实位置
的 trial 内后向差分速度进行初始 KF 拟合；否则创建零输出种子。校准块只用于
归一化，不作为零速度监督。

`adapt()` 保留真实相对速度大小，将方向转向目标，再加回目标速度；半径外加入
默认 50 px/s、接近目标时递减的启动下限；目标半径内相对意图速度为零。
每个 trial 首行的差分速度无效，排除于拟合。归一化始终继承父模型。

`RefitKFFitConfig` 和 `RefitKFAdaptConfig` 分别控制四处正则化，默认均为 `1e-4`：

```python
transition_ridge_regularization = 1e-4             # Mean-loss ridge penalty for A_v
observation_ridge_regularization = 1e-4            # Mean-loss ridge penalty for C, excluding bias
process_covariance_floor_regularization = 1e-4     # Diagonal loading of W_v
observation_covariance_floor_regularization = 1e-4 # Diagonal loading of Q
```

四项均须为有限正数。从实现版本 `3.2.0` 起，两处 ridge 的目标改为每个样本的平均
平方误差（对输出维度求和），偏置仍不惩罚，输入尺度不变：

```python
loss = ((target - design @ coefficients)**2).sum() / sample_count + ridge_penalty
gram = design.T @ design / sample_count + regularization * penalty
rhs = design.T @ target / sample_count
coefficients = solve(gram, rhs)
```

实现时将上述正规方程两边同乘 `sample_count`，用以下等价形式求解，避免重新缩放
数据矩阵造成额外浮点差异：

```python
gram = design.T @ design + (sample_count * regularization) * penalty
coefficients = solve(gram, design.T @ target)
```

两个 ridge 默认仍为 `1e-4`，但相对于旧求和损失的有效强度随样本数增加。若需复现
旧参数 `old_lam` 的结果，按实际使用的有效样本数换算；不使用总采集帧数：

```python
transition_ridge_regularization = old_transition_lam / valid_transition_count
observation_ridge_regularization = old_observation_lam / valid_observation_count
process_covariance_floor_regularization = old_process_floor
observation_covariance_floor_regularization = old_observation_floor
```

两个 covariance floor 默认仍使用固定绝对值。可通过一个布尔开关同时启用残差缩放，
两个相对系数独立设置；以下选项在 fit 和 adapt 配置中均可用：

```python
config = RefitKFFitConfig(
    auto_covariance_floor=False,  # Default: retain fixed absolute floors
    process_covariance_floor_relative_ratio=1e-4,
    observation_covariance_floor_relative_ratio=1e-4,
)

variance_scale = trace(raw_residual_covariance) / dimension
floor = absolute_floor
if auto_covariance_floor:
    floor = max(absolute_floor, relative_ratio * variance_scale)
covariance = raw_residual_covariance + floor * eye(dimension)
floor_fraction = floor / (variance_scale + floor)
```

现有 `process_covariance_floor_regularization` 和
`observation_covariance_floor_regularization` 始终作为绝对下限，相对系数为无量纲
有限非负数。开关关闭时相对系数不参与计算；零残差时使用绝对下限，占比为 1。
开启后只能保持或提高原有绝对 floor，不会自动减小它。

只要执行了监督拟合，报告都会包含以下六项，与开关是否开启无关：

```python
metrics = result.report.metrics  # result = trainer.fit(...) or trainer.adapt(...)
metrics["train.process_covariance.residual_variance_mean"]
metrics["train.process_covariance.floor_effective"]
metrics["train.process_covariance.floor_fraction"]
metrics["train.observation_covariance.residual_variance_mean"]
metrics["train.observation_covariance.floor_effective"]
metrics["train.observation_covariance.floor_fraction"]
```

占比为 `0.01` 表示 floor 占最终平均方差的 1%，不是误差率或调参质量评分。
过程指标只针对速度块；观测指标针对全部展平特征。自适应使用参数混合时，这些指标
描述混合前的本次候选拟合；仅校准、没有监督数据的种子不产生残差指标。

新训练配置同时保存开关、两个相对系数，以及
`ridge_loss="mean_squared_error_per_sample"`。历史 Artifact 直接使用已保存的矩阵，
不重新拟合，所以仍可加载。新 Artifact 分别记录四项；只有旧 `regularization` 字段的历史
Artifact 仍可加载，包含部分新字段的配置会被拒绝。`p0_velocity_regularization`
仍独立控制初始速度协方差，默认保持 `2500.0`。

可选参数混合为 `new = alpha * parent + (1 - alpha) * candidate`。
从零种子开始的第一次适配完整安装候选参数。保留原有矩阵稳定性、协方差和
有限值校验；候选参数与混合后参数均须通过检查。

Runtime 每个接受帧执行一次固定时间预测，写入当前反馈位置后再计算 Kalman
增益。状态检查通过后才提交 `x/P`。支持 trial reset 和 checkpoint 恢复。

## 四步示例

`examples/standalone_refit_kf/example.py` 的 `main()` 顺序调用：

1. `generate_train_test_data()`：按 spike、EMG 或 ECoG 的实际参数构造 example schema，
   生成校准数据和独立的训练、测试 trial，保留原始 feature shape。
2. `data = validate_decoder_inputs(data)`：检查三个数据块的结构、有限值、布局一致性和
   训练/测试 trial 不重叠，返回原对象；再用 `data = flatten_decoder_inputs(data)`
   将特征统一为 `(N,F)`，生成带来源信息的新数据对象，不修改原数据。
3. `train_decoder()`：初始训练及可选适配，仅使用校准和训练数据。
4. `infer()`：测试 trial 逐帧推理，trial 边界重置，trial 内保留滤波状态。

```python
from examples.standalone_refit_kf.data import generate_train_test_data
from examples.standalone_refit_kf.data.preparation import flatten_decoder_inputs
from examples.standalone_refit_kf.data.validation import validate_decoder_inputs
from examples.standalone_refit_kf.workflow import train_decoder
from decoders.standalone_refit_kf import infer

data = generate_train_test_data()  # 默认 ecog，16 通道、6 频带、2D
# 也可选择：
data = generate_train_test_data(feature_type="spike", channel_count=96)
data = generate_train_test_data(feature_type="emg", channel_count=32)
data = generate_train_test_data(
  feature_type="ecog", channel_count=16, band_count=4, action_dim=3,
  action_type="velocity", feedback_type="position",
)
data = validate_decoder_inputs(data)
data = flatten_decoder_inputs(data)
artifact = train_decoder(data)
actions = infer(artifact, data.test)
```

`feature_type` 接受 `spike`、`emg`、`ecog`。`channel_count` 默认 16。
`band_count` 只用于 ECoG，省略时为 6，接受任意正整数。
`band_labels` 由输入方提供，支持按数据顺序排列的字符串列表或元组。
生成器在 6 个频带且未提供标签时默认传入 `theta, mu_alpha, beta, low_gamma, high_gamma, very_high_gamma`。
其他频带数量未提供标签，或显式传入空列表/元组时，使用 `band_0, band_1, ...`。
这些默认生理频段标签只在 `generation.py` 中定义；通用 Spec 构造器不预设频段名称。
非空标签必须与频带数量一致，且非空、唯一。构造规格时只打印对应关系，
例如 `band0: theta, band1: alpha`，不再打印频带数量推荐或缺少标签的提示。
标签只命名现有数据，不重排数值；不同标签顺序产生不同 Layout ID，并随 Artifact 保存。
例如：`generate_train_test_data(band_count=5, band_labels=("theta", "alpha", "beta", "low_gamma", "high_gamma"))`。
`action_dim` 接受任意正整数，默认 2；非 2/3 的 action 维度仍打印推荐提示，不阻止运行。
两者仍拒绝零、负数、浮点数和布尔值。
原始 schema ID 按参数命名，例如 `example.feature.ecog.c16.b8.v1`、
`example.feature.spike.c96.v1` 和 `example.feature.emg.c32.v1`。
采样周期、数值域和频带顺序保存在属性中；action 维度不参与 feature 命名。
spike 为非负整数值 float32，EMG 为非负连续 float32，ECoG 为 log10 功率实数。
生成的是满足值域的合成特征，不模拟真实 spike 或上游 EMG/ECoG 信号处理过程。

`action_type`、`feedback_type` 均接受 `position`、`velocity`、`acc`，默认分别为
`velocity` 和 `position`。类型决定 action/feedback schema 的 ID、layout ID、
description、单位及对应的 `action_type` / `feedback_type` 属性。
例如 `example.action.velocity_2d.v1`、`example.feedback.position_2d.v1`，
加速度使用简写 `acc`，例如 `example.action.acc_2d.v1`。
position、velocity、acc 的单位依次为 `px`、`px/s`、`px/s²`。

当前模型仍使用 `[position, velocity, 1]`，计算输出是 velocity，反馈值按 position 使用。
类型、单位和其他 Schema 属性仅为描述信息，不切换模型方程、不转换单位，也不触发
通用输入限制。调用方应准确描述实际数据；将属性改成 acc 并不能获得加速度模型。
合成数据块始终保存实际位置。类型声明不改变合成运动；类型属性随 Artifact 一起保存。

`generate_train_test_data()` 的 spike/emg/ecog 选项仅控制合成数据生成方式。
外部调用方可提供任意 rank 的 FeatureSpec、任意自定义轴名及有限 float32 特征；
无需 feature_type、value_domain 或固定频带标签。负数和小数不会因描述属性被拒绝。
`validate_decoder_inputs()` 保留块结构、样本对齐、有限性、trial、时间戳与 Spec 匹配检查。
已展平数据必须与所记录的 source Spec 和 C 顺序一致；训练后不得更换特征维度或排列。

其他参数：`calibration_samples=100`、`train_trials=4`、`test_trials=2`、
`samples_per_trial=36`、`calibration_seed=42`、`tracking_seed=43`。
改变样本量时，仍须满足训练最低有效样本量。

生成器保留原来的恒速目标和追踪几何，只将相对速度创新的默认标准差设为
`motion_noise_std=0.5`（单位 px/s）。观测特征的高斯噪声保持原来的 `0.03`。
每个 seed 只生成一次，不根据训练稳定性重抽数据。训练的四项正则化默认均为 `1e-4`，
保留原有谱半径 `<1` 检查。

```python
data = generate_train_test_data()  # motion_noise_std=0.5
data_original = generate_train_test_data(motion_noise_std=1.0)  # 对照原运动噪声
```

降低运动噪声在默认三维案例中可避免候选速度矩阵略微越过稳定边界，
但不保证任意 seed 或配置都稳定。历史数值回归显式使用 `motion_noise_std=1.0`，
继续与原始基线文件比较。

ECoG 原始特征为 `(N,C,B)`，在第二步按 C 顺序变为 `(N,C*B)`；spike/EMG
原始特征为 `(N,C)`。三者进入第三、四步时均为 `(N,F)` / `(F,)`。
返回数据的 `source_feature_spec` 保留原始 example spec。通用向量 spec 使用独立
schema，记录原 schema/layout ID、原 shape、轴标签及 `flatten_order="C"`；这些
信息随 Artifact 保存。在线生产方须使用相同顺序，不能仅凭特征数相同判断兼容。
展平后 ID 例如 `example.feature.ecog.c16.b8.flat.v1`。新 ID 与旧 catalog ID 不同，
旧 Artifact 仍可运行，但不能与新生成的数据直接混用。

由于归一化按元素计算，先展平再估计均值/标准差与原来的先归一化后展平等价。
二维 ECoG 使用历史基线的 `tracking_seed=23` 和 `motion_noise_std=1.0` 时，数值与原基线保持一致。
非 2/3 维合成运动使用静止目标和向目标收敛的轨迹，覆盖每个坐标分量；
现有 2/3 维运动生成方式保持不变。维度增加时仍需足够样本，数值稳定性检查仍生效。

PowerShell 中从项目根目录运行：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m examples.standalone_refit_kf.example
```

PyCharm 使用 Module name `examples.standalone_refit_kf.example`，Working directory
设为项目根目录，将 `src` 加入 PYTHONPATH。

默认输出 144 个训练样本、72 个测试样本、首帧 action 和测试真实速度 RMSE。
合成数据仅用于验证代码与数值合同，不代表真实解码精度或闭环收益。

## 绘图依赖与结果检查

`pyproject.toml` 将 `matplotlib>=3.8` 声明为必需依赖；uv 的解析结果
记录在 `uv.lock`。普通同步即安装绘图依赖：

```powershell
uv sync
```

需要测试依赖时使用 `uv sync --extra test`。运行 Example 默认只显示图窗，不保存图片：

```powershell
uv run python -m examples.standalone_refit_kf.example
```

使用此命令时，同样需要将 `src` 加入 PYTHONPATH，或使用项目已安装的 editable 包。
绘图设置由 `plot_inference()` 的参数控制，不解析命令行。需要保存图片时显式指定
`output_path`；仅保存、不显示图窗时设置 `show=False`：

```python
from examples.standalone_refit_kf.evaluation import plot_inference

actions = infer(artifact, data.test)
figure = plot_inference(
  artifact, data.test, actions,
  output_path="outputs/inference_comparison.png", show=False,
)
```

每个速度分量各一行，比较 Infer、GT actual 和 ReFIT intended，最后一行显示向量
误差模长。控制台输出实际速度的 RMSE 和 R²。GT actual 使用实际光标位置的固定 20 ms trial 内后向差分；ReFIT
intended 使用 Artifact 的半径、启动速度和限速配置重构适配标签。图中显示分别
相对两种标签的向量 RMSE 和平均分量 R²。R² 按分量计算 `1 - SSE/SST` 后等权
平均，常量分量不参与平均；不足两行有效样本或全部分量为常量时，控制台输出
`null`，图上显示 `N/A`。R² 可以为负。trial 首行无效差分不画入标签/误差曲线，也不计入指标；
不同 trial 不连线。速度曲线和指标使用全部维度。

右侧显示 position trajectory：黑色为记录的 GT 位置，蓝色为预测速度积分得到的
位置，绿色标记各 trial 的起点。每个 trial 从 GT 初始位置开始，之后按
`p_hat[t] = p_hat[t-1] + v_hat[t] * 0.020` 累积，不逐帧用 GT 位置覆盖。
该对齐方式对应 GT 的后向差分速度，trial 首帧速度不参与积分。二维使用 XY
等比例坐标，三维使用 XYZ 轨迹；一维显示 position 随时间变化，高于三维只展示
前三维轨迹并在标题标注投影。trial 之间不连接。

该图检查记录位置反馈下的速度回放，不表示预测光标的闭环运动轨迹。绘图库只在
调用 `plot_inference()` 时导入，`train_decoder()` 和 `infer()` 无需 matplotlib。

## 回归与兼容性

模型标识为 `standalone_refit_kf`，实现版本为 `3.3.0`，与当前 decoder 目录名一致，
与教学示例的 `refit_kf` 注册分开。示例入口目录沿用 `examples/standalone_refit_kf/`。
兼容性检查要求 Artifact 的 `model_type` 匹配且实现主版本一致；过去使用
`refit_kf_decoder`、`ecog_refit_kf` 等其他标识的 Artifact 不会自动转换或加载。
Example 要求先接收第二步返回的向量数据再训练。

decoder 包提供正式输入类型、Trainer、Core 和批量 `infer`，模型校验统一从
`decoders.standalone_refit_kf.validation` 导入。示例数据和默认训练流程位于 examples，
模型训练、推理和校验均不依赖 examples 或顶层校验包。
绘图依赖仍在示例的 `plot_inference()` 内按需导入。

运行数值回归测试：

```powershell
python -m pytest tests/test_standalone_refit_kf_regression.py tests/test_standalone_refit_kf_feature_types.py -q
```

`tests/data/refit_decoder_baseline_v1.npz` 在本次重构前采集，保存输入、四条生命周期
路径的全部参数及逐帧 action、状态和协方差。测试将旧 ridge 权重按有效样本数换算后，
以该固定基线检查数值结果，不重新生成基线文件，
不要求更名后的模型版本字符串保持一致。
