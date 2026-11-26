# 数据分析 GUI（data-analysis-gui）

猕猴 512 通道量化事件计数 + CursorTasks 行为 XDF 的单日 / 跨天分析工作台。融合了
`macaque_data_analysis`（GUI 风格、进度条、缓存优先的计算与存储、物理布局、在线/离线对比、Grid 方向定义）
与 `neural_behavior`（trial QC、data_qc_v1、independent_recheck_v1 的分析方法）。代码结构参照
`neural_behavior/code/<topic>_v<N>/`：每个 section 一个带版本号的子目录。Python 包名 `data_analysis_gui`。

## 启动

在 BCI Tools 启动台选择「数据处理 → 数据分析 GUI」，或在本目录运行：

```bash
uv sync --locked
uv run --locked python app.py
```

独立 Python 3.11 环境，依赖 numpy / scipy / matplotlib / pyxdf / umap-learn；不需要安装 Decoder SDK（见「Decoder」）。
默认数据根目录 `E:/AitBCI/data/keaiduo`（顶栏可改，或设环境变量 `DATA_ANALYSIS_GUI_DATA_ROOT`）；物理布局默认
`channel_layout/天枢3.0_ChannelMap.xlsx`，读取其中的 `Channel# / Shank / Elec#` 表。
旧默认布局偏好自动升级为新表，自定义路径继续保留；原 128×8 数值表仍可手动选择。缺失时其他图退回电极 ID 网格，S14 明确提示缺少新表。界面偏好保存在 `.local/preferences.json`。

单记录分析新增 **S14 · 解码权重与电极深度**：继承 S10 的 bin、窗口、sqrt 和训练参数，
对当前 run 全量拟合 Ridge（默认）或 Velocity KF。输出 |wx|、|wy|、sqrt(wx²+wy²) 的热力图和数值图，共六张，
六图统一使用 `hot_r` 配色（高值深色、低值浅色）、固定 0–25 色条，不包含 position groups 副图。数值图统一 x1 倍率；
超过 25 仅颜色饱和，数字和 CSV 保留实际值。灰色表示未传输，零权重为白色，CSV 的 status 区分常数输入通道。
CSV 的 vx / vy 为绝对值，vx_signed / vy_signed 保留原始系数正负号。
同时导出逐位点权重与位置分组统计 CSV。默认 per_SD 权重用于比较输入一个标准差对应的模型敏感度；
KF 使用稳态解码增益而非观测矩阵。Elec# 为沿杆位置序号，未标定脑内深度；权重大小不等于删通道后的性能贡献。
S14 属于慢分析，可单独运行，或勾选包含慢分析后参与批量计算。

S14 第七张图为有效通道深度分布：整段加载的 run 中任意一帧 code > 0 的通道计 1，
默认按 100 µm 分箱，并按 Channel ID 对齐最终工作簿的 Shank 与沿轴坐标。深度权重与二值发放分布可用 scripts/export_depth_weights_and_firing.py 导出，code 1/2/3 每帧均计一次事件。
图中 `single unit count` 沿用参考图标签，统计对象是有效通道，不是 spike sorting 分离的神经元数。
深度 CSV 的内容纳入 S14 缓存签名，修改后自动重新计算。

所有电极位置图默认 Elec# 1 在下、128 在上，文件名不附方向后缀。
跨记录新增 **X09 · 解码权重 median / average**，继承 S14/S10 设置，对勾选记录独立拟合。
每通道等权汇总 |wx|、|wy|、逐 run 的 norm；未传输值排除、恒定输入的零权重保留。
输出六张 hot_r 热力图，六图按共同最大统计值自动统一色条范围，并导出实际参与 run 数的 CSV。
每份录制是一个样本，不按天或 bin 数加权。单记录 S14 色条仍固定 0–25。

## 多台电脑同步

本工具的共享分支是 [`origin/data-analysis-gui`](https://github.com/sillybun/bci_tools/tree/data-analysis-gui)。在本工具目录运行 Git 命令即可。各台电脑可以使用不同的盘符和数据目录。

### 开工和更新

首次配置时执行 `git clone --branch data-analysis-gui https://github.com/sillybun/bci_tools.git`，然后进入 `bci_tools/tools/data-analysis-gui`。已有仓库先确认 `git remote -v` 和 `git status --short --branch`。共享分支的工作区干净时执行：

```powershell
git switch data-analysis-gui
git fetch origin
git merge --ff-only origin/data-analysis-gui
git rev-parse HEAD
uv sync --locked
```

有本机修改时先检查 `git diff` 和 `git diff --cached`。需要跨机继续时，把待交接的改动纳入下一个发布批次，向用户确认后再提交和推送。遇到分支分叉或修改重叠时，先比较双方改动，再决定如何整合。启动前按本机目录设置 GUI 的数据根目录、布局与输出路径。

### 开发、提交和发布

1. 从更新后的 `data-analysis-gui` 创建 `codex/<任务名>` 分支。可以并行开发不同任务。同一任务在另一台电脑继续时，从远端检出原任务分支。
2. 小改动先留在本机工作区，积攒成一批可以一起说明和验证的改动。涉及启动方式、默认参数、布局、输出或兼容性时，把 README 和实现放在同一批。
3. 运行相关测试或最小复现，向用户说明待提交的范围、验证结果和拟推送的分支。**所有电脑上的 Codex 都要先取得用户对本批次的明确确认，再执行 `git commit` 或 `git push`。** 一次确认可以同时覆盖两项操作，按用户确认的范围执行。
4. 获得确认后逐路径暂存本批次文件，用 `git diff --cached --check` 和 `git diff --cached` 核对范围，再用 `feat:`、`fix:` 或 `docs:` 等前缀提交。获准发布时用 `git push -u origin HEAD` 推送任务分支。涉及多人并行开发或功能改动时，通过 PR 合入 `data-analysis-gui`。获准直接发布到共享分支时，先确认远端可快进。发布后用 `git ls-remote origin refs/heads/<分支>` 核对提交。

Git 跟踪代码、锁文件、文档和获准纳入项目的小型测试样本。原始 XDF、实验配置与日志、结果与缓存、`.venv` 和 `.local/preferences.json` 通过各自的数据存储或本机环境管理。独立分发的布局表和深度表用文件名及 SHA-256 对齐，例如 `Get-FileHash -Algorithm SHA256 <本机文件路径>`。交接报告时记录数据范围、运行参数、布局文件名及哈希、代码提交，并提供接收方可访问的数据或结果位置。

当前交接涉及的新版布局资料是 `channel_layout/TS3.0_ChannelMap_Final.xlsx`，使用「电极shank通道对应」工作表。已核对版本的 SHA-256 为 `459E3E845B2B6A7ADECB15F3D0783FE7C69A78540D4941FFBECE22642516CEEC`。新版读取器和 GUI 默认路径的切换是待实现的代码任务，布局文件需要在目标电脑的数据目录中单独准备。

### 发给另一台电脑上 Codex 的指令

在目标电脑打开本工具目录对应的 Codex 聊天，把尖括号替换为交接信息后发送：

```text
请同步 bci_tools 的 data-analysis-gui 工具。
远端 https://github.com/sillybun/bci_tools.git
目标分支 <分支名>，交接提交 <SHA>
同步后继续 <具体任务或仅同步验证>
数据和配置 <本机数据位置、依赖文件名及 SHA-256>
验证 <测试命令或日期、run 和预期输出>

如需继续开发，把小改动积攒到同一个批次，并在 commit 或 push 前请我确认。
先检查当前分支、远端、已暂存和未提交的修改。fetch 后确认交接提交
在目标远端分支历史里。工作区干净时切到目标分支，并用 --ff-only 更新。
遇到本机改动重叠或分支分叉，先报告文件和双方提交。
同步成功后执行 uv sync --locked 和上述验证，回复实际分支、HEAD、
验证结果及仍需从外部同步的数据。若需复现交接提交的精确版本，
用独立 checkout 或 worktree 检出该 SHA。
```

## 界面

命名备注仅标注有区别的实验条件：目标辅助开启时标 `ass速度`，手束缚时标 `restrained`；普通脑控不再附加 `arm-free` 或 `no-assist`，offline 不再重复附加 `manual`。完整控制参数仍保存在配置和 session_info.json 中。

当前统一归档的 20261106 / 20261109 / 20261110 / 20261111 使用扁平目录：`日期/run-001-CO8或Grid-online或offline-note/` 下直接放 `xdf/`、`behavior/`、`config/`、`logs/`、`initial_weights/`、`refit/` 及说明。拆分保留来源 run 编号，按 session 时间顺序增加字母，例如 `run-006-a-CO8-online-note`、`run-006-b-Grid-online-note`、`run-006-c-Grid-online-note`。XDF 文件也以同一 run 名开头，再附加日期、rec 编号、片段类型及信号类型。一个 run 对应一个任务 session，其身份和分段信息保存在根目录 `session_info.json`，不再增加 `sessions/session-…` 两层。旧命名和嵌套目录仍兼容；GUI 的 Run 列保留 `006-a/b/c`。`scripts/rename_run_first.py` 为这次名称迁移工具，`scripts/flatten_run_sessions.py` 为目录提升工具；历史审计保留原样，当前名称见每日索引和最新 `run_naming_audit_*`。

界面字体、配色、进度条和图像查看沿用 macaque GUI；Windows 上启用 per-monitor DPI 感知，避免模糊。

- **顶部进度条**：显示已结束/总任务数、等待数、失败数、停止数及当前任务。每个分析在独立 worker 子进程中执行；运行期间可继续提交，任务按顺序排队。任务提交时固定记录、section 参数、筛选、布局、是否强制重算及输出目录，后续修改界面不会改变已提交任务；同设置的等待/运行任务不会重复提交。
  「停止当前」直接终止当前分析进程，之后继续执行等待任务；「清空待执行」只移除等待任务。任务队列窗口显示每项状态和进度，双击可核对提交时的设置；「日志」查看输出。关闭程序会停止当前任务并清空等待任务，队列不跨重启恢复；已完整保存的结果保留，中断的 partial 结果不作为有效缓存。
- **标题栏**：`单记录分析 Single-session` / `跨记录分析 Cross-session`；`⚡ 一键计算 Compute All` 在后台运行当前模式下的全部分析项目
- **跨记录一键报告**：勾选记录后点击「一键报告」，按数据与实验范式、通道稳定性、通讯稳定性、离线解码、在线 CO8 行为、在线 Grid 行为六类整理 PNG。原计算目录保持不变，额外副本保存到 `E:\AitBCI\data_analysis_gui_output\keaiduo\report\起始日期_结束日期_Nruns_v1`，同日期范围和run数量再次生成时递增版本。离线解码仅使用离线 CO8/Grid，在线行为仅使用对应在线记录；缺少某类记录时明确跳过，计算失败时保留可用图片并标记未完成。「报告文件夹」可打开最近生成的报告；`report.json` 记录所选记录、来源和生成状态。
- **报告空间图**：X01 同时导出 default 电极 ID 网格与 physical 真实排列；真实排列采用 TS3.0_ChannelMap_Final.xlsx。新增 `07_方向调谐_S05` 和 `08_解码权重_S14`，均包含 `runs` 与 `summary_all_runs`。S05 输出 tuning 的 mean/median；S14 输出 |wx|、|wy|、norm 的 mean/median 和深度权重/二值事件率分布。支持 Electrode #、Depth、去掉 Shank1 的 7shank（保留2～8编号）。轴向距离取自最终工作簿；summary只包括本次勾选记录，缺测排除，未检验不当作未调谐。S14 沿用本报告 S10 参数独立拟合。
  （单日：所选多份记录各自的全部单日 section，勾选全部记录选项则使用所有可见记录；跨天：当前勾选记录 × 全部跨天 section）。
  S08 需要通道配置 INI 或 Neuralinsight 日志、S11 只对 online 记录运行，不满足时自动跳过。计算期间可继续浏览，完成一项显示一项。
- **Topbar · Preprocessing**：数据目录与数据类型（CO8 / Grid / CO8 online / Grid online）、
  `筛选设置…`（全部 trial 筛选参数的弹出窗口：基础 / Advanced / Session 级 / 规范，带预览、保存 / 载入配置）、
  `筛选详情…`（逐 trial 查看每条标准的 pass / fail / flag）、电极布局 xlsx（可取消勾选：按电极 index 排在 128×8）、结果存储目录、`Recompute / 强制重算`。
- **通道配置 INI / LOG**：支持 `configs.ini` / `config.ini`、日期命名的 INI，以及 `Neuralinsight.log` / `Neuralinsight_YYYYMMDD.log`。自动在日期目录、`shared_config/acquisition/`、`shared_logs/acquisition/neuralinsight/` 查找，优先 INI，缺少时使用当天日志；兼容 `20261109`、`2026-11-09`、`1109` 日期目录，不自动使用其他日期的日志。日志取文件中最后一次 `filter.*.results_applied` 的完整通道分类，校验三组分类覆盖 1024 个通道且互斥，以及编号、重复和数量；最后一条结果损坏时报错，不静默使用旧结果。good / uncertain / bad / 串扰编号从 0 转为 1 起始，实际传输列顺序以 XDF 为准，不从日志推造 `last_programmed_channels`。结果记录来源时间、行号和路径。同一天有多次分类或录制后重新筛选时，应选择对应时间的日志文件；当前自动规则使用日志最后一次保存结果。顶栏「…」可选择 INI 或 LOG，「自动」恢复查找。使用通道配置的 Section 按有效文件路径和 SHA-256 区分缓存。
- **1106 归档 run**：支持 `<日期>/CO8_run-.../xdf/`，保留原 CO8 / CO8 online 分类和筛选；单日、跨记录表增加「实验条件」，区分手控离线、自由手无辅助脑控、手束缚初始辅助 200 / 100 px/s。跨记录标签保留条件，辅助 200 / 100 使用紫色 / 青色；辅助数值代表初始值，可随 block 衰减。条件来自归档文件夹命名，不使用通用 `ASSIST.MODE_ENABLED` 判断目标辅助。`early_segment_old1` 保留为独立记录并沿用所属 run 的条件和模型目录，不与主 XDF 拼接。归档 run 优先在 XDF 的 `T_SESSION_END` 处结束；无结束事件时保留全部录制范围，不受旧 1106 测试记录的 25 分钟规则截断。行为 CSV 可覆盖整场任务，统计范围可能大于 XDF。
- **单日工作台**：左侧记录表支持 Ctrl/Shift 多选；「加载 / 计算」提交所选记录的当前 section，「Compute All」提交所选记录的全部适用 section。右侧始终预览所选记录中的焦点记录，卡片显示已选数量和当前预览；刷新及重启保留多选。单日分析逐记录独立计算，不把多份记录合并为跨天分析。
- **跨天工作台**：上方记录勾选表 + 日期时间轴（全选 / 仅离线 / 每天第一份），下方每个跨天 section 一个标签页。
- **结果查看**：每张图以 300 dpi PNG 预览（适应宽度 / 整页 / 100%，Ctrl+滚轮缩放、拖动平移），
  按钮打开 PNG / SVG / PDF / CSV、绘图数据表、「计算方式」和结果文件夹；另有 Summary 与统一的 Table View 标签页，所有结果表通过 Prev / Next 和下拉框切换，按需加载。

一键报告默认在所勾选的记录中保留有可读 XDF 且 Trial 数严格大于 50 的 run。优先采用可匹配的行为 CSV，缺失或无效时按 XDF 事件中的完整 trial 计数。同一 run 的匹配 CSV 与 XDF 分段分别提供计数来源，有可用 CSV 时取 CSV 计数。报告的 `run_filter` 记录排除原因，每日 BGA 和控制对比也遵守筛选。需要恢复原选择方式时，取消一键报告的筛选勾选。

## 计算与存储（优先读取存储结果，缺失时才计算）

结果写入「结果存储」目录。默认勾选「跟随数据目录」：与数据目录并行，路径中最近的 `data` 文件夹换成 `data_analysis_gui_output`，其余部分不变，例如数据目录 `E:\AitBCI\data\keaiduo` → 结果存储 `E:\AitBCI\data_analysis_gui_output\keaiduo`（路径中没有 `data` 文件夹时，放在数据目录旁边：`D:
ec\keaiduo` → `D:
ec\data_analysis_gui_output\keaiduo`）。用「…」手动选择目录后自动取消跟随；命令行 `--output` 默认同样规则。目录结构（下面的 `output/` 指结果存储目录）：

```text
output/single/<记录>/<sid>_<ver>/<签名12位>/      output/cross/<n>rec_<hash>/<sid>_<ver>/<签名12位>/
    result.json                     # 签名输入、summary、notes、全部文件 SHA-256
    figures/SID_NN_<标题>.png|.svg|.pdf
    figures/SID_NN_<标题>.csv           # 图中实际绘制的数值；# 开头的行写明计算方式、记录 SHA-256、筛选条件、参数
    figures/SID_NN_<标题>__<名称>.csv   # section 给出的逐图数据表
    figures/SID_NN_<标题>.figure.json
    tables/<名称>.csv
```

签名包含：记录文件 SHA-256、section 版本及其源码、`core/` 与 `decoders/` 源码、trial 筛选全部阈值（含版本号）、
section 参数、布局 xlsx 与 configs.ini 内容。打开或计算时先找签名一致且所有文件哈希核对无误的结果；
找不到才计算。任何输入或代码变化都会生成新目录，旧结果不会被误用。

### 开发要求：Section 级增量更新

- 每个 Section 独立计算、保存和加载。只改一个 Section 的绘图或分析代码时，重新执行该 Section 和实际依赖这项改动的 Section。其余 Section 复用与当前输入匹配且完整的缓存。
- 缓存签名跟踪记录、参数、配置、源码和依赖。跨 Section 的分析来源通过 `source_sections` 声明，额外使用的共享源码通过 `cache_code_dependencies` 声明。S03 和 S12 共用的缺帧算法独立于 S03 绘图。通用 `core/`、`decoders/` 和 `sections/base.py` 目前作为共享依赖保守跟踪。
- 单个 Section 的准备、计算或保存失败只报告该任务的错误。批量任务继续加载或计算其余 Section，已保存的结果仍可读取。
- 验证增量行为时使用默认的缓存优先模式。手动启用「强制重算」会跳过命中的缓存。
- 单日与跨天任务共用输出目录下的 `.session-cache`。通道统计、传输汇总、trial 特征、调谐统计及解码中间数据按记录 SHA-256 和实际计算设置保存；迁移拟合按有序记录对保存。跨天增删记录时复用可用条目，只为缺失的中间数据计算，并重新汇总及导出当前组合。日志显示每个任务复用和计算的条目数。首次计算会从原始记录建立数值缓存，已有图和 CSV 仍可查看。

## Trial 筛选规范（`core/filters.py`，版本 `trial-filter-v2`）

### 多 session XDF 拆分

`scripts/split_session_package.py` 可从整理包中选取多会话 run，生成独立的派生包：

```powershell
uv run --locked python scripts/split_session_package.py --source-package "E:\AitBCI\data\keaiduo_1109_1110_1111_整理包" --output "E:\AitBCI\data\keaiduo_1109_1110_1111_session_split" --run "1109/CO8_run-006" --run "1111/GRID_run-001"
```

输出目录必须尚不存在。按 `session_name` 保留各 session 的全部任务流；共享神经、metadata、decoder 流按实际任务流覆盖范围前后各 0.5 秒裁剪，相邻 session 的上下文不会重复。重复流、电极映射、原时间戳、clock offset 和样本值均保留，不去重、不归零、不重采样。输出与来源逐流逐样本核对，并校验原始文件 SHA-256 未变。会话间空闲段仍保存在原始 XDF 中。

派生包带对应 config、logs、behavior、初始权重、refit 和 Neuralinsight 日志，复制文件逐一校验 SHA-256。完整会话的 CSV/refit 可比 XDF 当前片段覆盖更长；启动或短片段照常保留，不能当作完整任务。`split_manifest.json` 及各 session 的 `xdf_split_manifest.json` 记录来源、时间边界、样本数和 GUI 检查。GUI 应单独选择派生包为数据根目录，并勾选所需的 CO8 online / Grid online，避免与原包重复统计。

每个 trial 在每条标准下记录 pass / fail / flag 和原因，并按固定优先级给出最终状态与一个主要原因：

| 优先级 | 标准 | 原因代码 | 用于 |
| --- | --- | --- | --- |
| 1 | Incomplete trial | 缺 START `end_without_start`、缺 END `start_without_end`、嵌套 `nested_start`、trial id 冲突 `trial_id_conflict`、无 active cursor、cursor 编号变化 | 方向 + 速度 |
| 2 | Success only | `not_success`；结局不明确 `ambiguous_outcome`（ACQUIRED ≥2 或 ACQUIRED+MISS，单列，默认不纳入） | 方向 + 速度 |
| 3 | 方向标签有效性 | trial 内 target 变化、CO8 角度偏离 45° 超过容差、CO8 target 距离超范围、CO8 起点不在中心、起点=目标 | 方向 + 速度 |
| 4 | Overtime / 过快 | `overtime`（默认 start→命中，固定秒数或 median + k·MAD）；`too_fast`（默认 flag） | 方向 + 速度 |
| 5 | 窗口长于 trial | `shorter_than_window`（窗口终点超过 END 会混入 ITI；随窗口变化） | 方向 + 速度 |
| 6 | 窗口内神经覆盖率 | 无帧、帧数比例 < 阈值（默认 80%，相对实测帧周期）、最大接收空档 > 阈值（默认 150 ms）、窗口在记录外 | 方向 + 速度 |
| 7 | block 范围 | `block_out_of_range` | 方向 + 速度 |
| 8 | 速度 bin 质量 | cursor 采样空档、cursor 不属于本 trial / 非 active、短窗口神经帧不足；坏 bin 去掉整个 trial（默认）或只去掉该 bin | 仅速度 / KF |
| Flag | 命中早于窗口结束 | `acquired_before_window_end`（记录命中后数据占窗口比例；主分析保留，可在敏感性分析中排除） | — |

Session 级：最少样本（速度 trial ≥ 80、每方向 ≥ 8、时间 block ≥ 10；不满足标为「不合格」而不中止）、
offline / online 标签（online 的 cursor 由解码器驱动，速度标签含义不同）、神经长时间断流（≥ 5 s）及影响的 trial。

- 所有阈值在「筛选设置…」中修改，可「保存配置 / 载入配置」为带版本号的 JSON；阈值应在查看结果之前确定，不根据结果调整。
- Trial 级问题只记录原因、不中止；只有 session 级数据错误（重复流不一致、时间戳不递增）才停止加载。
- 默认值复现两个来源项目的已核验 cohort（例如 1026 run-010：383 行 → 方向 347 → 速度 325；1027 th120：153 / 143）。
  最大接收空档改为 100 ms 时，1029 之后每份记录会多排除约 50–90 个 trial。

## Sections

| ID | 单日 | ID | 跨天 |
| --- | --- | --- | --- |
| S01 | Raw data：八方向 raster + 八方向 trial average + 筛选概览，可选全通道稳定性图册 | X01 | 跨天通道稳定性与活跃度 |
| S02 | 通道稳定性分析：最大编码、编码分布、饱和度、饱和分区 (saturation zones)、平均发放率 | X02 | 跨 session 传输质量 |
| S03 | 丢包率（window_index、接收间隔、神经缺口、FPGA 时钟丢帧） | X03 | 跨天 TDR + variance |
| S04 | 降维分析 PCA / UMAP / TDR（总览含 TDR 角度误差；随后详细分析） | X04 | 跨天方向解码准确性 |
| S05 | ANOVA（八方向调谐、BH-FDR） | X05 | 跨天速度解码 R² / RMSE（多 decoder） |
| S06 | PSTH | X06 | 离线模型跨天迁移（多 decoder） |
| S07 | 方向/时间调谐重叠、配置层级组成与八方向基线响应变化 | | |
| S08 | Compensation 通道筛选影响（configs.ini good/uncertain/bad） | | |
| S09 | 方向解码 Overview + examples | | |
| S10 | 速度解码 Overview + examples（多 decoder 对比图） | | |
| S11 | 在线闭环 decoder 结果复现与离线对比（多 decoder 对比图） | | |
| S12 | 单日一图流：通道活跃度 → 丢包率 → 降维 → 方向/时间-tuned → 解码 | | |

S12 首页按五大模块组织：①全记录平均率分布、活跃数量及码 3 占比；②接收/FPGA 缺口及 trial 窗口暴露；③PCA 方差与投影、UMAP、held-out TDR；④Direction / pooled Time / Both、偏好方向及方向内时间响应；⑤方向或 Grid 目标解码、速度指标和轨迹。在线 vs 离线的闭环对比归入第五模块，数据/筛选背景留在页头，通道子集解码影响移至补充页。S04/S05/S06 的相关设置自动继承，S07 方向内检验 helper 复用，详细控制检验仍看来源 section。全记录 ≥1 Hz 的活跃描述阈值与 S05/S06 active 定义明确区分；缺失/不可计算项不填零。

分析约定：接收时间戳不同步、不去抖；20 ms 计数 / 0.020 s 换算名义 Hz；计数为 2 bit，3 表示 ≥3；
交叉验证按时间顺序把 block 分 5 段，训练集去掉测试段两侧相邻 block；置换在 block 内打乱方向；
跨天比较按电极 ID 对齐（不同日期传输的 512/1024 电极不同）。跨记录结果是描述性比较。

S03 首先展示传输状态总览、相对投递延迟、按时间和 trial 的影响；原始间隔分布、时钟拟合、完整延迟图及逐缺口放大在后续 Supplementary 标签中。无 FPGA 时钟时缺帧和相对延迟标为未知；时钟间断不能独立定位丢失发生在哪个环节。

S07 前两组图汇总 S05 普通方向 ANOVA 和 S06 时间响应检验：Direction/Time 均包含 Both，分别报告占全部传输通道（512）和 Active 的比例。此处 Active 为 S05/S06 各自活跃集合的并集，原有基线变化图保留自身定义。有当日 configs.ini 时增加 Good / Uncertain / Bad / Unlisted 的互斥堆叠及重叠比例，空组为 N/A。GUI 自动继承 S05/S06 当前参数；CLI 可用 `--section s07 --param s05:alpha=0.01 --param s06:bin=0.1` 指定来源设置。来源参数、版本及代码一并参与签名，单独运行 S07 也能生成统计，无需依赖先前保存的图。

S07 第三组图按八方向统计唯一偏好方向、方向内时间响应以及两者交集；同时给出 S05 全局方向检验显著与方向内时间响应的交集，避免将全局 ANOVA 解释成逐方向选择性检验。方向内时间检验沿用 S06 设置，但独立按方向分组，bin 内 Bonferroni 后对全部可检验 channel×direction 对共同做 BH；每方向/每 bin 至少 8 个有效配对 trial，不足时标 N/A。第四组为显著通道 ANOVA/PSTH 配对分页图，每页 4×8 通道、每通道左右两轴；选取 S05 或 pooled S06 显著并集，PSTH 仅加粗方向内时间检验显著的曲线，ANOVA 独立标出偏好方向。第五组为真实通道探索性 PSTH 示例（相近窗口均值但峰时不同、时间响应但 ANOVA 未显著、方向与时间响应共同出现），无符合实例时明确显示缺失；不以峰时差直接宣称交互显著或可解码。原有基线变化图顺延至补充页。

S05 和 S06 分别提供“全部通道 / 显著通道”两套 8×8 分页图：S05 为八方向平均 Hz±SEM 调谐曲线，S06 为每通道八方向 PSTH。全部通道按物理电极 ID，显著通道按当前 section 的 q、原始 p、电极 ID 排序；S06 使用自身时间响应显著性，不替换成方向 ANOVA。每个子图独立 y 范围且显示自身刻度，不能直接按图形高度比较跨通道幅度。每套在 GUI 占一个标签页，用 Prev/Next 翻页，支持每页导出及“整组 PDF”。

S11 可通过 `online_model_dir` 指定保存的在线模型目录，留空时在当前 XDF 所在当天目录查找唯一更新 session。按实际 applied 日志加载初始与更新权重，核对窗口序号、神经计数、LSL 时间及在线输出后才关联；初始权重历史路径支持迁移到当天目录，亦可用 `initial_model_path` 显式指定。默认 `saved_position_feedback=recorded` 使用因果对齐的 XDF 位置反馈（最大年龄 `feedback_max_age=0.25` s），`decoded` 为仅 trial 起点位置初始化的全 KF 对照。所有实际模型、切换日志和 block 数据参与缓存签名，首图展示保存权重回放与记录输出的逐帧及分阶段误差；不做事后拟合或输出校正。当前支持 `model_type=position_velocity` 矩阵文件，不将其当作无状态 Madduri D/b 权重。原在线程序缺失时，图中明确标注重置/切换/后处理语义未验证，不能把高相关或高 R² 称为精确复现。2026-11-02 的 `keaiduo_1102_run003_512_lag6` 是训练命名，其日志与 `run-004_spike_online.xdf` 对应；`run-003` XDF 本身没有在线 decoder 流。

S04 首图为两行三列 Overview，三列依次 PCA、UMAP、TDR：上排为 PC1–2、UMAP 方向、held-out TDR 方向平面；下排为样本内残差 PCA、同一 UMAP 按 block 着色、TDR 角度误差散点图。普通/残差 PC1/PC2 的解释方差直接写在各自轴标签，残差比例以重新标准化后的残差方差为分母；补充页新增逐折 held-out 残差 PCA，通道选择、β、标准化和 PCA 均仅在训练集拟合，不混合不同折坐标系。随后每种方法各一页详细主图；载荷、PCA held-out 投影、TDR 样本内及读出细节保留为 Supplementary。
PCA 展示方差谱、PC1–PC2 / PC2–PC3、各 PC 与总发放率和真实经过时间的关系，以及回归掉整体增益后的方向结构和 block 内 max-F 置换对照，不预设 PC1 是增益轴。
UMAP 默认启用，同一嵌入按方向、block、总发放率着色，并展示相对 PCA2 的邻域可信度（最多固定种子抽样评价 2000 个 trial）；可调整邻居数（默认 15）、最小距离（默认 0.1）和种子。首次计算可能需要编译；该图为探索结果，不作为解码指标。
TDR 主图展示 held-out 平面、描述性角误差、置换检验、折间轴稳定性，以及在相同速度合格 trial 上加入实测 vx/vy/平均速率前后的对照（至少 80 个 trial）。时间项为实际经过秒数；在线 cursor 为闭环输出，控制结果不能解释为独立运动学验证。所有方向 centroid 投影均包含 1 SD 协方差椭圆。

## Decoder（`data_analysis_gui/decoders/`，每个 decoder 一个脚本）

| 脚本 | Decoder | 来源 |
| --- | --- | --- |
| `ridge.py` | Ridge 回归 | 本地 numpy |
| `velocity_kf.py` | 速度 Kalman filter（观测协方差向对角收缩） | 本地 numpy |
| `refit_kf_filter.py` | ReFIT-KF（SDK `standalone_refit_kf`，Stage-1 监督拟合 + SDK Runtime 回放） | vendored `sdk_b70297c` |
| `madduri_coadaptive_decoder.py` | Madduri SmoothBatch | vendored `sdk_b70297c` |
| `ecog_linear_velocity_decoder.py` | ECoG-linear（用其公开拟合函数处理 spike rate） | vendored `sdk_8d79df0` |

- `vendor/sdk_b70297c/`：`bci-decoder-api @ git+ssh://git@github.com/sillybun/bci-decoder-sdk.git@b70297c092157a0627fa8d5f97f1694d6043e2bc`
- `vendor/sdk_8d79df0/`：`bci-decoder-api @ git+ssh://git@github.com/sillybun/bci-decoder-sdk.git@8d79df0475b85ca859a02a8268f5262358a22afa`
- 快照逐文件原样复制，文件顶部注明仓库、完整 commit SHA、原路径和原 git blob SHA；唯一改动是把
  `from bci_decoder_api` 改为包内相对导入，使两个 SDK 版本可以共存（见各目录 `VENDOR.json`）。
  `tests/test_decoders.py` 去掉文件头、还原 import 行后核对 blob SHA；有本地 SDK 仓库时再与 `git show` 逐字比较。
  重新生成：`uv run --locked python -m data_analysis_gui.decoders.vendor.vendor_tool E:/AitBCI/bci-decoder-sdk`。
- S10、S11、S12、X05 和 X06 的速度模型比较固定使用 Ridge 与 Vel-KF。X05 在记录总览后展示 Velocity / Position、Velocity X / Velocity Y 两页，误差线为按 block 重采样的 95% 区间。
  ECoG-linear 原为 ECoG 频带功率设计，SDK Trainer 不接受 spike 特征，结果不代表 ECoG 表现。

## 代码结构

```text
data-analysis-gui/
├── app.py                         # GUI 入口
├── tool.toml / pyproject.toml / uv.lock
├── data_analysis_gui/
│   ├── config.py                  # 路径、常量、偏好
│   ├── core/                      # XDF 读取、目录扫描、trial 重建、筛选规范、特征、统计、布局、跨天对齐
│   ├── decoders/                  # 每个 decoder 一个脚本 + vendor/ SDK 快照
│   ├── sections/
│   │   ├── base.py                # Section / Param / AnalysisContext / SectionResult(add_figure(method, data))
│   │   ├── s01_raw_visualization_v1/section.py
│   │   ├── …                      # 每个 section 一个 <id>_<name>_v<N>/ 目录
│   │   └── x06_offline_model_transfer_v1/section.py
│   ├── storage.py                 # 签名、缓存查找与核验、PNG/SVG/PDF/CSV 导出
│   ├── worker.py                  # 计算子进程（GUI 与 CLI 共用）
│   ├── gui/                       # app.py、filter_dialog.py、widgets.py、canvas.py、theme.py
│   ├── runner.py / cli.py
└── tests/
```

**每个 section 的说明（`guide.md`）**：每个 section 目录里有一份中文说明。
开头用一句话概述分析目的；逐图以「计算公式 / 结果解读 / 质量评价标准」解释每个 subplot，并区分统计判据、配置阈值与描述性参考。计算后结果页顶部和 Summary 会显示基于实际指标的一句话摘要；旧缓存也可从现有指标生成摘要。GUI 中每个 section 都有「说明」标签页（计算前也可查看），每张图的工具栏有「说明」按钮只显示该图对应的一节；
导出结果时同时保存一份 `GUIDE.md`。格式：`## 图 N：<add_figure 的标题>` 下依次是 `### 计算公式`、`### 结果解读`、`### 质量评价标准`；
`tests/test_sections.py` 检查每张图都有对应说明、结果解读与质量标准的面板分组一一对应。说明不进入结果签名，修改说明不会触发重算。

**S15 任务回放**：选择一份 CO8 / Grid 记录并计算 S15，在结果的「任务回放」页播放，默认 1× 实时。支持暂停、倍速、进度拖动、上一/下一 trial、轨迹尾迹和保存速度箭头。任务布局读取与这份 XDF 唯一匹配的 `task_config.json` / `logs/task.log`；位置、速度、目标、trial 状态使用已加载的 XDF cursor_state 全段，不受成功 trial 或神经筛选限制。短且位置连续的 Grid trial 切换平滑衔接，真实复位和长缺口不插值。BPS按当前trial直接读取任务`rolling_metrics.csv`的保存结果，包括片段开始前的窗口历史，不从回放起点累计、不重算，缺失显示N/A。结果目录的 `attachments/task_replay.html` 可独立双击打开，JSON/HTML 也进入哈希校验和批次导出；原始窗口未记录时会注明估计范围。

S15 的「回放片段」可切换全程、BPS最高60s、该窗口前后各60s。最高窗口优先使用任务 `rolling_metrics.csv` 保存的完整60秒 achieved bitrate；只在本份录制覆盖范围内选择，并列取最早。它使用真实时间轴，包含ITI和反馈，不等同于 S13 的累积活动时间窗口。180秒范围遇到录制边界会明确标注实际时长。CLI 可用 `--section s15 --param s15:replay_range=BPS最高60s` 设置默认片段，生成后仍可在 GUI/HTML 内切换。

**新增或修改 section**：复制目录并把版本号加一（例如 `s05_anova_v2/`），在 `section.py` 中定义 `SECTION = ...`。
注册表自动发现目录，默认显示最高版本，旧版本可在「版本」下拉中切回。每张图用
`result.add_figure(title, fig, method='计算方式…', data={'表名': rows})` 注册；只用 matplotlib 面向对象接口。

## 命令行与测试

```bash
uv run --locked python -m data_analysis_gui.cli --list
uv run --locked python -m data_analysis_gui.cli --section s01 --match 20261026            # 写入结果存储目录，已存在则直接读取
uv run --locked python -m data_analysis_gui.cli --section x04 --match 20261029 --match 20261030 --recompute
uv run --locked python -m data_analysis_gui.cli --section s10 --param s10:decoders=ridge,velocity_kf,refit_kf_filter --match 20261029
uv run --locked python -m data_analysis_gui.cli --section s08 --match 20261102     # 读取 20261102/configs.ini
uv run --locked pytest -q            # 核心、筛选规范、decoder（含 vendor 完整性）、全部 section 合成数据端到端
uv run --locked python tests/ui_smoke.py .local/ui   # 需桌面：单日 / 筛选弹窗 / 跨天，经子进程计算并从存储读取
```

## 已知限制

- 只支持 512 列、20 ms 窗口的 `spike_counts` 及 CursorTasks 行为流；其他格式在记录列表中显示原因。
- FPGA 丢帧分析需要 metadata 第 5 通道 `device_timestamp_100us`（10-29 之后的记录）；更早记录只做接收端检查。
- 本地速度 KF 与 SDK ReFIT-KF（状态含位置）不是同一模型；ReFIT 的 `adapt()` 需要闭环数据，未运行。
- Section 11 复现在线 decoder 采用从记录的神经帧到 XDF decoder 输出的线性拟合，不读取在线模型文件。
- 目前只有 20261102 有 configs.ini。该文件的 good 列表中有 83 个电极未出现在 run-003 的传输通道里，列顺序也不是 good→uncertain→bad：它可能是之后重新编程时保存的版本，S08 会如实标注。
- 目前在 Windows 11 上验证。

### 在线滚动 BPS

S13保留原任务block成功率和到靶时间统计。所有BPS使用累计60秒有效尝试时间滚动窗口，按1秒步长更新。ITI、block休息、反馈等待从时间轴中排除；选择dwell、失败和超时耗时保留。长run从第60秒开始；仅整个run不足60秒时按实际有效时长折算，缺失计时或trial断档会重置窗口，不跨session/录制片段。

左边界trial只累计落在窗口内的耗时，选择结局按完成时刻计数。Grid仅画C×log₂(N)/60。CO8选择BPS采用同一T=60秒；Fitts保持其独立公式，用同一60秒范围内完整成功trial的难度/外向运动时间，边界不完整trial不参与Fitts。

X08按run汇总60秒、step 1秒窗口的median/IQR和peak（整个run不足60秒时用实际全程窗口折算）；Grid采用横向A4比例上下两个子图并共用run横轴：上图为蓝色BPS分布及橙色peak，下图为橘色成功trial到靶时间median/IQR（秒），均保留数值标注。峰值明细导出有效时间范围、trial范围。已不再输出原先按block或整场计算的BPS图；成功率和到靶时间保持原统计口径。一键报告使用相同规则。

一键报告追加离线行为目录 `09_离线行为_Grid_BPS_TTT` / `10_离线行为_CO8_BPS_TTT`：直接读取所选 run 的 trials.csv，可在日期范围报告中纳入无 XDF 的 run。每 run 输出 block 成功率/TTT、完整 60 秒有效任务时间窗口 BPS、成功 trial 的 TTT 分布；summary_all_runs 汇总各 run 的 BPS median/IQR/peak 和 TTT。空记录标为不可计算；不足 60 秒的整个 run 按实际时长 ratio 折算，长 run 从第 60 秒起按 1 秒步长滑动。电极位置图和 elec_index 统一为 1–128；Shank 保留 1–8，深度轴仍使用 µm。

导出字号固定：分析图标题 18 pt、坐标及色条标签 14 pt、刻度与图例 11 pt，不随旧字号百分比叠加。记录/实验范式示意图（X00、Session Overview）保留原有样式，不套用此规则；长标题自动换行。PNG、SVG、PDF 和一键报告共用该导出规则。

一键报告默认导出各分类下的 `summary_all_runs`，并附带每日 BGA 概览。每日 BGA 按 run 编号顺序查找当天首个具有可读 XDF 神经计数数据的 run，跳过无 XDF、损坏或没有神经计数的记录；图标题显示实际 run 编号，manifest 记录来源及跳过原因。当天所有 run 均无可用 XDF 时才显示无数据，排除日期仍不导出。日 BGA 不受“导出每个 run 的图”开关影响；勾选后才附带其余逐 run 分析图和行为图。汇总始终使用全部所选 run 的有效数据；此开关不改变统计口径。默认模式会复用已有缓存，缺少缓存时只保存单 run 的计算表和摘要，不保存其余逐 run 图文件。界面复选框及记录勾选列的选中标记统一为 √。

一键报告共用 `report/individual_runs` 单 run 缓存池：S05/S14/S10/S13 原分析结果、S05/S14 深度图、BGA 数据和图、BPS/TTT 数据和图分别按输入内容与设置建立缓存。再次导出相同 run 时先读取并校验缓存；原单记录缓存目录移除后，也可以从池子恢复读取。文件采用 SHA-256 校验，数据、mapping、参数或相关代码变化只重建对应缓存；“强制重算”会绕过池子重新生成。报告通过硬链接引用不可变的池子文件，同盘不重复占用图文件空间，不支持硬链接时才复制。跨 run 的 `summary_all_runs` 始终按本次选择重新汇总，不复用其他选择的汇总数值。已有报告不会因缓存更新而被覆盖。

缓存池先按 section 分类（`S05`、`S14`、`BGA`、`BPS_TTT` 等），再按 `日期_run编号_任务` 分类。每个 run 下按分析类型和输入/参数版本保存；版本内 `figures/` 存放图片，`data/` 存放 CSV、计算结果和绘图数据，`entry.json` 记录来源、参数和校验信息。打开该 run 的 `index.html` 可浏览图片并获取生成数据；`index.json` 提供机器可读索引。不同设置的版本分别保留，避免错误复用。

X10「脑控与离线摇杆 BPS / TTT 配置对照」读取所选日期范围内的正式编号 run，包含只有 CSV 的日期，默认排除 1116。蓝色圆点为脑控、橙色方点为离线摇杆；Grid 与 CO8 分开比较。BPS 采用共用 60 秒有效尝试时长、1 秒步长缓存；TTT 对成功 trial 仅计一次。一键报告新增 `11_脑控与离线摇杆_BPS_TTT/summary_all_runs`，包含一张 Grid 与 CO8 共用横轴的全 run BPS / TTT 图、逐 run 统计 Excel、当天配置变化、全部字段完整配置对比（相同设置也保留），用日期/任务/run 和图编号对应。总图使用各任务原有 BPS 口径，CO8 采用 Neuralink-style，Wolpaw 和 Fitts 保留在统计表。Excel 由 Codex bundled Artifact Tool 导出，支持 `AITBCI_ARTIFACT_NODE` / `AITBCI_ARTIFACT_MODULES` 路径覆盖。

## Keaiduo 命名与配置审计

顶栏「重命名文件夹…」可选择当前数据目录内的子文件夹，同时查看原名和建议名。
可直接编辑建议名，或在备注栏追加内容，确认后刷新记录列表。操作只更改所选文件夹的名称，
内部文件名与配置文件保持原样。正在执行或等待分析时，先结束相关任务，避免队列引用旧路径。

批量核对时可先生成预览：

```powershell
uv run --locked python scripts/audit_keaiduo_archive.py --data D:/AitBCI/keaiduo --out D:/AitBCI/data_analysis_gui_output/keaiduo/naming_config_review
```

审计输出逐 run 命名、XDF 信号证据、配置原始字段和逐份配置新增／缺失／值变化。已有正式 run 编号和拆分字母保留，
其余会话按日期内目录顺序预分配空闲编号。无配置、读不到信号流或出现命名冲突时，先核对审计结果。
明确标注 LFP 的记录保留 LFP；确认有 spike 计数流的 XDF 将文件名中的 eeg 改为 spike。审计命令只读原始数据。
归档 XDF 在原目录内使用短名，例如 `xdf/rec-001-spike.xdf` 或 `xdf/rec-001-lfp.xdf`，
同一 run 有多份 XDF 时递增录制序号，特殊分段可保留 `early`、`startup` 或 `aux`。
完整实验条件和日期由上层目录提供，在线状态还会结合 decoder 流。迁移时同步归档元数据中的录制路径，
原始来源字段保留。将数据目录复制到另一台电脑时一起复制 `.xdf_name_map.json`，供旧目录结构恢复历史 run 身份。
短文件名会产生新的结果缓存签名，既有结果和报告保留；报告的 `xdf_paths_after_rename.csv` 可用来查找迁移后的本机来源。
运行日志使用 `<run名>--<原日志文件名>`，GUI 兼容迁移前后的日志命名。

命名基础为 `run-NNN-任务-online/offline`，标签依次为尺寸、`sp`、`bc`、`dw`、`nextT`、`nearby`、`ass`。Grid 尺寸取 `GRID_ROWS` / `GRID_COLS`，`sp021` 取
`CONTROL.POS_SCALE_FACTOR=0.021`，`nearby3` 表示邻近辅助开启且 `NEARBY_TARGET_ASSIST_RADIUS=3`（模式 3，7×7）。模式 1、2、3 分别对应 3×3、5×5、7×7。
`dw001` 表示 `DWELL_TIME_SEC=0.01` 秒，`dw020` 表示 0.20 秒，已保存的 dwell 值都显示。`nextT` 表示 `PREVIEW_NEXT_TARGET=true`，
`bc125` 表示在线 `BCI_VELOCITY_SCALE=1.25`，1 倍省略。`ass150` 表示 `BCI_TARGET_ASSIST_ENABLED=true` 且 `BCI_TARGET_ASSIST_SPEED=150.0` px/s，目标辅助关闭时省略。配置未保存的字段保留缺失状态。
X10 Excel 将这些字段放在行为指标之前，保留完整配置，并按时间列出新增、缺失和变化。
物理布局可继续使用外部 `天枢3.0_ChannelMap20261122.xlsx`，SHA-256 为
`5164b80e417a1b42ec89756b0a67e553f19c9064a829984110bf1fb2a856a2df`。命名与配置审计本身不需要布局表。


### 最终通道映射与逐日汇总

GUI 默认从数据根目录的 `channel_layout/TS3.0_ChannelMap_Final.xlsx` 读取物理布局。一键报告要求此表并使用其总表的 Shank 和柄内位点，以及「Fanin芯片BGA位点对应」工作表的 BGA 位置。本机文件 SHA-256 为 `459e3e845b2b6a7adecb15f3d0783fe7c69a78540d4941ffbece22642516ceec`，报告清单记录实际使用的校验值。物理位点按工作簿编号显示为 1–128，Shank 显示为 1–8。轴向图使用工作簿 Y 坐标距各 Shank 第一个位点的距离（µm），属于芯片几何坐标。GUI 中可显式选择其他布局进行对照。

图中的 run 标签按单行显示，长横轴标签竖排。S05 和 S14 新增 `summary_by_day/YYYY-MM-DD`。同一 run 的多个 XDF 先合并到 run 均值，再按日期等权汇总。S05 仅汇总有效检验的二元显著性。S14 对每个日期、通道和权重指标，在至少 4 个有限 run 值且 IQR 大于 0 时剔除 1.5×IQR 界限之外的观测；`excluded_outliers.csv` 记录剔除项，`channel_statistics.csv` 提供均值、中位数和有效 run 数。


### 中文图表导出验收

- 在导出电脑上安装支持中文的字体，Windows 优先使用 Microsoft YaHei，其他环境可安装 Noto Sans CJK SC。安装后重新启动 GUI 或报告 worker，让 Matplotlib 重新扫描系统字体。
- PNG、PDF 和 SVG 导出遇到缺字会立即失败并给出字符和目标路径。导出写入临时文件，通过后才替换目标图。中文 SVG 使用字形轮廓，接收方打开图时无需安装同款字体。
- 一键报告对照目录保留 Excel 和一张总图。验收时打开工作簿核对中文表头及说明。
- 修改字体、绘图或运行环境后运行 `python -m pytest -q tests/test_fixed_font_policy.py tests/test_control_comparison.py`，并抽查最长 run 标签与中文说明之间的留白。出现缺字时修复导出机字体后重跑失败项。

离线解码汇总图 `Velocity.png` 额外要求各录制通过 S10 筛选的 trial 数大于 50，并且至少一个模型的 R² 有效。原始行为 CSV 的 trial 数用于一键报告的 run 入选检查。

Grid 的 BPS / TTT run 图按日期分段，下方表格按 run 对齐保存的配置。Cursor Speed 以归档中的 POS_SCALE_FACTOR 0.012 为 ×1；BC Speed 以 1.00 为 ×1；Nearby 半径 1、2、3 分别显示 3×3、5×5、7×7；在线图列出开启时的初始 Assist Speed（px/s）。配置未记录的项目显示 —。2026-11-11 归档缺少 Next Target 和 Nearby Target 标志时，按已确认的设置显示 Off。


一键报告也会在各分类的 `summary_all_runs` 导出每日方向调谐数量箱线图、单独的 Shank 位点图和离线双向定位的 BGA 映射 HTML、X/Y R² 独立趋势及 Vel-KF 双线趋势（按 run、按天、按天半宽），以及在线 Grid 的 Assist Off 图。R² 趋势沿用 S10 有效 trial 数大于 50 的筛选，同一 run 的有效录制先等权合并；逐日图显示中位数和 Q1–Q3。Assist Off 图只使用配置明确关闭辅助的 run。补图路径记录在 `report.json` 的 `supplementary_files`。S14 继续按选中日期生成逐日汇总，历史单独补画的 2026-11-10 不会进入未选中该日的新报告。
