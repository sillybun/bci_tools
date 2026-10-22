# 迁移记录

- 日期：2026-10-22。
- 原仓库：`git@github.com:sillybun/coadptation-emg-bci-cursor-control-NMI.git`。
- 来源分支：`feat/emg-bipolar32-decoder`。
- 固定来源提交：`ed07a2f3669ffeabf274b9b6bd2078138750c616`。
- 原入口：`scripts/emg_drone_task/monitor_raw_emg_stream.py`。
- 新入口：`tools/emg-signal-monitor/app.py`。

从 Git 对象读取源码；原项目工作目录、分支及未提交修改未改动。

## 范围

- Qt 入口迁为 `app.py`，接收/显示后端迁为 `raw_stream_monitor.py`。
- 保留 `packet.py`、`lsl_discovery.py`、`stream_metrics.py`；导入调整为工具内部模块。
- 从 `bipolar32.py` 提取固定 SAGA 标签、配对和 `Bipolar32Montage` 到 `montage.py`；不引入训练预处理或解码器。
- 迁入原分包和监控台测试，调整本地导入与入口路径。
- 新增独立依赖、锁文件、启动台配置、使用文档，以及真实 LSL / 原生窗口冒烟脚本。

依赖为 NumPy、SciPy、PySide6 Essentials、pylsl；不依赖原项目路径、PsychoPy 或 Decoder SDK。来源提交未发现仓库根目录许可文件；保留提取代码中的注释和来源记录。

## 验证记录

- macOS 独立环境：Python 3.11.15、NumPy 2.4.6、SciPy 1.17.1、PySide6 Essentials 6.12.0、pylsl 1.18.6，均由 `uv.lock` 固定。
- 工具测试 28 项通过（含原有分包/监控测试与旧连接排队事件的回归测试）；启动台测试 13 项通过。
- 隔离本机 LSL 流：接收 101 包、每包 2560 个值、20 ms 时间戳间隔，无无效包/乱序/间隙。
- 与工具箱现有 Mock EMG Data 发布器跨独立 Python 环境联调：接收 102 包，无无效包/乱序/间隙。
- 原生 Qt 窗口验证：真实 LSL 接收、Raw/Bipolar 切换、滤波关闭及改参、断开重连和正常关闭；检查主界面和设置窗口渲染图。
- 原生启动台验证：发现和搜索新卡片、按配置运行 `uv run --locked python app.py`、运行时防重复启动。
- 本次未验证 Windows/Linux、跨机器网络或真实采集设备。

## 提取后的修复

- 将拥挤的工具栏拆为两行，完整显示双极模式选项与分页信息。
- 每次连接使用独立标识过滤旧线程排队信号，避免快速重连时旧线程的结束/数据/错误事件覆盖新连接状态；自动回归及原生重连验证通过。
