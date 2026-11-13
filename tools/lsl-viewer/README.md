# LSL Viewer

独立的 Lab Streaming Layer 信号观测器。它发现当前可见的全部 LSL 流，显示广播源、流类型、通道数、格式、XML 元数据、实时样本和观测频率。

窗口右上角的 `EN` / `ZH` 按钮可即时切换中英文界面。语言切换只更新展示文案，不会重建 LSL inlet 或阻塞后台发现线程。手动刷新使用带本地化悬停提示的 `↻` 图标；操作按钮和左右主面板使用无边框圆角矩形，列表横纵滚动条统一为无边框、淡色、圆角的 Canvas 样式。

## 运行

```bash
uv run --locked lsl-viewer
```

状态颜色：

- 绿色 `LIVE`：有名义频率且最近仍收到数据，同时显示观测频率。
- 黄色 `LOST`：曾经收到周期数据，但最近已超时。
- 蓝色 `DATA`：无名义频率或不规则数据流。

## 说明

LSL 的流发现依赖本机网络可达性和 pylsl/liblsl。默认仅创建 inlet 读取预览数据，不发布或修改 LSL 样本。对匹配到的本机发布进程，界面另提供需要确认的终止操作。

默认列表和状态刷新频率为 **5 Hz**（每 200 ms 一次）。LSL 发现、XML 元数据解析、样本拉取与本机发布进程识别由单一后台工作线程独占，Tk 主线程只读取完成后的快照；“手动刷新”和“刷新进程识别”只发送异步刷新请求。每轮发现调用最多等待 100 ms，每条流每轮最多拉取 512 个样本，避免持续高频流让单轮解析无限延长。LSL 发现可能在网络或发布器启动时短暂返回不完整快照，因此流需要连续缺失 2 秒才会从列表移除并关闭对应 inlet，避免把常驻流误删。LIVE/LOST/WAIT 状态在每个刷新周期重新计算。

每条流默认保留最近 **5 秒**的样本历史。样本按 LSL 时间戳排序并按时间戳去重；延迟到达或重连后从 inlet 缓冲区恢复的样本会插入正确位置。观测频率使用该历史窗口的时间跨度计算，而不是简单按 UI 刷新次数计数。

Viewer 会读取 Integrated Term 的系统级活动进程列表，用流的 `name` 和 `type` 为当前信号匹配最可能的本机管理进程并绑定 PID。活动列表只接受当前仍存在的本机 PID，以避免崩溃遗留条目造成误绑定；没有 registry 或本机命令行候选时，发布进程显示为“来自其他电脑节点”。

活动列表在 macOS 使用 `~/Library/Application Support/BCIControl/active_lsl_processes.json`，Windows 使用 `%APPDATA%\BCIControl\active_lsl_processes.json`，Linux 使用 `$XDG_CONFIG_HOME/BCIControl/active_lsl_processes.json` 或 `~/.config/BCIControl/active_lsl_processes.json`。

## 无硬件示例与测试

在另一个终端运行 `uv run --locked python examples/demo_stream.py`，观测器应发现
`AIT-Demo`（Demo，1 通道，名义 10 Hz），展示样本 0–99；示例十秒后自动退出。

```bash
uv run --locked --extra dev python -m pytest -q
```

macOS 原生 Tk 窗口与模拟流在本次接入时验证；Windows/Linux 仍需原生验证。
使用 `.python-version` 指定的 Python 3.11.15。若 pylsl 找不到 liblsl，先安装对应平台的
liblsl，并按 pylsl 提示设置 `PYLSL_LIB`。进程终止按钮会要求确认，请勿用于非测试进程。
