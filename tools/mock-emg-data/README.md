# Mock EMG Data

通过鼠标拖动弹簧小球生成模拟肌电，以 LSL 数据流发送。适合调试数据接收、可视化及解码链路，无需连接肌电设备、XDF 文件或原来的实验仓库。

## 启动

在 BCI Tools 启动台中选择“采集与连接”→“Mock EMG Data”→“打开”。已有启动台窗口点击右上角刷新图标即可发现。

也可在本工具目录直接运行：

```bash
uv run --locked python app.py
```

工具使用独立的 Python 3.11.15 环境，依赖 NumPy、PySide6 Essentials 和 pylsl。首次启动需要下载依赖。窗口打开后自动开始发送模拟数据。

## 操作

- 按住蓝色小球拖动：右为 +x，上为 +y，按住时保持位置。
- 松开鼠标或切换窗口：小球平滑回到中心。
- **Center Ball**：立即回中，继续发送静息载波和噪声。
- **Stop Stream / Start Stream**：停止或重新开始发送。
- 右侧 **Source Parameters**：修改幅度、噪声、回中时间、随机种子或 Source ID，点击 **Apply Parameters**（或在输入框按回车）生效。
- 左上角侧栏按钮：切换完整参数视图与紧凑小球视图。
- 关闭窗口：停止数据流并退出；仅关闭工具箱不会停止这个工具。

应用参数时，正在发送的数据流会先停止，再按新参数重新启动；计数、载波相位和噪声序列从头开始。已停止的流保持停止，下次点击 **Start Stream** 使用新参数。小球当前位置连续，回中速度按新的时间常数变化。

输入错误会显示在参数区，原有参数和数据流继续使用。幅度和噪声须为有限非负数，回中时间须大于 0，随机种子须为非负整数，Source ID 不能为空。设置仅在当前窗口会话内保留。更换 Source ID 后，接收端需要选择新的源。

## 参数与启动默认值

```bash
uv run --locked python app.py --mapping-norm 100 --noise-level 5 --return-tau 1.0 --seed 42
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--mapping-norm` | `100` | 四方向幅度映射矩阵的 Frobenius 范数 |
| `--noise-level` | `5` | 每通道、每采样点独立高斯噪声标准差 |
| `--return-tau` | `1.0` | 回中时间常数，单位秒 |
| `--channels` | `64` | 固定 64 通道，沿用 SAGA 顺序与 Bipolar-32 配对 |
| `--seed` | `42` | 载波初相和噪声的随机种子 |
| `--source-id` | `mouse_mock_emg_raw` | LSL 数据源标识；多实例时分别指定唯一值 |

命令行参数决定窗口的初始值，打开后可在侧栏修改。通道数、采样率、包大小和载波频率保持固定。运行 `uv run --locked python app.py --help` 查看说明。

## 输出协议

| 项目 | 值 |
| --- | --- |
| LSL stream name | `emg_raw_signed_2khz` |
| LSL stream type | `EMG` |
| 物理通道数 / 采样率 | 64 / 2000 Hz |
| 数据包 | 每通道 40 点，每 20 ms 一包 |
| LSL channel count / nominal rate | 2560 / 50 Hz |
| 数值布局 | `float32[64,40]` 按 NumPy C-order 展平为 2560 个值 |
| 时间戳 | 包内最后一个采样点的 LSL 时间 |

信号沿用原工具的四方向幅度编码与连续 80 Hz 载波，中心位置保留静息信号。调度与时间戳统一使用 `pylsl.local_clock()`，发送延迟时跳过过期时隙。

这是合成数据源，数值不代表实测微伏。`Streaming` 表示源正在发送，接收端是否连接及解码结果由下游工具确认。

## 验证

```bash
uv run --locked python -m pytest tests -q
uv run --locked python tests/lsl_smoke.py
uv run --locked python tests/ui_smoke.py
```

第一条运行信号、时间戳与 Qt 离屏测试，包含参数校验、运行中应用、停止时应用及应用期间关闭窗口。第二条使用独立的测试 source ID，在本机创建真实 LSL outlet/inlet，检查 100 个数据包。第三条需桌面环境，验证参数应用后的真实 LSL 重启、接收 20 包新数据和侧栏切换。后两项结束后均关闭测试数据流。

如窗口显示 liblsl 加载错误，检查当前平台的 pylsl 原生库支持；加载错误会显示在底栏和启动台日志中。Windows/Linux 仍需在对应系统上验证。

代码来源和提取范围见 [迁移记录](MIGRATION.md)。
