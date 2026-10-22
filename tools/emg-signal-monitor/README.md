# EMG 信号调试

实时查看 LSL 分包肌电，支持原始通道、signed Bipolar-32 双极差分、在线带通显示和接收状态诊断。用于采集链路检查与 EMG 信号调试。

## 启动

在 BCI Tools 启动台选择“信号查看 → EMG 信号调试”。也可在本工具目录运行：

```bash
uv run --locked python app.py
```

首次启动需要 uv 下载 Python 和依赖；本工具使用独立 Python 3.11 环境。支持状态以迁移记录中的实际验证为准，Windows/Linux 仍需原生验证。

## 和模拟肌电源配合

1. 在启动台打开 **Mock EMG Data**，保持默认 64 通道并开始发送。
2. 打开 **EMG 信号调试**，默认自动连接 `emg_raw_signed_2khz`。
3. 拖动模拟源的小球，观察波形和差分结果。
4. 在监控台切换 Raw-64 / signed Bipolar-32，调节显示时间窗、幅度、通道分页；点击通道查看细节。
5. 在设置中调整显示带通或关闭滤波；断开、重新连接可检查流恢复。

同名流有多个时，使用设置中的 Source ID 指定目标，或启动时传入：

```bash
uv run --locked python app.py --source-id my-emg-source
```

## 输入协议与通道顺序

- 默认流名 `emg_raw_signed_2khz`，类型 `EMG`，LSL 格式 `float32`。
- 物理采样率固定为 **2000 Hz**，每包每通道 **40 点**，LSL 标称包率 **50 Hz**。
- 一个 LSL sample 是 C-order 展平的 `[C, 40]`；64 通道时长度为 **2560**。
- 包时间戳表示该包最后一个物理采样点；监控台据此重建包内时间轴。
- 这不是任意 LSL EMG 流的通用查看器：逐采样点发送、其他采样率或展平顺序需要先适配。

64 通道默认采用本工具 `montage.py` 中的 SAGA 存储顺序，与 Mock EMG Data 一致。原始分包协议不携带电极标签；接真实设备前，应核对发布端实际顺序。若不同，通过 `--channel-labels` 或设置输入按实际顺序排列的标签。

双极模式按标签做 `R1C1−R2C1` 等固定 32 对正端减负端。非完整 SAGA 标签布局禁用双极模式，仍可查看原始通道。不要仅凭“64 通道”认定真实设备的电极顺序正确。

## 显示与诊断

- 默认 signed Bipolar-32、每页 32 通道、5 秒可见窗、10 秒历史。
- 默认启用 1–200 Hz 因果 Butterworth 带通；可在设置中修改截止频率。
- 滤波有跨包状态；间隙、时间戳重叠和重新连接会重置状态。
- 查看包率、间隙、乱序、无效包、断开状态，以及波形统计量。
- 差分和滤波用于显示，原始历史保留；不改写源流，不发送控制命令。
- 波形幅值沿用输入数值，不自动换算成 µV；没有 CAR、notch、自动坏道剔除、录制或解码。

更多流身份、通道数及历史参数见 `uv run --locked python app.py --help`。

## 验证

```bash
uv run --locked python -m pytest
uv run --locked python tests/lsl_smoke.py
uv run --locked python tests/ui_smoke.py
```

自动化测试覆盖分包校验、双极配对、显示滤波状态、历史缓存和 Qt 界面。LSL 冒烟使用隔离名称的模拟流；UI 冒烟需桌面环境。

来源和本次验证结果见 [MIGRATION.md](MIGRATION.md)。
