# 迁移记录

- 迁移日期：2026-10-20。
- 原仓库：`git@github.com:sillybun/coadptation-emg-bci-cursor-control-NMI.git`。
- 来源分支：`chore/pre-refactor-emg-motion-control`。
- 来源提交：`16624ddd85c636822f07cebf6db2ea24df10190d`（`chore(emg-drone): park WIP mock mouse EMG source`）。
- 原入口：`python -m scripts.emg_drone_task.mock_mouse_emg_source`。
- 新入口：在 `tools/mock-emg-data/` 运行 `uv run --locked python app.py`。

提取时，原工作目录在 `feat/emg-drone-control`，其中没有鼠标源模块，并有与本工具无关的未提交修改。本次从上述 Git 提交读取源码，原工作目录及分支未改动。

## 提取范围

- `scripts/emg_drone_task/mock_mouse_emg_source.py` → `app.py`：保留 Qt 界面与 LSL 发布逻辑，调整本地导入及启动说明。
- `scripts/emg_drone_task/mouse_emg.py` → `mouse_emg.py`：保留运动、方向幅度、载波与噪声模型，调整本地导入。
- `trainer/emg_drone_task/packet.py` → `packet.py`：保留数据包校验与解码，加入原 `mock_signal.py` 的 `flatten_packet`。
- 原 `bipolar32.py` 的固定电极顺序、配对函数和 `Bipolar32Montage` → `montage.py`；只提取生成信号需要的部分。
- 原 `test_mock_mouse_emg_source.py` 的数据源测试 → `tests/`：调整导入路径。依赖 Fish Pursuit、训练器和 Decoder SDK 的端到端学习测试保留在原仓库，不纳入本工具。

运行依赖限于 NumPy、Qt Essentials 与 pylsl，环境独立于启动台。不迁入实验数据、解码器、训练流程或原项目完整依赖。

原提交标记为 WIP；本次验证范围以独立数据源、界面和本机 LSL 传输为准，不沿用原项目的闭环或解码验收结论。

## 后续调整

- 参数区增加幅度、噪声、回中时间、随机种子和 Source ID 的编辑与校验。
- 运行中应用参数时，等待旧发布线程关闭 outlet 后再启动新流；停止状态下仅更新配置。关闭窗口会取消待执行的重启。
- `SpringBall.set_tau()` 在锁内按旧时间常数计算当前位置，再切换时间常数，避免位置跳变。
- 输出协议、方向映射、80 Hz 载波和噪声生成算法沿用原实现。

## 验证记录

- 独立环境：Python 3.11.15、NumPy 2.4.6、Qt Essentials 6.12.0、pylsl 1.18.6；版本由 `uv.lock` 固定。
- 初次提取的 18 项源测试通过；加入参数编辑后共 32 项测试通过，覆盖新配置实际生成的数据、无效输入、无重复发布及关闭期间取消重启；启动台的 13 项测试通过。
- 初次提取时对照来源代码验证运行逻辑一致，仅调整导入路径和入口说明；此后按上述“后续调整”扩展界面与参数应用逻辑。
- 本机真实 LSL 收到 100 包，每包 2560 个值；相邻包时间戳间隔中位数 20 ms，最大包龄约 17 ms。
- 通过启动台卡片在工具自己的 `.venv` 中打开 Qt 窗口，验证发送、300×340 紧凑模式、恢复和正常退出；检查完整与紧凑界面的 Qt 渲染图。
- 参数编辑的 macOS 原生窗口检查通过：运行中应用新参数，保留 Source ID 重启，通过真实 LSL inlet 收到 20 包应用之后的新数据。
- 验证平台为 macOS。Windows/Linux、跨机器接收及实际解码器连接未在本次验证。
