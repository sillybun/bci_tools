# BCI Tools

团队共同维护的 BCI 小工具集合，用于实验搭建、进程管理、数据检查和日常调试。

在统一启动台中按分类浏览、搜索并打开工具。每个工具放在 `tools/<工具名>/`，各自维护说明、依赖和测试。

## 工具目录

| 工具 | 用途 | 文档 |
| --- | --- | --- |
| Integrated Term（集成终端控制台） | 集中启动、停止和分组管理采集、解码、设备桥接及可视化进程，查看终端输出 | [使用说明](tools/integrated-term/README.md) |

## 打开工具箱

安装 uv 后，在本仓库根目录运行：

```bash
uv run --locked python app.py
```

首次运行会按需准备 Python 3.13.14 和启动台自己的 `.venv`，需要访问 Python 下载源。启动台使用 Python 标准库和 Tk，无额外 pip 运行依赖。

- 左侧选择分类，顶部按名称、用途或标签搜索；`Cmd/Ctrl+F` 聚焦搜索，`Esc` 清空搜索。
- 点击工具卡片的“打开工具”启动，点击“使用说明”用系统默认应用打开 README。
- 工具在自己的目录和环境中运行。首次运行可能需要准备环境；可通过“查看日志”了解进度或失败原因。
- 同一次启动台会话内，正在运行的工具会禁用重复启动。关闭启动台后，已打开的工具继续运行；重新打开启动台不会接管之前的进程。
- 新增或修改工具配置后点击“刷新”。分类从配置中自动生成，某个工具配置出错时，其余工具仍可使用。

启动日志存放在 `.local/logs/`，只保留各工具最近一次启动的输出，并由 Git 忽略。卡片的运行状态表示启动命令是否仍在运行；设备连接、数据质量等状态由工具自身展示。

当前已在 macOS 上验证；Windows/Linux 使用同一入口，仍需在对应系统上做原生验证。

## 使用集成终端

首次启动显示空的进程列表。可通过“导入配置”选择工具目录下的 `examples/processes.example.json`，运行示例命令确认输出，再用“设置根目录”选择自己的项目目录，添加实际进程。

已有 Integrated Term 用户可导入原工具导出的 JSON 配置；本机配置保存在新工具目录的 `processes.json`，由 Git 忽略。

也可直接启动单个工具：

```bash
cd tools/integrated-term
uv run --locked python app.py
```

集成终端管理的采集、解码等程序仍需各自准备环境。

## 仓库结构

```text
bci_tools/
├── README.md
├── CONTRIBUTING.md
├── app.py                    # 统一入口
├── pyproject.toml            # 启动台环境
├── uv.lock
├── launcher/                 # 发现、展示、启动工具
├── tests/                    # 启动台测试
└── tools/
    └── integrated-term/
        ├── tool.toml         # 工具介绍、分类与启动命令
        ├── README.md
        ├── MIGRATION.md
        ├── app.py
        ├── lsl_registry.py
        ├── pyproject.toml
        ├── uv.lock
        ├── examples/
        ├── assets/
        ├── packaging/
        └── tests/
```

## 共同维护

新增工具或修复现有工具时，参照 [贡献说明](CONTRIBUTING.md)。共享可复用的代码、示例和使用说明；设备地址、个人路径、凭据、实验数据和本地运行状态保留在各自机器。

启动台测试在仓库根目录运行：

```bash
uv run --locked python -m unittest discover -s tests -v
```
