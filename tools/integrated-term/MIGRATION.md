# 迁移记录

- 迁移日期：2026-10-20。
- 原仓库：`git@github.com:Bertie97/integrated-term.git`。
- 来源分支：`feat/uv-project-startup`。
- 来源提交：`7b896d38871abfdae5a88235040cf6dc58cb411d`（`fix: keep process configuration local`）。
- 目标位置：`bci_tools/tools/integrated-term/`。

本次以源工作区中 Git 跟踪的 61 个文件为基线导入，包括代码、图标、字体、测试、uv 环境声明和打包脚本。源工作区的已跟踪文件在复制时没有未提交修改。来源 Git 历史保留在原仓库，本仓库从该快照继续维护。

原目录、本机的 `processes.json`、`.venv`、缓存和 `.git` 未纳入复制。运行代码 `app.py`、`lsl_registry.py` 及 Python 版本、依赖锁文件沿用来源版本。

迁入后补充了团队使用说明、通用示例配置及打包输出忽略规则；Linux/Windows 打包清单去除了对本机 `processes.json` 的依赖，避免新克隆缺文件或携带某位成员的本机配置。进程管理测试使用临时注册表，避免影响机器上其他 BCI 工具的活动进程记录。

## 继续使用已有配置

启动新目录中的控制台，点击“导入配置”，选择旧工具导出的 JSON 或旧目录的 `processes.json`。导入后会保存到新目录的本机配置文件；确认“设置根目录”显示的位置符合当前机器。

## 验证边界

本次在 macOS、uv 管理的 Python 3.13.14 下验证：

- `uv run --offline --locked python -m unittest discover -s tests -v`：31 项测试通过，无跳过。原生桌面与进程树访问需在沙箱外运行。
- 实际创建 Tk 控制台，验证首次启动为空配置、导入示例、两条示例命令成功退出并输出预期内容、配置保存及活动注册表清理；测试配置和注册表均使用临时目录，未测试其他应用窗口的显示/隐藏。
- 核对来源的 61 个已跟踪文件均已迁入；运行代码、资源和依赖文件内容保持一致。
- 检查 Linux/Windows 打包清单中的资源路径均存在，且不再包含本机配置；确认本机配置、虚拟环境和打包输出被 Git 忽略。

Python 版本固定为 3.13.14，沿用原工具的 Tk 环境选择。源代码包含 Windows、macOS 和 Linux 适配，但这些适配仍需在团队实际使用的平台上分别验证；在一个平台通过测试不代表其他平台或打包产物已验证。
