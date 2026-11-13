# Integrated GitHub Repos

一个基于本地 Python、面向 GitHub 与私有 GitLab 仓库的跨平台桌面客户端。应用调用系统 `git`；token 不写入仓库 URL 或仓库配置。

## uv 启动

```bash
uv run --locked integrated-github-repos
uv run --locked --extra dev python -m pytest -q
```

`.python-version` 选择 Python 3.13.13；复用既有 `pyproject.toml` 与 `uv.lock`，需要系统 Git。原有 pip 启动方式仍可使用。

## 当前功能

- 添加本地仓库、从指定目录递归扫描仓库、克隆网络仓库。
- 克隆时可保持 `none`，或创建 `feat/*`、`fix/*` 等规范前缀分支。
- 展示未提交、同步、过期、断线状态；窗口重新聚焦时自动 `fetch --all --prune`。
- commit 从新到旧展示，每次展开 5 项；支持提交全部改动、快进拉取、推送和切换本地分支。
- 展示贡献者、当前分支、远端信息和最近 35 天更新方格。
- 仓库详情下方直接渲染根目录 Markdown README，并支持相对图片。
- commit 左侧根据真实 parent SHA 绘制分支/合并曲线；本地未推送行直接显示上传按钮，远端未抓取行直接显示抓取按钮，已同步行不显示多余操作。
- 展示 worktree/commit 文件列表与统一 diff，并按新增、删除、区块头着色。
- 读取和编辑 `.codex/commit-acceptance/<commit-sha>.json` 验收表。
- 未提交工作区的空验收表可复制 AI 审阅提示词，结果写入 `.codex/commit-acceptance/pending.json`。
- 工具栏“设置”可管理 GitHub/私有 GitLab HTTP(S) token：手动输入并保存到应用私有目录的纯文本文件，或选择已有的单行 token 文本文件。不会读取钥匙串；SSH 仓库仍使用系统 SSH 密钥。
- GitHub 凭据设置默认填写 `https://github.com`，身份包含用户名和邮箱；SSH 仓库按远端主机匹配设置中的身份，并在仓库标题右侧显示为 `SSH:user<email>`。SSH URL 中固定的 `git` 传输用户不会被误作实际账号。身份列表再次点击当前项可取消选中。
- 仓库介绍页可展开“仓库设置”，重新绑定列表中的本地仓库目录、实际添加或修改 `origin`，并切换本地或已抓取的 `origin/*` 分支；远端独有分支会自动建立同名本地跟踪分支，修改 Git 状态前会显示命令并要求确认。
- 分支下拉菜单提供“添加分支…”，输入合法名称后创建并切换；尚无首个 commit 的仓库归类为“未提交”，并显示包含初始文件的工作区提交卡片。

私有 GitLab 可填写完整仓库地址，如 `https://gitlab.example.com/group/repo.git`。手动输入的 token 以明文形式保存于应用数据目录的 `tokens/` 下（macOS/Linux 尽可能设置为仅当前用户可读）；选择文件模式只记录绝对路径，源文件权限和备份由用户管理。请勿共享或同步 token 文件。若服务仅提供 HTTP，token 还将在网络中以明文传输；应用默认拒绝，必须在设置中明确勾选不安全传输选项。不要把 token 写进仓库 URL。设置的 token 仅用于对应主机的 Git HTTP(S) 操作，现有 SSH 仓库不受影响。

贡献者只统计当前 `HEAD` 可达历史。应用不会使用 `--all`，因此单独保存的 `upstream/*`、其他远端分支或无关历史不会污染当前仓库的贡献者列表。

## 开发运行

需要 Python 3.10+ 和可在终端调用的 Git。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
integrated-github-repos
```

Windows 激活环境可使用 `.venv\Scripts\Activate.ps1`，其余命令不变。

## 测试

```bash
ruff check .
pytest --cov=integrated_github_repos --cov-report=term-missing
```

功能门禁与已知未测边界见 `docs/feature-registry.md`，跨模块契约见 `docs/code-relations.md`。

## 打包

PyInstaller 不能可靠地跨操作系统生成原生包，因此三个脚本分别在目标系统运行：

```bash
# macOS
./packaging/macos/build_app.sh

# Ubuntu
./packaging/ubuntu/build_app.sh
./packaging/ubuntu/install_app.sh

# Windows PowerShell
.\packaging\windows\build_app.ps1
```

macOS 脚本优先使用环境变量 `CODESIGN_IDENTITY`，未提供时采用 ad-hoc 签名。对外分发仍需 Developer ID 证书和 notarization。Windows 正式分发建议在生成后用组织代码签名证书签署 `.exe`。

应用图标源文件是 `assets/app-icon-source-v3.png`。`packaging/generate_icons.py` 会确定性生成运行时 PNG、macOS ICNS 和 Windows ICO，打包脚本会自动调用。

## 验收表格式

每个 commit 的表格位于 `.codex/commit-acceptance/<完整 SHA>.json`。文件可由 Codex 的 commit 审核流程生成，也可在本应用的 commit 详情页编辑。应用不会自动提交该文件。

Ubuntu 安装脚本将应用复制到 `~/.local/opt/integrated-github-repos` 并注册应用菜单入口。需要 `apt-get`、`dpkg-deb` 和可用的 Ubuntu 软件源；脚本下载 `libxcb-cursor0` 并将运行库放入 PyInstaller 的 `_internal` 目录，无需 sudo。构建前请安装 `python3-venv`。
