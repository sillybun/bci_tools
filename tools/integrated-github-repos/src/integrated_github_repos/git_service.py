from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

from .credentials import CredentialStore
from .models import (
    AcceptanceItem,
    AcceptanceReview,
    CommitInfo,
    CommitState,
    ContributionDay,
    RepoInfo,
    RepoState,
)


class GitError(RuntimeError):
    pass


def localize_git_error(message: str) -> str:
    """Translate common filesystem and Git failures shown in the desktop UI."""
    text = message.strip()
    lowered = text.casefold()
    translations = (
        (("no such file or directory", "cannot chdir", "does not exist"), "找不到仓库目录，文件夹可能已被移动、重命名或删除。"),
        (("not a git repository",), "所选目录不是有效的 Git 仓库。"),
        (("could not resolve host", "failed to connect", "connection timed out"), "无法连接远端服务器，请检查网络或服务器状态。"),
        (("authentication failed", "access denied", "403 forbidden"), "身份验证失败或当前身份没有仓库访问权限。"),
    )
    for needles, translated in translations:
        if any(needle in lowered for needle in needles):
            return translated
    return text or "Git 操作失败。"


BRANCH_PREFIXES = (
    "none",
    "feat",
    "fix",
    "hotfix",
    "refactor",
    "perf",
    "style",
    "test",
    "docs",
    "chore",
    "ci",
    "build",
    "release",
    "develop",
    "wip",
    "revert",
)

ACCEPTANCE_ITEMS = (
    "改了什么",
    "为什么改",
    "如何验证",
    "是否涉及删除",
    "是否涉及迁移",
    "是否涉及配置",
    "是否涉及权限",
    "是否涉及数据",
    "是否涉及模型",
    "是否涉及大文件",
    "是否有兼容性风险",
)


def _git_environment() -> dict[str, str]:
    """Give Git hooks the same common package-manager tools as an interactive shell."""
    environment = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": str(Path(sys.executable if getattr(sys, "frozen", False) else sys.argv[0]).resolve()),
        "IGR_GIT_ASKPASS": "1",
    }
    path_entries = environment.get("PATH", "").split(os.pathsep) if environment.get("PATH") else []
    candidates = [
        Path.home() / ".bun" / "bin",
        Path.home() / ".local" / "bin",
        Path("/opt/homebrew/bin"),
        Path("/usr/local/bin"),
    ]
    for candidate in candidates:
        value = str(candidate)
        if candidate.is_dir() and value not in path_entries:
            path_entries.insert(0, value)
    environment["PATH"] = os.pathsep.join(path_entries)
    return environment


def _run(
    repo: Path | None,
    *args: str,
    timeout: float = 30,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = ["git"]
    for item in CredentialStore().hosts():
        command += ["-c", f"credential.https://{item.host}.helper="]
        if item.allow_http:
            command += ["-c", f"credential.http://{item.host}.helper="]
    if repo is not None:
        command += ["-C", str(repo)]
    command += list(args)
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=creationflags,
            env=_git_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitError(str(exc)) from exc
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "Git command failed"
        raise GitError(localize_git_error(detail))
    return result


def is_repository(path: Path) -> bool:
    return path.is_dir() and _run(path, "rev-parse", "--is-inside-work-tree", check=False).returncode == 0


def discover_repositories(roots: Iterable[Path], max_depth: int = 5) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            continue
        for current, dirs, _files in os.walk(root):
            current_path = Path(current)
            depth = len(current_path.relative_to(root).parts)
            if ".git" in dirs:
                found.add(current_path)
                dirs.remove(".git")
            if depth >= max_depth:
                dirs[:] = []
                continue
            dirs[:] = [
                item
                for item in dirs
                if item not in {"node_modules", ".venv", "venv", "Library", "$RECYCLE.BIN"}
            ]
    return sorted(found, key=lambda item: str(item).casefold())


def repository_name_from_url(url: str) -> str:
    value = url.strip().rstrip("/")
    if not value:
        raise GitError("仓库地址不能为空")
    if ":" in value and not value.startswith(("http://", "https://", "ssh://")):
        tail = value.rsplit(":", 1)[-1]
    else:
        tail = urlparse(value).path
    name = Path(tail).name
    name = name.removesuffix(".git")
    if not name or name in {".", ".."}:
        raise GitError("无法从地址识别仓库名称")
    return name


def normalize_remote_url(url: str) -> str:
    value = url.strip()
    if not value:
        raise GitError("仓库地址不能为空")
    if value.startswith(("http://", "https://")):
        parsed = urlparse(value)
        if not parsed.hostname or parsed.username or parsed.password:
            raise GitError("HTTP(S) 仓库地址无效")
        path = parsed.path.rstrip("/")
        # GitLab web clients sometimes pass the fragment-style smart HTTP endpoint.
        # Convert that form to the repository endpoint before Git sees it.
        if not path.endswith(".git"):
            path += ".git"
        return parsed._replace(path=path, query="", fragment="").geturl()
    return value


def clone_repository(
    url: str,
    parent: Path,
    branch_prefix: str = "none",
    branch_name: str = "",
) -> Path:
    if branch_prefix not in BRANCH_PREFIXES:
        raise GitError(f"不支持的分支类型: {branch_prefix}")
    url = normalize_remote_url(url)
    parent = parent.expanduser().resolve()
    if not parent.is_dir():
        raise GitError("本地父文件夹不存在")
    destination = parent / repository_name_from_url(url)
    if destination.exists():
        raise GitError(f"目标已存在: {destination}")
    _run(None, "clone", "--", url.strip(), str(destination), timeout=300)
    if branch_prefix != "none":
        leaf = branch_name.strip().strip("/")
        if not leaf:
            raise GitError("选择分支类型后必须填写分支名称")
        _run(destination, "switch", "-c", f"{branch_prefix}/{leaf}")
    return destination


def validate_initial_remote(url: str) -> None:
    """Allow only an empty remote or one containing an empty README."""
    remote = normalize_remote_url(url)
    if not remote:
        raise GitError("仓库地址不能为空")
    refs = _run(None, "ls-remote", "--refs", remote, timeout=90, check=False)
    if refs.returncode:
        raise GitError(refs.stderr.strip() or "无法读取远端仓库")
    if not refs.stdout.strip():
        return
    with tempfile.TemporaryDirectory(prefix="integrated-github-repos-remote-") as directory:
        checkout = Path(directory) / "remote"
        result = _run(
            None, "clone", "--depth=1", "--no-tags", "--", remote, str(checkout),
            timeout=300, check=False,
        )
        if result.returncode:
            raise GitError(result.stderr.strip() or "无法检查远端仓库内容")
        files = _run(checkout, "ls-tree", "-r", "--name-only", "HEAD").stdout.splitlines()
        allowed = {"README", "README.md", "README.rst", "README.txt"}
        if not files or any(item not in allowed for item in files):
            raise GitError("远端不是空仓库，且不符合仅含空 README 的初始化条件")
        for name in files:
            if _run(checkout, "show", f"HEAD:{name}").stdout.strip():
                raise GitError("远端 README 不是空文件，无法作为初始化目标")


def initialize_from_local(
    path: Path, remote_url: str, branch_prefix: str = "none", branch_name: str = ""
) -> Path:
    remote_url = normalize_remote_url(remote_url)
    path = path.expanduser().resolve()
    if not is_repository(path):
        raise GitError("本地初始仓库不是有效的 Git 仓库")
    validate_initial_remote(remote_url)
    reconnect_origin(path, remote_url)
    has_head = _run(path, "rev-parse", "--verify", "HEAD", check=False).returncode == 0
    has_changes = bool(_run(path, "status", "--porcelain").stdout.strip())
    if not has_head or has_changes:
        commit_all_changes(path, "init commit")
    if branch_prefix != "none":
        leaf = branch_name.strip().strip("/")
        if not leaf:
            raise GitError("选择分支类型后必须填写分支名称")
        _run(path, "switch", "-c", f"{branch_prefix}/{leaf}")
    # Publish every local branch so the new repository reflects the forked history,
    # while the current branch remains the upstream branch for normal push/pull.
    _run(path, "push", "--set-upstream", "origin", "--all", timeout=300)
    return path


def fetch_repository(path: Path) -> None:
    _run(path, "fetch", "--all", "--prune", timeout=90)


def _tracking_counts(path: Path) -> tuple[int, int]:
    upstream = _run(path, "rev-parse", "--abbrev-ref", "@{upstream}", check=False)
    if upstream.returncode:
        return 0, 0
    counts = _run(path, "rev-list", "--left-right", "--count", "HEAD...@{upstream}").stdout.split()
    return (int(counts[0]), int(counts[1])) if len(counts) == 2 else (0, 0)


def inspect_repository(path: Path, fetch: bool = False) -> RepoInfo:
    path = path.expanduser().resolve()
    if not path.is_dir():
        return RepoInfo(
            path=path,
            name=path.name,
            state=RepoState.DISCONNECTED,
            error="找不到仓库目录，文件夹可能已被移动、重命名或删除。",
        )
    if not is_repository(path):
        return RepoInfo(
            path=path,
            name=path.name,
            state=RepoState.DISCONNECTED,
            error="所选目录不是有效的 Git 仓库。",
        )
    fetch_error = ""
    remote = _run(path, "remote", "get-url", "origin", check=False).stdout.strip()
    if fetch and remote:
        try:
            fetch_repository(path)
        except GitError as exc:
            fetch_error = str(exc)
    branch = _run(path, "branch", "--show-current", check=False).stdout.strip() or "(detached)"
    has_head = _run(path, "rev-parse", "--verify", "HEAD", check=False).returncode == 0
    dirty = bool(_run(path, "status", "--porcelain", "--untracked-files=no").stdout.strip())
    ahead, behind = _tracking_counts(path)
    if fetch_error:
        state = RepoState.DISCONNECTED
    elif not has_head or dirty:
        state = RepoState.DIRTY
    elif behind:
        state = RepoState.STALE
    else:
        state = RepoState.SYNCED
    return RepoInfo(
        path=path,
        name=path.name,
        branch=branch,
        state=state,
        ahead=ahead,
        behind=behind,
        remote_url=remote,
        contributors=list_contributors(path),
        contributions=list_contributions(path),
        error=fetch_error,
    )


def has_tracked_changes(path: Path) -> bool:
    has_head = _run(path, "rev-parse", "--verify", "HEAD", check=False).returncode == 0
    untracked = "no" if has_head else "normal"
    return bool(_run(path, "status", "--porcelain", f"--untracked-files={untracked}").stdout.strip())


def list_commits(path: Path, limit: int = 200) -> list[CommitInfo]:
    commits: list[CommitInfo] = []
    if has_tracked_changes(path):
        commits.append(
            CommitInfo(
                sha="",
                subject="工作区中尚未提交的跟踪文件改动",
                author="本地工作区",
                authored_at="",
                states=(CommitState.UNCOMMITTED,),
                is_worktree=True,
                graph_edges=((0, 0),),
            )
        )
    result = _run(
        path,
        "log",
        "--topo-order",
        "--branches",
        "--remotes=origin",
        f"--max-count={limit}",
        "--date=iso-strict",
        "--format=%H%x1f%s%x1f%an%x1f%ad%x1f%P%x1f%D%x1e",
        check=False,
    )
    if result.returncode:
        return commits
    upstream = _run(path, "rev-parse", "--verify", "@{upstream}", check=False)
    has_origin = bool(_run(path, "remote", "get-url", "origin", check=False).stdout.strip())
    head_commits = set(_run(path, "rev-list", "HEAD", check=False).stdout.splitlines())
    upstream_commits = (
        set(_run(path, "rev-list", "@{upstream}", check=False).stdout.splitlines())
        if upstream.returncode == 0
        else set()
    )
    for record in result.stdout.strip("\x1e\r\n").split("\x1e"):
        fields = record.strip("\r\n").split("\x1f")
        if len(fields) != 6:
            continue
        sha, subject, author, authored_at, parents, refs = fields
        in_head = sha in head_commits
        states: list[CommitState] = []
        needs_push = False
        needs_pull = False
        if upstream.returncode == 0:
            in_upstream = sha in upstream_commits
            if in_head and in_upstream:
                states.append(CommitState.SYNCED)
            elif in_head:
                states.append(CommitState.UNPUSHED)
                needs_push = True
            elif in_upstream:
                states.extend((CommitState.FETCHED, CommitState.UNFETCHED))
                needs_pull = True
        elif in_head:
            if has_origin:
                states.append(CommitState.UNPUSHED)
                needs_push = True
            else:
                states.append(CommitState.COMMITTED)
        commits.append(
            CommitInfo(
                sha,
                subject,
                author,
                authored_at,
                tuple(states),
                parents=tuple(parents.split()),
                refs=tuple(ref.strip() for ref in refs.split(",") if ref.strip()),
                needs_push=needs_push,
                needs_pull=needs_pull,
            )
        )
    return _with_graph_layout(commits)


def _with_graph_layout(commits: list[CommitInfo]) -> list[CommitInfo]:
    from dataclasses import replace

    laid_out: list[CommitInfo] = []
    lanes: list[str] = []
    for commit in commits:
        if commit.is_worktree:
            laid_out.append(replace(commit, lane_count=max(1, len(lanes)), graph_edges=((0, 0),)))
            continue
        if commit.sha not in lanes:
            lanes.append(commit.sha)
        node_lane = lanes.index(commit.sha)
        before = list(lanes)
        after = list(lanes)
        after.pop(node_lane)
        for parent in reversed(commit.parents):
            if parent not in after:
                after.insert(node_lane, parent)
        edges: list[tuple[int, int]] = []
        for source_lane, source_sha in enumerate(before):
            if source_sha == commit.sha:
                edges.extend((source_lane, after.index(parent)) for parent in commit.parents if parent in after)
            elif source_sha in after:
                edges.append((source_lane, after.index(source_sha)))
        lane_count = max(1, len(before), len(after))
        laid_out.append(
            replace(commit, lane=node_lane, lane_count=lane_count, graph_edges=tuple(edges))
        )
        lanes = after
    return laid_out


def list_contributors(path: Path, limit: int = 8) -> list[str]:
    output = _run(path, "shortlog", "-sn", "HEAD", check=False).stdout
    return [line.strip() for line in output.splitlines() if line.strip()][:limit]


def read_readme(path: Path, max_bytes: int = 2_000_000) -> tuple[Path, str] | None:
    candidates = sorted(
        (
            candidate
            for candidate in path.iterdir()
            if candidate.is_file()
            and candidate.stem.casefold() == "readme"
            and candidate.suffix.casefold() in {".md", ".markdown", ".mdown", ".mkd"}
        ),
        key=lambda candidate: candidate.name.casefold(),
    )
    if not candidates:
        return None
    selected = candidates[0]
    if selected.stat().st_size > max_bytes:
        raise GitError(f"README 超过 {max_bytes // 1_000_000} MB，已停止渲染")
    return selected, selected.read_text(encoding="utf-8", errors="replace")


def list_contributions(path: Path, days: int = 35) -> list[ContributionDay]:
    today = datetime.now().astimezone().date()
    start = today - timedelta(days=today.weekday() + max(0, days - 7))
    output = _run(
        path,
        "log",
        f"--since={start.isoformat()} 00:00:00",
        "--date=short",
        "--format=%ad",
        check=False,
    ).stdout
    counts = Counter(line.strip() for line in output.splitlines() if line.strip())
    return [
        ContributionDay(
            (start + timedelta(days=offset)).isoformat(),
            counts.get((start + timedelta(days=offset)).isoformat(), 0)
            if start + timedelta(days=offset) <= today
            else 0,
        )
        for offset in range(days)
    ]


def _untracked_files(path: Path) -> list[str]:
    output = _run(
        path,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        check=False,
    ).stdout
    return [line[3:] for line in output.splitlines() if line.startswith("?? ")]


def _untracked_diff(path: Path, file_path: str) -> str:
    result = _run(
        path,
        "diff",
        "--no-ext-diff",
        "--no-index",
        "--",
        "/dev/null",
        file_path,
        check=False,
    )
    return result.stdout


def changed_files(path: Path, revision: str | None) -> list[tuple[str, str]]:
    if revision is None:
        has_head = _run(path, "rev-parse", "--verify", "HEAD", check=False).returncode == 0
        args = ("diff", "HEAD", "--name-status") if has_head else ("diff", "--cached", "--name-status")
    else:
        args = ("show", "--format=", "--name-status", revision)
    output = _run(path, *args, check=False).stdout
    changed: list[tuple[str, str]] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            changed.append((parts[0], parts[-1]))
    if revision is None:
        tracked_paths = {file_path for _status, file_path in changed}
        changed.extend(
            ("未跟踪", file_path)
            for file_path in _untracked_files(path)
            if file_path not in tracked_paths
        )
    return changed


def diff_text(path: Path, revision: str | None, file_path: str | None = None) -> str:
    if revision is None:
        untracked = _untracked_files(path)
        if file_path and file_path in untracked:
            return f"未跟踪文件: {file_path}\n\n{_untracked_diff(path, file_path)}"
        has_head = _run(path, "rev-parse", "--verify", "HEAD", check=False).returncode == 0
        args = ["diff", "--no-ext-diff", "HEAD"] if has_head else ["diff", "--cached", "--no-ext-diff"]
    else:
        args = ["show", "--format=", "--no-ext-diff", revision]
    if file_path:
        args += ["--", file_path]
        return _run(path, *args, check=False).stdout
    content = _run(path, *args, check=False).stdout
    if revision is None:
        for untracked_path in untracked:
            content += f"\n--- 未跟踪文件: {untracked_path} ---\n"
            content += _untracked_diff(path, untracked_path)
    return content


def commit_all_changes(path: Path, message: str) -> None:
    if not message.strip():
        raise GitError("提交说明不能为空")
    _run(path, "add", "--all")
    _run(path, "commit", "-m", message.strip())


def _push_hook_requires_bun(path: Path) -> bool:
    candidates = [path / ".husky" / "pre-push"]
    git_hooks = _run(path, "rev-parse", "--git-path", "hooks", check=False).stdout.strip()
    if git_hooks:
        hooks_path = Path(git_hooks)
        if not hooks_path.is_absolute():
            hooks_path = path / hooks_path
        candidates.append(hooks_path / "pre-push")
    for hook in candidates:
        try:
            if hook.is_file() and "bun" in hook.read_text(encoding="utf-8", errors="replace"):
                return True
        except OSError:
            continue
    return False


def _ensure_push_hook_runtime(path: Path) -> None:
    if _push_hook_requires_bun(path) and shutil.which("bun", path=_git_environment().get("PATH", "")) is None:
        raise GitError(
            "当前仓库的 pre-push hook 依赖 Bun，但应用环境中找不到 bun。\n"
            "请先安装 Bun（macOS 可执行 `brew install oven-sh/bun/bun`），"
            "或将 bun 所在目录加入 PATH 后重新启动应用。"
        )


def push(path: Path) -> None:
    _ensure_push_hook_runtime(path)
    branch = _run(path, "branch", "--show-current", check=False).stdout.strip()
    if not branch:
        raise GitError("当前处于 detached HEAD，无法确定要推送的分支")
    # Do not let a stale upstream such as origin/dev redirect the current branch.
    _run(path, "push", "--set-upstream", "origin", "HEAD", timeout=300)


def pull(path: Path) -> None:
    _run(path, "pull", "--ff-only", timeout=300)


def list_branches(path: Path) -> list[str]:
    output = _run(
        path,
        "for-each-ref",
        "--format=%(refname:short)",
        "refs/heads/",
        "refs/remotes/origin/",
    ).stdout
    branches: list[str] = []
    for line in output.splitlines():
        name = line.strip()
        if name.startswith("origin/"):
            name = name.removeprefix("origin/")
        # `origin` is the remote name, not a user-switchable branch.
        if name and name not in {"HEAD", "origin"} and name not in branches:
            branches.append(name)
    return branches


def branch_switch_args(path: Path, branch: str) -> tuple[str, ...]:
    valid = _run(path, "check-ref-format", "--branch", branch, check=False).returncode == 0
    if not valid:
        raise GitError("分支名称无效")
    local = _run(path, "show-ref", "--verify", f"refs/heads/{branch}", check=False).returncode == 0
    if local:
        return "switch", "--", branch
    remote = _run(
        path, "show-ref", "--verify", f"refs/remotes/origin/{branch}", check=False
    ).returncode == 0
    if remote:
        return "switch", "--track", "-c", branch, f"origin/{branch}"
    raise GitError(f"找不到分支: {branch}")


def switch_branch(path: Path, branch: str) -> None:
    _run(path, *branch_switch_args(path, branch))


def create_branch_args(path: Path, branch: str) -> tuple[str, ...]:
    branch = branch.strip()
    valid = _run(path, "check-ref-format", "--branch", branch, check=False).returncode == 0
    if not valid:
        raise GitError("分支名称无效")
    for ref in (f"refs/heads/{branch}", f"refs/remotes/origin/{branch}"):
        if _run(path, "show-ref", "--verify", ref, check=False).returncode == 0:
            raise GitError(f"分支已存在: {branch}")
    return "switch", "-c", branch


def create_branch(path: Path, branch: str) -> None:
    _run(path, *create_branch_args(path, branch))


def configure_origin(path: Path, remote_url: str) -> None:
    remote_url = remote_url.strip()
    if not remote_url or remote_url.startswith("-") or any(char in remote_url for char in "\r\n"):
        raise GitError("远端仓库地址无效")
    has_origin = _run(path, "remote", "get-url", "origin", check=False).returncode == 0
    if has_origin:
        _run(path, "remote", "set-url", "origin", remote_url)
    else:
        _run(path, "remote", "add", "origin", remote_url)


def reconnect_origin(path: Path, remote_url: str) -> None:
    """Replace the cloned origin and discard its tracking metadata before initialization."""
    remote_url = remote_url.strip()
    if not remote_url or remote_url.startswith("-") or any(char in remote_url for char in "\r\n"):
        raise GitError("远端仓库地址无效")
    if _run(path, "remote", "get-url", "origin", check=False).returncode == 0:
        _run(path, "remote", "remove", "origin")
    _run(path, "remote", "add", "origin", remote_url)
    branch = _run(path, "branch", "--show-current", check=False).stdout.strip()
    if branch:
        _run(path, "branch", "--unset-upstream", check=False)


def acceptance_path(path: Path, commit: str) -> Path:
    safe_commit = commit if commit else "pending"
    return path / ".codex" / "commit-acceptance" / f"{safe_commit}.json"


def acceptance_review_prompt(path: Path) -> str:
    rows = "\n".join(f"- {item}" for item in ACCEPTANCE_ITEMS)
    target = acceptance_path(path, "pending")
    return f"""请检查 {path} 文件夹下项目中的未提交文件，执行提交前代码审阅。

核心约定：
1. 检查 git status、已跟踪与未跟踪文件的实际 diff，并说明改动目的、验证证据和风险；结论必须基于实际内容，不确定项明确写明，不要臆测。
2. 检查删除、迁移、配置、权限或密钥、数据、模型、大文件与兼容性风险；不要把 token、密钥、构建产物、缓存、日志或本机专用配置纳入提交。
3. 只执行审阅和必要的只读检查；不要执行 git add、git commit、git push、git pull、merge、rebase 或修改远端。
4. 运行与改动相称的测试或静态检查，并在“如何验证”中记录实际命令和结果；未执行的验证必须明确标注。

审阅表必须按以下顺序包含全部 11 项：
{rows}

请将审阅结果保存为 UTF-8 JSON：{target}
JSON 格式：
{{
  "schema_version": 1,
  "commit": "pending",
  "reviewed_at": "带时区的 ISO-8601 时间",
  "items": [{{"item": "改了什么", "answer": "基于实际审阅的答案"}}],
  "notes": "可选补充说明"
}}
items 必须包含上述全部 11 项并保持顺序。不要覆盖已有的人工填写内容；如文件已存在，先读取并在保留有效内容的基础上补全。"""


def load_acceptance(path: Path, commit: str) -> AcceptanceReview | None:
    review_path = acceptance_path(path, commit)
    if not review_path.is_file():
        return None
    try:
        payload = json.loads(review_path.read_text(encoding="utf-8"))
        items = tuple(AcceptanceItem(str(item["item"]), str(item["answer"])) for item in payload["items"])
        return AcceptanceReview(
            schema_version=int(payload.get("schema_version", 1)),
            commit=str(payload.get("commit", commit)),
            reviewed_at=str(payload.get("reviewed_at", "")),
            items=items,
            notes=str(payload.get("notes", "")),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise GitError(f"验收表格式无效: {review_path}: {exc}") from exc


def save_acceptance(path: Path, review: AcceptanceReview) -> Path:
    target = acceptance_path(path, review.commit)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": review.schema_version,
        "commit": review.commit,
        "reviewed_at": review.reviewed_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "items": [{"item": item.item, "answer": item.answer} for item in review.items],
        "notes": review.notes,
    }
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target
