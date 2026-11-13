from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from integrated_github_repos.git_service import (
    ACCEPTANCE_ITEMS,
    GitError,
    _git_environment,
    acceptance_review_prompt,
    changed_files,
    clone_repository,
    commit_all_changes,
    configure_origin,
    create_branch,
    diff_text,
    discover_repositories,
    initialize_from_local,
    inspect_repository,
    list_branches,
    list_commits,
    list_contributors,
    load_acceptance,
    normalize_remote_url,
    read_readme,
    repository_name_from_url,
    save_acceptance,
    switch_branch,
    validate_initial_remote,
)
from integrated_github_repos.models import AcceptanceItem, AcceptanceReview, CommitState, RepoState


def test_repository_name_from_common_urls() -> None:
    assert repository_name_from_url("git@github.com:Bertie97/project.git") == "project"
    assert repository_name_from_url("https://github.com/Bertie97/project") == "project"
    with pytest.raises(GitError):
        repository_name_from_url("")


def test_git_environment_includes_existing_common_tool_paths() -> None:
    environment = _git_environment()
    paths = environment["PATH"].split(os.pathsep)
    for candidate in (Path.home() / ".bun" / "bin", Path("/opt/homebrew/bin"), Path("/usr/local/bin")):
        if candidate.is_dir():
            assert str(candidate) in paths


def test_normalize_remote_url_removes_gitlab_web_fragment() -> None:
    assert normalize_remote_url(
        "http://192.168.120.121/software-team/ai/opencode#/info/refs?service=git-upload-pack"
    ) == "http://192.168.120.121/software-team/ai/opencode.git"
    assert normalize_remote_url("https://gitlab.example.com/group/project") == "https://gitlab.example.com/group/project.git"


def test_discovers_nested_repositories_once(git_repo: Path, tmp_path: Path) -> None:
    assert discover_repositories([tmp_path]) == [git_repo]


def test_dirty_state_only_uses_tracked_files(git_repo: Path) -> None:
    (git_repo / "untracked.txt").write_text("new\n", encoding="utf-8")
    assert inspect_repository(git_repo).state == RepoState.SYNCED
    (git_repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    assert inspect_repository(git_repo).state == RepoState.DIRTY
    commits = list_commits(git_repo)
    assert commits[0].is_worktree
    assert CommitState.UNCOMMITTED in commits[0].states


def test_worktree_diff_appends_and_labels_untracked_files(git_repo: Path) -> None:
    (git_repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    (git_repo / "untracked.txt").write_text("new\n", encoding="utf-8")
    files = changed_files(git_repo, None)
    assert files[-1] == ("未跟踪", "untracked.txt")
    content = diff_text(git_repo, None)
    assert "--- 未跟踪文件: untracked.txt ---" in content
    assert "+new" in content
    assert diff_text(git_repo, None, "untracked.txt").startswith("未跟踪文件: untracked.txt")


def test_unborn_repository_with_untracked_files_is_not_synced(tmp_path: Path) -> None:
    repo = tmp_path / "unborn"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    (repo / "first.txt").write_text("first\n", encoding="utf-8")
    assert inspect_repository(repo).state == RepoState.DIRTY
    commits = list_commits(repo)
    assert len(commits) == 1
    assert commits[0].is_worktree
    assert CommitState.UNCOMMITTED in commits[0].states


def test_missing_repository_is_disconnected_with_chinese_error(tmp_path: Path) -> None:
    missing = tmp_path / "renamed-repository"
    info = inspect_repository(missing, fetch=True)
    assert info.state == RepoState.DISCONNECTED
    assert info.error == "找不到仓库目录，文件夹可能已被移动、重命名或删除。"


def test_list_commits_keeps_records_with_empty_refs_and_root_parent(git_repo: Path) -> None:
    for index in range(2):
        tracked = git_repo / "tracked.txt"
        tracked.write_text(f"revision {index}\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(git_repo), "add", "tracked.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(git_repo), "commit", "-m", f"revision {index}"],
            check=True,
            capture_output=True,
        )

    commits = list_commits(git_repo)
    assert [commit.subject for commit in commits] == ["revision 1", "revision 0", "initial"]
    assert commits[-1].parents == ()
    assert commits[1].refs == ()


def test_local_commit_with_origin_but_no_upstream_needs_push(git_repo: Path, tmp_path: Path) -> None:
    subprocess.run(
        ["git", "-C", str(git_repo), "remote", "add", "origin", str(tmp_path / "remote.git")],
        check=True,
    )
    commit = next(item for item in list_commits(git_repo) if not item.is_worktree)
    assert commit.states == (CommitState.UNPUSHED,)
    assert commit.needs_push


def test_list_commits_uses_synced_when_direction_is_not_in_history(git_repo: Path, tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    branch = subprocess.run(
        ["git", "-C", str(git_repo), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(remote), "symbolic-ref", "HEAD", f"refs/heads/{branch}"],
        check=True,
    )
    subprocess.run(["git", "-C", str(git_repo), "remote", "add", "origin", str(remote)], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "push", "--set-upstream", "origin", "HEAD"],
        check=True,
        capture_output=True,
    )

    commit = list_commits(git_repo)[0]
    assert commit.states == (CommitState.SYNCED,)

    (git_repo / "tracked.txt").write_text("local change\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(git_repo), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "commit", "-m", "local only"],
        check=True,
        capture_output=True,
    )
    local_only = next(item for item in list_commits(git_repo) if item.subject == "local only")
    assert local_only.states == (CommitState.UNPUSHED,)
    assert local_only.needs_push

    other = tmp_path / "other"
    subprocess.run(["git", "clone", str(remote), str(other)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(other), "config", "user.name", "Remote User"], check=True)
    subprocess.run(
        ["git", "-C", str(other), "config", "user.email", "remote@example.com"], check=True
    )
    (other / "remote.txt").write_text("remote change\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(other), "add", "remote.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(other), "commit", "-m", "remote only"],
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "-C", str(other), "push"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(git_repo), "fetch", "origin"], check=True, capture_output=True)
    remote_only = next(item for item in list_commits(git_repo) if item.subject == "remote only")
    upstream_shas = subprocess.run(
        ["git", "-C", str(git_repo), "rev-list", "@{upstream}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert remote_only.sha in upstream_shas
    assert remote_only.states == (CommitState.FETCHED, CommitState.UNFETCHED)
    assert remote_only.needs_pull


def test_clone_and_create_prefixed_branch(git_repo: Path, tmp_path: Path) -> None:
    source = tmp_path / "remote-source.git"
    subprocess.run(["git", "clone", "--bare", str(git_repo), str(source)], check=True, capture_output=True)
    parent = tmp_path / "clones"
    parent.mkdir()
    cloned = clone_repository(str(source), parent, "feat", "desktop-ui")
    branch = subprocess.run(
        ["git", "-C", str(cloned), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == "feat/desktop-ui"


def test_configure_origin_adds_and_updates_remote(git_repo: Path, tmp_path: Path) -> None:
    first = str(tmp_path / "first.git")
    second = "https://gitlab.example.com/group/repo.git"
    configure_origin(git_repo, first)
    assert subprocess.run(
        ["git", "-C", str(git_repo), "remote", "get-url", "origin"],
        check=True, capture_output=True, text=True,
    ).stdout.strip() == first
    configure_origin(git_repo, second)
    assert subprocess.run(
        ["git", "-C", str(git_repo), "remote", "get-url", "origin"],
        check=True, capture_output=True, text=True,
    ).stdout.strip() == second
    with pytest.raises(GitError, match="无效"):
        configure_origin(git_repo, "--upload-pack=bad")


def test_validate_initial_remote_allows_empty_and_empty_readme(tmp_path: Path) -> None:
    empty = tmp_path / "empty.git"
    subprocess.run(["git", "init", "--bare", str(empty)], check=True, capture_output=True)
    validate_initial_remote(str(empty))

    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True)
    (source / "README.md").write_text("", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-m", "empty readme"], check=True, capture_output=True)
    remote = tmp_path / "readme.git"
    subprocess.run(["git", "clone", "--bare", str(source), str(remote)], check=True, capture_output=True)
    validate_initial_remote(str(remote))


def test_initialize_from_local_sets_origin_and_creates_init_commit(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()
    subprocess.run(["git", "init", str(local)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(local), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(local), "config", "user.email", "test@example.com"], check=True)
    (local / "main.txt").write_text("init\n", encoding="utf-8")
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    initialize_from_local(local, str(remote))
    assert subprocess.run(["git", "-C", str(local), "log", "-1", "--format=%s"], check=True, capture_output=True, text=True).stdout.strip() == "init commit"
    assert subprocess.run(["git", "-C", str(local), "remote", "get-url", "origin"], check=True, capture_output=True, text=True).stdout.strip() == str(remote)


def test_initialize_from_local_accepts_existing_clean_history(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()
    subprocess.run(["git", "init", str(local)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(local), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(local), "config", "user.email", "test@example.com"], check=True)
    (local / "existing.txt").write_text("existing\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(local), "add", "existing.txt"], check=True)
    subprocess.run(["git", "-C", str(local), "commit", "-m", "existing"], check=True, capture_output=True)
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    initialize_from_local(local, str(remote))
    assert subprocess.run(["git", "-C", str(local), "log", "-1", "--format=%s"], check=True, capture_output=True, text=True).stdout.strip() == "existing"


def test_initialize_from_local_replaces_origin_and_pushes_all_branches(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(local)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(local), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(local), "config", "user.email", "test@example.com"], check=True)
    (local / "main.txt").write_text("main\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(local), "add", "main.txt"], check=True)
    subprocess.run(["git", "-C", str(local), "commit", "-m", "main history"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(local), "switch", "-c", "legacy"], check=True, capture_output=True)
    (local / "legacy.txt").write_text("legacy\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(local), "add", "legacy.txt"], check=True)
    subprocess.run(["git", "-C", str(local), "commit", "-m", "legacy history"], check=True, capture_output=True)
    old_remote = tmp_path / "old.git"
    target_remote = tmp_path / "target.git"
    subprocess.run(["git", "init", "--bare", str(old_remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", "--bare", str(target_remote)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(local), "remote", "add", "origin", str(old_remote)], check=True)
    subprocess.run(["git", "-C", str(local), "push", "-u", "origin", "--all"], check=True, capture_output=True)

    initialize_from_local(local, str(target_remote))

    assert subprocess.run(
        ["git", "-C", str(local), "remote", "get-url", "origin"],
        check=True, capture_output=True, text=True,
    ).stdout.strip() == str(target_remote)
    refs = subprocess.run(
        ["git", "--git-dir", str(target_remote), "for-each-ref", "--format=%(refname:short)", "refs/heads/"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert set(refs) == {"main", "legacy"}
    assert subprocess.run(
        ["git", "-C", str(local), "rev-parse", "--abbrev-ref", "@{upstream}"],
        check=True, capture_output=True, text=True,
    ).stdout.strip() == "origin/legacy"


def test_remote_only_branch_becomes_switchable_candidate(git_repo: Path, tmp_path: Path) -> None:
    current = subprocess.run(
        ["git", "-C", str(git_repo), "branch", "--show-current"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    remote = tmp_path / "origin.git"
    subprocess.run(["git", "clone", "--bare", str(git_repo), str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(git_repo), "remote", "add", "origin", str(remote)], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "switch", "-c", "remote-topic"],
        check=True, capture_output=True,
    )
    (git_repo / "remote-topic.txt").write_text("topic\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(git_repo), "add", "remote-topic.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "commit", "-m", "remote topic"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(git_repo), "push", "-u", "origin", "remote-topic"],
        check=True, capture_output=True,
    )
    subprocess.run(["git", "-C", str(git_repo), "switch", current], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(git_repo), "branch", "-D", "remote-topic"], check=True)

    assert "remote-topic" in list_branches(git_repo)
    switch_branch(git_repo, "remote-topic")
    assert subprocess.run(
        ["git", "-C", str(git_repo), "branch", "--show-current"],
        check=True, capture_output=True, text=True,
    ).stdout.strip() == "remote-topic"
    assert subprocess.run(
        ["git", "-C", str(git_repo), "rev-parse", "--abbrev-ref", "@{upstream}"],
        check=True, capture_output=True, text=True,
    ).stdout.strip() == "origin/remote-topic"


def test_list_branches_excludes_remote_name_origin(git_repo: Path, tmp_path: Path) -> None:
    remote = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(git_repo), "remote", "add", "origin", str(remote)], check=True)
    subprocess.run(["git", "-C", str(git_repo), "branch", "origin"], check=True, capture_output=True)
    assert "origin" not in list_branches(git_repo)


def test_switch_branch_rejects_missing_or_invalid_candidate(git_repo: Path) -> None:
    with pytest.raises(GitError, match="找不到"):
        switch_branch(git_repo, "does-not-exist")
    with pytest.raises(GitError, match="无效"):
        switch_branch(git_repo, "--discard-changes")


def test_create_branch_creates_and_switches_or_rejects_existing(git_repo: Path) -> None:
    create_branch(git_repo, "feat/new-branch")
    assert subprocess.run(
        ["git", "-C", str(git_repo), "branch", "--show-current"],
        check=True, capture_output=True, text=True,
    ).stdout.strip() == "feat/new-branch"
    with pytest.raises(GitError, match="已存在"):
        create_branch(git_repo, "feat/new-branch")
    with pytest.raises(GitError, match="无效"):
        create_branch(git_repo, "bad branch")


def test_acceptance_round_trip(git_repo: Path) -> None:
    review = AcceptanceReview(
        schema_version=1,
        commit="abc123",
        reviewed_at="2026-09-14T12:00:00+08:00",
        items=tuple(AcceptanceItem(item, f"answer-{index}") for index, item in enumerate(ACCEPTANCE_ITEMS)),
        notes="reviewed",
    )
    target = save_acceptance(git_repo, review)
    assert target.name == "abc123.json"
    assert load_acceptance(git_repo, "abc123") == review


def test_acceptance_review_prompt_contains_contract_and_pending_path(git_repo: Path) -> None:
    prompt = acceptance_review_prompt(git_repo)
    assert str(git_repo) in prompt
    assert str(git_repo / ".codex" / "commit-acceptance" / "pending.json") in prompt
    assert '"commit": "pending"' in prompt
    assert "不要执行 git add、git commit、git push" in prompt
    for item in ACCEPTANCE_ITEMS:
        assert f"- {item}" in prompt


def test_commit_all_includes_tracked_and_untracked_changes(git_repo: Path) -> None:
    (git_repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    (git_repo / "new.txt").write_text("new\n", encoding="utf-8")
    assert "+changed" in diff_text(git_repo, None)
    commit_all_changes(git_repo, "save all")
    names = subprocess.run(
        ["git", "-C", str(git_repo), "show", "--format=", "--name-only", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert {"tracked.txt", "new.txt"}.issubset(names)


def test_push_reports_missing_bun_for_husky_hook(git_repo: Path, monkeypatch) -> None:
    husky = git_repo / ".husky"
    husky.mkdir()
    (husky / "pre-push").write_text("#!/bin/sh\nbun run lint\n", encoding="utf-8")
    monkeypatch.setattr(
        "integrated_github_repos.git_service._git_environment",
        lambda: {"PATH": "/usr/bin:/bin"},
    )
    with pytest.raises(GitError, match="依赖 Bun"):
        from integrated_github_repos.git_service import push

        push(git_repo)


def test_push_targets_current_branch_instead_of_stale_upstream(git_repo: Path, tmp_path: Path) -> None:
    from integrated_github_repos.git_service import push

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(git_repo), "remote", "add", "origin", str(remote)], check=True)
    branch = subprocess.run(
        ["git", "-C", str(git_repo), "branch", "--show-current"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    subprocess.run(["git", "-C", str(git_repo), "push", "-u", "origin", "HEAD"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(git_repo), "switch", "-c", "feature/publish"], check=True, capture_output=True)
    (git_repo / "feature.txt").write_text("feature\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(git_repo), "add", "feature.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "commit", "-m", "feature commit"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(git_repo), "branch", "--set-upstream-to", f"origin/{branch}"],
        check=True, capture_output=True,
    )

    push(git_repo)

    refs = subprocess.run(
        ["git", "--git-dir", str(remote), "for-each-ref", "--format=%(refname:short)", "refs/heads/"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert "feature/publish" in refs
    assert subprocess.run(
        ["git", "-C", str(git_repo), "rev-parse", "--abbrev-ref", "@{upstream}"],
        check=True, capture_output=True, text=True,
    ).stdout.strip() == "origin/feature/publish"


def test_contributors_ignore_unrelated_upstream_history(git_repo: Path) -> None:
    tree = subprocess.run(
        ["git", "-C", str(git_repo), "mktree"],
        input="",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Unrelated Google Author",
        "GIT_AUTHOR_EMAIL": "unrelated@google.com",
        "GIT_COMMITTER_NAME": "Unrelated Google Author",
        "GIT_COMMITTER_EMAIL": "unrelated@google.com",
    }
    unrelated = subprocess.run(
        ["git", "-C", str(git_repo), "commit-tree", tree, "-m", "unrelated upstream"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(git_repo), "update-ref", "refs/remotes/upstream/main", unrelated],
        check=True,
    )
    contributors = list_contributors(git_repo)
    assert contributors == ["1\tTest User"]
    assert all("google.com" not in contributor for contributor in contributors)


def test_merge_commit_produces_two_graph_edges(git_repo: Path) -> None:
    current = subprocess.run(
        ["git", "-C", str(git_repo), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "-C", str(git_repo), "switch", "-c", "topic"], check=True, capture_output=True)
    (git_repo / "topic.txt").write_text("topic\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(git_repo), "add", "topic.txt"], check=True)
    subprocess.run(["git", "-C", str(git_repo), "commit", "-m", "topic"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(git_repo), "switch", current], check=True, capture_output=True)
    (git_repo / "main.txt").write_text("main\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(git_repo), "add", "main.txt"], check=True)
    subprocess.run(["git", "-C", str(git_repo), "commit", "-m", "main"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "merge", "--no-ff", "topic", "-m", "merge topic"],
        check=True,
        capture_output=True,
    )
    commits = list_commits(git_repo)
    merge = next(commit for commit in commits if commit.subject == "merge topic")
    assert len(merge.parents) == 2
    assert len(merge.graph_edges) == 2
    assert merge.lane_count >= 2


def test_commit_sync_direction_distinguishes_push_from_pull(git_repo: Path, tmp_path: Path) -> None:
    branch = subprocess.run(
        ["git", "-C", str(git_repo), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "clone", "--bare", str(git_repo), str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(git_repo), "remote", "add", "origin", str(remote)], check=True)
    subprocess.run(["git", "-C", str(git_repo), "fetch", "origin"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(git_repo), "branch", "--set-upstream-to", f"origin/{branch}"],
        check=True,
        capture_output=True,
    )
    (git_repo / "local.txt").write_text("local\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(git_repo), "add", "local.txt"], check=True)
    subprocess.run(["git", "-C", str(git_repo), "commit", "-m", "local only"], check=True, capture_output=True)

    peer = tmp_path / "peer"
    subprocess.run(["git", "clone", str(remote), str(peer)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(peer), "config", "user.name", "Remote User"], check=True)
    subprocess.run(["git", "-C", str(peer), "config", "user.email", "remote@example.com"], check=True)
    (peer / "remote.txt").write_text("remote\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(peer), "add", "remote.txt"], check=True)
    subprocess.run(["git", "-C", str(peer), "commit", "-m", "remote only"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(peer), "push"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(git_repo), "fetch", "origin"], check=True, capture_output=True)

    commits = {commit.subject: commit for commit in list_commits(git_repo)}
    assert commits["local only"].needs_push
    assert not commits["local only"].needs_pull
    assert commits["remote only"].needs_pull
    assert not commits["remote only"].needs_push


def test_readme_markdown_is_loaded_case_insensitively(git_repo: Path) -> None:
    readme = git_repo / "ReadMe.MD"
    readme.write_text("# Sample\n\nRendered text.\n", encoding="utf-8")
    loaded = read_readme(git_repo)
    assert loaded == (readme, "# Sample\n\nRendered text.\n")
