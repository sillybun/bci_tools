from __future__ import annotations

import os
from pathlib import Path
from time import monotonic

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QHeaderView,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QStyle,
    QStyleOptionViewItem,
)

from integrated_github_repos.credentials import CredentialStore
from integrated_github_repos.git_service import (
    ACCEPTANCE_ITEMS,
    inspect_repository,
    list_commits,
    save_acceptance,
)
from integrated_github_repos.models import (
    AcceptanceItem,
    AcceptanceReview,
    CommitInfo,
    CommitState,
    RepoInfo,
    RepoState,
)
from integrated_github_repos.repo_store import RepoStore
from integrated_github_repos.ui import main_window as main_window_module
from integrated_github_repos.ui.main_window import (
    CloneDialog,
    CommitDialog,
    CredentialDialog,
    MainWindow,
    RepoStateDelegate,
)
from integrated_github_repos.ui.theme import APP_STYLE
from integrated_github_repos.ui.widgets import CommitRow, format_commit_time


def test_window_and_clone_defaults(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(RepoStore(tmp_path / "repos.json"))
    assert window.windowTitle() == "Integrated GitHub Repos"
    assert window.commit_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert window.repo_list.verticalScrollMode() == QAbstractItemView.ScrollPerPixel
    assert window.files_list.verticalScrollMode() == QAbstractItemView.ScrollPerPixel
    dialog = CloneDialog(window)
    assert dialog.prefix.currentText() == "none"
    assert not dialog.branch_name.isEnabled()
    assert not dialog.prefix_drop_button.icon().isNull()
    assert not dialog.initialize_from_local.isChecked()
    dialog.initialize_from_local.setChecked(True)
    assert dialog.path_label.text() == "本地初始仓库"
    dialog.initialize_from_local.setChecked(False)
    assert dialog.path_label.text() == "本地父文件夹"
    dialog.prefix.setCurrentText("feat")
    assert dialog.branch_name.isEnabled()
    dialog.close()
    window.close()
    assert not window._worktree_timer.isActive()
    app.processEvents()


def test_remove_repo_requires_confirmation_and_keeps_local_files(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    window = MainWindow(RepoStore(tmp_path / "repos.json"))
    window.paths = [repo]
    window.current_repo = repo
    monkeypatch.setattr(main_window_module.QMessageBox, "question", lambda *args, **kwargs: main_window_module.QMessageBox.No)
    window.remove_current()
    assert window.paths == [repo]
    monkeypatch.setattr(main_window_module.QMessageBox, "question", lambda *args, **kwargs: main_window_module.QMessageBox.Yes)
    window.remove_current()
    assert window.paths == []
    assert repo.is_dir()
    window.close()
    app.processEvents()


def test_commit_dialog_allows_commit_without_acceptance_answers(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    dialog = CommitDialog(tmp_path)
    dialog.message.setText("commit without review")
    dialog._validate()
    assert dialog.result() == QDialog.Accepted
    assert dialog.review() is None
    dialog.close()
    app.processEvents()


def test_commit_dialog_prefills_pending_acceptance_review(git_repo: Path) -> None:
    app = QApplication.instance() or QApplication([])
    save_acceptance(
        git_repo,
        AcceptanceReview(
            schema_version=1,
            commit="pending",
            reviewed_at="2026-09-30T10:00:00+08:00",
            items=tuple(
                AcceptanceItem(item, f"answer-{index}")
                for index, item in enumerate(ACCEPTANCE_ITEMS)
            ),
            notes="existing notes",
        ),
    )
    dialog = CommitDialog(git_repo)
    assert dialog.table.item(0, 1).text() == "answer-0"
    assert dialog.table.item(len(ACCEPTANCE_ITEMS) - 1, 1).text() == "answer-10"
    assert dialog.notes.toPlainText() == "existing notes"
    dialog.close()
    app.processEvents()


def test_empty_worktree_review_exposes_copyable_agent_prompt(
    git_repo: Path, tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    (git_repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    window = MainWindow(RepoStore(tmp_path / "repos.json"))
    window.current_repo = git_repo
    window._show_repo(inspect_repository(git_repo))
    worktree = CommitInfo(
        sha="",
        subject="工作区中尚未提交的跟踪文件改动",
        author="本地工作区",
        authored_at="",
        states=(CommitState.UNCOMMITTED,),
        is_worktree=True,
    )

    window.show_commit(worktree)

    assert not window.acceptance_section.isHidden()
    assert window.acceptance_table.rowCount() == len(ACCEPTANCE_ITEMS)
    assert all(
        window.acceptance_table.item(row, 1).text() == ""
        for row in range(window.acceptance_table.rowCount())
    )
    assert not window.copy_review_prompt_button.isHidden()
    window.copy_review_prompt_button.click()
    prompt = QApplication.clipboard().text()
    assert str(git_repo) in prompt
    assert ".codex/commit-acceptance/pending.json" in prompt
    assert "不要执行 git add、git commit、git push" in prompt
    window.close()
    app.processEvents()


def test_worktree_diff_shows_loading_state_then_populates_files(
    git_repo: Path, tmp_path: Path, monkeypatch,
) -> None:
    app = QApplication.instance() or QApplication([])
    (git_repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    window = MainWindow(RepoStore(tmp_path / "repos.json"))
    window.current_repo = git_repo
    worktree = list_commits(git_repo)[0]
    streams = []
    monkeypatch.setattr(
        window,
        "_start",
        lambda work, done, label, failed=None: streams.append((work, done, failed)),
    )

    window.show_commit(worktree)

    assert len(streams) == 1
    assert not window.diff_progress.isHidden()
    assert "正在加载工作区文件差异" in window.diff.toPlainText()
    streams[0][1](streams[0][0]())
    assert window.diff_progress.isHidden()
    assert window.files_list.count() == 1
    assert "tracked.txt" in window.files_list.item(0).text()
    assert "+changed" in window.diff.toPlainText()
    window.close()
    app.processEvents()


def test_periodic_worktree_diff_refresh_does_not_show_progress(
    git_repo: Path, tmp_path: Path, monkeypatch,
) -> None:
    app = QApplication.instance() or QApplication([])
    (git_repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    window = MainWindow(RepoStore(tmp_path / "repos.json"))
    window.current_repo = git_repo
    worktree = list_commits(git_repo)[0]
    streams = []
    monkeypatch.setattr(
        window,
        "_start",
        lambda work, done, label, failed=None: streams.append((work, done, failed)),
    )

    window.show_commit(worktree)
    streams[0][1](streams[0][0]())
    assert window.diff_progress.isHidden()
    window.show_commit(worktree, force_acceptance=False, show_diff_progress=False)
    assert window.diff_progress.isHidden()
    assert len(streams) == 2
    streams[1][1](streams[1][0]())
    window.close()
    app.processEvents()


def test_repo_page_settings_expand_and_follow_repository(git_repo: Path, tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(RepoStore(tmp_path / "repos.json"))
    info = inspect_repository(git_repo)
    window._show_repo(info)
    assert window.repo_local_path.text() == str(git_repo)
    assert window.repo_remote_url.text() == ""
    assert window.repo_branch.currentText() == info.branch
    assert window.login_identity.objectName() == "loginIdentityTag"
    assert window.login_identity.textFormat() == Qt.PlainText
    assert "默认登录身份" not in window.login_identity.text()
    assert str(git_repo) not in window.repo_meta.text()
    assert "origin" not in window.repo_meta.text()
    assert window.repo_meta.text() == (
        f"分支  {info.branch}    |    {info.state.value}    |    "
        f"ahead {info.ahead} / behind {info.behind}"
    )
    assert window.repo_branch.itemData(window.repo_branch.count() - 1) == "add-branch"
    assert window.repo_branch.itemText(window.repo_branch.count() - 1) == "添加分支…"
    assert not window.branch_drop_button.icon().isNull()
    assert "max-width:100%" in window.readme.document().defaultStyleSheet()
    assert not window.repo_settings_panel.isVisible()
    window.repo_settings_button.setChecked(True)
    assert not window.repo_settings_panel.isHidden()
    window.repo_settings_button.setChecked(False)
    assert window.repo_settings_panel.isHidden()
    window.close()
    app.processEvents()


def test_refresh_keeps_latest_selection_and_ignores_stale_result(
    tmp_path: Path, monkeypatch
) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(RepoStore(tmp_path / "repos.json"))
    first = (tmp_path / "first").resolve()
    second = (tmp_path / "second").resolve()
    first.mkdir()
    second.mkdir()
    window.paths = [first, second]
    streams = []
    selections = []
    monkeypatch.setattr(
        window,
        "_start_stream",
        lambda work, progress, done, _label: streams.append((work, progress, done)),
    )
    monkeypatch.setattr(window, "_repo_selected", lambda row: selections.append(row))

    window.current_repo = second
    window.refresh_all(fetch=False)
    window.current_repo = first
    window.refresh_all(fetch=False)
    streams[1][1](RepoInfo(first, "first-new", state=RepoState.SYNCED))
    assert window.repo_list.item(0).text().startswith("first-new")
    assert window.repo_list.item(1).text().endswith("正在读取...")
    assert Path(window.repo_list.currentItem().data(Qt.UserRole)) == first
    assert selections == []

    streams[0][1](RepoInfo(first, "first-stale", state=RepoState.DISCONNECTED))
    assert Path(window.repo_list.currentItem().data(Qt.UserRole)) == first
    assert window.repo_infos[first].name == "first-new"
    assert selections == []
    streams[1][1](RepoInfo(second, "second-new", state=RepoState.SYNCED))
    assert window.repo_list.item(1).text().startswith("second-new")
    assert selections == []
    window.close()
    app.processEvents()


def test_background_repo_refresh_preserves_open_commit_detail(
    git_repo: Path, tmp_path: Path, monkeypatch,
) -> None:
    app = QApplication.instance() or QApplication([])
    (git_repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    window = MainWindow(RepoStore(tmp_path / "repos.json"))
    window.paths = [git_repo]
    window.current_repo = git_repo
    info = inspect_repository(git_repo)
    window.repo_infos[git_repo] = info
    window._populate_repo_placeholders()
    window._show_repo(info)
    window.show_commit(
        CommitInfo(
            sha="",
            subject="工作区中尚未提交的跟踪文件改动",
            author="本地工作区",
            authored_at="",
            states=(CommitState.UNCOMMITTED,),
            is_worktree=True,
        )
    )
    assert window.details.currentWidget() is window.commit_page
    stream = {}
    monkeypatch.setattr(
        window,
        "_start_stream",
        lambda work, progress, done, _label: stream.update(
            work=work, progress=progress, done=done
        ),
    )

    window.refresh_all(fetch=False)
    stream["progress"](RepoInfo(git_repo, git_repo.name, state=RepoState.DIRTY))

    assert window.details.currentWidget() is window.commit_page
    assert window.commit_title.text() == "工作区中尚未提交的跟踪文件改动"
    window.close()
    app.processEvents()


def test_refresh_publishes_all_local_results_before_remote_fetch(
    tmp_path: Path, monkeypatch
) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(RepoStore(tmp_path / "repos.json"))
    first = (tmp_path / "first").resolve()
    second = (tmp_path / "second").resolve()
    window.paths = [first, second]
    stream = {}
    calls = []
    emitted = []

    def inspect(path, fetch=False):
        calls.append((path, fetch))
        return RepoInfo(path, path.name, state=RepoState.SYNCED)

    monkeypatch.setattr(main_window_module, "inspect_repository", inspect)
    monkeypatch.setattr(
        window,
        "_start_stream",
        lambda work, progress, done, _label: stream.update(
            work=work, progress=progress, done=done
        ),
    )
    window.refresh_all(fetch=True)
    stream["work"](emitted.append)
    stream["done"](None)

    assert calls[:2] == [(first, False), (second, False)]
    assert sorted(calls[2:]) == sorted([(first, True), (second, True)])
    assert [info.path for info in emitted[:2]] == [first, second]
    assert not window._fetch_in_flight
    window.close()
    app.processEvents()


def test_focus_refresh_uses_cooldown_and_coalesces_in_flight_fetch(
    tmp_path: Path, monkeypatch
) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(RepoStore(tmp_path / "repos.json"))
    window.paths = [(tmp_path / "repo").resolve()]
    starts = []
    monkeypatch.setattr(
        window,
        "_start_stream",
        lambda work, progress, done, label: starts.append((work, progress, done, label)),
    )

    window._fetch_in_flight = True
    window.refresh_all(fetch=True)
    assert starts == []

    window._fetch_in_flight = False
    window._last_fetch_completed = monotonic()
    window.refresh_all(fetch=True)
    assert len(starts) == 1
    assert starts[0][3] == "正在读取仓库..."
    window.close()
    app.processEvents()


def test_automatic_refresh_skips_disconnected_repository(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(RepoStore(tmp_path / "repos.json"))
    missing = (tmp_path / "renamed-repository").resolve()
    disconnected = RepoInfo(
        missing,
        missing.name,
        state=RepoState.DISCONNECTED,
        error="找不到仓库目录，文件夹可能已被移动、重命名或删除。",
    )
    window.paths = [missing]
    window.current_repo = missing
    window.repo_infos[missing] = disconnected
    stream = {}
    calls = []
    monkeypatch.setattr(
        main_window_module,
        "inspect_repository",
        lambda path, fetch=False: calls.append((path, fetch)),
    )
    monkeypatch.setattr(
        window,
        "_start_stream",
        lambda work, progress, done, label: stream.update(
            work=work, progress=progress, done=done, label=label
        ),
    )

    window.refresh_all(fetch=True)
    emitted = []
    stream["work"](emitted.append)

    assert calls == []
    assert emitted == [disconnected]
    window._poll_worktree_commits()
    assert not window._commit_refresh_in_flight
    window.close()
    app.processEvents()


def test_stale_fetch_completion_still_releases_in_flight_guard(
    tmp_path: Path, monkeypatch,
) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(RepoStore(tmp_path / "repos.json"))
    repo = (tmp_path / "repo").resolve()
    repo.mkdir()
    window.paths = [repo]
    streams = []
    monkeypatch.setattr(
        window,
        "_start_stream",
        lambda work, progress, done, label: streams.append((work, progress, done, label)),
    )

    window.refresh_all(fetch=True)
    assert window._fetch_in_flight
    window.refresh_all(fetch=False)
    streams[0][2](None)

    assert not window._fetch_in_flight
    window.close()
    app.processEvents()


def test_worktree_commit_refreshes_while_window_remains_open(
    git_repo: Path, tmp_path: Path, monkeypatch,
) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(RepoStore(tmp_path / "repos.json"))
    window.current_repo = git_repo
    streams = []
    monkeypatch.setattr(
        window,
        "_start",
        lambda work, done, label, failed=None: streams.append((work, done, failed)),
    )

    window._request_commits(git_repo)
    assert len(streams) == 1
    assert window._commit_refresh_in_flight
    streams[0][1](list_commits(git_repo))
    assert not window._commit_refresh_in_flight
    assert not window.commits[0].is_worktree

    (git_repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    window._request_commits(git_repo)
    streams[1][1](list_commits(git_repo))
    assert window.commits[0].is_worktree
    window.close()
    app.processEvents()


def test_selected_worktree_acceptance_reloads_when_pending_file_changes(
    git_repo: Path, tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    (git_repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    window = MainWindow(RepoStore(tmp_path / "repos.json"))
    window.current_repo = git_repo
    worktree = list_commits(git_repo)[0]
    window.show_commit(worktree)
    assert window.acceptance_table.item(0, 1).text() == ""
    assert window.acceptance_path_label.isReadOnly()
    window.acceptance_table.item(0, 1).setText("unsaved draft")
    window._poll_selected_acceptance()
    assert window.acceptance_table.item(0, 1).text() == "unsaved draft"

    save_acceptance(
        git_repo,
        AcceptanceReview(
            schema_version=1,
            commit="pending",
            reviewed_at="2026-09-21T12:00:00+08:00",
            items=tuple(
                AcceptanceItem(item, f"answer-{index}")
                for index, item in enumerate(ACCEPTANCE_ITEMS)
            ),
        ),
    )
    window._poll_selected_acceptance()

    assert window.acceptance_table.item(0, 1).text() == "answer-0"
    assert window.acceptance_table.item(len(ACCEPTANCE_ITEMS) - 1, 1).text() == "answer-10"
    assert window.acceptance_path_label.text().endswith(
        ".codex/commit-acceptance/pending.json"
    )
    window.close()
    app.processEvents()


def test_add_branch_option_prompts_and_prepares_real_command(
    git_repo: Path, tmp_path: Path, monkeypatch
) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(RepoStore(tmp_path / "repos.json"))
    info = inspect_repository(git_repo)
    window.current_repo = git_repo
    window.repo_infos = {git_repo: info}
    window._show_repo(info)
    monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("feat/from-menu", True))
    observed = {}

    def capture(title, command, warning, operation):
        observed.update(title=title, command=command, warning=warning, operation=operation)

    monkeypatch.setattr(window, "_confirm_git", capture)
    window._repo_branch_activated(window.repo_branch.count() - 1)
    assert observed["title"] == "确认添加分支"
    assert observed["command"] == "git switch -c feat/from-menu"
    assert "创建并切换" in observed["warning"]
    window.close()
    app.processEvents()


def test_repo_settings_show_unborn_current_branch(tmp_path: Path) -> None:
    import subprocess

    app = QApplication.instance() or QApplication([])
    repo = tmp_path / "unborn"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    window = MainWindow(RepoStore(tmp_path / "repos.json"))
    window._show_repo(inspect_repository(repo))
    assert window.repo_branch.currentText() == "main"
    window.close()
    app.processEvents()


def test_settings_dialog_does_not_display_saved_token(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    store = CredentialStore(tmp_path / "hosts.json")
    store.save("https://github.com", "alice", "private-token", email="alice@example.com")
    dialog = CredentialDialog(store)
    assert dialog.hosts.verticalScrollMode() == QAbstractItemView.ScrollPerPixel
    dialog.hosts.setCurrentRow(0)
    assert dialog.address.text() == "https://github.com"
    assert dialog.username.text() == "alice"
    assert dialog.email.text() == "alice@example.com"
    assert dialog.token.text() == ""
    assert dialog.source.currentIndex() == 0
    assert dialog.token_file.text() == store.hosts()[0].token_file
    dialog.source.setCurrentIndex(1)
    assert not dialog.file_row.isHidden()
    dialog.close()
    app.processEvents()


def test_settings_defaults_to_github_and_clicking_record_does_not_delete_it(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    store = CredentialStore(tmp_path / "hosts.json")
    store.save("https://github.com", "alice", "private-token", email="alice@example.com")
    dialog = CredentialDialog(store)
    assert dialog.address.text() == "https://github.com"
    dialog.resize(600, 500)
    dialog.show()
    app.processEvents()
    item = dialog.hosts.item(0)
    QTest.mouseClick(dialog.hosts.viewport(), Qt.LeftButton, pos=dialog.hosts.visualItemRect(item).center())
    app.processEvents()
    assert len(store.hosts()) == 1
    assert dialog.hosts.count() == 1
    assert dialog.hosts.item(0).text().startswith("github.com")
    assert dialog.hosts.currentItem() is item
    assert dialog.email.text() == "alice@example.com"
    QTest.mouseClick(dialog.hosts.viewport(), Qt.LeftButton, pos=dialog.hosts.visualItemRect(item).center())
    app.processEvents()
    assert dialog.hosts.currentItem() is None
    assert dialog.address.text() == "https://github.com"
    assert dialog.username.text() == ""
    assert dialog.email.text() == ""
    assert len(store.hosts()) == 1
    dialog.close()
    app.processEvents()


def test_settings_dialog_saves_selected_file_and_manual_token(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    store = CredentialStore(tmp_path / "hosts.json")
    source = tmp_path / "token.txt"
    source.write_text("fixture-token\n", encoding="utf-8")
    dialog = CredentialDialog(store)
    dialog.address.setText("https://gitlab.example.com")
    dialog.username.setText("alice")
    dialog.email.setText("alice@example.com")
    dialog.source.setCurrentIndex(1)
    dialog.token_file.setText(str(source))
    dialog._save()
    assert store.hosts()[0].token_file == str(source)
    assert dialog.token.text() == ""
    dialog.source.setCurrentIndex(0)
    dialog.token.setText("replacement-token")
    dialog._save()
    assert store.hosts()[0].managed
    assert store.answer_prompt("Password for 'https://alice@gitlab.example.com': ") == "replacement-token"
    assert source.read_text(encoding="utf-8").strip() == "fixture-token"
    dialog.hosts.setCurrentRow(0)
    dialog.email.setText("updated@example.com")
    dialog.token.clear()
    dialog._save()
    assert store.hosts()[0].email == "updated@example.com"
    assert store.answer_prompt("Password for 'https://alice@gitlab.example.com': ") == "replacement-token"
    dialog.close()
    app.processEvents()


def test_commit_row_shows_only_the_applicable_material_sync_action() -> None:
    app = QApplication.instance() or QApplication([])
    selected: list[str] = []
    commit = CommitInfo(
        sha="a" * 40,
        subject="merge feature",
        author="Test User",
        authored_at="2026-09-14T12:00:00+08:00",
        states=(CommitState.COMMITTED,),
        parents=("b" * 40, "c" * 40),
        lane_count=2,
        graph_edges=((0, 0), (0, 1)),
        needs_push=True,
    )
    row = CommitRow(
        commit,
        lambda: selected.append(commit.sha),
        lambda: None,
        lambda: None,
        lambda: None,
    )
    row.resize(380, 108)
    canvas = QPixmap(row.size())
    canvas.fill()
    row.render(canvas)
    assert row.pull_button is None
    assert row.push_button is not None
    assert row.commit_button is None
    assert not row.push_button.icon().isNull()
    assert row.surface.width() < row.width()
    assert row.version_label.text() == "aaaaaaaa"
    assert row.time_label is not None
    assert row.time_label.text() == "2026-09-14 12:00"
    assert not row.surface.property("selected")
    row.set_selected(True)
    assert row.surface.property("selected") is True
    assert row.subject_label.wordWrap()
    row.surface.click()
    assert selected == [commit.sha]
    row.close()

    remote = CommitInfo(
        sha="d" * 40,
        subject="remote change",
        author="Remote User",
        authored_at="2026-09-15T00:00:00+08:00",
        states=(CommitState.UNFETCHED,),
        needs_pull=True,
    )
    remote_row = CommitRow(remote, lambda: None, lambda: None, lambda: None, lambda: None)
    assert remote_row.pull_button is not None
    assert remote_row.push_button is None
    remote_row.close()

    worktree = CommitInfo(
        sha="",
        subject="tracked changes",
        author="Local",
        authored_at="",
        states=(CommitState.UNCOMMITTED,),
        is_worktree=True,
    )
    worktree_row = CommitRow(worktree, lambda: None, lambda: None, lambda: None, lambda: None)
    assert worktree_row.commit_button is not None
    assert worktree_row.pull_button is None
    assert worktree_row.time_label is None
    worktree_row.close()
    app.processEvents()


def test_acceptance_corner_uses_the_shared_blue_green_table_surface() -> None:
    assert "QTableCornerButton::section" in APP_STYLE
    assert "#e2f0ef" in APP_STYLE


def test_acceptance_table_allows_horizontal_scrolling_for_long_content(
    tmp_path: Path,
) -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(RepoStore(tmp_path / "repos.json"))
    assert window.acceptance_table.horizontalScrollBarPolicy() == Qt.ScrollBarAsNeeded
    assert window.acceptance_table.verticalScrollMode() == QAbstractItemView.ScrollPerPixel
    assert window.acceptance_table.horizontalScrollMode() == QAbstractItemView.ScrollPerPixel
    assert not window.acceptance_table.wordWrap()
    assert window.acceptance_table.horizontalHeader().sectionResizeMode(0) == QHeaderView.ResizeToContents
    assert window.acceptance_table.horizontalHeader().sectionResizeMode(1) == QHeaderView.Stretch
    assert window.acceptance_table.columnWidth(0) < 300
    assert window.acceptance_table.columnWidth(1) >= 500
    window.close()
    app.processEvents()


def test_commit_time_format_keeps_unrecognized_values() -> None:
    assert format_commit_time("2026-09-20T09:08:07+08:00") == "2026-09-20 09:08"
    assert format_commit_time("unknown") == "unknown"


def test_repo_state_delegate_paints_text_when_selected() -> None:
    app = QApplication.instance() or QApplication([])
    repo_list = QListWidget()
    item = QListWidgetItem("断线仓库")
    item.setBackground(QColor("#e8ebee"))
    item.setForeground(QColor("#53606c"))
    repo_list.addItem(item)
    canvas = QPixmap(240, 50)
    canvas.fill(Qt.white)
    option = QStyleOptionViewItem()
    option.rect = canvas.rect()
    option.widget = repo_list
    option.state = QStyle.State_Enabled | QStyle.State_Selected
    painter = QPainter(canvas)
    RepoStateDelegate(repo_list).paint(painter, option, repo_list.model().index(0, 0))
    painter.end()

    dark_pixels = 0
    image = canvas.toImage()
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            dark_pixels += int(max(color.red(), color.green(), color.blue()) < 150)
    assert dark_pixels > 20
    border = image.pixelColor(4, image.height() // 2)
    assert border.green() > border.red()
    assert border.green() > border.blue()
    app.processEvents()
