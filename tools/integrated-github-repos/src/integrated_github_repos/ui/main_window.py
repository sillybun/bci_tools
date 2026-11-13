from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any

from PySide6.QtCore import QEvent, QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QPalette,
    QPen,
    QTextCharFormat,
    QTextCursor,
    QTextOption,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..credentials import CredentialStore, normalize_host
from ..git_service import (
    ACCEPTANCE_ITEMS,
    BRANCH_PREFIXES,
    GitError,
    acceptance_path,
    acceptance_review_prompt,
    branch_switch_args,
    changed_files,
    clone_repository,
    commit_all_changes,
    configure_origin,
    create_branch,
    create_branch_args,
    diff_text,
    discover_repositories,
    initialize_from_local,
    inspect_repository,
    list_branches,
    list_commits,
    load_acceptance,
    normalize_remote_url,
    pull,
    push,
    read_readme,
    repository_name_from_url,
    save_acceptance,
    switch_branch,
    validate_initial_remote,
)
from ..models import (
    AcceptanceItem,
    AcceptanceReview,
    CommitInfo,
    ContributionDay,
    RepoInfo,
    RepoState,
)
from ..repo_store import RepoStore
from .material_icons import material_icon
from .theme import STATE_COLORS
from .widgets import CommitRow

FETCH_COOLDOWN_SECONDS = 30.0


class WorkerSignals(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(object)


class Worker(QRunnable):
    def __init__(self, function: Callable[[], Any]) -> None:
        super().__init__()
        self.function = function
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            self.signals.done.emit(self.function())
        except Exception as exc:  # noqa: BLE001 - worker must relay all failures to the GUI.
            self.signals.failed.emit(str(exc))


class StreamingWorker(QRunnable):
    def __init__(self, function: Callable[[Callable[[Any], None]], Any]) -> None:
        super().__init__()
        self.function = function
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            self.signals.done.emit(self.function(self.signals.progress.emit))
        except Exception as exc:  # noqa: BLE001 - worker must relay all failures to the GUI.
            self.signals.failed.emit(str(exc))


class RepoStateDelegate(QStyledItemDelegate):
    """Keep each repository's state color visible even while it is selected."""

    def paint(self, painter: Any, option: QStyleOptionViewItem, index: Any) -> None:
        option = QStyleOptionViewItem(option)
        self.initStyleOption(option, index)
        background = index.data(Qt.BackgroundRole)
        foreground = index.data(Qt.ForegroundRole)
        selected = bool(option.state & QStyle.State_Selected)
        surface = option.rect.adjusted(3, 2, -3, -2)
        if background:
            painter.save()
            painter.setPen(Qt.NoPen)
            painter.setBrush(background)
            painter.drawRoundedRect(surface, 8, 8)
            painter.restore()
        if selected:
            painter.save()
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("#168a78"), 2))
            painter.drawRoundedRect(surface.adjusted(1, 1, -1, -1), 7, 7)
            painter.restore()
        option.state &= ~QStyle.State_Selected
        option.state &= ~QStyle.State_MouseOver
        option.state &= ~QStyle.State_HasFocus
        if foreground:
            color = foreground.color() if hasattr(foreground, "color") else foreground
            option.palette.setColor(QPalette.Text, color)
        option.backgroundBrush = QBrush(Qt.NoBrush)
        option.rect = option.rect.adjusted(12, 0, -6, 0)
        QApplication.style().drawControl(QStyle.CE_ItemViewItem, option, painter)


class CloneDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("添加网络仓库")
        self.setMinimumWidth(520)
        form = QFormLayout(self)
        self.url = QLineEdit()
        self.url.setPlaceholderText("GitHub / GitLab 的 HTTPS 或 SSH 仓库地址")
        self.initialize_from_local = QCheckBox("从本地仓库为初始化")
        self.initialize_from_local.toggled.connect(self._toggle_initial_mode)
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        self.parent_path = QLineEdit()
        browse = QPushButton("选择...")
        browse.clicked.connect(self._browse)
        path_layout.addWidget(self.parent_path, 1)
        path_layout.addWidget(browse)
        self.prefix = QComboBox()
        self.prefix.setStyleSheet(
            "QComboBox::drop-down { width: 0px; border: 0; }"
            "QComboBox::down-arrow { width: 0px; height: 0px; }"
        )
        for value in BRANCH_PREFIXES:
            self.prefix.addItem(value)
            if value in {"hotfix", "test", "build"}:
                self.prefix.insertSeparator(self.prefix.count())
        self.branch_name = QLineEdit()
        self.branch_name.setPlaceholderText("例如 repo-manager；none 时留空")
        self.prefix.currentTextChanged.connect(
            lambda value: self.branch_name.setEnabled(value != "none")
        )
        self.branch_name.setEnabled(False)
        form.addRow("仓库地址", self.url)
        form.addRow("", self.initialize_from_local)
        self.path_label = QLabel("本地父文件夹")
        form.addRow(self.path_label, path_row)
        branch_row = QWidget()
        branch_layout = QHBoxLayout(branch_row)
        branch_layout.setContentsMargins(0, 0, 0, 0)
        self.prefix_drop_button = QToolButton()
        self.prefix_drop_button.setObjectName("cloneBranchDropButton")
        self.prefix_drop_button.setIcon(material_icon("menu-down", "#111820"))
        self.prefix_drop_button.setIconSize(QSize(20, 20))
        self.prefix_drop_button.setToolTip("展开分支类型")
        self.prefix_drop_button.clicked.connect(self.prefix.showPopup)
        branch_layout.addWidget(self.prefix, 1)
        branch_layout.addWidget(self.prefix_drop_button)
        form.addRow("新建分支类型", branch_row)
        form.addRow("分支名称", self.branch_name)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _browse(self) -> None:
        title = "选择本地初始仓库" if self.initialize_from_local.isChecked() else "选择本地父文件夹"
        selected = QFileDialog.getExistingDirectory(self, title)
        if selected:
            self.parent_path.setText(selected)

    def _toggle_initial_mode(self, enabled: bool) -> None:
        self.path_label.setText("本地初始仓库" if enabled else "本地父文件夹")
        self.parent_path.setPlaceholderText(
            "选择已有的本地 Git 仓库" if enabled else "选择用于创建克隆目录的父文件夹"
        )

    def _validate(self) -> None:
        if not self.url.text().strip() or not Path(self.parent_path.text()).is_dir():
            target = "本地初始仓库" if self.initialize_from_local.isChecked() else "本地父文件夹"
            QMessageBox.warning(self, "信息不完整", f"请填写仓库地址并选择有效的{target}。")
            return
        if self.initialize_from_local.isChecked() and not (Path(self.parent_path.text()) / ".git").exists():
            QMessageBox.warning(self, "信息不完整", "本地初始仓库必须包含 .git 目录。")
            return
        if self.prefix.currentText() != "none" and not self.branch_name.text().strip():
            QMessageBox.warning(self, "信息不完整", "选择分支类型后需要填写分支名称。")
            return
        self.accept()


class CredentialDialog(QDialog):
    def __init__(self, store: CredentialStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store
        self.setWindowTitle("Git 凭据设置")
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)
        self.hosts = QListWidget()
        self.hosts.setObjectName("credentialHosts")
        self.hosts.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.hosts.verticalScrollBar().setSingleStep(6)
        self.hosts.currentRowChanged.connect(self._selected)
        self.hosts.itemClicked.connect(self._item_clicked)
        self.hosts.viewport().installEventFilter(self)
        self._clear_on_click = False
        layout.addWidget(self.hosts)
        form = QFormLayout()
        self.address = QLineEdit()
        self.address.setText("https://github.com")
        self.address.setPlaceholderText("https://github.com 或 https://gitlab.example.com")
        self.username = QLineEdit()
        self.username.setPlaceholderText("GitHub 用户名 / GitLab 用户名")
        self.email = QLineEdit()
        self.email.setPlaceholderText("用于完整身份显示，例如 you@example.com")
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.Password)
        self.token.setPlaceholderText("Personal Access Token（保存为本地纯文本）")
        self.source = QComboBox()
        self.source.addItems(["手动输入", "选择已有文本文件"])
        self.source.currentIndexChanged.connect(self._source_changed)
        file_row = QWidget()
        file_layout = QHBoxLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.token_file = QLineEdit()
        self.token_file.setPlaceholderText("选择只含一行 token 的文本文件")
        browse = QPushButton("选择...")
        browse.clicked.connect(self._browse_token)
        file_layout.addWidget(self.token_file, 1)
        file_layout.addWidget(browse)
        self.file_row = file_row
        self.allow_http = QCheckBox("允许通过未加密 HTTP 传输 token（不安全）")
        form.addRow("服务地址", self.address)
        form.addRow("用户名", self.username)
        form.addRow("邮箱", self.email)
        form.addRow("token 来源", self.source)
        form.addRow("新 token", self.token)
        form.addRow("文本文件", file_row)
        form.addRow("", self.allow_http)
        self.source_form = form
        layout.addLayout(form)
        actions = QHBoxLayout()
        save = QPushButton("保存凭据")
        save.clicked.connect(self._save)
        remove = QPushButton("删除选中凭据")
        remove.clicked.connect(self._delete)
        actions.addWidget(save)
        actions.addWidget(remove)
        actions.addStretch()
        layout.addLayout(actions)
        note = QLabel(
            "不读取钥匙串。手动输入会保存在应用数据目录的纯文本文件中；选择文件只保存路径，"
            "原文件由你管理。请限制文件权限并避免同步或分享。HTTP 会在网络中明文传输 token；"
            "SSH 仓库继续使用系统 SSH 密钥。"
        )
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        self._source_changed()
        self._reload()

    def _source_changed(self) -> None:
        manual = self.source.currentIndex() == 0
        self.source_form.setRowVisible(self.token, manual)
        self.source_form.setRowVisible(self.file_row, not manual)

    def _browse_token(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(self, "选择 token 文本文件")
        if selected:
            self.token_file.setText(selected)

    def _reload(self) -> None:
        self.hosts.clear()
        for item in self.store.hosts():
            protocol = "HTTP（不安全）" if item.allow_http else "HTTPS"
            identity = f"{item.username} <{item.email}>" if item.email else item.username
            entry = QListWidgetItem(f"{item.host}    {identity}    {protocol}")
            entry.setData(Qt.UserRole, item.host)
            self.hosts.addItem(entry)

    def _selected(self, row: int) -> None:
        item = self.hosts.item(row)
        if item is None:
            self._clear_form()
            return
        record = next((host for host in self.store.hosts() if host.host == item.data(Qt.UserRole)), None)
        if record:
            self.address.setText(f"{record.scheme}://{record.host}")
            self.username.setText(record.username)
            self.email.setText(record.email)
            self.allow_http.setChecked(record.allow_http)
            self.token.clear()
            self.source.setCurrentIndex(0 if record.managed else 1)
            self.token_file.setText(record.token_file)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.hosts.viewport() and event.type() == QEvent.MouseButtonPress:
            item = self.hosts.itemAt(event.position().toPoint())
            self._clear_on_click = item is not None and item.isSelected()
        return super().eventFilter(watched, event)

    def _item_clicked(self, _item: QListWidgetItem) -> None:
        if self._clear_on_click:
            self.hosts.clearSelection()
            self.hosts.setCurrentRow(-1)
        self._clear_on_click = False

    def _clear_form(self) -> None:
        self.address.setText("https://github.com")
        self.username.clear()
        self.email.clear()
        self.token.clear()
        self.token_file.clear()
        self.allow_http.setChecked(False)
        self.source.setCurrentIndex(0)

    def _save(self) -> None:
        try:
            _host, scheme = normalize_host(self.address.text())
            if scheme == "http" and not self.allow_http.isChecked():
                raise ValueError("HTTP 会明文传输 token；请使用 HTTPS，或明确勾选风险选项")
            self.store.save(
                self.address.text(), self.username.text(),
                email=self.email.text(),
                token=self.token.text() if self.source.currentIndex() == 0 else "",
                allow_http=self.allow_http.isChecked(),
                token_file=Path(self.token_file.text()) if self.source.currentIndex() == 1 else None,
            )
        except Exception as exc:  # noqa: BLE001 - report inaccessible token files in the UI.
            QMessageBox.warning(self, "无法保存凭据", str(exc))
            return
        self.token.clear()
        self._reload()

    def _delete(self) -> None:
        item = self.hosts.currentItem()
        if item is None:
            return
        host = item.data(Qt.UserRole)
        if QMessageBox.question(
            self, "删除凭据", f"删除 {host} 的配置？应用创建的 token 文件将删除；自选文件不受影响。"
        ) != QMessageBox.Yes:
            return
        try:
            self.store.delete(host)
        except Exception as exc:  # noqa: BLE001 - filesystem failures need a visible error.
            QMessageBox.warning(self, "无法删除凭据", str(exc))
            return
        self.token.clear()
        self.address.clear()
        self.username.clear()
        self.email.clear()
        self.token_file.clear()
        self._reload()


class CommitDialog(QDialog):
    def __init__(self, repo: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        pending_review = load_acceptance(repo, "")
        self.setWindowTitle("提交所有已跟踪改动")
        self.setMinimumSize(700, 560)
        layout = QVBoxLayout(self)
        warning = QLabel(
            "将执行 git add --all 和 git commit，包含未跟踪文件；不会自动 push 到远端。验收表可选，填写后会保存为 pending.json，待产生 commit 后转存到对应 SHA。"
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        self.message = QLineEdit()
        self.message.setPlaceholderText("提交说明")
        layout.addWidget(self.message)
        self.table = QTableWidget(len(ACCEPTANCE_ITEMS), 2)
        self.table.setHorizontalHeaderLabels(["项目", "回答"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        pending_answers = (
            {entry.item: entry.answer for entry in pending_review.items}
            if pending_review is not None
            else {}
        )
        for row, item in enumerate(ACCEPTANCE_ITEMS):
            fixed = QTableWidgetItem(item)
            fixed.setFlags(fixed.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, fixed)
            self.table.setItem(row, 1, QTableWidgetItem(pending_answers.get(item, "")))
        layout.addWidget(self.table, 1)
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("补充说明（可选）")
        if pending_review is not None:
            self.notes.setPlainText(pending_review.notes)
        self.notes.setMaximumHeight(80)
        layout.addWidget(self.notes)
        path_label = QLabel(f"仓库：{repo}")
        path_label.setObjectName("muted")
        layout.addWidget(path_label)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        buttons.button(QDialogButtonBox.Ok).setText("确认提交")
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate(self) -> None:
        if not self.message.text().strip():
            QMessageBox.warning(self, "缺少提交说明", "提交说明不能为空。")
            return
        self.accept()

    def review(self) -> AcceptanceReview | None:
        answers = tuple(
            self.table.item(row, 1).text().strip() for row in range(self.table.rowCount())
        )
        if not any(answers) and not self.notes.toPlainText().strip():
            return None
        return AcceptanceReview(
            schema_version=1,
            commit="pending",
            reviewed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            items=tuple(
                AcceptanceItem(self.table.item(row, 0).text(), self.table.item(row, 1).text().strip())
                for row in range(self.table.rowCount())
            ),
            notes=self.notes.toPlainText().strip(),
        )


class ContributionGrid(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.days: list[ContributionDay] = []
        self.setFixedSize(112, 110)

    def set_days(self, days: list[ContributionDay]) -> None:
        self.days = days
        self.update()

    def paintEvent(self, event: Any) -> None:
        from PySide6.QtGui import QPainter

        super().paintEvent(event)
        painter = QPainter(self)
        cell, gap, left, top = 12, 3, 28, 3
        maximum = max((day.count for day in self.days), default=0)
        colors = ["#edf4f4", "#c9e5e1", "#91cbc4", "#4da49d", "#176b63"]
        painter.setPen(QColor("#66727f"))
        for index, label in enumerate(("一", "三", "五", "日")):
            painter.drawText(4, top + index * 2 * (cell + gap) + 11, label)
        for index, day in enumerate(self.days):
            col, row = divmod(index, 7)
            level = 0 if day.count == 0 or maximum == 0 else min(4, 1 + int(day.count * 3 / maximum))
            painter.setBrush(QColor(colors[level]))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(
                left + col * (cell + gap), top + row * (cell + gap), cell, cell, 3, 3
            )


class MainWindow(QMainWindow):
    def __init__(self, store: RepoStore | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Integrated GitHub Repos")
        self.resize(1280, 800)
        self.setMinimumSize(980, 640)
        self.store = store or RepoStore()
        self.paths = self.store.load()
        self.repo_infos: dict[Path, RepoInfo] = {}
        self.current_repo: Path | None = None
        self.commits: list[CommitInfo] = []
        self.visible_commits = 5
        self._refresh_generation = 0
        self._fetch_in_flight = False
        self._last_fetch_completed = 0.0
        self._commit_refresh_in_flight = False
        self._commit_refresh_generation = 0
        self._acceptance_signature: tuple[object, ...] | None = None
        self._diff_load_generation = 0
        self._diff_load_in_flight = False
        self._diff_load_in_flight_generation: int | None = None
        self._worktree_diff_loaded_repo: Path | None = None
        self.pool = QThreadPool.globalInstance()
        self.workers: set[Worker] = set()
        self._build_toolbar()
        self._build_ui()
        self._worktree_timer = QTimer(self)
        self._worktree_timer.setInterval(1500)
        self._worktree_timer.timeout.connect(self._poll_worktree_commits)
        self._worktree_timer.timeout.connect(self._poll_selected_acceptance)
        self._worktree_timer.start()
        self._populate_repo_placeholders()
        if self.paths:
            self.refresh_all(fetch=False)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(17, 17))
        self.addToolBar(toolbar)
        add_folder = QAction(material_icon("folder-plus-outline"), "添加文件夹", self)
        add_folder.triggered.connect(self.add_folder)
        scan = QAction(material_icon("folder-search-outline"), "扫描磁盘", self)
        scan.triggered.connect(self.scan_disk)
        clone = QAction(material_icon("cloud-download-outline"), "克隆网络仓库", self)
        clone.triggered.connect(self.clone_remote)
        refresh = QAction(material_icon("refresh"), "刷新", self)
        refresh.triggered.connect(lambda: self.refresh_all(fetch=True, force_fetch=True))
        settings = QAction(material_icon("cog-outline"), "设置", self)
        settings.triggered.connect(self.open_settings)
        toolbar.addActions([add_folder, scan, clone])
        toolbar.addSeparator()
        toolbar.addAction(refresh)
        toolbar.addAction(settings)

    def open_settings(self) -> None:
        CredentialDialog(CredentialStore(), self).exec()
        if self.current_repo and (info := self.repo_infos.get(self.current_repo)):
            self._show_repo(info, navigate=self.details.currentWidget() is not self.commit_page)

    def _build_ui(self) -> None:
        shell = QWidget()
        shell.setObjectName("shell")
        shell_layout = QHBoxLayout(shell)
        shell_layout.setContentsMargins(14, 14, 14, 14)
        splitter = QSplitter()
        left = QFrame()
        left.setObjectName("leftPane")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 10)
        title = QLabel("仓库")
        title.setObjectName("sectionTitle")
        left_layout.addWidget(title)
        self.repo_list = QListWidget()
        self.repo_list.setSpacing(2)
        self.repo_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.repo_list.verticalScrollBar().setSingleStep(6)
        self.repo_list.horizontalScrollBar().setSingleStep(6)
        self.repo_list.setItemDelegate(RepoStateDelegate(self.repo_list))
        self.repo_list.currentRowChanged.connect(self._repo_selected)
        self.repo_list.itemClicked.connect(self._repo_item_clicked)
        left_layout.addWidget(self.repo_list, 2)
        action_row = QHBoxLayout()
        self.commit_button = QToolButton()
        self.commit_button.setIcon(material_icon("content-save-check-outline"))
        self.commit_button.setToolTip("提交全部改动（不会自动 push）")
        self.commit_button.clicked.connect(self.commit_all)
        self.branch_button = QToolButton()
        self.branch_button.setIcon(material_icon("source-branch"))
        self.branch_button.setToolTip("切换分支")
        self.branch_button.clicked.connect(self.choose_branch)
        self.remove_button = QToolButton()
        self.remove_button.setIcon(material_icon("delete-outline", "#a12622"))
        self.remove_button.setToolTip("从列表移除，不删除仓库")
        self.remove_button.clicked.connect(self.remove_current)
        action_row.addWidget(self.commit_button)
        action_row.addWidget(self.branch_button)
        action_row.addStretch()
        action_row.addWidget(self.remove_button)
        left_layout.addLayout(action_row)
        commits_title = QLabel("提交记录")
        commits_title.setObjectName("sectionTitle")
        left_layout.addWidget(commits_title)
        self.commit_scroll = QScrollArea()
        self.commit_scroll.setObjectName("commitScroll")
        self.commit_scroll.setWidgetResizable(True)
        self.commit_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.commit_scroll.verticalScrollBar().setSingleStep(12)
        self.commit_container = QWidget()
        self.commit_container.setObjectName("commitContainer")
        self.commit_container.setMinimumWidth(0)
        self.commit_layout = QVBoxLayout(self.commit_container)
        self.commit_layout.setContentsMargins(0, 0, 0, 0)
        self.commit_layout.setSpacing(5)
        self.commit_scroll.setWidget(self.commit_container)
        left_layout.addWidget(self.commit_scroll, 3)

        self.details = QStackedWidget()
        self.empty_page = QLabel("从左侧添加或选择一个 Git 仓库")
        self.empty_page.setAlignment(Qt.AlignCenter)
        self.empty_page.setObjectName("muted")
        self.repo_page = self._build_repo_page()
        self.commit_page = self._build_commit_page()
        self.details.addWidget(self.empty_page)
        self.details.addWidget(self.repo_page)
        self.details.addWidget(self.commit_page)
        splitter.addWidget(left)
        splitter.addWidget(self.details)
        splitter.setSizes([410, 870])
        splitter.setStretchFactor(1, 1)
        shell_layout.addWidget(splitter)
        self.setCentralWidget(shell)
        self.statusBar().showMessage("就绪")

    def _build_repo_page(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setObjectName("detailPanel")
        scroll.setWidgetResizable(True)
        page = QWidget()
        page.setObjectName("repoPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 26, 30, 26)
        info_split = QHBoxLayout()
        info_split.setSpacing(28)
        identity_box = QVBoxLayout()
        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        self.repo_name = QLabel()
        self.repo_name.setObjectName("repoTitle")
        title_row.addWidget(self.repo_name, 0, Qt.AlignVCenter)
        self.login_identity = QLabel()
        self.login_identity.setObjectName("loginIdentityTag")
        self.login_identity.setTextFormat(Qt.PlainText)
        self.login_identity.setWordWrap(False)
        title_row.addWidget(self.login_identity, 0, Qt.AlignVCenter)
        title_row.addStretch()
        identity_box.addLayout(title_row)
        self.repo_meta = QLabel()
        self.repo_meta.setObjectName("muted")
        self.repo_meta.setWordWrap(True)
        identity_box.addWidget(self.repo_meta)
        contributors_label = QLabel("贡献者")
        contributors_label.setObjectName("sectionTitle")
        identity_box.addSpacing(14)
        identity_box.addWidget(contributors_label)
        self.contributors = QLabel()
        self.contributors.setWordWrap(True)
        identity_box.addWidget(self.contributors)
        identity_box.addStretch()
        info_split.addLayout(identity_box, 1)
        activity_card = QFrame()
        activity_card.setObjectName("activityCard")
        activity_box = QVBoxLayout(activity_card)
        activity_box.setContentsMargins(14, 12, 14, 10)
        activity_box.setSpacing(5)
        activity_label = QLabel("最近 35 天更新")
        activity_label.setObjectName("sectionTitle")
        activity_box.addWidget(activity_label)
        self.grid = ContributionGrid()
        activity_box.addWidget(self.grid)
        info_split.addWidget(activity_card, 0, Qt.AlignTop | Qt.AlignRight)
        layout.addLayout(info_split)
        self.repo_settings_button = QToolButton()
        self.repo_settings_button.setObjectName("repoSettingsToggle")
        self.repo_settings_button.setIcon(material_icon("cog-outline"))
        self.repo_settings_button.setText("仓库设置")
        self.repo_settings_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.repo_settings_button.setCheckable(True)
        self.repo_settings_button.toggled.connect(self._toggle_repo_settings)
        layout.addWidget(self.repo_settings_button, 0, Qt.AlignLeft)
        self.repo_settings_panel = QFrame()
        self.repo_settings_panel.setObjectName("repoSettingsPanel")
        settings_form = QFormLayout(self.repo_settings_panel)
        settings_form.setContentsMargins(14, 12, 14, 12)
        local_row = QWidget()
        local_layout = QHBoxLayout(local_row)
        local_layout.setContentsMargins(0, 0, 0, 0)
        self.repo_local_path = QLineEdit()
        choose_local = QToolButton()
        choose_local.setIcon(material_icon("folder-outline"))
        choose_local.setToolTip("选择本地 Git 仓库")
        choose_local.clicked.connect(self._browse_repo_path)
        apply_local = QPushButton("应用")
        apply_local.clicked.connect(self._apply_repo_path)
        local_layout.addWidget(self.repo_local_path, 1)
        local_layout.addWidget(choose_local)
        local_layout.addWidget(apply_local)
        settings_form.addRow("本地目录", local_row)
        remote_row = QWidget()
        remote_layout = QHBoxLayout(remote_row)
        remote_layout.setContentsMargins(0, 0, 0, 0)
        self.repo_remote_url = QLineEdit()
        self.repo_remote_url.setPlaceholderText("https://... 或 git@host:group/repo.git")
        apply_remote = QPushButton("应用")
        apply_remote.clicked.connect(self._apply_repo_remote)
        remote_layout.addWidget(self.repo_remote_url, 1)
        remote_layout.addWidget(apply_remote)
        settings_form.addRow("远端 origin", remote_row)
        branch_row = QWidget()
        branch_layout = QHBoxLayout(branch_row)
        branch_layout.setContentsMargins(0, 0, 0, 0)
        branch_layout.setSpacing(4)
        self.repo_branch = QComboBox()
        self.repo_branch.setObjectName("branchSelector")
        self.repo_branch.activated.connect(self._repo_branch_activated)
        self.branch_drop_button = QToolButton()
        self.branch_drop_button.setObjectName("branchDropButton")
        self.branch_drop_button.setIcon(material_icon("menu-down", "#111820"))
        self.branch_drop_button.setIconSize(QSize(20, 20))
        self.branch_drop_button.setFixedWidth(30)
        self.branch_drop_button.setToolTip("展开分支候选")
        self.branch_drop_button.clicked.connect(self.repo_branch.showPopup)
        switch_button = QPushButton("切换")
        switch_button.clicked.connect(self._switch_repo_branch)
        branch_layout.addWidget(self.repo_branch, 1)
        branch_layout.addWidget(self.branch_drop_button)
        branch_layout.addWidget(switch_button)
        settings_form.addRow("当前分支", branch_row)
        self.repo_settings_panel.setVisible(False)
        layout.addWidget(self.repo_settings_panel)
        layout.addSpacing(16)
        self.readme_section = QWidget()
        readme_layout = QVBoxLayout(self.readme_section)
        readme_layout.setContentsMargins(0, 0, 0, 0)
        readme_label = QLabel("README")
        readme_label.setObjectName("sectionTitle")
        readme_layout.addWidget(readme_label)
        self.readme = QTextBrowser()
        self.readme.setObjectName("readme")
        self.readme.setOpenExternalLinks(True)
        self.readme.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.readme.setLineWrapMode(QTextBrowser.WidgetWidth)
        text_option = QTextOption()
        text_option.setWrapMode(QTextOption.WrapAnywhere)
        self.readme.document().setDefaultTextOption(text_option)
        self.readme.document().setDefaultStyleSheet(
            "body{color:#173f46;font-family:-apple-system,Segoe UI,sans-serif;}"
            "h1,h2,h3{color:#123e47;} code,pre{font-family:Menlo,Consolas,monospace;}"
            "body{background:transparent;} pre{padding:10px;}"
            "a{color:#1e7777;} img{max-width:100%; height:auto;}"
        )
        readme_layout.addWidget(self.readme)
        layout.addWidget(self.readme_section, 1)
        scroll.setWidget(page)
        return scroll

    def _build_commit_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("commitPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        self.commit_title = QLabel()
        self.commit_title.setObjectName("repoTitle")
        self.commit_title.setWordWrap(True)
        layout.addWidget(self.commit_title)
        self.commit_meta = QLabel()
        self.commit_meta.setObjectName("muted")
        layout.addWidget(self.commit_meta)
        self.acceptance_section = QWidget()
        acceptance_layout = QVBoxLayout(self.acceptance_section)
        acceptance_layout.setContentsMargins(0, 0, 0, 0)
        self.acceptance_label = QLabel("验收表")
        self.acceptance_label.setObjectName("sectionTitle")
        acceptance_layout.addWidget(self.acceptance_label)
        self.acceptance_table = QTableWidget(0, 2)
        self.acceptance_table.setHorizontalHeaderLabels(["项目", "回答"])
        acceptance_header = self.acceptance_table.horizontalHeader()
        acceptance_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        acceptance_header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.acceptance_table.setColumnWidth(0, 120)
        self.acceptance_table.setColumnWidth(1, 520)
        self.acceptance_table.setWordWrap(False)
        self.acceptance_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.acceptance_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.acceptance_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.acceptance_table.verticalScrollBar().setSingleStep(6)
        self.acceptance_table.horizontalScrollBar().setSingleStep(6)
        self.acceptance_table.setMaximumHeight(260)
        acceptance_layout.addWidget(self.acceptance_table)
        acceptance_actions = QHBoxLayout()
        self.acceptance_path_label = QLineEdit()
        self.acceptance_path_label.setObjectName("muted")
        self.acceptance_path_label.setReadOnly(True)
        self.acceptance_path_label.setToolTip("验收表文件路径，可水平滚动查看完整路径")
        self.save_acceptance_button = QPushButton("保存验收表")
        self.save_acceptance_button.clicked.connect(self.save_current_acceptance)
        self.copy_review_prompt_button = QPushButton("复制审阅提示词")
        self.copy_review_prompt_button.clicked.connect(self.copy_acceptance_review_prompt)
        acceptance_actions.addWidget(self.acceptance_path_label, 1)
        acceptance_actions.addWidget(self.copy_review_prompt_button)
        acceptance_actions.addWidget(self.save_acceptance_button)
        acceptance_layout.addLayout(acceptance_actions)
        layout.addWidget(self.acceptance_section)
        files_label = QLabel("文件改动")
        files_label.setObjectName("sectionTitle")
        layout.addWidget(files_label)
        file_diff_splitter = QSplitter(Qt.Vertical)
        self.files_list = QListWidget()
        self.files_list.setMaximumHeight(150)
        self.files_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.files_list.verticalScrollBar().setSingleStep(6)
        self.files_list.horizontalScrollBar().setSingleStep(6)
        self.files_list.currentRowChanged.connect(self._file_selected)
        self.diff = QPlainTextEdit()
        self.diff.setReadOnly(True)
        self.diff.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.diff.verticalScrollBar().setSingleStep(12)
        self.diff.horizontalScrollBar().setSingleStep(12)
        fixed = QFont("Menlo" if os.name != "nt" else "Consolas")
        fixed.setStyleHint(QFont.Monospace)
        self.diff.setFont(fixed)
        self.diff_progress = QProgressBar()
        self.diff_progress.setObjectName("diffProgress")
        self.diff_progress.setRange(0, 0)
        self.diff_progress.setTextVisible(False)
        self.diff_progress.setVisible(False)
        layout.addWidget(self.diff_progress)
        file_diff_splitter.addWidget(self.files_list)
        file_diff_splitter.addWidget(self.diff)
        file_diff_splitter.setSizes([110, 350])
        layout.addWidget(file_diff_splitter, 1)
        return page

    def _start(
        self,
        function: Callable[[], Any],
        done: Callable[[Any], None],
        label: str,
        failed: Callable[[], None] | None = None,
    ) -> None:
        worker = Worker(function)
        self.workers.add(worker)
        self.statusBar().showMessage(label)

        def finish(value: Any) -> None:
            self.workers.discard(worker)
            self.statusBar().showMessage("就绪", 2500)
            done(value)

        def fail(message: str) -> None:
            self.workers.discard(worker)
            self.statusBar().showMessage("操作失败", 4000)
            if failed:
                failed()
            QMessageBox.critical(self, "Git 操作失败", message)

        worker.signals.done.connect(finish)
        worker.signals.failed.connect(fail)
        self.pool.start(worker)

    def _start_stream(
        self,
        function: Callable[[Callable[[Any], None]], Any],
        progress: Callable[[Any], None],
        done: Callable[[Any], None],
        label: str,
    ) -> None:
        worker = StreamingWorker(function)
        self.workers.add(worker)
        self.statusBar().showMessage(label)

        def finish(value: Any) -> None:
            self.workers.discard(worker)
            self.statusBar().showMessage("就绪", 2500)
            done(value)

        def fail(message: str) -> None:
            self.workers.discard(worker)
            self.statusBar().showMessage("操作失败", 4000)
            QMessageBox.critical(self, "Git 操作失败", message)

        worker.signals.progress.connect(progress)
        worker.signals.done.connect(finish)
        worker.signals.failed.connect(fail)
        self.pool.start(worker)

    def _populate_repo_placeholders(self) -> None:
        self.repo_list.clear()
        for path in self.paths:
            item = QListWidgetItem(f"{path.name}\n正在读取...")
            item.setData(Qt.UserRole, str(path))
            self.repo_list.addItem(item)

    def _repo_item_text(self, info: RepoInfo) -> str:
        suffix = f"  ↑{info.ahead}" if info.ahead else ""
        suffix += f"  ↓{info.behind}" if info.behind else ""
        return f"{info.name}    {info.state.value}{suffix}\n{info.branch or '(detached)'}"

    def refresh_all(self, fetch: bool = True, force_fetch: bool = False) -> None:
        if not self.paths:
            return
        if fetch and self._fetch_in_flight:
            return
        if fetch and not force_fetch and monotonic() - self._last_fetch_completed < FETCH_COOLDOWN_SECONDS:
            fetch = False
        if fetch:
            self._fetch_in_flight = True
        self._refresh_generation += 1
        generation = self._refresh_generation
        paths = tuple(self.paths)

        selected = self.current_repo if self.current_repo in paths else paths[0]
        self.current_repo = selected
        self.repo_list.blockSignals(True)
        try:
            existing_paths = [
                Path(self.repo_list.item(row).data(Qt.UserRole))
                for row in range(self.repo_list.count())
                if self.repo_list.item(row) is not None
            ]
            if existing_paths != list(paths):
                self.repo_list.clear()
                for path in paths:
                    item = QListWidgetItem(f"{path.name}\n正在读取...")
                    item.setData(Qt.UserRole, str(path))
                    self.repo_list.addItem(item)
            for row, path in enumerate(paths):
                item = self.repo_list.item(row)
                if item is None:
                    continue
                info = self.repo_infos.get(path)
                if info:
                    item.setText(self._repo_item_text(info))
                    background, foreground = STATE_COLORS[info.state.value]
                    item.setBackground(QColor(background))
                    item.setForeground(QColor(foreground))
                    item.setToolTip(info.error)
                if path == selected:
                    self.repo_list.setCurrentRow(row)
        finally:
            self.repo_list.blockSignals(False)

        def work(emit: Callable[[RepoInfo], None]) -> None:
            fetchable: list[Path] = []
            for path in paths:
                previous = self.repo_infos.get(path)
                if fetch and not force_fetch and previous and previous.state == RepoState.DISCONNECTED:
                    emit(previous)
                    continue
                try:
                    info = inspect_repository(path, fetch=False)
                except GitError as exc:
                    info = RepoInfo(
                        path=path, name=path.name, state=RepoState.DISCONNECTED, error=str(exc)
                    )
                emit(info)
                if info.state != RepoState.DISCONNECTED:
                    fetchable.append(path)
            if not fetch:
                return

            def fetch_one(path: Path) -> RepoInfo:
                try:
                    return inspect_repository(path, fetch=True)
                except GitError as exc:
                    return RepoInfo(path=path, name=path.name, error=str(exc))

            if not fetchable:
                return
            with ThreadPoolExecutor(max_workers=min(3, len(fetchable))) as executor:
                futures = {executor.submit(fetch_one, path): path for path in fetchable}
                for future in as_completed(futures):
                    try:
                        info = future.result()
                    except Exception as exc:  # noqa: BLE001 - preserve per-repository progress.
                        path = futures[future]
                        info = RepoInfo(path=path, name=path.name, error=str(exc))
                    emit(info)

        def progress(info: RepoInfo) -> None:
            if generation != self._refresh_generation:
                return
            self.repo_infos[info.path] = info
            try:
                row = paths.index(info.path)
            except ValueError:
                return
            item = self.repo_list.item(row)
            if item is None:
                return
            item.setText(self._repo_item_text(info))
            background, foreground = STATE_COLORS[info.state.value]
            item.setBackground(QColor(background))
            item.setForeground(QColor(foreground))
            item.setToolTip(info.error)
            if info.path == self.current_repo:
                self._show_repo(info, navigate=self.details.currentWidget() is not self.commit_page)

        def done(_value: None) -> None:
            if fetch:
                self._fetch_in_flight = False
                self._last_fetch_completed = monotonic()
            if generation != self._refresh_generation:
                return
            path = self.current_repo
            if path is not None and path.is_dir():
                self._request_commits(path)

        self._start_stream(
            work,
            progress,
            done,
            "正在抓取并刷新仓库..." if fetch else "正在读取仓库...",
        )

    def _repo_selected(self, row: int) -> None:
        item = self.repo_list.item(row)
        if not item:
            self.current_repo = None
            self.details.setCurrentWidget(self.empty_page)
            self._render_commits([])
            return
        path = Path(item.data(Qt.UserRole))
        self.current_repo = path
        info = self.repo_infos.get(path)
        if info:
            self._show_repo(info)
        self.visible_commits = 5
        if path.is_dir() and (info is None or info.state != RepoState.DISCONNECTED):
            self._request_commits(path)
        else:
            self._render_commits([])

    def _repo_item_clicked(self, item: QListWidgetItem) -> None:
        """Return to the repository overview even when the clicked row is already selected."""
        path = Path(item.data(Qt.UserRole))
        info = self.repo_infos.get(path)
        if info is None:
            return
        self.current_repo = path
        self._show_repo(info, navigate=True)

    def _poll_worktree_commits(self) -> None:
        info = self.repo_infos.get(self.current_repo) if self.current_repo else None
        if (
            self.isVisible()
            and self.current_repo
            and self.current_repo.is_dir()
            and (info is None or info.state != RepoState.DISCONNECTED)
            and not self._commit_refresh_in_flight
        ):
            self._request_commits(self.current_repo)

    def _poll_selected_acceptance(self) -> None:
        if (
            self.details.currentWidget() is self.commit_page
            and self.current_repo
            and hasattr(self, "selected_commit")
            and self.selected_commit.is_worktree
        ):
            self._load_acceptance(self.selected_commit)

    def _request_commits(self, path: Path) -> None:
        info = self.repo_infos.get(path)
        if not path.is_dir() or (info and info.state == RepoState.DISCONNECTED):
            self._commit_refresh_in_flight = False
            if self.commits:
                self._commits_loaded([])
            return
        if self._commit_refresh_in_flight:
            return
        self._commit_refresh_in_flight = True
        self._commit_refresh_generation += 1
        generation = self._commit_refresh_generation

        def loaded(commits: list[CommitInfo]) -> None:
            self._commit_refresh_in_flight = False
            if generation != self._commit_refresh_generation or self.current_repo != path:
                return
            changed = self._commits_loaded(commits)
            if (
                changed
                and
                self.details.currentWidget() is self.commit_page
                and hasattr(self, "selected_commit")
                and self.selected_commit.is_worktree
            ):
                worktree = next((commit for commit in commits if commit.is_worktree), None)
                if worktree is None:
                    self.details.setCurrentWidget(self.repo_page)
                    self.acceptance_section.setVisible(False)
                else:
                    self.show_commit(
                        worktree, force_acceptance=False, show_diff_progress=False
                    )

        self._start(
            lambda: list_commits(path),
            loaded,
            "正在读取提交记录...",
            failed=lambda: setattr(self, "_commit_refresh_in_flight", False),
        )

    def _show_repo(self, info: RepoInfo, navigate: bool = True) -> None:
        self.repo_name.setText(info.name)
        self.repo_meta.setText(
            f"分支  {info.branch}    |    {info.state.value}    |    "
            f"ahead {info.ahead} / behind {info.behind}"
        )
        identity = CredentialStore().identity_for_remote(info.remote_url)
        self.login_identity.setText(identity)
        self.repo_local_path.setText(str(info.path))
        self.repo_remote_url.setText(info.remote_url)
        self.repo_branch.clear()
        try:
            branches = list_branches(info.path)
        except GitError:
            branches = []
        if info.branch and info.branch != "(detached)" and info.branch not in branches:
            branches.insert(0, info.branch)
        self.repo_branch.addItems(branches)
        if info.branch in branches:
            self.repo_branch.setCurrentText(info.branch)
        self.repo_branch.insertSeparator(self.repo_branch.count())
        self.repo_branch.addItem(material_icon("plus"), "添加分支…", "add-branch")
        self.contributors.setText("\n".join(info.contributors) or "暂无提交者信息")
        self.grid.set_days(info.contributions)
        readme = None
        try:
            readme = read_readme(info.path)
        except GitError as exc:
            self.readme_section.setVisible(True)
            self.readme.setPlainText(str(exc))
        else:
            self.readme_section.setVisible(readme is not None)
        if readme is not None:
            readme_path, markdown = readme
            self.readme.document().setBaseUrl(QUrl.fromLocalFile(f"{readme_path.parent}{os.sep}"))
            self.readme.setMarkdown(markdown)
        if navigate:
            self.details.setCurrentWidget(self.repo_page)

    def _toggle_repo_settings(self, expanded: bool) -> None:
        self.repo_settings_panel.setVisible(expanded)
        self.repo_settings_button.setIcon(
            material_icon("chevron-up" if expanded else "cog-outline")
        )

    def _browse_repo_path(self) -> None:
        start = self.repo_local_path.text() or str(Path.home())
        selected = QFileDialog.getExistingDirectory(self, "选择本地 Git 仓库", start)
        if selected:
            self.repo_local_path.setText(selected)

    def _apply_repo_path(self) -> None:
        if not self.current_repo:
            return
        candidate = Path(self.repo_local_path.text()).expanduser().resolve()
        try:
            inspect_repository(candidate, fetch=False)
        except GitError as exc:
            QMessageBox.warning(self, "无法设置本地目录", str(exc))
            return
        old = self.current_repo
        if candidate == old:
            return
        if QMessageBox.question(
            self, "确认本地目录", f"将列表中的仓库目录从：\n{old}\n\n改为：\n{candidate}\n\n不会移动磁盘文件。"
        ) != QMessageBox.Yes:
            return
        self.paths = [candidate if path == old else path for path in self.paths]
        self.paths = list(dict.fromkeys(self.paths))
        self.store.save(self.paths)
        self.current_repo = candidate
        self.refresh_all(fetch=False)

    def _apply_repo_remote(self) -> None:
        if not self.current_repo:
            return
        remote_url = self.repo_remote_url.text().strip()
        if not remote_url or remote_url.startswith("-") or any(char in remote_url for char in "\r\n"):
            QMessageBox.warning(self, "无法修改远端", "请输入有效的远端仓库地址。")
            return
        repo = self.current_repo
        info = self.repo_infos.get(repo)
        action = "set-url" if info and info.remote_url else "add"
        command = f"git remote {action} origin {remote_url}"
        self._confirm_git(
            "确认修改远端",
            command,
            "此操作会实际修改当前仓库的 origin 地址。",
            lambda: configure_origin(repo, remote_url),
        )

    def _switch_repo_branch(self) -> None:
        if not self.current_repo or not self.repo_branch.currentText():
            return
        branch = self.repo_branch.currentText()
        repo = self.current_repo
        current = self.repo_infos.get(repo)
        if current and current.branch == branch:
            return
        try:
            command = "git " + " ".join(branch_switch_args(repo, branch))
        except GitError as exc:
            QMessageBox.warning(self, "无法切换分支", str(exc))
            return
        self._confirm_git(
            "确认切换分支",
            command,
            "此操作会实际切换当前工作区；Git 会阻止覆盖冲突的未提交改动。",
            lambda: switch_branch(repo, branch),
        )

    def _repo_branch_activated(self, index: int) -> None:
        if self.repo_branch.itemData(index) != "add-branch":
            return
        self._prompt_create_branch()

    def _prompt_create_branch(self) -> None:
        if not self.current_repo:
            return
        repo = self.current_repo
        current = self.repo_infos.get(repo)
        branch, accepted = QInputDialog.getText(self, "添加分支", "新分支名称")
        if not accepted:
            if current:
                self.repo_branch.setCurrentText(current.branch)
            return
        try:
            args = create_branch_args(repo, branch)
        except GitError as exc:
            QMessageBox.warning(self, "无法添加分支", str(exc))
            if current:
                self.repo_branch.setCurrentText(current.branch)
            return
        command = "git " + " ".join(args)
        self._confirm_git(
            "确认添加分支",
            command,
            "此操作会创建并切换到新的本地分支。",
            lambda: create_branch(repo, branch),
        )

    def _commits_loaded(self, commits: list[CommitInfo]) -> bool:
        if commits == self.commits:
            return False
        self.commits = commits
        self._render_commits(commits)
        return True

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _render_commits(self, commits: list[CommitInfo]) -> None:
        self._clear_layout(self.commit_layout)
        selected_commit = getattr(self, "selected_commit", None)
        for commit in commits[: self.visible_commits]:
            row = CommitRow(
                commit,
                lambda _checked=False, value=commit: self.show_commit(value),
                self.commit_all,
                self.pull_current,
                self.push_current,
            )
            row.set_selected(commit == selected_commit)
            self.commit_layout.addWidget(row)
        if len(commits) > self.visible_commits:
            more = QPushButton("展开更多")
            more.clicked.connect(self._show_more)
            self.commit_layout.addWidget(more)
        self.commit_layout.addStretch()

    def _show_more(self) -> None:
        self.visible_commits += 5
        self._render_commits(self.commits)

    def _update_commit_selection(self) -> None:
        selected_commit = getattr(self, "selected_commit", None)
        for index in range(self.commit_layout.count()):
            widget = self.commit_layout.itemAt(index).widget()
            if isinstance(widget, CommitRow):
                widget.set_selected(widget.commit == selected_commit)

    def _populate_diff(self, changed: list[tuple[str, str]], content: str) -> None:
        self.files_list.clear()
        for status, path in changed:
            item = QListWidgetItem(f"{status}  {path}")
            item.setData(Qt.UserRole, path)
            self.files_list.addItem(item)
        self.diff.setPlainText(content or "未检测到文件差异。")
        self._color_diff()

    def _load_commit_diff(self, commit: CommitInfo, show_progress: bool = True) -> None:
        if not self.current_repo:
            return
        repo = self.current_repo
        revision = None if commit.is_worktree else commit.sha
        self._diff_load_generation += 1
        generation = self._diff_load_generation
        self.files_list.clear()
        if not commit.is_worktree:
            self._populate_diff(changed_files(repo, revision), diff_text(repo, revision))
            self.diff_progress.setVisible(False)
            return

        if not show_progress and self._diff_load_in_flight:
            return

        first_load = self._worktree_diff_loaded_repo != repo
        if first_load and show_progress:
            self.diff_progress.setVisible(True)
            self.diff.setPlainText("正在加载工作区文件差异…")
            self.statusBar().showMessage("正在加载工作区差异…")
        elif not self._diff_load_in_flight:
            self.diff_progress.setVisible(False)
        self._diff_load_in_flight = True
        self._diff_load_in_flight_generation = generation

        def loaded(result: tuple[list[tuple[str, str]], str]) -> None:
            if self._diff_load_in_flight_generation == generation:
                self._diff_load_in_flight = False
            if (
                generation != self._diff_load_generation
                or self.current_repo != repo
                or getattr(self, "selected_commit", None) != commit
            ):
                return
            self._worktree_diff_loaded_repo = repo
            self.diff_progress.setVisible(False)
            self._populate_diff(*result)

        def failed() -> None:
            if self._diff_load_in_flight_generation == generation:
                self._diff_load_in_flight = False
            if generation == self._diff_load_generation:
                self.diff_progress.setVisible(False)
                self.diff.setPlainText("工作区差异加载失败。")

        self._start(
            lambda: (changed_files(repo, revision), diff_text(repo, revision)),
            loaded,
            "正在加载工作区差异…",
            failed=failed,
        )

    def show_commit(
        self,
        commit: CommitInfo,
        force_acceptance: bool = True,
        show_diff_progress: bool = True,
    ) -> None:
        if not self.current_repo:
            return
        self.selected_commit = commit
        self._update_commit_selection()
        if force_acceptance:
            self._acceptance_signature = None
        self.commit_title.setText(commit.subject)
        self.commit_meta.setText(f"{commit.short_sha}  {commit.author}  {commit.authored_at}")
        self._load_acceptance(commit, force=force_acceptance)
        self.details.setCurrentWidget(self.commit_page)
        self._load_commit_diff(commit, show_progress=show_diff_progress)

    def _load_acceptance(self, commit: CommitInfo, force: bool = False) -> None:
        assert self.current_repo is not None
        review_path = acceptance_path(self.current_repo, commit.sha)
        try:
            stat = review_path.stat()
            signature: tuple[object, ...] = (
                str(review_path), True, stat.st_mtime_ns, stat.st_size
            )
        except FileNotFoundError:
            signature = (str(review_path), False)
        if not force and self._acceptance_signature == signature:
            return
        self._acceptance_signature = signature
        review = load_acceptance(self.current_repo, commit.sha)
        show_empty_worktree_review = commit.is_worktree and review is None
        self.acceptance_section.setVisible(review is not None or show_empty_worktree_review)
        if review is None and not show_empty_worktree_review:
            self.acceptance_table.setRowCount(0)
            self.copy_review_prompt_button.setVisible(False)
            return
        rows = review.items if review is not None else tuple(
            AcceptanceItem(item, "") for item in ACCEPTANCE_ITEMS
        )
        self.acceptance_table.setRowCount(len(rows))
        for row, entry in enumerate(rows):
            fixed = QTableWidgetItem(entry.item)
            fixed.setFlags(fixed.flags() & ~Qt.ItemIsEditable)
            self.acceptance_table.setItem(row, 0, fixed)
            self.acceptance_table.setItem(row, 1, QTableWidgetItem(entry.answer))
        target = acceptance_path(self.current_repo, commit.sha)
        self.acceptance_path_label.setText(str(target))
        is_empty = all(not entry.answer.strip() for entry in rows)
        self.copy_review_prompt_button.setVisible(commit.is_worktree and is_empty)

    def copy_acceptance_review_prompt(self) -> None:
        if not self.current_repo:
            return
        QApplication.clipboard().setText(acceptance_review_prompt(self.current_repo))
        self.statusBar().showMessage("审阅提示词已复制", 3000)

    def save_current_acceptance(self) -> None:
        if not self.current_repo or not hasattr(self, "selected_commit"):
            return
        review = AcceptanceReview(
            schema_version=1,
            commit=self.selected_commit.sha,
            reviewed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            items=tuple(
                AcceptanceItem(
                    self.acceptance_table.item(row, 0).text(),
                    self.acceptance_table.item(row, 1).text(),
                )
                for row in range(self.acceptance_table.rowCount())
            ),
        )
        target = save_acceptance(self.current_repo, review)
        self.acceptance_path_label.setText(str(target))
        self.statusBar().showMessage("验收表已保存", 3000)

    def _file_selected(self, row: int) -> None:
        item = self.files_list.item(row)
        if not item or not self.current_repo or not hasattr(self, "selected_commit"):
            return
        revision = None if self.selected_commit.is_worktree else self.selected_commit.sha
        self.diff.setPlainText(diff_text(self.current_repo, revision, item.data(Qt.UserRole)))
        self._color_diff()

    def _color_diff(self) -> None:
        document = self.diff.document()
        cursor = QTextCursor(document)
        cursor.beginEditBlock()
        block = document.firstBlock()
        while block.isValid():
            text = block.text()
            color = None
            background = None
            if text.startswith("+") and not text.startswith("+++"):
                color, background = "#17663f", "#e5f5eb"
            elif text.startswith("-") and not text.startswith("---"):
                color, background = "#a12622", "#fde8e7"
            elif text.startswith("@@"):
                color, background = "#245b7a", "#e7f1f8"
            if color:
                fmt = QTextCharFormat()
                fmt.setForeground(QColor(color))
                fmt.setBackground(QColor(background))
                cursor.setPosition(block.position())
                cursor.setPosition(block.position() + block.length() - 1, QTextCursor.KeepAnchor)
                cursor.mergeCharFormat(fmt)
            block = block.next()
        cursor.endEditBlock()

    def add_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择 Git 仓库")
        if not selected:
            return
        path = Path(selected).resolve()
        try:
            inspect_repository(path)
        except GitError as exc:
            QMessageBox.warning(self, "无法添加", str(exc))
            return
        self._add_paths([path])

    def scan_disk(self) -> None:
        roots = QFileDialog.getExistingDirectory(self, "选择扫描起点")
        if not roots:
            return
        self._start(
            lambda: discover_repositories([Path(roots)]),
            lambda paths: self._scan_finished(paths),
            "正在扫描 Git 仓库...",
        )

    def _scan_finished(self, paths: list[Path]) -> None:
        if not paths:
            QMessageBox.information(self, "扫描完成", "未找到 Git 仓库。")
            return
        self._add_paths(paths)
        self.statusBar().showMessage(f"已发现 {len(paths)} 个仓库", 4000)

    def clone_remote(self) -> None:
        dialog = CloneDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        url = normalize_remote_url(dialog.url.text())
        parent = Path(dialog.parent_path.text())
        prefix = dialog.prefix.currentText()
        branch_name = dialog.branch_name.text()
        if dialog.initialize_from_local.isChecked():
            repo = parent.resolve()
            answer = QMessageBox.question(
                self,
                "确认初始化远端仓库",
                f"将先检查远端是否为空或仅含空 README：\n{url}\n\n"
                f"检查通过后会连接本地仓库：\n{repo}\n\n继续检查吗？",
            )
            if answer != QMessageBox.Yes:
                return
            self._start(
                lambda: validate_initial_remote(url),
                lambda _value: self._confirm_initial_commit(repo, url, prefix, branch_name),
                "正在检查远端初始化条件...",
            )
            return
        destination = parent / repository_name_from_url(url)
        command = f"git clone -- {url} {destination}"
        if prefix != "none":
            command += f"\n然后执行 git switch -c {prefix}/{branch_name.strip().strip('/')}"
        answer = QMessageBox.question(
            self,
            "确认克隆",
            f"将创建本地仓库并可能创建分支：\n\n{command}\n\n继续吗？",
        )
        if answer != QMessageBox.Yes:
            return
        self._start(
            lambda: clone_repository(url, parent, prefix, branch_name),
            lambda path: self._add_paths([path]),
            "正在克隆仓库...",
        )

    def _confirm_initial_commit(
        self, repo: Path, url: str, prefix: str, branch_name: str
    ) -> None:
        branch = ""
        if prefix != "none":
            branch = f"\n并在 init commit 后创建分支 {prefix}/{branch_name.strip().strip('/')}"
        answer = QMessageBox.question(
            self,
            "确认创建 init commit",
            f"远端符合初始化条件。将执行：\n\n"
            f"移除旧 origin 跟踪信息并连接 origin {url}\n"
            "git add --all\n"
            'git commit -m "init commit"（如有需要）\n'
            "git push --set-upstream origin --all"
            f"{branch}\n\n是否继续？",
        )
        if answer != QMessageBox.Yes:
            return
        self._start(
            lambda: initialize_from_local(repo, url, prefix, branch_name),
            lambda path: self._add_paths([path]),
            "正在连接远端并创建 init commit...",
        )

    def _add_paths(self, paths: list[Path]) -> None:
        known = {path.resolve() for path in self.paths}
        for path in paths:
            resolved = path.resolve()
            if resolved not in known:
                self.paths.append(resolved)
                known.add(resolved)
        self.store.save(self.paths)
        self._populate_repo_placeholders()
        self.refresh_all(fetch=False)

    def remove_current(self) -> None:
        if not self.current_repo:
            return
        repo = self.current_repo
        if QMessageBox.question(
            self,
            "确认移除仓库",
            f"仅从仓库列表移除：\n{repo}\n\n不会删除本地仓库文件。是否继续？",
        ) != QMessageBox.Yes:
            return
        self.paths = [path for path in self.paths if path != repo]
        self.store.save(self.paths)
        self.current_repo = None
        self._populate_repo_placeholders()
        self.details.setCurrentWidget(self.empty_page)
        self._render_commits([])
        if self.paths:
            self.refresh_all(fetch=False)

    def commit_all(self) -> None:
        if not self.current_repo:
            return
        dialog = CommitDialog(self.current_repo, self)
        if dialog.exec() != QDialog.Accepted:
            return
        repo, message, review = self.current_repo, dialog.message.text(), dialog.review()

        def work() -> str:
            if review is not None:
                save_acceptance(repo, review)
            commit_all_changes(repo, message)
            from ..git_service import _run

            sha = _run(repo, "rev-parse", "HEAD").stdout.strip()
            if review is not None:
                pending = acceptance_path(repo, "pending")
                final = acceptance_path(repo, sha)
                final.parent.mkdir(parents=True, exist_ok=True)
                pending.replace(final)
                payload = load_acceptance(repo, sha)
                if payload and payload.commit != sha:
                    save_acceptance(repo, AcceptanceReview(1, sha, payload.reviewed_at, payload.items, payload.notes))
            return sha

        self._start(work, lambda _sha: self.refresh_all(fetch=False), "正在提交改动...")

    def _confirm_git(self, title: str, command: str, warning: str, operation: Callable[[], None]) -> None:
        if QMessageBox.question(self, title, f"{warning}\n\n将执行：\n{command}\n\n继续吗？") != QMessageBox.Yes:
            return
        self._start(lambda: operation(), lambda _value: self.refresh_all(fetch=False), f"正在执行 {command}...")

    def push_current(self) -> None:
        if self.current_repo:
            self._confirm_git(
                "确认推送",
                "git push --set-upstream origin HEAD（当前分支）",
                "此操作会把当前分支的本地 commit 写入已配置的上游远端。",
                lambda: push(self.current_repo),
            )

    def pull_current(self) -> None:
        if self.current_repo:
            self._confirm_git(
                "确认拉取",
                "git pull --ff-only",
                "此操作会更新当前本地分支；仅允许快进，不会自动创建 merge commit。",
                lambda: pull(self.current_repo),
            )

    def choose_branch(self) -> None:
        if not self.current_repo:
            return
        branches = list_branches(self.current_repo)
        if not branches:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("切换分支")
        layout = QVBoxLayout(dialog)
        combo = QComboBox()
        combo.addItems(branches)
        layout.addWidget(combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return
        branch = combo.currentText()
        try:
            command = "git " + " ".join(branch_switch_args(self.current_repo, branch))
        except GitError as exc:
            QMessageBox.warning(self, "无法切换分支", str(exc))
            return
        self._confirm_git(
            "确认切换分支",
            command,
            "工作区必须允许切换；Git 会阻止覆盖未提交的冲突改动。",
            lambda: switch_branch(self.current_repo, branch),
        )

    def changeEvent(self, event: Any) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow() and self.paths:
            self.refresh_all(fetch=True)

    def closeEvent(self, event: Any) -> None:
        self._worktree_timer.stop()
        super().closeEvent(event)
