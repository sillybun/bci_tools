from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..models import CommitInfo
from .material_icons import material_icon

GRAPH_COLORS = ("#197b76", "#2e8290", "#3c9b91", "#24706f", "#5b9fa3", "#17606d")


def format_commit_time(value: str) -> str:
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


class BranchGlyph(QWidget):
    def __init__(self, commit: CommitInfo, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.commit = commit
        self.setFixedWidth(min(58, 14 * max(1, commit.lane_count) + 12))
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        spacing = max(8.0, min(13.0, (self.width() - 10) / max(1, self.commit.lane_count)))

        def lane_x(lane: int) -> float:
            return 6.0 + lane * spacing

        middle = self.height() / 2
        drawn_node_stem = False
        for source, target in self.commit.graph_edges:
            color = QColor(GRAPH_COLORS[source % len(GRAPH_COLORS)])
            painter.setPen(QPen(color, 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            source_x, target_x = lane_x(source), lane_x(target)
            if source == self.commit.lane:
                if not drawn_node_stem:
                    painter.drawLine(QPoint(int(source_x), 0), QPoint(int(source_x), int(middle)))
                    drawn_node_stem = True
                path = QPainterPath()
                path.moveTo(source_x, middle)
                path.cubicTo(source_x, middle + 12, target_x, self.height() - 12, target_x, self.height())
            else:
                path = QPainterPath()
                path.moveTo(source_x, 0)
                path.cubicTo(source_x, 12, target_x, self.height() - 12, target_x, self.height())
            painter.drawPath(path)
        if not self.commit.graph_edges:
            color = QColor(GRAPH_COLORS[self.commit.lane % len(GRAPH_COLORS)])
            painter.setPen(QPen(color, 2.2, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPoint(int(lane_x(self.commit.lane)), 0), QPoint(int(lane_x(self.commit.lane)), int(middle)))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(GRAPH_COLORS[self.commit.lane % len(GRAPH_COLORS)]))
        painter.drawEllipse(QPoint(int(lane_x(self.commit.lane)), int(middle)), 4, 4)


class CommitRow(QWidget):
    def __init__(
        self,
        commit: CommitInfo,
        selected: Callable[[], None],
        commit_requested: Callable[[], None],
        pull_requested: Callable[[], None],
        push_requested: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.commit = commit
        self.setFixedHeight(108)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 3, 8, 3)
        layout.setSpacing(0)
        badges = " · ".join(state.value for state in commit.states)
        refs = " · ".join(commit.refs) or ("工作区" if commit.is_worktree else "无分支标记")
        self.surface = QPushButton(self)
        self.surface.setMinimumWidth(0)
        self.surface.setObjectName("commitCard")
        self.surface.setProperty("selected", False)
        self.surface.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.surface.setToolTip(f"{commit.author}  {commit.authored_at}")
        self.surface.clicked.connect(selected)
        layout.addWidget(self.surface, 1)

        card_layout = QHBoxLayout(self.surface)
        card_layout.setContentsMargins(10, 8, 8, 8)
        card_layout.setSpacing(8)
        self.graph = BranchGlyph(commit, self.surface)
        card_layout.addWidget(self.graph)
        content = QVBoxLayout()
        content.setSpacing(3)
        meta = QHBoxLayout()
        meta.setSpacing(6)
        self.version_label = QLabel(commit.short_sha)
        self.version_label.setObjectName("commitVersion")
        self.version_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        meta.addWidget(self.version_label)
        self.refs_label = QLabel(refs)
        self.refs_label.setObjectName("commitRefs")
        self.refs_label.setMinimumWidth(0)
        self.refs_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.refs_label.setToolTip(refs)
        self.refs_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        meta.addWidget(self.refs_label, 1)
        content.addLayout(meta)
        self.subject_label = QLabel(commit.subject)
        self.subject_label.setObjectName("commitSubject")
        self.subject_label.setWordWrap(True)
        self.subject_label.setMaximumHeight(36)
        self.subject_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        content.addWidget(self.subject_label)
        state_row = QHBoxLayout()
        self.state_label = QLabel(badges)
        self.state_label.setObjectName("commitState")
        self.state_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        state_row.addWidget(self.state_label)
        state_row.addStretch()
        self.time_label: QLabel | None = None
        if not commit.is_worktree:
            self.time_label = QLabel(format_commit_time(commit.authored_at))
            self.time_label.setObjectName("commitTime")
            self.time_label.setAttribute(Qt.WA_TransparentForMouseEvents)
            state_row.addWidget(self.time_label)
        self.pull_button: QToolButton | None = None
        self.push_button: QToolButton | None = None
        self.commit_button: QToolButton | None = None
        if commit.is_worktree:
            self.commit_button = QToolButton(self.surface)
            self.commit_button.setIcon(material_icon("content-save-check-outline"))
            self.commit_button.setToolTip("提交全部改动（不会自动 push）")
            self.commit_button.clicked.connect(commit_requested)
            state_row.addWidget(self.commit_button)
        if commit.needs_pull:
            self.pull_button = QToolButton(self.surface)
            self.pull_button.setIcon(material_icon("download"))
            self.pull_button.setToolTip("抓取并快进到远端提交")
            self.pull_button.clicked.connect(pull_requested)
            state_row.addWidget(self.pull_button)
        if commit.needs_push:
            self.push_button = QToolButton(self.surface)
            self.push_button.setIcon(material_icon("upload"))
            self.push_button.setToolTip("推送尚未发布的本地提交")
            self.push_button.clicked.connect(push_requested)
            state_row.addWidget(self.push_button)
        content.addLayout(state_row)
        card_layout.addLayout(content, 1)

    def set_selected(self, selected: bool) -> None:
        self.surface.setProperty("selected", selected)
        self.surface.style().unpolish(self.surface)
        self.surface.style().polish(self.surface)
        self.surface.update()
