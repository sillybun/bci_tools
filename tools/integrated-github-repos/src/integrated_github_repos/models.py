from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class RepoState(str, Enum):
    DIRTY = "未提交"
    SYNCED = "同步"
    STALE = "过期"
    DISCONNECTED = "断线"


class CommitState(str, Enum):
    SYNCED = "已同步"
    COMMITTED = "已提交"
    FETCHED = "已抓取"
    UNCOMMITTED = "未提交"
    UNFETCHED = "未抓取"
    UNPUSHED = "未推送"


@dataclass(frozen=True)
class CommitInfo:
    sha: str
    subject: str
    author: str
    authored_at: str
    states: tuple[CommitState, ...] = ()
    is_worktree: bool = False
    parents: tuple[str, ...] = ()
    refs: tuple[str, ...] = ()
    lane: int = 0
    lane_count: int = 1
    graph_edges: tuple[tuple[int, int], ...] = ()
    needs_push: bool = False
    needs_pull: bool = False

    @property
    def short_sha(self) -> str:
        return self.sha[:8] if self.sha else "WORKTREE"


@dataclass(frozen=True)
class ContributionDay:
    date: str
    count: int


@dataclass
class RepoInfo:
    path: Path
    name: str
    branch: str = ""
    state: RepoState = RepoState.DISCONNECTED
    ahead: int = 0
    behind: int = 0
    remote_url: str = ""
    contributors: list[str] = field(default_factory=list)
    contributions: list[ContributionDay] = field(default_factory=list)
    error: str = ""


@dataclass(frozen=True)
class AcceptanceItem:
    item: str
    answer: str


@dataclass(frozen=True)
class AcceptanceReview:
    schema_version: int
    commit: str
    reviewed_at: str
    items: tuple[AcceptanceItem, ...]
    notes: str = ""
