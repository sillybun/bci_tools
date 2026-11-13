from pathlib import Path

from integrated_github_repos.repo_store import RepoStore


def test_store_deduplicates_and_drops_missing_paths(tmp_path: Path) -> None:
    first = tmp_path / "first"
    first.mkdir()
    missing = tmp_path / "missing"
    store = RepoStore(tmp_path / "state" / "repositories.json")
    store.save([first, first, missing])
    assert store.load() == [first.resolve()]

