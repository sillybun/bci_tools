from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from integrated_github_repos import git_service
from integrated_github_repos.credentials import (
    CredentialStore,
    normalize_host,
    remote_host_and_user,
)


def test_credential_store_uses_plaintext_files_and_isolates_hosts(tmp_path) -> None:
    store = CredentialStore(tmp_path / "hosts.json")
    store.save("https://github.com", "alice", "github-secret", email="alice@example.com")
    store.save("https://gitlab.example.com", "bob", "gitlab-secret", email="bob@example.com")

    assert store.answer_prompt("Username for 'https://github.com': ") == "alice"
    assert store.answer_prompt("Password for 'https://alice@github.com': ") == "github-secret"
    assert store.answer_prompt("Password for 'https://bob@gitlab.example.com': ") == "gitlab-secret"
    assert store.answer_prompt("Password for 'https://alice@github.com.evil': ") is None
    assert store.answer_prompt("Password for 'http://alice@github.com': ") is None
    assert "github-secret" not in store.path.read_text(encoding="utf-8")
    assert "gitlab-secret" not in store.path.read_text(encoding="utf-8")
    assert len(json.loads(store.path.read_text(encoding="utf-8"))["hosts"]) == 2
    managed_path = next(Path(item.token_file) for item in store.hosts() if item.host == "github.com")
    assert managed_path.read_text(encoding="utf-8").strip() == "github-secret"
    assert managed_path.stat().st_mode & 0o077 == 0

    store.delete("github.com")
    assert store.answer_prompt("Username for 'https://github.com': ") is None
    assert not managed_path.exists()


def test_http_requires_explicit_opt_in(tmp_path) -> None:
    store = CredentialStore(tmp_path / "hosts.json")
    with pytest.raises(ValueError, match="HTTP"):
        store.save("http://gitlab.example.com", "bob", "private-token", email="bob@example.com")
    store.save(
        "http://gitlab.example.com", "bob", "private-token",
        email="bob@example.com", allow_http=True,
    )
    assert store.answer_prompt("Password for 'http://bob@gitlab.example.com': ") == "private-token"
    assert store.hosts()[0].scheme == "http"


def test_existing_file_is_referenced_but_not_deleted(tmp_path) -> None:
    source = tmp_path / "my-token.txt"
    source.write_text("external-secret\n", encoding="utf-8")
    store = CredentialStore(tmp_path / "hosts.json")
    store.save("https://github.com", "alice", email="alice@example.com", token_file=source)
    assert store.answer_prompt("Password for 'https://alice@github.com': ") == "external-secret"
    assert store.hosts()[0].token_file == str(source)
    assert not store.hosts()[0].managed
    store.delete("github.com")
    assert source.exists()


@pytest.mark.parametrize(
    ("remote", "expected"),
    [
        ("https://github.com/group/repo.git", ("github.com", "", "https")),
        ("ssh://alice@gitlab.example.com/group/repo.git", ("gitlab.example.com", "alice", "ssh")),
        ("git@github.com:group/repo.git", ("github.com", "git", "ssh")),
        ("/tmp/repo.git", ("", "", "local")),
    ],
)
def test_remote_host_and_user(remote, expected) -> None:
    assert remote_host_and_user(remote) == expected


def test_identity_for_remote_uses_host_scoped_config(tmp_path) -> None:
    store = CredentialStore(tmp_path / "hosts.json")
    store.save("https://github.com", "alice", "private-token", email="alice@example.com")
    assert store.identity_for_remote("https://github.com/team/repo.git") == (
        "alice <alice@example.com> @ github.com（应用 token）"
    )
    assert store.identity_for_remote("https://gitlab.example.com/team/repo.git") == (
        "未在应用中配置 gitlab.example.com 的身份"
    )
    assert store.identity_for_remote("git@github.com:team/repo.git") == (
        "SSH:alice<alice@example.com>"
    )
    assert store.identity_for_remote("git@gitlab.example.com:team/repo.git") == (
        "SSH:未配置 gitlab.example.com 的身份"
    )
    assert store.identity_for_remote("ssh://alice@gitlab.example.com/team/repo.git") == (
        "SSH:未配置 gitlab.example.com 的身份"
    )


def test_identity_update_keeps_existing_token_and_legacy_config_loads(tmp_path) -> None:
    store = CredentialStore(tmp_path / "hosts.json")
    store.save("https://github.com", "alice", "private-token", email="old@example.com")
    token_path = Path(store.hosts()[0].token_file)
    store.save("https://github.com", "alice-new", email="new@example.com")
    assert token_path.read_text(encoding="utf-8").strip() == "private-token"
    assert store.hosts()[0].email == "new@example.com"
    assert store.answer_prompt("Username for 'https://github.com': ") == "alice-new"

    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["hosts"][0].pop("email")
    store.path.write_text(json.dumps(payload), encoding="utf-8")
    assert store.hosts()[0].email == ""


def test_new_identity_requires_email(tmp_path) -> None:
    with pytest.raises(ValueError, match="邮箱"):
        CredentialStore(tmp_path / "hosts.json").save(
            "https://github.com", "alice", "private-token"
        )


@pytest.mark.parametrize("contents", ["", "first\nsecond\n", "x" * 4097])
def test_reject_invalid_token_file(tmp_path, contents) -> None:
    source = tmp_path / "invalid.txt"
    source.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        CredentialStore(tmp_path / "hosts.json").save(
            "https://github.com", "alice", email="alice@example.com", token_file=source
        )


@pytest.mark.parametrize("url", ["https://u:secret@github.com", "https://github.com/team/repo", "file:///tmp"])
def test_reject_credential_urls_and_non_host_addresses(url) -> None:
    with pytest.raises(ValueError):
        normalize_host(url)


def test_git_overrides_only_configured_host_helper(tmp_path, monkeypatch) -> None:
    store = CredentialStore(tmp_path / "hosts.json")
    store.save("https://gitlab.example.com", "bob", "private-token", email="bob@example.com")
    monkeypatch.setattr(git_service, "CredentialStore", lambda: store)
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(git_service.subprocess, "run", fake_run)
    git_service._run(None, "ls-remote", "https://gitlab.example.com/group/repo.git")
    assert "credential.https://gitlab.example.com.helper=" in observed["command"]
    assert not any("github.com.helper" in arg for arg in observed["command"])
    assert "private-token" not in repr(observed)
    assert observed["env"]["GIT_TERMINAL_PROMPT"] == "0"


def test_macos_bundle_askpass_reads_plaintext_file(tmp_path) -> None:
    bundle = Path(__file__).resolve().parents[1] / (
        "dist/Integrated GitHub Repos.app/Contents/MacOS/Integrated GitHub Repos"
    )
    if sys.platform != "darwin" or not bundle.exists():
        pytest.skip("macOS bundle not built here")
    store = CredentialStore(tmp_path / "hosts.json")
    store.save(
        "https://codex-askpass-test.invalid", "qa-user", "synthetic-token",
        email="qa@example.com",
    )
    result = subprocess.run(
        [str(bundle), "Password for 'https://qa-user@codex-askpass-test.invalid': "],
        env={**os.environ, "IGR_GIT_ASKPASS": "1", "IGR_CREDENTIAL_HOSTS_PATH": str(store.path)},
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == b"synthetic-token"
