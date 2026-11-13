from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from PySide6.QtCore import QStandardPaths


@dataclass(frozen=True)
class CredentialHost:
    host: str
    username: str
    token_file: str
    email: str = ""
    allow_http: bool = False
    scheme: str = "https"
    managed: bool = False


def normalize_host(value: str) -> tuple[str, str]:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("请输入不含账号信息的 HTTP(S) 服务地址")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("请填写服务根地址，不要填写仓库路径")
    return parsed.netloc.lower(), parsed.scheme


def read_token(path: Path) -> str:
    if not path.is_file() or path.stat().st_size > 4096:
        raise ValueError("token 文件不存在或超过 4 KB")
    token = path.read_text(encoding="utf-8").strip()
    if not token or "\n" in token or "\r" in token:
        raise ValueError("token 文件应只包含一行非空 token")
    return token


def remote_host_and_user(remote_url: str) -> tuple[str, str, str]:
    value = remote_url.strip()
    if not value:
        return "", "", ""
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https", "ssh"} and parsed.hostname:
        return parsed.hostname.lower(), parsed.username or "", parsed.scheme
    match = re.match(r"(?:(?P<user>[^@:/]+)@)?(?P<host>[^:/]+):.+", value)
    if match:
        return match.group("host").lower(), match.group("user") or "", "ssh"
    return "", "", "local"


class CredentialStore:
    def __init__(self, path: Path | None = None) -> None:
        root = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation))
        self.path = path or Path(os.environ.get("IGR_CREDENTIAL_HOSTS_PATH", root / "credential-hosts.json"))

    def hosts(self) -> list[CredentialHost]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return [CredentialHost(**item) for item in payload.get("hosts", [])]
        except (OSError, ValueError, TypeError, KeyError):
            return []

    def _managed_path(self, host: str) -> Path:
        return self.path.parent / "tokens" / (hashlib.sha256(host.encode()).hexdigest() + ".txt")

    def _save_hosts(self, hosts: list[CredentialHost]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.path.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"hosts": [vars(host) for host in hosts]}, stream, ensure_ascii=False, indent=2)
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)

    def save(
        self, address: str, username: str, token: str = "", email: str = "", allow_http: bool = False,
        token_file: Path | None = None,
    ) -> None:
        host, scheme = normalize_host(address)
        if scheme == "http" and not allow_http:
            raise ValueError("HTTP 会明文传输 token，请明确启用 HTTP 凭据传输")
        if not username.strip() or not email.strip():
            raise ValueError("用户名和邮箱不能为空")
        old = next((item for item in self.hosts() if item.host == host), None)
        if token.strip() and token_file:
            raise ValueError("请选择手动输入 token 或现有 token 文件")
        if not token.strip() and token_file is None and old:
            file_path = Path(old.token_file)
            read_token(file_path)
            managed = old.managed
        elif token_file is not None:
            file_path = token_file.expanduser().resolve(strict=True)
            read_token(file_path)
            managed = False
        elif token.strip():
            if "\n" in token or "\r" in token:
                raise ValueError("token 只能占一行")
            file_path = self._managed_path(host)
            file_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(token.strip() + "\n")
            os.chmod(file_path, 0o600)
            managed = True
        else:
            raise ValueError("请选择手动输入 token 或现有 token 文件")
        self._save_hosts(
            [item for item in self.hosts() if item.host != host]
            + [CredentialHost(host, username.strip(), str(file_path), email.strip(), allow_http, scheme, managed)]
        )
        if old and old.managed and old.token_file != str(file_path):
            self._remove_managed(old)

    def _remove_managed(self, item: CredentialHost) -> None:
        if item.managed and Path(item.token_file) == self._managed_path(item.host):
            Path(item.token_file).unlink(missing_ok=True)

    def delete(self, host: str) -> None:
        old = next((item for item in self.hosts() if item.host == host), None)
        self._save_hosts([item for item in self.hosts() if item.host != host])
        if old:
            self._remove_managed(old)

    def answer_prompt(self, prompt: str) -> str | None:
        match = re.search(r"(?:https?://)(?:[^@'\s]+@)?([^/'\s]+)", prompt)
        if not match:
            return None
        parsed = urlparse(match.group(0))
        host = parsed.netloc.lower().rsplit("@", 1)[-1]
        item = next((entry for entry in self.hosts() if entry.host == host), None)
        if item is None or (parsed.scheme == "http" and not item.allow_http):
            return None
        if prompt.lower().startswith("username"):
            return item.username
        if prompt.lower().startswith("password"):
            try:
                return read_token(Path(item.token_file))
            except (OSError, UnicodeError, ValueError):
                return None
        return None

    def identity_for_remote(self, remote_url: str) -> str:
        host, _remote_user, scheme = remote_host_and_user(remote_url)
        if scheme == "local":
            return "本地远端，无需登录"
        item = next((entry for entry in self.hosts() if entry.host == host), None)
        if scheme == "ssh":
            if item:
                return f"SSH:{item.username}<{item.email}>"
            return f"SSH:未配置 {host} 的身份"
        if item:
            identity = f"{item.username} <{item.email}>" if item.email else f"{item.username}（邮箱未配置）"
            return f"{identity} @ {host}（应用 token）"
        if host:
            return f"未在应用中配置 {host} 的身份"
        return "未配置远端登录身份"
