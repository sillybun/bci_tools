"""Copy a pushed subproject commit into bci_tools/tools/<name>."""
from __future__ import annotations
import argparse, subprocess, tempfile, tarfile, tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    args = p.parse_args()
    source = args.source.resolve()
    manifest = source / "tool.toml"
    if not manifest.is_file():
        raise SystemExit("source tool.toml is required")
    config = tomllib.loads(manifest.read_text())
    target = config.get("bci_tools", {}).get("target")
    if not isinstance(target, str) or not target.startswith("tools/") or target.count("/") != 1:
        raise SystemExit("tool.toml must declare [bci_tools].target = 'tools/<name>'")
    if target in {"tools/integrated-term", "tools/integrated-term-2"} and config.get("name") == "集成终端 2" and target != "tools/integrated-term-2":
        raise SystemExit("集成终端 2 must target tools/integrated-term-2")
    dirty = [line for line in git(source, "status", "--porcelain").splitlines()
             if not line.startswith("?? .codex/")]
    if dirty:
        raise SystemExit("source worktree must be clean except ignored local .codex records")
    sha = git(source, "rev-parse", "HEAD")
    upstream = git(source, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    remote_ref = upstream.removeprefix("origin/")
    remote_lines = git(source, "ls-remote", "origin", f"refs/heads/{remote_ref}").splitlines()
    if not remote_lines:
        raise SystemExit(f"published upstream branch not found: origin/{remote_ref}")
    remote = remote_lines[0].split()[0]
    if sha != remote:
        raise SystemExit(f"local HEAD {sha} is not published origin/{remote_ref} {remote}")
    destination = ROOT / target
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp) / "snapshot.tar"
        with archive.open("wb") as output:
            subprocess.run(["git", "archive", sha], cwd=source, check=True, stdout=output)
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                if member.name == ".codex" or member.name.startswith(".codex/") or member.name == "processes.json":
                    continue
                bundle.extract(member, destination)
    print(f"synced {target} at {sha}")
    subprocess.run([".venv/bin/python", "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
