"""Discover independently runnable tools from their TOML manifests."""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Tool:
    id: str
    directory: Path
    name: str
    description: str
    category: str
    tags: tuple[str, ...]
    command: tuple[str, ...]
    readme: Path
    icon: Path | None = None

    def matches(self, query: str, category: str | None = None) -> bool:
        text = " ".join((self.name, self.description, self.category, *self.tags)).casefold()
        return (category is None or self.category == category) and all(
            word in text for word in query.casefold().split()
        )


def _text(data: dict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} 必须是非空字符串")
    return value.strip()


def _strings(data: dict, key: str, *, required: bool = False) -> tuple[str, ...]:
    value = data.get(key, [])
    if not isinstance(value, list) or (required and not value) or any(
        not isinstance(item, str) or not item.strip() or "\0" in item for item in value
    ):
        raise ValueError(f"{key} 必须是字符串数组" + ("，且不能为空" if required else ""))
    return tuple(value)


def _file(directory: Path, value: str) -> Path:
    path = (directory / value).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError(f"文件不存在或不在工具目录内：{value}")
    return path


def load_tool(manifest: Path) -> Tool:
    directory = manifest.parent.resolve()
    with manifest.open("rb") as handle:
        data = tomllib.load(handle)
    return Tool(
        id=directory.name,
        directory=directory,
        name=_text(data, "name"),
        description=_text(data, "description"),
        category=_text(data, "category"),
        tags=_strings(data, "tags"),
        command=_strings(data, "command", required=True),
        readme=_file(directory, _text(data, "readme")),
        icon=_file(directory, _text(data, "icon")) if "icon" in data else None,
    )


def discover_tools(directory: Path) -> tuple[list[Tool], list[str]]:
    tools, errors = [], []
    for manifest in sorted(directory.glob("*/tool.toml")):
        try:
            tools.append(load_tool(manifest))
        except (OSError, ValueError) as exc:
            errors.append(f"{manifest.parent.name}/tool.toml：{exc}")
    return sorted(tools, key=lambda tool: (tool.category, tool.name.casefold())), errors
