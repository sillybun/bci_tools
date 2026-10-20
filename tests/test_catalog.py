from pathlib import Path
import tempfile
import unittest

from launcher.catalog import discover_tools


MANIFEST = '''\
name = "集成终端"
description = "集中管理实验进程"
category = "进程与环境"
tags = ["Integrated Term", "LSL"]
command = ["uv", "run", "--locked", "python", "app.py"]
readme = "README.md"
'''


class CatalogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def add_tool(self, name="terminal", manifest=MANIFEST):
        directory = self.root / name
        directory.mkdir()
        (directory / "README.md").write_text("说明", encoding="utf-8")
        (directory / "tool.toml").write_text(manifest, encoding="utf-8")
        return directory

    def test_discovers_tools_and_searches_names_tags_and_categories(self):
        directory = self.add_tool()
        tools, errors = discover_tools(self.root)
        self.assertEqual(errors, [])
        self.assertEqual(len(tools), 1)
        tool = tools[0]
        self.assertEqual(tool.directory, directory.resolve())
        self.assertTrue(tool.matches("终端"))
        self.assertTrue(tool.matches("INTEGRATED lsl"))
        self.assertTrue(tool.matches("环境", "进程与环境"))
        self.assertFalse(tool.matches("未知工具"))
        self.assertFalse(tool.matches("", "数据处理"))
        self.assertEqual(tool.command, ("uv", "run", "--locked", "python", "app.py"))

    def test_bad_manifest_does_not_hide_other_tools(self):
        self.add_tool()
        self.add_tool("broken", "name = [")
        (self.root / "unregistered").mkdir()
        tools, errors = discover_tools(self.root)
        self.assertEqual([tool.id for tool in tools], ["terminal"])
        self.assertEqual(len(errors), 1)
        self.assertIn("broken/tool.toml", errors[0])

    def test_rejects_invalid_commands_and_missing_metadata(self):
        for old, new in [
            ('["uv", "run", "--locked", "python", "app.py"]', '"uv run python app.py"'),
            ('["uv", "run", "--locked", "python", "app.py"]', "[]"),
            ('["uv", "run", "--locked", "python", "app.py"]', '["uv", 4]'),
            ('category = "进程与环境"', 'category = ""'),
        ]:
            with self.subTest(new=new):
                directory = self.root / "tool"
                directory.mkdir(exist_ok=True)
                (directory / "README.md").write_text("")
                (directory / "tool.toml").write_text(MANIFEST.replace(old, new), encoding="utf-8")
                tools, errors = discover_tools(self.root)
                self.assertEqual(tools, [])
                self.assertEqual(len(errors), 1)

    def test_readme_and_icon_must_exist_inside_tool(self):
        directory = self.add_tool()
        for manifest in [
            MANIFEST.replace("README.md", "missing.md"),
            MANIFEST.replace("README.md", "../outside.md"),
            MANIFEST + 'icon = "missing.png"\n',
        ]:
            with self.subTest(manifest=manifest):
                (self.root / "outside.md").write_text("outside")
                (directory / "tool.toml").write_text(manifest, encoding="utf-8")
                tools, errors = discover_tools(self.root)
                self.assertEqual(tools, [])
                self.assertEqual(len(errors), 1)

    def test_refresh_reads_new_categories_without_code_changes(self):
        self.add_tool()
        tools, _ = discover_tools(self.root)
        self.assertEqual(len(tools), 1)
        self.add_tool("viewer", MANIFEST.replace("进程与环境", "信号查看"))
        tools, errors = discover_tools(self.root)
        self.assertEqual(errors, [])
        self.assertEqual({tool.category for tool in tools}, {"进程与环境", "信号查看"})

    def test_empty_directory_is_valid(self):
        self.assertEqual(discover_tools(self.root / "not-created"), ([], []))
