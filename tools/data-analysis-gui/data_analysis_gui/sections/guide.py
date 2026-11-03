"""Per-section reading guide: `<section folder>/guide.md`.

Structure (Chinese, plain-text formulas, no LaTeX):

    # S05 · ANOVA
    **一句话总结：**（本节目的、数据来自哪里）
    ## 图 1：<exact figure title passed to add_figure>
    ### 计算公式
    ### 结果解读
    - **面板名称/位置**：逐 subplot 解读。
    ### 质量评价标准
    - **相同面板名称/位置**：统计/配置判据及解释限制。
    ## 图 2：...
    ## 注意事项

`figure_guide()` returns the part for one figure (matched by title), so each figure
tab can show its own explanation.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import re

HEADING = re.compile(r'^##\s+(?:图\s*\d+\s*[：:]\s*)?(?P<title>.+?)\s*$')


def guide_path(section) -> Path:
    module = importlib.import_module(type(section).__module__)
    return Path(module.__file__).resolve().parent / 'guide.md'


def load_guide(section) -> str:
    try:
        text = guide_path(section).read_text(encoding='utf-8')
    except OSError:
        return ''
    if section.mode == 'cross' and section.id not in ('x08', 'x09'):
        overview = '''## 图 0：Session Overview

### 计算公式
**符号**
- 每个标记代表一份记录。方形为 Center Out 8，圆形为 Grid。

**步骤**
1. 按日期和记录文件排序，等距画出各 session，在日期切换处画虚线。

### 结果解读
- **记录时间轴**：读日期、任务和在线状态。灰色方块的 Offline CO8 标签使用黑色。

### 质量评价标准
- **记录时间轴**：先确认各日期的记录数量与任务组成，再比较后续指标。相邻点的图上距离代表排列顺序。

'''
        if section.id == 'x00':
            from .session_overview import PAGES

            return text.rstrip() + '\n\n' + ''.join(
                overview.replace('## 图 0：Session Overview', f'## 图 {index}：{title}', 1)
                for index, (_, title) in enumerate(PAGES, 1))
        first = text.find('## 图')
        if first >= 0:
            text = text[:first] + overview + text[first:]
    return text


def _norm(text: str) -> str:
    return re.sub(r'[\s·:：_\-–—()（）]+', '', text).lower()


def figure_guide(text: str, title: str) -> str:
    """The '## 图 n：<title>' block (up to the next '## ' heading), or ''.

    An exact (normalised) title match wins; otherwise the first heading that contains, or is contained
    in, the title. Exact-first matters when one title extends another, e.g. 'Channel saturation zones'
    and 'Channel saturation zones (linear axes)'.
    """
    lines = text.splitlines()
    target = _norm(title)
    headings = [(i, _norm(m['title'])) for i, line in enumerate(lines) if (m := HEADING.match(line))]
    start = next((i for i, h in headings if h == target), None)
    if start is None:
        start = next((i for i, h in headings if h and (target in h or h in target)), None)
    if start is None:
        return ''
    end = next((i for i, _ in headings if i > start), len(lines))
    return '\n'.join(lines[start:end]).strip()


def figure_titles(text: str) -> list[str]:
    return [m['title'] for line in text.splitlines() if line.startswith('## 图') and (m := HEADING.match(line))]
