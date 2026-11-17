"""Electrode identity: physical workbook layout, 128×8 id grid, configs.ini tiers.

Columns stay in acquisition order everywhere; these helpers only annotate them.
Untransmitted electrodes are NaN (grey), never zero.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
import zipfile
from xml.etree import ElementTree as ET

import numpy as np

from .. import config as C

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
TIERS = ('good', 'uncertain', 'bad')
INI_KEYS = ('good', 'uncertain', 'bad', 'crosstalk_channels', 'last_programmed_channels', 'last_selected_channels')


# ---------------------------------------------------------------- workbook layout
@lru_cache(maxsize=4)
def _read_layout(path: str, mtime: float) -> dict:
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        sheet = workbook.find('m:sheets/m:sheet', NS)
        rels = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        target = next(r.attrib['Target'] for r in rels if r.attrib['Id'] == sheet.attrib[REL])
        target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
        root = ET.fromstring(archive.read(target))
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = ET.fromstring(archive.read('xl/sharedStrings.xml'))
            strings = [''.join(t.text or '' for t in item.findall('.//m:t', NS)) for item in shared]
    raw_cells = {}
    for cell in root.findall('.//m:sheetData/m:row/m:c', NS):
        value = cell.find('m:v', NS)
        kind = cell.attrib.get('t', 'n')
        text = (''.join(t.text or '' for t in cell.findall('.//m:t', NS)) if kind == 'inlineStr'
                else strings[int(value.text)] if kind == 's' and value is not None
                else value.text if value is not None else '')
        raw_cells[cell.attrib['r']] = text
    # New Tianshu workbooks are channel tables, not 128x8 numeric matrices.
    if [raw_cells.get(f'{col}1') for col in 'ABC'] == ['Channel#', 'Shank', 'Elec#']:
        position = np.full((C.N_ELECTRODES, 2), -1, int)
        seen = set()
        records = []
        for row in range(2, C.N_ELECTRODES + 2):
            try:
                numbers = [float(raw_cells[f'{col}{row}']) for col in 'ABC']
                if any(not np.isfinite(v) or v != int(v) for v in numbers):
                    raise ValueError()
                channel, shank, electrode = map(int, numbers)
            except (KeyError, ValueError, OverflowError) as exc:
                raise ValueError(f'布局表第 {row} 行需包含整数 Channel#、Shank、Elec#') from exc
            records.append((row, channel, shank, electrode))
        electrode_numbers = {record[3] for record in records}
        if electrode_numbers == set(range(128)):
            electrode_base = 0
        elif electrode_numbers == set(range(1, 129)):
            electrode_base = 1
        else:
            raise ValueError('布局 Elec# 需完整采用 0–127 或 1–128 编号')
        for row, channel, shank, electrode in records:
            if not (0 <= channel < C.N_ELECTRODES and 1 <= shank <= 8):
                raise ValueError(f'布局表第 {row} 行通道或 shank / Elec# 越界')
            if position[channel, 0] >= 0 or (electrode, shank) in seen:
                raise ValueError(f'布局表第 {row} 行通道或位置重复')
            position[channel] = (electrode - electrode_base, shank - 1)
            seen.add((electrode, shank))
        if len(seen) != C.N_ELECTRODES:
            raise ValueError('布局文件需覆盖 1024 个通道和 shank 位点')
        bga = None
        if raw_cells.get('D1') == 'BGA Row' and raw_cells.get('E1') == 'BGA Col':
            bga = np.full((C.N_ELECTRODES, 2), -1, int)
            for row, channel, _, _ in records:
                values = [float(raw_cells[f'{col}{row}']) for col in 'DE']
                if any(v != int(v) for v in values):
                    raise ValueError(f'布局表第 {row} 行 BGA 坐标需为整数')
                bga[channel] = values
            if (not ((bga[:,0]>=1)&(bga[:,0]<=30)&(bga[:,1]>=1)&(bga[:,1]<=38)).all()
                    or len(set(map(tuple,bga))) != C.N_ELECTRODES):
                raise ValueError('BGA 坐标需覆盖 1024 个唯一位点，行1–30、列1–38')
        return dict(source=path, rows=128, columns=8, position=position, kind='shank_table', bga_position=bga,
                    row_label='Elec# (0 near fan-in; 127 tip)', column_label='Shank', index_base=0,
                    electrode_index_base=electrode_base)
    cells = []
    for cell in root.findall('.//m:sheetData/m:row/m:c', NS):
        value = cell.find('m:v', NS)
        if value is None or cell.attrib.get('t', 'n') != 'n':
            continue
        number = float(value.text)
        if not np.isfinite(number) or number != int(number):
            raise ValueError('布局通道编号需为有限整数')
        match = re.fullmatch(r'([A-Z]+)([1-9][0-9]*)', cell.attrib['r'])
        column = 0
        for char in match[1]:
            column = column * 26 + ord(char) - 64
        cells.append((int(match[2]), column, int(number)))
    labels = [c[2] for c in cells]
    if len(labels) != C.N_ELECTRODES or len(set(labels)) != C.N_ELECTRODES:
        raise ValueError('布局文件需包含 1024 个唯一通道编号')
    base = 0 if min(labels) == 0 else 1
    if set(labels) != set(range(base, C.N_ELECTRODES + base)):
        raise ValueError('布局通道编号需为 0–1023 或 1–1024')
    rows, cols = max(c[0] for c in cells), max(c[1] for c in cells)
    position = np.zeros((C.N_ELECTRODES, 2), int)          # electrode-1 -> (row-1, col-1)
    for r, c, label in cells:
        position[label - base] = (r - 1, c - 1)
    return dict(source=path, rows=rows, columns=cols, position=position, kind='grid', index_base=base)


def read_layout(path=None) -> dict | None:
    """Workbook layout; None → default workbook path; '' → None (no workbook: maps use the
    default index arrangement, electrode e at column (e-1)//128, row (e-1)%128 of 128 × 8)."""
    if path == '':
        return None
    path = Path(path or C.DEFAULT_LAYOUT_XLSX)
    if not path.is_file():
        return None
    return _read_layout(str(path.resolve()), path.stat().st_mtime)


def id_grid(electrode_ids, values, total=C.N_ELECTRODES, n_cols=8) -> np.ndarray:
    """128 rows × 8 columns by electrode id, column-major (ids 1–128 form column 1)."""
    full = np.full(total, np.nan)
    full[np.asarray(electrode_ids) - 1] = values
    return full.reshape(n_cols, total // n_cols).T


def layout_grid(layout, electrode_ids, values) -> np.ndarray:
    """Rows indexed by position 0–127; plots place row 0 at the bottom."""
    grid = np.full((layout['rows'], layout['columns']), np.nan)
    pos = layout['position'][np.asarray(electrode_ids) - 1]
    grid[pos[:, 0], pos[:, 1]] = values
    return grid


CELL_EDGE_COLOR = (0.5, 0.5, 0.5, 0.35)     # faint mid grey: visible on white and on black cells
CELL_EDGE_LW = 0.25


def draw_channel_map(ax, electrode_ids, values, *, layout=None, cmap='viridis', norm=None,
                     title='', colorbar=True, cbar_label='', missing='#d6d4cf', extend='neither',
                     cell_edges=True, show_labels=True):
    """Physical layout if available, else the 128×8 electrode-id grid.

    Every electrode is a square cell. Cell edges can be shown independently of labels.
    """
    from matplotlib import colormaps
    if isinstance(cmap, str):
        cmap = colormaps[cmap]
    cmap = cmap.with_extremes(bad=missing)
    if layout is not None:
        grid = layout_grid(layout, electrode_ids, values)
        ylabel = layout.get('row_label', 'Workbook row')
    else:
        grid = id_grid(electrode_ids, values)
        ylabel = 'Row'
    image = ax.imshow(np.ma.masked_invalid(grid), aspect='equal', cmap=cmap, norm=norm,
                      interpolation='nearest', origin='lower')
    ticks = np.unique(np.r_[np.arange(0, grid.shape[0], 16), grid.shape[0]-1])
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    if show_labels:
        ax.set_yticks(ticks, ticks, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
    else:
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
    if cell_edges:      # faint border around every cell so rows/columns are easy to follow
        ax.set_xticks(np.arange(-0.5, grid.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0]), minor=True)
        ax.tick_params(which='minor', length=0)
        ax.grid(False, which='major')
        ax.grid(True, which='minor', color=CELL_EDGE_COLOR, lw=CELL_EDGE_LW)
        ax.set_axisbelow(False)
    if title:
        ax.set_title(title, fontsize=8)
    if colorbar:
        cb = ax.figure.colorbar(image, ax=ax, fraction=0.08, pad=0.03, extend=extend)
        cb.ax.tick_params(labelsize=6)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=7)
    ax.set_ylim(0, grid.shape[0]-1)
    return image


# ---------------------------------------------------------------- configs.ini tiers
def read_channel_config(path) -> dict:
    """Channel tiers from INI or the last saved Neuralinsight log classification.

    Both sources use zero-based indices; returned electrode IDs are one-based.
    Logs do not provide last_selected/programmed_channels: use XDF for mapping.
    """
    path = Path(path)
    text = path.read_text(encoding='utf-8-sig', errors='replace')
    if path.suffix.lower() == '.log':
        return _read_log_config(text, path)
    out = {}
    for key in INI_KEYS:
        match = re.search(rf'^[ \t]*{key}[ \t]*=[ \t]*"?([^"\r\n]*)"?[ \t]*$', text, re.M)
        out[key] = [int(x) + 1 for x in (match.group(1) if match else '').split(',') if x.strip()]
    if not out['good']:
        raise ValueError('通道配置 INI 中未找到 good 通道列表')
    return out


def _read_log_config(text, path):
    records = [(number, line) for number, line in enumerate(text.splitlines(), 1)
               if re.search(r'\bfilter\.[\w.]+\.results_applied:', line)]
    if not records:
        raise ValueError('Neuralinsight 日志中未找到已保存的通道筛选结果（results_applied）')
    number, line = records[-1]
    out = {key: [] for key in INI_KEYS}
    for field, key in (('good', 'good'), ('uncertain', 'uncertain'), ('bad', 'bad'),
                       ('crosstalk', 'crosstalk_channels')):
        match = re.search(rf'\b{field}_channel_ids=(\[[^\]]*\])', line)
        if not match:
            if field == 'crosstalk':
                continue
            raise ValueError(f'{path.name}:{number} 最后一次筛选结果缺少 {field}_channel_ids')
        try:
            ids = json.loads(match[1])
        except ValueError as exc:
            raise ValueError(f'{path.name}:{number} 的 {field} 列表损坏') from exc
        if (any(type(i) is not int or not 0 <= i < C.N_ELECTRODES for i in ids)
                or len(ids) != len(set(ids))):
            raise ValueError(f'{path.name}:{number} 的 {field} 通道编号越界或重复')
        count = re.search(rf'\b{field}_count=(\d+)', line)
        if count and int(count[1]) != len(ids):
            raise ValueError(f'{path.name}:{number} 的 {field} 通道数量与列表不一致')
        out[key] = [i + 1 for i in ids]
    tiers = [set(out[key]) for key in TIERS]
    if (set.union(*tiers) != set(range(1, C.N_ELECTRODES + 1))
            or sum(map(len, tiers)) != C.N_ELECTRODES):
        raise ValueError(f'{path.name}:{number} 的 good/uncertain/bad 未形成完整且互斥的 1024 通道分类')
    out['_source'] = dict(kind='neuralinsight_log', path=str(path), line=number,
                          timestamp=line.split(' [', 1)[0], selection='last_results_applied',
                          missing_fields=[key for key in INI_KEYS if f'{key}=' not in line
                                          and key.startswith('last_')])
    return out


def column_tiers(electrode_ids, config: dict) -> np.ndarray:
    """Tier name per stored column: good / uncertain / bad / unlisted."""
    lookup = {e: t for t in TIERS for e in config.get(t, [])}
    return np.array([lookup.get(int(e), 'unlisted') for e in electrode_ids])
