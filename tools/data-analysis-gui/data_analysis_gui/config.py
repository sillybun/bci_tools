"""Paths, fixed acquisition constants and user preference storage."""
from __future__ import annotations

import json
import os
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DIR = TOOL_ROOT / '.local'
PREFERENCES_PATH = LOCAL_DIR / 'preferences.json'
EXPORT_DIR = LOCAL_DIR / 'exports'

# Overridable via preferences or the environment; never required to exist.
DEFAULT_DATA_ROOT = Path(os.environ.get('DAGUI_DATA_ROOT', 'E:/AitBCI/data/keaiduo'))
BASE_LAYOUT_XLSX = DEFAULT_DATA_ROOT / 'channel_layout' / '天枢3.0_ChannelMap.xlsx'
_DATED_LAYOUTS = sorted(BASE_LAYOUT_XLSX.parent.glob('天枢3.0_ChannelMap[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].xlsx'))
DEFAULT_LAYOUT_XLSX = (_DATED_LAYOUTS[-1] if _DATED_LAYOUTS else BASE_LAYOUT_XLSX)
LEGACY_DEPTH_CSV = DEFAULT_DATA_ROOT / 'channel_layout' / '20260917_kad_channel_depth_TS-3.0.csv'
DEFAULT_DEPTH_CSV = DEFAULT_DATA_ROOT / 'channel_layout' / '20260917_kad_channel_depth_TS-3.0_corrected.csv'
LEGACY_LAYOUT_XLSX = DEFAULT_DATA_ROOT / 'channel_layout' / 'TS-3.0 ChannelMap-4code.xlsx'
BGA_SHANK_PROJECTION = DEFAULT_DATA_ROOT / 'channel_layout' / 'BGA_shank_projection.png'
OUTPUT_DIR_NAME = 'data_analysis_gui_output'


def default_output_root(data_root) -> Path:
    """Result folder parallel to the data folder.

    The nearest ancestor named `data` is replaced by a sibling `data_analysis_gui_output`, keeping the
    rest of the path: E:/AitBCI/data/keaiduo -> E:/AitBCI/data_analysis_gui_output/keaiduo. Without a
    `data` folder in the path the result folder sits next to the data folder:
    D:/rec/keaiduo -> D:/rec/data_analysis_gui_output/keaiduo.
    """
    path = Path(data_root).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    parts = path.parts
    for i in range(len(parts) - 1, 0, -1):
        if parts[i].lower() == 'data':
            return Path(*parts[:i], OUTPUT_DIR_NAME, *parts[i + 1:])
    if path.parent == path:                         # drive / filesystem root
        return path / OUTPUT_DIR_NAME
    return path.parent / OUTPUT_DIR_NAME / path.name

# Acquisition facts shared by all recordings handled here.
N_COLUMNS = 512            # transmitted channels per frame
N_ELECTRODES = 1024        # sites on the array
FRAME_SECONDS = 0.020      # configured integration window per frame
MAX_CODE = 3               # 2-bit counts: 3 means ">= 3"
DIRECTION_LABELS = ('0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°')
DIRECTION_ANGLES = tuple(range(0, 360, 45))

REQUIRED_STREAMS = ('spike_counts', 'spike_counts_metadata', 'cursortasks_events', 'cursortasks_cursor_state')
DECODER_STREAM = 'cursortasks_decoder_velocity'

# Cursor-state columns (see XDF channel labels).
C_X, C_Y, C_VX, C_VY, C_TX, C_TY, C_BLOCK, C_ORDER, C_TRIAL, C_ACTIVE = 2, 3, 4, 5, 6, 7, 8, 9, 10, 11


def load_preferences() -> dict:
    try:
        values = json.loads(PREFERENCES_PATH.read_text(encoding='utf-8'))
        # Upgrade the previous built-in default; retain explicit custom layouts.
        saved = values.get('layout_path', '')
        if saved and Path(saved).resolve() in (LEGACY_LAYOUT_XLSX.resolve(), BASE_LAYOUT_XLSX.resolve()):
            values['layout_path'] = str(DEFAULT_LAYOUT_XLSX)
        def upgrade_depth(node):
            if not isinstance(node, dict):
                return
            if node.get('depth_csv') and Path(node['depth_csv']).resolve() == LEGACY_DEPTH_CSV.resolve():
                node['depth_csv'] = str(DEFAULT_DEPTH_CSV)
            for child in node.values():
                upgrade_depth(child)
        upgrade_depth(values)
        return values
    except (OSError, ValueError):
        return {}


def save_preferences(values: dict) -> None:
    try:
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        PREFERENCES_PATH.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding='utf-8')
    except OSError:
        pass
