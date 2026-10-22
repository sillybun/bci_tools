"""Native desktop smoke: real LSL, mode/filter changes, reconnect and close."""

import argparse
from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import RawStreamMonitorWindow
from raw_stream_monitor import DISPLAY_MODE_RAW, DISPLAY_MODE_BIPOLAR32
from lsl_smoke import sample_source, wait_until


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    qt = QApplication.instance() or QApplication([])
    with sample_source() as config:
        window = RawStreamMonitorWindow(config, auto_connect=False)
        try:
            window.show()
            window.connect_stream()
            wait_until(qt, lambda: window._last_health.get("rx_total", 0) >= 100)
            assert window._display_mode == DISPLAY_MODE_BIPOLAR32
            assert len(window._labels) == 32
            assert window._last_health["invalid_total"] == 0
            if args.output:
                args.output.mkdir(parents=True, exist_ok=True)
                qt.processEvents()
                assert window.grab().save(str(args.output / "emg-monitor-bipolar.png"))
            window.display_mode_combo.setCurrentText(DISPLAY_MODE_RAW)
            assert len(window._labels) == 64
            window.bandpass_enable_check.setChecked(False)
            assert window._apply_bandpass_settings()
            assert not window._bandpass_enabled
            window.bandpass_enable_check.setChecked(True)
            window.bandpass_low_spin.setValue(10)
            window.bandpass_high_spin.setValue(250)
            assert window._apply_bandpass_settings()
            assert window._bandpass_config.low_hz == 10
            window.display_mode_combo.setCurrentText(DISPLAY_MODE_BIPOLAR32)
            assert window.disconnect_stream()
            window.connect_stream()
            wait_until(qt, lambda: window._last_health.get("rx_total", 0) >= 30)
            assert window._last_health["invalid_total"] == 0
            assert window.disconnect_button.isEnabled()
            window.settings_dialog.show()
            qt.processEvents()
            if args.output:
                assert window.settings_dialog.grab().save(str(args.output / "emg-monitor-settings.png"))
            print(f"Native EMG UI passed: real LSL, raw/bipolar, filter apply, reconnect; environment={sys.prefix}")
        finally:
            window.close()
            qt.processEvents()
            assert window.worker is None or not window.worker.isRunning()


if __name__ == "__main__":
    main()
