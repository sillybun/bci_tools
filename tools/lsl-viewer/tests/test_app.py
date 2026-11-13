import json
import threading
import time

from lsl_viewer.app import (
    UI_TEXT,
    AsyncLslStreamCatalog,
    LslStreamCatalog,
    TkLslViewer,
    active_registry_path,
    format_label,
    format_sample_preview,
    human_timestamp,
    match_publisher_process,
    match_registry_process,
    publisher_origin_label,
    rounded_rectangle_points,
    scrollbar_thumb_bounds,
    stream_record,
    ui_text,
)


class Info:
    def name(self):
        return "EEG"

    def type(self):
        return "EEG"

    def channel_count(self):
        return 2

    def nominal_srate(self):
        return 250

    def channel_format(self):
        return "float32"

    def source_id(self):
        return "muse-01"

    def hostname(self):
        return "lab-mac"

    def uid(self):
        return "uid-1"

    def created_at(self):
        return 1.0

    def desc(self):
        return None


class Inlet:
    def __init__(self, *_args, **_kwargs):
        self.samples = [([1.0, 2.0], 10.5), (None, None)]

    def pull_sample(self, timeout=0.0):
        return self.samples.pop(0)


def test_record_and_live_sample():
    record = stream_record(Info())
    assert record.source_id == "muse-01"
    catalog = LslStreamCatalog(resolve=lambda: [Info()], inlet_factory=Inlet)
    result = catalog.refresh()[0]
    assert result.latest_sample == [1.0, 2.0]
    assert result.signal_state() == "periodic-live"


def test_record_sample_orders_delayed_samples_and_uses_timestamp_window():
    record = stream_record(Info())
    record.record_sample([3.0], 3.0, history_seconds=10.0)
    record.record_sample([1.0], 1.0, history_seconds=10.0)
    record.record_sample([2.0], 2.0, history_seconds=10.0)
    assert [timestamp for timestamp, _sample in record.sample_history] == [1.0, 2.0, 3.0]
    assert record.latest_timestamp == 3.0
    assert record.latest_sample == [3.0]


def test_catalog_frequency_uses_timestamped_history_not_refresh_arrival_rate():
    class DelayedInlet:
        def __init__(self, *_args, **_kwargs):
            self.samples = [([3.0], 3.0), ([1.0], 1.0), ([2.0], 2.0), (None, None)]

        def pull_sample(self, timeout=0.0):
            return self.samples.pop(0)

    catalog = LslStreamCatalog(resolve=lambda: [Info()], inlet_factory=DelayedInlet)
    record = catalog.refresh()[0]
    assert record.sample_rate_hz == 1.0
    assert [timestamp for timestamp, _sample in record.sample_history] == [1.0, 2.0, 3.0]


def test_catalog_bounds_samples_drained_per_stream_refresh():
    class ContinuousInlet:
        def __init__(self, *_args, **_kwargs):
            self.sequence = 0

        def pull_sample(self, timeout=0.0):
            self.sequence += 1
            return [self.sequence], float(self.sequence)

    catalog = LslStreamCatalog(
        resolve=lambda: [Info()],
        inlet_factory=ContinuousInlet,
        max_samples_per_stream=3,
    )

    record = catalog.refresh()[0]

    assert record.sample_count == 3
    assert record.latest_sample == [3]


def test_async_catalog_keeps_refresh_requests_non_blocking():
    entered = threading.Event()
    release = threading.Event()

    def blocking_resolve():
        entered.set()
        assert release.wait(timeout=1.0)
        return [Info()]

    catalog = LslStreamCatalog(resolve=blocking_resolve, inlet_factory=Inlet)
    async_catalog = AsyncLslStreamCatalog(catalog, refresh_interval_s=60.0)
    async_catalog.start()
    try:
        assert entered.wait(timeout=1.0)
        started_at = time.monotonic()
        async_catalog.request_refresh()
        assert time.monotonic() - started_at < 0.05
        assert async_catalog.latest() is None

        release.set()
        deadline = time.monotonic() + 1.0
        snapshot = None
        while snapshot is None and time.monotonic() < deadline:
            snapshot = async_catalog.latest()
            time.sleep(0.001)
        assert snapshot is not None
        assert list(snapshot.records) == ["uid-1"]
        assert snapshot.error is None
    finally:
        release.set()
        async_catalog.close()


def test_stale_and_irregular_states():
    record = stream_record(Info())
    record.sample_count = 1
    record.last_sample_monotonic = time.monotonic() - 5
    assert record.signal_state() == "periodic-stale"
    record.nominal_srate = 0
    assert record.signal_state() == "irregular"


def test_periodic_stream_without_samples_is_waiting():
    record = stream_record(Info())
    assert record.signal_state() == "periodic-waiting"


def test_publisher_match_uses_stream_identity_and_excludes_viewer():
    record = stream_record(Info())
    matches = match_publisher_process(
        record,
        [(101, "python eeg_publisher.py --name EEG"), (102, "lsl-viewer"), (103, "unrelated")],
    )
    assert matches == [(101, "python eeg_publisher.py --name EEG")]


def test_registry_match_prefers_declared_lsl_stream():
    record = stream_record(Info())
    assert match_registry_process(
        record, [{"pid": 77, "name": "EMG source", "streams": [{"name": "EEG", "type": "EEG"}]}]
    ) == [(77, "EMG source")]


def test_registry_match_uses_newest_start_for_equal_candidates():
    record = stream_record(Info())
    stream = {"name": "EEG", "type": "EEG"}
    entries = [
        {"pid": 10, "name": "old", "started_at": 1.0, "streams": [stream]},
        {"pid": 20, "name": "new", "started_at": 2.0, "streams": [stream]},
    ]
    assert match_registry_process(record, entries)[0] == (20, "new")


def test_registry_path_uses_macos_application_support(monkeypatch):
    monkeypatch.setattr("lsl_viewer.app.platform.system", lambda: "Darwin")
    assert active_registry_path().endswith(
        "Library/Application Support/BCIControl/active_lsl_processes.json"
    )


def test_catalog_binds_live_local_registry_pid(monkeypatch):
    class LocalInfo(Info):
        def hostname(self):
            import platform

            return platform.node()

    monkeypatch.setattr("lsl_viewer.app.local_processes", lambda: [(77, "python publisher.py")])
    catalog = LslStreamCatalog(resolve=lambda: [LocalInfo()], inlet_factory=Inlet)
    catalog.registry_reader = lambda: [
        {"pid": 77, "name": "Managed publisher", "streams": [{"name": "EEG", "type": "EEG"}]}
    ]
    record = catalog.refresh()[0]
    assert record.publisher_pid == 77
    assert record.publisher_origin == "registry"


def test_catalog_marks_unmatched_remote_stream():
    catalog = LslStreamCatalog(resolve=lambda: [Info()], inlet_factory=Inlet)
    catalog.registry_reader = list
    record = catalog.refresh()[0]
    assert record.publisher_candidates == []
    assert record.publisher_origin == "remote"


def test_catalog_removes_stream_missing_from_latest_resolver_snapshot():
    snapshots = [[Info()], []]

    class CloseableInlet(Inlet):
        closed = False

        def close_stream(self):
            self.closed = True

    catalog = LslStreamCatalog(
        resolve=lambda: snapshots.pop(0), inlet_factory=CloseableInlet, missing_grace_s=0.0
    )
    assert [record.uid for record in catalog.refresh()] == ["uid-1"]
    inlet = catalog.inlets["uid-1"]
    assert catalog.refresh() == []
    assert "uid-1" not in catalog.records
    assert "uid-1" not in catalog.inlets
    assert inlet.closed is True


def test_catalog_updates_discovered_metadata_for_existing_uid():
    class UpdatedInfo(Info):
        def name(self):
            return "EEG-updated"

        def channel_count(self):
            return 4

    snapshots = [[Info()], [UpdatedInfo()]]
    catalog = LslStreamCatalog(resolve=lambda: snapshots.pop(0), inlet_factory=Inlet)
    catalog.refresh()
    record = catalog.refresh()[0]
    assert record.name == "EEG-updated"
    assert record.channel_count == 4


def test_viewer_refreshes_at_five_hz_by_default():
    assert TkLslViewer().refresh_ms == 200


def test_refresh_publishers_marks_unmatched_local_stream(monkeypatch):
    record = stream_record(Info())
    monkeypatch.setattr("lsl_viewer.app.platform.node", lambda: "lab-mac")
    catalog = LslStreamCatalog(resolve=list)
    catalog.registry_reader = list
    assert catalog.refresh_publishers(record, [(999, "unrelated")]) == []
    assert record.publisher_origin == "unmatched"


def test_format_and_timestamp_are_human_readable():
    assert format_label("5") == "float32"
    assert format_label("float32") == "float32"
    assert human_timestamp(0).startswith("1970-01-01 08:00:00.000")


def test_sample_preview_pretty_prints_single_json_channel():
    preview = format_sample_preview(['{"state":"ready","position":[1,2,3]}'])
    assert preview == '{\n  "state": "ready",\n  "position": [\n    1,\n    2,\n    3\n  ]\n}'


def test_sample_preview_preserves_multichannel_shape_and_plain_strings():
    preview = format_sample_preview(['{"active":["forward"]}', "plain", 3.5])
    assert json.loads(preview) == [{"active": ["forward"]}, "plain", 3.5]


def test_sample_preview_handles_empty_sample():
    assert format_sample_preview([]) == "[waiting for samples]"


def test_language_catalogs_have_matching_keys_and_english_is_fully_english():
    assert set(UI_TEXT["zh"]) == set(UI_TEXT["en"])
    assert not any(
        "\u4e00" <= character <= "\u9fff" for text in UI_TEXT["en"].values() for character in text
    )


def test_localized_dynamic_text_covers_non_widget_ui_content():
    assert ui_text("en", "stream_count", count=3) == "03 STREAMS · LIVE"
    assert publisher_origin_label("remote", "en") == "Remote computer"
    assert human_timestamp(None, "en") == "No timestamp received"
    assert format_sample_preview([], "zh") == "[等待样本]"


def test_rounded_rectangle_points_clamp_radius_to_geometry():
    points = rounded_rectangle_points(0, 0, 10, 6, radius=20)
    assert points[:4] == [3.0, 0, 7.0, 0]
    assert min(points[::2]) == 0
    assert max(points[::2]) == 10
    assert min(points[1::2]) == 0
    assert max(points[1::2]) == 6


def test_scrollbar_thumb_geometry_preserves_endpoints_and_minimum_size():
    assert scrollbar_thumb_bounds(100, 0.0, 0.1) == (2.0, 26.0)
    assert scrollbar_thumb_bounds(100, 0.9, 1.0) == (74.0, 98.0)
    assert scrollbar_thumb_bounds(100, 0.0, 1.0) == (2.0, 98.0)
