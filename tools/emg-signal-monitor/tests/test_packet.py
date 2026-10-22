from __future__ import annotations

import numpy as np
import pytest

from packet import (
    PACKET_SAMPLES,
    PacketContractError,
    PacketDecoder,
    decode_pull_chunk,
)


def test_c_order_flatten_round_trip_preserves_every_element():
    source = np.arange(3 * PACKET_SAMPLES, dtype=np.float32).reshape(3, 40)
    decoded = PacketDecoder(3).decode(source.flatten(order="C"), 10.0)

    assert decoded.raw_ct.dtype == np.float32
    np.testing.assert_array_equal(decoded.raw_ct, source)
    assert decoded.sample_timestamps[-1] == pytest.approx(10.0)
    assert decoded.sample_timestamps[0] == pytest.approx(10.0 - 39 / 2_000)
    np.testing.assert_allclose(np.diff(decoded.sample_timestamps), 1 / 2_000)


def test_pull_chunk_decodes_each_row_to_n_c_40():
    matrices = [
        np.arange(80, dtype=np.float32).reshape(2, 40) + offset
        for offset in (0, 1000, 2000)
    ]
    raw, timestamps = decode_pull_chunk(
        [matrix.ravel(order="C").tolist() for matrix in matrices],
        [1.00, 1.02, 1.04],
        2,
    )

    assert raw.shape == (3, 2, 40)
    np.testing.assert_array_equal(raw, np.stack(matrices))
    np.testing.assert_array_equal(timestamps, [1.00, 1.02, 1.04])


def test_wrong_length_nonfinite_and_wrong_dtype_are_rejected():
    decoder = PacketDecoder(2)
    with pytest.raises(PacketContractError, match="length"):
        decoder.decode(np.zeros(79, dtype=np.float32), 1.0)
    with pytest.raises(PacketContractError, match="NaN"):
        decoder.decode(np.full(80, np.nan, dtype=np.float32), 1.0)
    with pytest.raises(PacketContractError, match="numeric dtype"):
        decoder.decode(np.full(80, "1"), 1.0)


def test_timestamp_must_be_finite_and_strictly_increasing():
    decoder = PacketDecoder(1)
    decoder.decode(np.zeros(40, dtype=np.float32), 2.0)
    with pytest.raises(PacketContractError, match="not later"):
        decoder.decode(np.zeros(40, dtype=np.float32), 2.0)
    with pytest.raises(PacketContractError, match="finite"):
        PacketDecoder(1).decode(np.zeros(40, dtype=np.float32), np.inf)
    with pytest.raises(PacketContractError, match="non-negative"):
        PacketDecoder(1).decode(np.zeros(40, dtype=np.float32), -1.0)


def test_pull_chunk_requires_one_timestamp_per_sample():
    with pytest.raises(PacketContractError, match="different sample"):
        PacketDecoder(1).decode_chunk([np.zeros(40), np.zeros(40)], [1.0])
