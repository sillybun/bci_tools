# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: 8d79df0475b85ca859a02a8268f5262358a22afa
# original path: src/bci_decoder_api/schema_catalog.py
# original git blob: 34408aadcafad0679a069226bfc0eef930933f49
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Built-in public tensor semantics shared by producers and decoders.

The catalog owns reusable SchemaDefinition objects only. Concrete shapes,
channel identities, and ordering remain deployment-specific TensorLayouts and
are embedded together with the schema snapshot in each ModelArtifact.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .specs import SchemaDefinition


_SCHEMA_DEFINITIONS: Mapping[str, SchemaDefinition] = MappingProxyType(
    {
        definition.schema_id: definition
        for definition in (
            SchemaDefinition(
                schema_id="bci.feature.intracortical_spike_count_20ms.v1",
                tensor_role="feature",
                dtype="float32",
                axes=("channel",),
                description=(
                    "Nonnegative per-channel detected-spike counts accumulated "
                    "over one nominal non-overlapping 20 ms bin, with one vector "
                    "emitted per Decoder update."
                ),
                value_unit="count_per_20ms_bin",
                attributes={
                    "modality": "intracortical_spike",
                    "aggregation": "count",
                    "nominal_bin_width_ns": "20000000",
                    "bin_overlap": "none",
                    "timestamp_semantics": "right_edge_bin_available_time",
                    "value_domain": "nonnegative_integer_valued_float32",
                    "channel_identity": "defined_by_tensor_layout",
                },
            ),
            SchemaDefinition(
                schema_id="bci.feature.emg_causal_envelope_50hz.v1",
                tensor_role="feature",
                dtype="float32",
                axes=("channel",),
                description=(
                    "Per-channel causal surface-EMG linear envelope: a 4th-order "
                    "10-130 Hz Butterworth band-pass, full-wave rectification, then "
                    "a non-overlapping 40-sample mean, emitting one nonnegative "
                    "vector per 20 ms Decoder update at 50 Hz. Any reference method "
                    "and the band edges are frozen by this schema version; the "
                    "concrete channel count and identity live in the TensorLayout."
                ),
                value_unit="microvolt",
                attributes={
                    "modality": "surface_emg",
                    "aggregation": "rectified_bandpass_mean",
                    "band_pass_hz": "10-130",
                    "filter": "causal_butterworth_order4",
                    "nominal_bin_width_ns": "20000000",
                    "bin_overlap": "none",
                    "timestamp_semantics": "right_edge_bin_available_time",
                    "value_domain": "nonnegative_real_valued_float32",
                    "channel_identity": "defined_by_tensor_layout",
                },
            ),
            SchemaDefinition(
                schema_id="bci.feature.ecog_iir_bandpower_20ms.v1",
                tensor_role="feature",
                dtype="float32",
                axes=("channel", "band"),
                description=(
                    "Per-channel, per-band log10-transformed spectral band power "
                    "computed by an IIR band-pass filter bank over a sliding 2 s "
                    "power-integration window, emitting one (channel, band) matrix "
                    "every 20 ms. Adjacent 20 ms update bins are non-overlapping in "
                    "emission cadence, but successive 2 s analysis windows overlap. "
                    "The six frequency bands and their edges are frozen by this "
                    "schema version and given in fixed order by the 'band' axis; "
                    "the concrete channel count, channel identity, and channel "
                    "order live in the TensorLayout."
                ),
                value_unit="log10_band_power",
                attributes={
                    "modality": "ecog",
                    "feature_type": "iir_bandpower",
                    "filter": "iir_bandpass_bank",
                    "value_transform": "log10",
                    "band_count": "6",
                    "band_order": (
                        "theta,mu_alpha,beta,low_gamma,high_gamma,very_high_gamma"
                    ),
                    "band_edges_hz": (
                        "theta:4-8,mu_alpha:8-13,beta:13-30,low_gamma:30-70,"
                        "high_gamma:70-150,"
                        "very_high_gamma:150-200"
                    ),
                    "nominal_bin_width_ns": "20000000",
                    "bin_overlap": "none",
                    "analysis_window_ns": "2000000000",
                    "analysis_window_overlap": "sliding_2s_window_hop_20ms",
                    "timestamp_semantics": "right_edge_bin_available_time",
                    "value_domain": "real_valued_float32",
                    "channel_identity": "defined_by_tensor_layout",
                },
            ),
            SchemaDefinition(
                schema_id="bci.action.categorical_probability.v1",
                tensor_role="action",
                dtype="float32",
                axes=("class",),
                description=(
                    "Categorical class probabilities in the exact class-label order "
                    "declared by the deployed TensorLayout."
                ),
                value_unit="probability",
                attributes={
                    "quantity": "categorical_probability",
                    "value_domain": "finite_nonnegative_sum_to_one",
                    "class_identity": "defined_by_tensor_layout",
                },
            ),
            SchemaDefinition(
                schema_id="bci.action.cursor_velocity_px_per_s.v1",
                tensor_role="action",
                dtype="float32",
                axes=("component",),
                description=(
                    "Cartesian cursor velocity command expressed in screen "
                    "pixels per second."
                ),
                value_unit="pixel_per_second",
                attributes={
                    "quantity": "velocity",
                    "coordinate_frame": "cursor_task_screen_xy",
                },
            ),
            SchemaDefinition(
                schema_id="bci.feedback.cursor_position_px.v1",
                tensor_role="feedback",
                dtype="float32",
                axes=("component",),
                description="Cartesian cursor center position expressed in screen pixels.",
                value_unit="pixel",
                attributes={
                    "quantity": "position",
                    "coordinate_frame": "cursor_task_screen_xy",
                },
            ),
        )
    }
)


def load(schema_id: str) -> SchemaDefinition:
    """Return one immutable public schema definition by exact ID."""

    try:
        return _SCHEMA_DEFINITIONS[schema_id]
    except KeyError as exc:
        raise KeyError(f"unknown schema_id: {schema_id}") from exc


def list_ids() -> tuple[str, ...]:
    """Return all built-in public schema IDs in stable sorted order."""

    return tuple(sorted(_SCHEMA_DEFINITIONS))
