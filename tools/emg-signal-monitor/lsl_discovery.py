#!/usr/bin/env python3
"""Shared LSL discovery for externally operated input devices.

Experiment-internal streams are selected by their run-scoped ``source_id``.
External devices are different: they may start on another computer before they
know an experiment ``run_id``.  Those streams are discovered by a stable
name/type contract, then their publisher-provided identity is recorded and
locked for reconnects.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Any

from pylsl import StreamInfo, resolve_byprop

from stream_metrics import lsl_channel_format_name


class StreamAmbiguityError(RuntimeError):
    """Raised when discovery cannot select exactly one valid input stream."""


@dataclass(frozen=True)
class ExternalStreamContract:
    """Properties that an externally operated LSL stream must satisfy."""

    name: str
    stream_type: str
    channel_count: int
    channel_format: str
    nominal_srate: float
    nominal_srate_tolerance_hz: float = 0.5
    allow_stream_type_mismatch: bool = False
    #: Treat ``channel_count`` as a minimum. For consumers that select the
    #: channels they need by label, a publisher carrying extra columns is
    #: usable; only a publisher carrying too few can never satisfy them.
    allow_extra_channels: bool = False
    #: Some fixed external publishers do not set a source_id. Name plus the
    #: complete wire contract is sufficient when this is false.
    require_source_id: bool = True
    source_option: str = "--emg-source-id"


def stream_identity(stream: StreamInfo) -> dict[str, Any]:
    """Return the publisher identity and wire contract for logs/errors."""

    return {
        "name": stream.name(),
        "type": stream.type(),
        "source_id": stream.source_id(),
        "uid": stream.uid(),
        "hostname": stream.hostname(),
        "channel_count": stream.channel_count(),
        "channel_format": lsl_channel_format_name(stream.channel_format()),
        "nominal_hz": stream.nominal_srate(),
    }


def _contract_rejections(
    stream: StreamInfo,
    contract: ExternalStreamContract,
    *,
    source_id: str | None,
) -> list[str]:
    reasons: list[str] = []
    if stream.name() != contract.name:
        reasons.append(f"name={stream.name()!r}")
    if (
        stream.type() != contract.stream_type
        and not contract.allow_stream_type_mismatch
    ):
        reasons.append(f"type={stream.type()!r}")
    if contract.allow_extra_channels:
        if stream.channel_count() < contract.channel_count:
            reasons.append(
                f"channel_count={stream.channel_count()} "
                f"(need at least {contract.channel_count})"
            )
    elif stream.channel_count() != contract.channel_count:
        reasons.append(f"channel_count={stream.channel_count()}")
    actual_format = lsl_channel_format_name(stream.channel_format())
    if actual_format != contract.channel_format:
        reasons.append(f"channel_format={actual_format!r}")
    if (
        abs(float(stream.nominal_srate()) - contract.nominal_srate)
        > contract.nominal_srate_tolerance_hz
    ):
        reasons.append(f"nominal_hz={stream.nominal_srate()}")
    if contract.require_source_id and not stream.source_id():
        reasons.append("source_id is empty")
    if source_id is not None and stream.source_id() != source_id:
        reasons.append(f"source_id={stream.source_id()!r}")
    return reasons


def resolve_external_stream(
    *,
    stop_event: Event,
    log: Any,
    contract: ExternalStreamContract,
    description: str,
    source_id: str | None = None,
    resolve_timeout_s: float = 1.0,
) -> StreamInfo | None:
    """Resolve one external stream by name and validate its complete contract.

    ``source_id`` is an optional operator selector.  It is deliberately not
    derived from the experiment run ID.  After the first connection, callers
    can pass the resolved source ID back here to lock reconnects to the same
    physical/logical publisher.
    """

    waiting_logged = False
    last_rejection_signature: tuple[Any, ...] | None = None
    while not stop_event.is_set():
        discovered = resolve_byprop(
            "name",
            contract.name,
            minimum=1,
            timeout=resolve_timeout_s,
        )

        # liblsl should return each outlet once.  Deduplicate defensively so a
        # repeated discovery record cannot create a false ambiguity.
        candidates: list[StreamInfo] = []
        seen: set[tuple[str, str, str]] = set()
        for stream in discovered:
            identity = stream_identity(stream)
            key = (
                str(identity["uid"]),
                str(identity["source_id"]),
                str(identity["hostname"]),
            )
            if key not in seen:
                candidates.append(stream)
                seen.add(key)

        accepted: list[StreamInfo] = []
        rejected: list[dict[str, Any]] = []
        for stream in candidates:
            reasons = _contract_rejections(stream, contract, source_id=source_id)
            if reasons:
                rejected.append(
                    {
                        "identity": stream_identity(stream),
                        "reasons": reasons,
                    }
                )
            else:
                accepted.append(stream)

        if len(accepted) == 1:
            selected = accepted[0]
            identity = stream_identity(selected)
            if (
                selected.type() != contract.stream_type
                and contract.allow_stream_type_mismatch
            ):
                log.bind(
                    component="lsl",
                    event="stream_contract_warning",
                    role="subscriber",
                    description=description,
                    stream=contract.name,
                    reason_code="stream_type_mismatch_accepted",
                    expected_type=contract.stream_type,
                    actual_type=selected.type(),
                    source_id_selector=source_id,
                    **identity,
                ).warning("Stream contract warning")
            log.bind(
                component="lsl",
                event="stream_connected",
                role="subscriber",
                description=description,
                stream=contract.name,
                discovery_property="name",
                source_id_selector=source_id,
                **identity,
            ).info("Stream connected")
            return selected

        if len(accepted) > 1:
            identities = [stream_identity(stream) for stream in accepted]
            log.bind(
                component="lsl",
                event="stream_ambiguous",
                role="subscriber",
                description=description,
                stream=contract.name,
                source_id_selector=source_id,
                candidates=identities,
            ).error("Stream ambiguous")
            raise StreamAmbiguityError(
                f"{contract.name!r} matched {len(accepted)} valid LSL streams; "
                f"set {contract.source_option} to select one explicitly"
            )

        rejection_signature = tuple(
            (
                item["identity"]["uid"],
                item["identity"]["source_id"],
                tuple(item["reasons"]),
            )
            for item in rejected
        )
        if rejected and rejection_signature != last_rejection_signature:
            log.bind(
                component="lsl",
                event="stream_candidates_rejected",
                role="subscriber",
                description=description,
                stream=contract.name,
                source_id_selector=source_id,
                candidates=rejected,
            ).warning("Stream candidates rejected")
            last_rejection_signature = rejection_signature

        if not waiting_logged:
            log.bind(
                component="lsl",
                event="stream_waiting",
                role="subscriber",
                description=description,
                stream=contract.name,
                source_id_selector=source_id,
                discovery_property="name",
            ).info("Stream waiting")
            waiting_logged = True
        # A wrong same-name outlet makes resolve_byprop return immediately.
        # Back off so contract rejection cannot turn into a CPU-tight loop.
        if stop_event.wait(min(max(resolve_timeout_s, 0.1), 0.25)):
            break
    return None
