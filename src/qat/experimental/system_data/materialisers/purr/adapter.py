# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Boundary adapter for PuRR source payloads.

The adapter sits between raw source payload loading and ingress DTO validation. It is
responsible for light structural checks, format detection, source decoding, and
projection into an acyclic ingress-friendly shape.

Adapter responsibilities
========================

This module normalises source payloads into a stable shape expected by ingress
DTO validation. It intentionally keeps this logic compiler-local so ingress DTO
models can remain strict while legacy source quirks are handled in one place.

Current normalisation includes:

1. Top-level key presence checks for required entity collections.
2. Defaults for optional top-level collections.
3. Legacy ``default_acquire_mode`` coercion into ``str | None``.
4. Projection of decoded source graphs into an acyclic structure suitable for
    deterministic DTO validation.
5. Compatibility rewrite for legacy CR/CRC ownership conventions.
"""

from typing import Any

from qat.experimental.system_data.materialisers.errors import SourceValidationError
from qat.experimental.system_data.materialisers.purr.decoder import (
    decode_jsonpickle_payload,
)

_REQUIRED_TOP_LEVEL_KEYS = (
    "quantum_devices",
    "pulse_channels",
    "physical_channels",
    "basebands",
)

_DEFAULT_TOP_LEVEL_VALUES: dict[str, Any] = {
    "qubit_direction_couplings": [],
}

_FORCE_EXPAND_TOP_LEVEL_COLLECTIONS = {
    "quantum_devices",
    "pulse_channels",
    "physical_channels",
    "basebands",
}

_REFERENCE_ID_KEYS = (
    "id",
    "instrument_id",
)


def _raise_adapter_error(message: str, *, details: dict[str, Any]) -> None:
    """Raise a structured source-validation error for adapter-stage failures."""

    raise SourceValidationError(
        message,
        source_type="purr",
        path="$",
        details=details,
    )


def _detect_source_profile(payload: dict[str, Any]) -> str:
    """Classify the incoming payload so the adapter can choose a decode strategy.

    Current support distinguishes raw jsonpickle-like PuRR payloads from already-plain
    mappings that may come from tests or future upstream boundary layers.
    """

    if any(key.startswith("py/") for key in payload):
        return "jsonpickle"
    return "plain"


def _normalise_top_level(decoded_payload: dict[str, Any]) -> dict[str, Any]:
    """Normalise top-level payload shape expected by the ingress DTO.

    This step fills a small set of defaults and enforces the presence of the top-level
    entity collections that the downstream materialiser requires.
    """

    normalised = dict(decoded_payload)

    for key, value in _DEFAULT_TOP_LEVEL_VALUES.items():
        normalised.setdefault(key, value)

    _normalise_default_acquire_mode(normalised)

    missing_keys = [key for key in _REQUIRED_TOP_LEVEL_KEYS if key not in normalised]
    if missing_keys:
        _raise_adapter_error(
            "Decoded payload missing required top-level keys.",
            details={
                "missing_keys": tuple(missing_keys),
                "present_keys": tuple(sorted(normalised.keys())),
            },
        )

    return normalised


def _normalise_default_acquire_mode(normalised_payload: dict[str, Any]) -> None:
    """Normalise legacy default-acquire-mode shapes to string-or-none.

    Some legacy payloads encode an unset default as an empty list, while newer payloads
    provide a mode string. This adapter coercion keeps ingress typing strict by producing a
    consistent string-or-none value.
    """

    value = normalised_payload.get("default_acquire_mode")
    if value is None or isinstance(value, str):
        return

    if isinstance(value, list):
        if not value:
            normalised_payload["default_acquire_mode"] = None
            return

        first = value[0]
        if isinstance(first, str):
            normalised_payload["default_acquire_mode"] = first
            return

    _raise_adapter_error(
        "Unsupported default_acquire_mode payload shape.",
        details={
            "default_acquire_mode_type": type(value).__name__,
            "default_acquire_mode": value,
        },
    )


def _normalise_legacy_crc_ownership(normalised_payload: dict[str, Any]) -> None:
    """Normalise CR/CRC ownership into target-owned CRC form.

    Legacy payloads may encode CRC channels on the control/source device under
    ``"{target}.cross_resonance_cancellation"`` with channel id
    ``"{source}.{target}.cross_resonance_cancellation"``.

    The current ingress contract expects CRC ownership on the target device
    under ``"{control}.cross_resonance_cancellation"`` while preserving the same
    channel id. This normaliser also synthesises missing CRC counterparts from CR
    mappings.

    It removes only CRC entries whose key/id pairing is parseable but matches
    neither the canonical target-owned form nor the legacy source-owned form.
    Other malformed payload shapes (for example missing/non-dict
    ``pulse_channel``, missing/non-string ``id``, or non-3-part ids) are left
    unchanged for downstream validators.
    """

    quantum_devices = normalised_payload.get("quantum_devices")
    if not isinstance(quantum_devices, dict):
        return

    # Each entry is (source_id, old_key, target_id, pulse_view); target_id=None means removal only.
    deferred: list[tuple[str, str, str | None, dict[str, Any] | None]] = []
    cr_pairs: set[tuple[str, str]] = set()

    def _ensure_target_entry(
        *,
        source_id: str,
        target_id: str,
        pulse_view: dict[str, Any],
    ) -> None:
        target_payload = quantum_devices.get(target_id)
        if not isinstance(target_payload, dict):
            target_payload = {"id": target_id, "pulse_channels": {}}
            quantum_devices[target_id] = target_payload

        target_channels = target_payload.get("pulse_channels")
        if not isinstance(target_channels, dict):
            target_channels = {}
            target_payload["pulse_channels"] = target_channels

        target_key = f"{source_id}.cross_resonance_cancellation"
        target_channels.setdefault(target_key, pulse_view)

    for source_id, source_payload in quantum_devices.items():
        if not isinstance(source_payload, dict):
            continue

        pulse_channels = source_payload.get("pulse_channels")
        if not isinstance(pulse_channels, dict):
            continue

        for old_key, pulse_view in pulse_channels.items():
            if not (isinstance(old_key, str) and isinstance(pulse_view, dict)):
                continue

            if old_key.endswith(".cross_resonance"):
                key_peer = old_key.split(".")[0]
                if key_peer != source_id:
                    cr_pairs.add((source_id, key_peer))
                continue

            if not old_key.endswith(".cross_resonance_cancellation"):
                continue

            pulse_channel = pulse_view.get("pulse_channel")
            if not isinstance(pulse_channel, dict):
                continue

            pulse_channel_id = pulse_channel.get("id")
            if not isinstance(pulse_channel_id, str):
                continue

            id_parts = pulse_channel_id.split(".")
            if len(id_parts) != 3:
                continue
            id_source, id_target, id_suffix = id_parts
            if id_suffix != "cross_resonance_cancellation":
                continue

            key_peer = old_key.split(".")[0]

            # Already in canonical target-owned form.
            if source_id == id_target and key_peer == id_source:
                continue

            # Legacy source-owned form to rewrite.
            if source_id == id_source and key_peer == id_target:
                deferred.append((source_id, old_key, id_target, pulse_view))
                continue

            # Malformed CRC mapping that matches neither canonical nor legacy forms.
            deferred.append((source_id, old_key, None, None))

    for source_id, old_key, target_id, pulse_view in deferred:
        if target_id is not None:
            _ensure_target_entry(
                source_id=source_id,
                target_id=target_id,
                pulse_view=pulse_view,
            )
        source_payload = quantum_devices.get(source_id)
        if not isinstance(source_payload, dict):
            continue
        source_channels = source_payload.get("pulse_channels")
        if isinstance(source_channels, dict):
            source_channels.pop(old_key, None)

    for source_id, target_id in sorted(cr_pairs):
        source_payload = quantum_devices.get(source_id)
        if not isinstance(source_payload, dict):
            continue
        source_channels = source_payload.get("pulse_channels")
        if not isinstance(source_channels, dict):
            continue

        cr_key = f"{target_id}.cross_resonance"
        cr_view = source_channels.get(cr_key)
        if not isinstance(cr_view, dict):
            continue
        cr_channel = cr_view.get("pulse_channel")
        if not isinstance(cr_channel, dict):
            continue

        crc_channel = dict(cr_channel)
        crc_channel["id"] = f"{source_id}.{target_id}.cross_resonance_cancellation"
        crc_view = {"pulse_channel": crc_channel}
        _ensure_target_entry(source_id=source_id, target_id=target_id, pulse_view=crc_view)


def _build_reference_stub(node: Any) -> Any:
    """Build a lightweight stand-in for a repeated container in the source graph.

    The goal is to preserve identifying context while breaking back-references that would
    otherwise recreate cyclic structures in the ingress payload.
    """

    if isinstance(node, dict):
        for key in _REFERENCE_ID_KEYS:
            value = node.get(key)
            if isinstance(value, str):
                return {key: value}
        return {"_adapter_reference": "mapping"}

    if isinstance(node, list):
        return []

    return node


def _resolve_next_root_key(
    *,
    path: tuple[str, ...],
    root_key: tuple[str, str] | None,
) -> tuple[str, str] | None:
    """Select the root-collection tracking scope for repeated node detection."""

    if len(path) == 2 and path[0] in _FORCE_EXPAND_TOP_LEVEL_COLLECTIONS:
        return (path[0], path[1])
    return root_key


def _get_active_seen_nodes(
    *,
    root_key: tuple[str, str] | None,
    seen_nodes: set[int],
    seen_nodes_by_root: dict[tuple[str, str], set[int]],
) -> set[int]:
    """Return the seen-node bucket scoped to the active root collection."""

    if root_key is None:
        return seen_nodes
    return seen_nodes_by_root.setdefault(root_key, set())


def _is_repeated_or_cyclic(
    *,
    node_id: int,
    stack: frozenset[int],
    active_seen: set[int],
) -> bool:
    """Check whether a container is cyclic in-stack or repeated in active scope."""

    if node_id in stack:
        return True
    if node_id in active_seen:
        return True
    return False


def _project_mapping_node(
    *,
    node: dict[str, Any],
    path: tuple[str, ...],
    root_key: tuple[str, str] | None,
    stack: frozenset[int],
    seen_nodes: set[int],
    seen_nodes_by_root: dict[tuple[str, str], set[int]],
) -> dict[str, Any]:
    """Project a mapping node while preserving only acyclic structure."""

    next_root_key = _resolve_next_root_key(path=path, root_key=root_key)
    active_seen = _get_active_seen_nodes(
        root_key=next_root_key,
        seen_nodes=seen_nodes,
        seen_nodes_by_root=seen_nodes_by_root,
    )

    node_id = id(node)
    if _is_repeated_or_cyclic(
        node_id=node_id,
        stack=stack,
        active_seen=active_seen,
    ):
        return _build_reference_stub(node)

    active_seen.add(node_id)
    return {
        key: _project_acyclic_payload(
            node=value,
            path=path + (key,),
            root_key=next_root_key,
            stack=stack | {node_id},
            seen_nodes=seen_nodes,
            seen_nodes_by_root=seen_nodes_by_root,
        )
        for key, value in node.items()
    }


def _project_sequence_node(
    *,
    node: list[Any],
    path: tuple[str, ...],
    root_key: tuple[str, str] | None,
    stack: frozenset[int],
    seen_nodes: set[int],
    seen_nodes_by_root: dict[tuple[str, str], set[int]],
) -> list[Any]:
    """Project a sequence node while preserving only acyclic structure."""

    active_seen = _get_active_seen_nodes(
        root_key=root_key,
        seen_nodes=seen_nodes,
        seen_nodes_by_root=seen_nodes_by_root,
    )

    node_id = id(node)
    if _is_repeated_or_cyclic(
        node_id=node_id,
        stack=stack,
        active_seen=active_seen,
    ):
        return _build_reference_stub(node)

    active_seen.add(node_id)
    return [
        _project_acyclic_payload(
            node=item,
            path=path + (str(idx),),
            root_key=root_key,
            stack=stack | {node_id},
            seen_nodes=seen_nodes,
            seen_nodes_by_root=seen_nodes_by_root,
        )
        for idx, item in enumerate(node)
    ]


def _project_acyclic_payload(
    *,
    node: Any,
    path: tuple[str, ...] = (),
    root_key: tuple[str, str] | None = None,
    stack: frozenset[int] = frozenset(),
    seen_nodes: set[int] | None = None,
    seen_nodes_by_root: dict[tuple[str, str], set[int]] | None = None,
) -> Any:
    """Project a decoded payload graph into an acyclic ingress-friendly structure.

    The PuRR decoder reconstructs shared references from jsonpickle ``py/id`` links,
    which recreates the original object graph and can introduce cycles. For ingress DTO
    validation and debugger inspection we only need an acyclic structural view, so any
    repeated container encountered after its first expansion is collapsed to a lightweight
    identifier stub.
    """

    if seen_nodes is None:
        seen_nodes = set()

    if seen_nodes_by_root is None:
        seen_nodes_by_root = {}

    if isinstance(node, dict):
        return _project_mapping_node(
            node=node,
            path=path,
            root_key=root_key,
            stack=stack,
            seen_nodes=seen_nodes,
            seen_nodes_by_root=seen_nodes_by_root,
        )

    if isinstance(node, list):
        return _project_sequence_node(
            node=node,
            path=path,
            root_key=root_key,
            stack=stack,
            seen_nodes=seen_nodes,
            seen_nodes_by_root=seen_nodes_by_root,
        )

    return node


def adapt_purr_payload(
    payload: dict[str, Any],
    *,
    extra_reduce_target_types: set[str] | None = None,
    extra_reduce_target_suffixes: set[str] | None = None,
) -> dict[str, Any]:
    """Adapt raw PuRR payload into decoder-normalised plain data.

    :param payload: Raw parsed JSON payload.
    :param extra_reduce_target_types: Optional extra fully-qualified
        ``py/reduce`` target types allowed by the source decoder at runtime.
    :param extra_reduce_target_suffixes: Optional extra terminal type-name
        suffixes allowed by the source decoder at runtime.
    :returns: Acyclic, boundary-normalised payload ready for ingress DTO validation. The
        returned mapping is the source-payload representation consumed by the ingress DTO.
        Compiler-owned enrichment happens after this stage.
    """

    if not isinstance(payload, dict):
        _raise_adapter_error(
            "PuRR payload root must be a dictionary.",
            details={"root_type": type(payload).__name__},
        )

    if not payload:
        _raise_adapter_error(
            "PuRR payload is empty.",
            details={"root_type": type(payload).__name__},
        )

    source_profile = _detect_source_profile(payload)
    source_hint = (
        payload.get("py/object") if isinstance(payload.get("py/object"), str) else None
    )

    if source_profile == "jsonpickle":
        decoded = decode_jsonpickle_payload(
            payload,
            extra_reduce_target_types=extra_reduce_target_types,
            extra_reduce_target_suffixes=extra_reduce_target_suffixes,
        )
    else:
        decoded = dict(payload)

    normalised = _normalise_top_level(decoded)
    _normalise_legacy_crc_ownership(normalised)
    acyclic_payload = _project_acyclic_payload(node=normalised)
    if source_hint is not None:
        acyclic_payload.setdefault("_adapter_source_hint", source_hint)
    return acyclic_payload
