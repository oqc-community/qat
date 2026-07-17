# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Coupling and CR/CRC validation rules for PuRR ingress payloads.

This module validates both coupling-list integrity and cross-resonance channel
consistency. In particular it covers the following responsibilities:

1. Coupling directions must reference known qubit indices.
2. Missing coupling quality values are surfaced as warnings.
3. Every CR channel mapping must have a corresponding CRC counterpart.
4. Every declared coupling edge must have both CR and CRC channel mappings.

Additional checks enforce consistency between CR/CRC key names, nested channel
identifiers, and optional auxiliary-target metadata.
"""

from __future__ import annotations

from typing import Any

from qat.experimental.system_data.materialisers.errors import SourceConsistencyError
from qat.experimental.system_data.materialisers.purr.ingress.v0_1_0 import PurrIngressV010
from qat.experimental.system_data.materialisers.purr.validators.common import (
    CRCRC_CHANNEL_ID_PATTERN,
    CRCRC_KEY_PATTERN,
    _collect_qubit_index_by_id,
    _iter_device_pulse_views,
    _iter_parsed_coupling_directions,
    _raise_validation_error,
)
from qat.experimental.utils.logging import get_logger

logger = get_logger(__name__)


def _validate_coupling_direction_entry(
    *,
    index: int,
    entry: Any,
    qubit_indices: set[int],
) -> None:
    """Validate one qubit-direction coupling record against known qubit indices."""

    if not isinstance(entry, dict):
        _raise_validation_error(
            "Coupling entry must be a mapping.",
            path=f"$.qubit_direction_couplings[{index}]",
            details={"entry_type": type(entry).__name__},
        )

    direction = entry.get("direction")
    if (
        not isinstance(direction, list)
        or len(direction) != 2
        or not all(isinstance(value, int) for value in direction)
    ):
        _raise_validation_error(
            "Coupling direction must be a two-element integer list.",
            path=f"$.qubit_direction_couplings[{index}].direction",
            details={"value": direction},
        )

    missing = [value for value in direction if value not in qubit_indices]
    if missing:
        raise SourceConsistencyError(
            "Coupling direction references unknown qubit indices.",
            source_type="purr",
            path=f"$.qubit_direction_couplings[{index}].direction",
            details={"missing_indices": tuple(missing)},
        )


def _validate_coupling_indices(
    qubit_direction_couplings: list[Any],
    qubit_indices: set[int],
) -> None:
    """Validate coupling directions against the set of declared qubit indices."""

    for index, entry in enumerate(qubit_direction_couplings):
        _validate_coupling_direction_entry(
            index=index,
            entry=entry,
            qubit_indices=qubit_indices,
        )


def _iter_cr_crc_pulse_views(dto: PurrIngressV010):
    """Yield pulse views that correspond to CR/CRC pulse-channel keys."""

    for device_id, _device_payload, pulse_key, pulse_view in _iter_device_pulse_views(dto):
        if not isinstance(pulse_key, str) or "cross_resonance" not in pulse_key:
            continue
        path_prefix = f"$.quantum_devices.{device_id}.pulse_channels.{pulse_key}"
        yield device_id, pulse_key, pulse_view, path_prefix


def _parse_cr_crc_key(pulse_key: str) -> tuple[str, str] | None:
    """Parse a CR/CRC key into ``(target, suffix)`` or return ``None``."""

    key_match = CRCRC_KEY_PATTERN.fullmatch(pulse_key)
    if key_match is None:
        return None
    return key_match.groups()


def _parse_cr_crc_channel_id(pulse_channel_id: str) -> tuple[str, str, str] | None:
    """Parse a CR/CRC channel id into ``(source, target, suffix)`` or return ``None``."""

    id_match = CRCRC_CHANNEL_ID_PATTERN.fullmatch(pulse_channel_id)
    if id_match is None:
        return None
    return id_match.groups()


def _normalise_cr_crc_auxiliary_target(pulse_channel: dict[str, Any]) -> str | None:
    """Return a parseable auxiliary qubit id when one is explicitly present."""

    auxiliary_target = pulse_channel.get("auxiliary_qubit")
    if isinstance(auxiliary_target, str) and CRCRC_KEY_PATTERN.match(
        f"{auxiliary_target}.cross_resonance"
    ):
        return auxiliary_target
    return None


def _validate_cr_crc_channel_mapping_keys(dto: PurrIngressV010) -> None:
    """Validate CR/CRC keys against the nested pulse-channel identifiers.

    This enforces consistency between the per-device pulse-channel key and the nested pulse-
    channel identifier, without requiring fully typed nested CR/CRC ingress structures.
    """

    for device_id, pulse_key, pulse_view, path_prefix in _iter_cr_crc_pulse_views(dto):
        parsed_key = _parse_cr_crc_key(pulse_key)
        if parsed_key is None:
            _raise_validation_error(
                "CR/CRC pulse-channel key must match '<target>.cross_resonance(_cancellation)'.",
                path=path_prefix,
                details={"pulse_key": pulse_key},
            )

        pulse_channel = pulse_view.get("pulse_channel")
        if not isinstance(pulse_channel, dict):
            _raise_validation_error(
                "CR/CRC pulse-channel view must contain a pulse_channel mapping.",
                path=f"{path_prefix}.pulse_channel",
                details={"entry_type": type(pulse_channel).__name__},
            )

        pulse_channel_id = pulse_channel.get("id")
        if not isinstance(pulse_channel_id, str):
            _raise_validation_error(
                "CR/CRC pulse_channel.id must be a string.",
                path=f"{path_prefix}.pulse_channel.id",
                details={"value_type": type(pulse_channel_id).__name__},
            )

        parsed_channel_id = _parse_cr_crc_channel_id(pulse_channel_id)
        if parsed_channel_id is None:
            _raise_validation_error(
                "CR/CRC pulse_channel.id must match '<source>.<target>.cross_resonance(_cancellation)'.",
                path=f"{path_prefix}.pulse_channel.id",
                details={"pulse_channel_id": pulse_channel_id},
            )

        key_target, key_suffix = parsed_key
        id_source, id_target, id_suffix = parsed_channel_id

        if id_source != device_id:
            raise SourceConsistencyError(
                "CR/CRC pulse channel source in id does not match owning device id.",
                source_type="purr",
                path=f"{path_prefix}.pulse_channel.id",
                details={"device_id": device_id, "id_source": id_source},
            )

        if id_target != key_target:
            raise SourceConsistencyError(
                "CR/CRC target qubit in key does not match pulse_channel.id target.",
                source_type="purr",
                path=path_prefix,
                details={
                    "key_target": key_target,
                    "id_target": id_target,
                    "pulse_channel_id": pulse_channel_id,
                },
            )

        if id_suffix != key_suffix:
            raise SourceConsistencyError(
                "CR/CRC channel type in key does not match pulse_channel.id suffix.",
                source_type="purr",
                path=path_prefix,
                details={
                    "key_suffix": key_suffix,
                    "id_suffix": id_suffix,
                    "pulse_channel_id": pulse_channel_id,
                },
            )


def _warn_missing_coupling_quality(qubit_direction_couplings: list[Any]) -> None:
    """Warn when coupling entries omit the quality value used as ZX fidelity.

    The PuRR source provides coupling quality as a fidelity metric for the ZX gate on each
    qubit pair. Missing quality values will be defaulted to 0.0 during materialisation,
    which may indicate incomplete source payloads.
    """

    missing_quality = []
    for entry in qubit_direction_couplings:
        if not isinstance(entry, dict):
            continue

        quality = entry.get("quality")
        if quality is None:
            direction = entry.get("direction")
            if isinstance(direction, list) and len(direction) == 2:
                missing_quality.append(tuple(direction))

    if missing_quality:
        logger.warning(
            "Coupling entries are missing quality fidelity values. "
            "These will default to 0.0 during materialisation. "
            "Affected qubit pairs: %s",
            missing_quality,
        )


def _validate_cr_crc_counterparts(dto: PurrIngressV010) -> None:
    """Require both CR and CRC keys for every source-target pair.

    A source-target pair must provide both cross-resonance and cross-resonance-cancellation
    mappings. Missing either side indicates an incomplete control definition and is treated
    as a consistency error.
    """

    present_keys: set[tuple[str, str, str]] = set()

    for device_id, pulse_key, _pulse_view, _path_prefix in _iter_cr_crc_pulse_views(dto):
        parsed_key = _parse_cr_crc_key(pulse_key)
        if parsed_key is None:
            continue
        target, suffix = parsed_key
        present_keys.add((device_id, target, suffix))

    missing_counterparts: set[tuple[str, str, str]] = set()
    for source, target, suffix in present_keys:
        counterpart_suffix = (
            "cross_resonance_cancellation"
            if suffix == "cross_resonance"
            else "cross_resonance"
        )
        if (source, target, counterpart_suffix) not in present_keys:
            missing_counterparts.add((source, target, suffix))

    if missing_counterparts:
        raise SourceConsistencyError(
            "CR/CRC pulse-channel mappings are missing counterpart keys for some pairs.",
            source_type="purr",
            path="$.quantum_devices.*.pulse_channels",
            details={"missing_counterparts": tuple(sorted(missing_counterparts))},
        )


def _iter_normalized_cr_crc_entries(dto: PurrIngressV010):
    """Yield CR/CRC entries that have consistent key, id, and device mappings.

    This helper reuses key/id parsing so downstream validators can operate on source/target
    semantics without requiring nested typed DTOs.
    """

    for device_id, pulse_key, pulse_view, path_prefix in _iter_cr_crc_pulse_views(dto):
        parsed_key = _parse_cr_crc_key(pulse_key)
        if parsed_key is None:
            continue
        key_target, key_suffix = parsed_key

        pulse_channel = pulse_view.get("pulse_channel")
        if not isinstance(pulse_channel, dict):
            continue

        pulse_channel_id = pulse_channel.get("id")
        if not isinstance(pulse_channel_id, str):
            continue

        parsed_channel_id = _parse_cr_crc_channel_id(pulse_channel_id)
        if parsed_channel_id is None:
            continue

        id_source, id_target, id_suffix = parsed_channel_id
        if id_source != device_id or id_target != key_target or id_suffix != key_suffix:
            continue

        yield {
            "source_id": id_source,
            "target_id": id_target,
            "suffix": id_suffix,
            "pulse_channel_id": pulse_channel_id,
            "path_prefix": path_prefix,
            "auxiliary_target": _normalise_cr_crc_auxiliary_target(pulse_channel),
        }


def _validate_cr_crc_matches_coupling_graph(dto: PurrIngressV010) -> None:
    """Validate that each declared coupling edge has both CR and CRC channel mappings.

    Legacy payloads may contain additional CR/CRC channel mappings beyond the coupling list.
    This check therefore enforces the reliable direction: every declared coupling edge must
    have both CR and CRC mappings.
    """

    qubit_index_by_id = _collect_qubit_index_by_id(dto)
    coupling_edges = set(_iter_parsed_coupling_directions(dto.qubit_direction_couplings))

    present_channel_mappings: set[tuple[int, int, str]] = set()
    for entry in _iter_normalized_cr_crc_entries(dto):
        source_index = qubit_index_by_id.get(entry["source_id"])
        target_index = qubit_index_by_id.get(entry["target_id"])

        if not isinstance(source_index, int) or not isinstance(target_index, int):
            continue

        present_channel_mappings.add((source_index, target_index, entry["suffix"]))

    missing_edges: list[tuple[int, int, tuple[str, ...]]] = []
    for source_index, target_index in sorted(coupling_edges):
        missing_suffixes: list[str] = []
        if (source_index, target_index, "cross_resonance") not in present_channel_mappings:
            missing_suffixes.append("cross_resonance")
        if (
            source_index,
            target_index,
            "cross_resonance_cancellation",
        ) not in present_channel_mappings:
            missing_suffixes.append("cross_resonance_cancellation")
        if missing_suffixes:
            missing_edges.append((source_index, target_index, tuple(missing_suffixes)))

    if missing_edges:
        raise SourceConsistencyError(
            "Declared coupling edges are missing CR/CRC pulse-channel mappings.",
            source_type="purr",
            path="$.qubit_direction_couplings",
            details={"missing_edges": tuple(missing_edges)},
        )


def _validate_cr_crc_auxiliary_targets(dto: PurrIngressV010) -> None:
    """Validate explicit auxiliary-target metadata when it is present and parseable."""

    for entry in _iter_normalized_cr_crc_entries(dto):
        auxiliary_target = entry["auxiliary_target"]
        if auxiliary_target is None:
            continue
        if auxiliary_target != entry["target_id"]:
            raise SourceConsistencyError(
                "CR/CRC auxiliary target does not match pulse-channel target mapping.",
                source_type="purr",
                path=entry["path_prefix"],
                details={
                    "source_id": entry["source_id"],
                    "target_id": entry["target_id"],
                    "auxiliary_target": auxiliary_target,
                    "pulse_channel_id": entry["pulse_channel_id"],
                },
            )
