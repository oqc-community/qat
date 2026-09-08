# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Decode legacy PuRR Qblox payload fragments into typed canonical extensions.

PuRR serialises ``qat.purr.backends.qblox.config`` objects into the ``baseband`` payload of
each physical channel. This module owns that source syntax: it strips serialisation
markers, drops the ``None`` placeholders PuRR writes for unset fields, and validates the
remaining values into the typed extension defined by
:mod:`qat.experimental.system_data.hardware.qblox.configuration`.

Keeping the decoding here means the derived Qblox hardware view never parses source
payloads, and later compiler layers only ever see validated, immutable configuration.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from qat.experimental.system_data.hardware.qblox.configuration import (
    QBLOX_CONFIGURATION_ATTRIBUTE,
    AcquisitionPathConnection,
    ConfigValue,
    OutputPathConnection,
    PortConnection,
    QbloxModuleConfiguration,
    QbloxSequencerConfiguration,
    QbloxSuppliedConfiguration,
    SequencerConnection,
)
from qat.experimental.system_data.hardware.qblox.models import (
    DirectionKind,
    PortReference,
    QbloxModuleKind,
    QbloxModuleLocation,
    SignalPath,
    connection_output_ids,
)
from qat.experimental.system_data.hardware.qblox.target import DEFAULT_QBLOX_TARGET

QBLOX_PORT_REFERENCE_ATTRIBUTE = "qblox"
"""Canonical external-resource attribute key holding a :class:`PortReference`."""

_SERIALISATION_KEYS = frozenset({"py/object", "py/obj_ref_id", "py/id", "py/ref"})
_ADAPTER_REFERENCE_KEY = "_adapter_reference"
_CONFIGURATION_KEYS = frozenset({"module", "sequencers"})
_IGNORED_CONFIGURATION_KEYS = frozenset({"slot_idx"})
_OUTPUT_PATH_FIELD = re.compile(r"out(?P<output>\d+)\Z")
_OUTPUT_CONNECTION = re.compile(r"out(?P<first>\d+)(?:_(?P<second>\d+))?\Z")
_INPUT_CONNECTION = re.compile(r"in(?P<first>\d+)(?:_(?P<second>\d+))?\Z")
_IO_CONNECTION = re.compile(r"io(?P<first>\d+)(?:_(?P<second>\d+))?\Z")
_ACQUISITION_INPUT_FIELD = re.compile(r"in(?P<input>\d+)\Z")
_OFF = "off"
_ACQUISITION_PATH_FIELDS = (("acq_I", SignalPath.i), ("acq_Q", SignalPath.q))
_CONNECTION_FIELDS = frozenset({"bulk_value", "acq", "acq_I", "acq_Q"})


def _decode_value(value: Any, path: str) -> ConfigValue | None:
    """Decode one source value, dropping unset fields and serialisation markers.

    :param value: Decoded PuRR payload value.
    :param path: Source path of the value, used to describe validation failures.
    :returns: The sparse configuration value, or ``None`` when the source supplies none.
    :raises ValueError: If the value uses an unsupported type or an unresolved reference.
    """

    if value is None:
        return None
    if isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, Mapping):
        if _ADAPTER_REFERENCE_KEY in value:
            raise ValueError(f"{path} is an unresolved shared-object reference")
        entries: dict[str, ConfigValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} has a non-string field {key!r}")
            if key in _SERIALISATION_KEYS:
                continue
            decoded = _decode_value(item, f"{path}.{key}")
            if decoded is not None:
                entries[key] = decoded
        return entries or None
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        items = tuple(
            decoded
            for index, item in enumerate(value)
            if (decoded := _decode_value(item, f"{path}[{index}]")) is not None
        )
        return items or None
    raise ValueError(f"{path} holds an unsupported {type(value).__name__} value")


def _decode_mapping(value: Any, path: str) -> dict[str, ConfigValue]:
    """Decode a source mapping into sparse configuration values.

    :param value: Decoded PuRR payload mapping.
    :param path: Source path of the mapping, used to describe validation failures.
    :returns: The sparse decoded values, empty when the source supplies none.
    :raises ValueError: If the value is not a mapping of supported values.
    """

    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return _decode_value(value, path) or {}


def _is_reference_stub(value: Mapping[str, Any]) -> bool:
    """Return whether a payload only refers to configuration supplied elsewhere.

    PuRR serialises a shared configuration once and replaces every later occurrence with a
    marker, because the same ``QbloxConfig`` object is reachable from several physical
    channels.
    """

    return _ADAPTER_REFERENCE_KEY in value


def _port_connection(connection: str, path: str) -> PortConnection:
    """Decode one entry of a source ``bulk_value`` list.

    Every direction accepts one I/O port for real mode or two for complex mode, so
    ``in0_1`` is decoded exactly like ``out0_1`` and ``io0_1``. Whether the installed
    module accepts the decoded routing is settled by target-aware validation.

    :param connection: Source connection string such as ``out0_1``, ``in0``, or ``io0``.
    :param path: Source path of the entry, used to describe validation failures.
    :returns: The typed connection.
    :raises ValueError: If the entry is not a supported connection string.
    """

    for pattern, direction in (
        (_OUTPUT_CONNECTION, DirectionKind.output),
        (_INPUT_CONNECTION, DirectionKind.input),
        (_IO_CONNECTION, DirectionKind.io),
    ):
        match = pattern.fullmatch(connection)
        if match is not None:
            return PortConnection(
                direction=direction,
                port_ids=tuple(int(group) for group in match.groups() if group is not None),
            )
    raise ValueError(f"{path} is not a Qblox connection string, got {connection!r}")


def _acquisition_input(connection: str, path: str) -> int:
    """Decode a source acquisition field naming one physical input.

    The ``acq``, ``acq_I``, and ``acq_Q`` fields each select a single physical input, so
    they never carry the complex ``in<I>_<Q>`` form a ``bulk_value`` entry may use.

    :param connection: Source connection string such as ``in0``.
    :param path: Source path of the field, used to describe validation failures.
    :returns: The physical input index.
    :raises ValueError: If the field does not name a physical input.
    """

    match = _ACQUISITION_INPUT_FIELD.fullmatch(connection)
    if match is None:
        raise ValueError(f"{path} must name a physical input, got {connection!r}")
    return int(match["input"])


def _decode_bulk_value(value: Any, path: str) -> tuple[PortConnection, ...]:
    """Decode the source ``bulk_value`` list of connection strings.

    :param value: Decoded ``bulk_value`` payload.
    :param path: Source path of the list, used to describe validation failures.
    :returns: The ordered typed connections.
    :raises ValueError: If the payload is not a list of connection strings.
    """

    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a list of connection strings")
    connections = []
    for index, entry in enumerate(value):
        entry_path = f"{path}[{index}]"
        if not isinstance(entry, str):
            raise ValueError(f"{entry_path} must be a connection string")
        connections.append(_port_connection(entry, entry_path))
    return tuple(connections)


def _decode_output_paths(
    connection: Mapping[str, Any], path: str
) -> tuple[tuple[OutputPathConnection, ...], frozenset[int]]:
    """Decode the source ``outN`` fields selecting sequencer paths.

    :param connection: Decoded source connection payload.
    :param path: Source path of the payload, used to describe validation failures.
    :returns: The bound output paths and the outputs configured as ``off``.
    :raises ValueError: If a field names an unsupported output or path.
    """

    bound: list[OutputPathConnection] = []
    disabled: set[int] = set()
    for key, value in connection.items():
        match = _OUTPUT_PATH_FIELD.fullmatch(key)
        if match is None or value is None:
            continue
        field_path = f"{path}.{key}"
        if not isinstance(value, str):
            raise ValueError(f"{field_path} must be a sequencer path or {_OFF!r}")
        if value == _OFF:
            disabled.add(int(match["output"]))
            continue
        try:
            signal_path = SignalPath(value)
        except ValueError as error:
            raise ValueError(
                f"{field_path} must be a sequencer path or {_OFF!r}, got {value!r}"
            ) from error
        bound.append(OutputPathConnection(output_id=int(match["output"]), path=signal_path))
    return tuple(bound), frozenset(disabled)


def _decode_acquisition_paths(
    connection: Mapping[str, Any], path: str
) -> tuple[tuple[AcquisitionPathConnection, ...], frozenset[SignalPath]]:
    """Decode the source ``acq_I`` and ``acq_Q`` fields.

    :param connection: Decoded source connection payload.
    :param path: Source path of the payload, used to describe validation failures.
    :returns: The bound acquisition paths and the paths configured as ``off``.
    :raises ValueError: If a field does not name a physical input.
    """

    bound: list[AcquisitionPathConnection] = []
    disabled: set[SignalPath] = set()
    for key, signal_path in _ACQUISITION_PATH_FIELDS:
        value = connection.get(key)
        if value is None:
            continue
        field_path = f"{path}.{key}"
        if not isinstance(value, str):
            raise ValueError(f"{field_path} must name a physical input or be {_OFF!r}")
        if value == _OFF:
            disabled.add(signal_path)
            continue
        bound.append(
            AcquisitionPathConnection(
                input_id=_acquisition_input(value, field_path), path=signal_path
            )
        )
    return tuple(bound), frozenset(disabled)


def _decode_connection(value: Any, path: str) -> SequencerConnection | None:
    """Decode the routing PuRR supplies for one sequencer.

    :param value: Decoded source ``connection`` payload.
    :param path: Source path of the payload, used to describe validation failures.
    :returns: The typed routing, or ``None`` when the source supplies none.
    :raises ValueError: If the payload carries unsupported fields or values.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    if _ADAPTER_REFERENCE_KEY in value:
        raise ValueError(f"{path} is an unresolved shared-object reference")
    unsupported = sorted(
        key
        for key in value
        if key not in _SERIALISATION_KEYS
        and key not in _CONNECTION_FIELDS
        and _OUTPUT_PATH_FIELD.fullmatch(key) is None
    )
    if unsupported:
        raise ValueError(f"{path} has unsupported fields {unsupported!r}")

    connections = _decode_bulk_value(value.get("bulk_value"), f"{path}.bulk_value")
    output_paths, disabled_outputs = _decode_output_paths(value, path)
    acquisition_paths, disabled_acquisition_paths = _decode_acquisition_paths(value, path)

    acquisition = value.get("acq")
    acquisition_enabled = acquisition if isinstance(acquisition, bool) else None
    acquisition_disabled = None
    if isinstance(acquisition, str):
        if acquisition == _OFF:
            acquisition_disabled = True
        else:
            acquisition_paths += (
                AcquisitionPathConnection(
                    input_id=_acquisition_input(acquisition, f"{path}.acq"),
                    path=SignalPath.iq,
                ),
            )
    elif acquisition is not None and not isinstance(acquisition, bool):
        raise ValueError(f"{path}.acq must be a boolean or name a physical input")

    connection = SequencerConnection(
        connections=connections,
        output_path_connections=output_paths,
        acquisition_path_connections=acquisition_paths,
        acquisition_enabled=acquisition_enabled,
        disabled_outputs=disabled_outputs,
        disabled_acquisition_paths=disabled_acquisition_paths,
        acquisition_disabled=acquisition_disabled,
    )
    if connection == SequencerConnection():
        return None
    if acquisition_disabled and connection.input_ids:
        raise ValueError(f"{path} disables acquisition while binding physical inputs")
    bound_outputs = {
        output_id
        for entry in connections
        for output_id in connection_output_ids(entry.direction, entry.port_ids)
    } | {entry.output_id for entry in output_paths}
    if overlap := disabled_outputs & bound_outputs:
        raise ValueError(f"{path} both binds and disables outputs {sorted(overlap)}")
    return connection


def _decode_sequencer(index: int, value: Any, path: str) -> QbloxSequencerConfiguration:
    """Decode the configuration PuRR supplies for one sequencer.

    :param index: Physical sequencer index.
    :param value: Decoded source sequencer payload.
    :param path: Source path of the payload, used to describe validation failures.
    :returns: The typed sequencer configuration.
    :raises ValueError: If the payload is not a mapping of supported values.
    """

    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    values = {
        key: decoded
        for key, item in value.items()
        if key not in _SERIALISATION_KEYS
        and key != "connection"
        and (decoded := _decode_value(item, f"{path}.{key}")) is not None
    }
    return QbloxSequencerConfiguration(
        index=index,
        connection=_decode_connection(value.get("connection"), f"{path}.connection"),
        values=values,
    )


def _sequencer_index(key: Any, path: str) -> int:
    """Decode a source sequencer-bank key.

    :param key: Source mapping key naming a physical sequencer.
    :param path: Source path of the bank, used to describe validation failures.
    :returns: The physical sequencer index.
    :raises ValueError: If the key does not name a physical sequencer.
    """

    if isinstance(key, int) and not isinstance(key, bool) and key >= 0:
        return key
    if isinstance(key, str) and key.isdigit():
        return int(key)
    raise ValueError(f"{path} is keyed by sequencer index, got {key!r}")


def decode_qblox_configuration(
    value: Any, path: str = "baseband.config"
) -> QbloxSuppliedConfiguration | None:
    """Decode the ``baseband.config`` payload of one PuRR physical channel.

    :param value: Decoded source configuration payload.
    :param path: Source path of the payload, used to describe validation failures.
    :returns: The typed configuration, or ``None`` when the channel supplies none.
    :raises ValueError: If the payload carries unsupported fields or values.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    if _is_reference_stub(value):
        unsupported = sorted(
            set(value)
            - _SERIALISATION_KEYS
            - _IGNORED_CONFIGURATION_KEYS
            - {_ADAPTER_REFERENCE_KEY}
        )
        if unsupported:
            raise ValueError(
                f"{path} refers to another port but carries fields {unsupported!r}"
            )
        return QbloxSuppliedConfiguration(is_reference=True)

    unsupported = sorted(
        set(value) - _SERIALISATION_KEYS - _IGNORED_CONFIGURATION_KEYS - _CONFIGURATION_KEYS
    )
    if unsupported:
        raise ValueError(f"{path} has unsupported fields {unsupported!r}")

    module_values = value.get("module")
    module_configuration = (
        QbloxModuleConfiguration(values=_decode_mapping(module_values, f"{path}.module"))
        if module_values is not None
        else None
    )

    bank = value.get("sequencers")
    if bank is not None and not isinstance(bank, Mapping):
        raise ValueError(f"{path}.sequencers must be a mapping")
    sequencers: dict[int, QbloxSequencerConfiguration] = {}
    for key, sequencer_value in (bank or {}).items():
        if key in _SERIALISATION_KEYS:
            continue
        index = _sequencer_index(key, f"{path}.sequencers")
        if index in sequencers:
            raise ValueError(f"{path}.sequencers repeats sequencer {index}")
        sequencers[index] = _decode_sequencer(
            index, sequencer_value, f"{path}.sequencers[{index}]"
        )
    return QbloxSuppliedConfiguration(
        module_configuration=module_configuration,
        sequencers=tuple(sequencers[index] for index in sorted(sequencers)),
    )


def decode_qblox_port_reference(payload: Mapping[str, Any]) -> PortReference | None:
    """Read the physical identity of one PuRR physical channel.

    Physical channels of other targets return ``None`` so the generic PuRR materialiser
    keeps their source attributes untouched.

    :param payload: Decoded PuRR physical-channel payload.
    :returns: The typed port reference, or ``None`` for a non-Qblox channel.
    :raises ValueError: If a Qblox channel carries malformed physical identity.
    """

    port_id = payload.get("id")
    kind = QbloxModuleKind.from_qblox_identifier(port_id)
    if kind is None:
        return None

    baseband = payload.get("baseband")
    if not isinstance(baseband, Mapping):
        raise ValueError(f"Qblox physical channel {port_id!r} has no baseband payload")
    oscillator_id = baseband.get("id")
    if oscillator_id is not None and (
        not isinstance(oscillator_id, str) or not oscillator_id
    ):
        raise ValueError(f"Qblox physical channel {port_id!r} has an invalid baseband id")
    instrument_id = baseband.get("instrument_id")
    if not isinstance(instrument_id, str) or not instrument_id.strip():
        raise ValueError(f"Qblox physical channel {port_id!r} has no instrument identifier")
    slot = baseband.get("slot_idx")
    if isinstance(slot, bool) or not isinstance(slot, int):
        raise ValueError(f"Qblox physical channel {port_id!r} has no module slot")
    module_location = QbloxModuleLocation(instrument_id, slot)
    DEFAULT_QBLOX_TARGET.validate_module_location(module_location)
    return PortReference(
        kind=kind,
        module_location=module_location,
        oscillator_id=oscillator_id,
    )


def decode_qblox_port_attributes(
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Materialise the typed Qblox extensions of one PuRR physical channel.

    :param payload: Decoded PuRR physical-channel payload.
    :returns: The typed attributes to attach, or ``None`` for a non-Qblox channel.
    :raises ValueError: If the channel carries malformed Qblox metadata.
    """

    reference = decode_qblox_port_reference(payload)
    if reference is None:
        return None
    attributes: dict[str, Any] = {QBLOX_PORT_REFERENCE_ATTRIBUTE: reference}
    configuration = decode_qblox_configuration(payload["baseband"].get("config"))
    if configuration is not None:
        attributes[QBLOX_CONFIGURATION_ATTRIBUTE] = configuration
    return attributes
