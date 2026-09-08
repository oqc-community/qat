# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Typed Qblox configuration extension carried by canonical system data.

A source materialiser decodes the configuration its calibration supplies for one physical
channel and attaches the result to that channel's canonical external resource under
:data:`QBLOX_CONFIGURATION_ATTRIBUTE`. The records here are the stable boundary between
that source-specific decoding and the compiler layers that resolve Q1 configuration: they
are immutable, sparse, and free of source syntax such as jsonpickle markers.

The extension deliberately stays port-scoped. Two canonical ports may share one physical
Qblox module - a QRC exposes its control and readout ports that way - yet each port still
supplies its own sequencer bank. Grouping the fragments of one module is a compilation
concern, not a system-data one.

Values absent from the source stay absent here. A consumer therefore cannot confuse an
unset value with one the calibration deliberately configured.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from frozendict import frozendict

from qat.experimental.system_data.canonical.schema import (
    CanonicalSystemData,
    ExternalResourceData,
)
from qat.experimental.system_data.hardware.qblox.models import (
    DirectionKind,
    SignalPath,
    connection_input_ids,
    connection_output_ids,
)

QBLOX_CONFIGURATION_ATTRIBUTE = "qblox_config"
"""Canonical external-resource attribute key holding a
:class:`QbloxSuppliedConfiguration`."""

ConfigScalar: TypeAlias = str | int | float | bool
ConfigValue: TypeAlias = (
    ConfigScalar | tuple["ConfigValue", ...] | frozendict[str, "ConfigValue"]
)


def immutable_config_value(value: Any, path: str = "config") -> ConfigValue:
    """Return a recursively immutable copy of a decoded configuration value.

    :param value: Decoded configuration value using plain Python containers and scalars.
    :param path: Source path of the value, used to describe validation failures.
    :returns: An equal value built from immutable containers.
    :raises ValueError: If the value cannot be represented by the typed extension.
    """

    if isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, Mapping):
        entries: dict[str, ConfigValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} has a non-string mapping key {key!r}")
            entries[key] = immutable_config_value(item, f"{path}.{key}")
        return frozendict(entries)
    if isinstance(value, Sequence):
        return tuple(
            immutable_config_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValueError(f"{path} holds an unsupported {type(value).__name__} value")


def _immutable_config_mapping(values: Any, path: str) -> frozendict[str, ConfigValue]:
    """Return an immutable mapping of configuration values.

    :param values: Decoded mapping of configuration values.
    :param path: Source path of the mapping, used to describe validation failures.
    :returns: The frozen mapping.
    :raises ValueError: If the value is not a mapping of supported values.
    """

    if not isinstance(values, Mapping):
        raise ValueError(f"{path} must be a mapping of configuration values")
    return immutable_config_value(values, path)


@dataclass(frozen=True, slots=True)
class PortConnection:
    """One connection string accepted by the Qblox sequencer connection API.

    A connection names one I/O port to put the sequencer in real mode, or an I and a Q
    port to put it in complex mode. Complex mode is available in every direction, so
    ``in0_1`` is as valid as ``out0_1`` and ``io0_1``. Which of those a particular module
    accepts is a target concern, not a syntactic one. See
    ``qblox_instruments.qcodes_drivers.sequencer.Sequencer.validate_connections``.

    :ivar direction: Whether the connection carries output, input, or bidirectional data.
    :ivar port_ids: Ordered I/O ports bound to the sequencer paths.
    """

    direction: DirectionKind
    port_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "port_ids", tuple(self.port_ids))
        if not self.port_ids:
            raise ValueError("A Qblox connection requires at least one I/O port")
        if any(
            isinstance(port_id, bool) or not isinstance(port_id, int) or port_id < 0
            for port_id in self.port_ids
        ):
            raise ValueError("Qblox connection I/O ports must be non-negative integers")
        if len(set(self.port_ids)) != len(self.port_ids):
            raise ValueError("Qblox connection I/O ports must be distinct")
        if len(self.port_ids) > 2:
            raise ValueError("A Qblox connection binds at most two I/O ports")

    @property
    def connection(self) -> str:
        """Return the string accepted by ``connect_sequencer``."""

        return self.direction.value + "_".join(str(port_id) for port_id in self.port_ids)


@dataclass(frozen=True, slots=True)
class OutputPathConnection:
    """One physical output bound to a sequencer output path.

    :ivar output_id: Physical output configured by the source ``outN`` field.
    :ivar path: Sequencer path driving the output.
    """

    output_id: int
    path: SignalPath

    def __post_init__(self) -> None:
        if (
            isinstance(self.output_id, bool)
            or not isinstance(self.output_id, int)
            or self.output_id < 0
        ):
            raise ValueError("A Qblox output path connection needs a physical output")


@dataclass(frozen=True, slots=True)
class AcquisitionPathConnection:
    """One physical input bound to a sequencer acquisition path.

    :ivar input_id: Physical input selected by the source ``acq`` field.
    :ivar path: Acquisition path receiving the input.
    """

    input_id: int
    path: SignalPath

    def __post_init__(self) -> None:
        if (
            isinstance(self.input_id, bool)
            or not isinstance(self.input_id, int)
            or self.input_id < 0
        ):
            raise ValueError("A Qblox acquisition path connection needs a physical input")


@dataclass(frozen=True, slots=True)
class SequencerConnection:
    """Routing the source supplies for one physical sequencer.

    :ivar connections: Ordered entries of the source ``bulk_value`` list.
    :ivar output_path_connections: Output path selected for each physical output.
    :ivar acquisition_path_connections: Physical input selected for each acquisition path.
    :ivar acquisition_enabled: Explicit acquisition enable supplied as a boolean.
    :ivar disabled_outputs: Physical outputs the source configures as ``off``.
    :ivar disabled_acquisition_paths: Acquisition paths the source configures as ``off``.
    :ivar acquisition_disabled: Whether the combined acquisition path is ``off``.
    """

    connections: tuple[PortConnection, ...] = ()
    output_path_connections: tuple[OutputPathConnection, ...] = ()
    acquisition_path_connections: tuple[AcquisitionPathConnection, ...] = ()
    acquisition_enabled: bool | None = None
    disabled_outputs: frozenset[int] = frozenset()
    disabled_acquisition_paths: frozenset[SignalPath] = frozenset()
    acquisition_disabled: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "connections", tuple(self.connections))
        object.__setattr__(
            self, "output_path_connections", tuple(self.output_path_connections)
        )
        object.__setattr__(
            self,
            "acquisition_path_connections",
            tuple(self.acquisition_path_connections),
        )
        object.__setattr__(self, "disabled_outputs", frozenset(self.disabled_outputs))
        object.__setattr__(
            self,
            "disabled_acquisition_paths",
            frozenset(self.disabled_acquisition_paths),
        )
        output_ids = [connection.output_id for connection in self.output_path_connections]
        if len(set(output_ids)) != len(output_ids):
            raise ValueError("A Qblox output may select only one sequencer path")
        paths = [connection.path for connection in self.acquisition_path_connections]
        if len(set(paths)) != len(paths):
            raise ValueError("A Qblox acquisition path may select only one input")

    @property
    def output_ids(self) -> frozenset[int]:
        """Return every physical output the connection binds or disables."""

        bound = {
            output_id
            for connection in self.connections
            for output_id in connection_output_ids(
                connection.direction, connection.port_ids
            )
        }
        bound.update(connection.output_id for connection in self.output_path_connections)
        return frozenset(bound | self.disabled_outputs)

    @property
    def input_ids(self) -> frozenset[int]:
        """Return every physical input the connection binds."""

        bound = {
            input_id
            for connection in self.connections
            for input_id in connection_input_ids(connection.direction, connection.port_ids)
        }
        bound.update(
            connection.input_id for connection in self.acquisition_path_connections
        )
        return frozenset(bound)

    @property
    def uses_acquisition(self) -> bool:
        """Return whether the connection places the sequencer on an acquisition path.

        Acquisition is implied by binding a physical input, but also by any explicit
        acquisition routing state the source supplied: enabling or disabling acquisition
        outright, and selecting or disabling individual acquisition signal paths. A module
        without acquisition support cannot honour any of those, so they must all be treated
        as acquisition use rather than silently ignored.
        """

        return bool(
            self.input_ids
            or self.acquisition_enabled is not None
            or self.acquisition_disabled is not None
            or self.disabled_acquisition_paths
        )


@dataclass(frozen=True, slots=True)
class QbloxModuleConfiguration:
    """Sparse module configuration supplied through one canonical port.

    :ivar values: Decoded module values keyed by their source field names.
    """

    values: frozendict[str, ConfigValue] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _immutable_config_mapping(self.values, "module"))


@dataclass(frozen=True, slots=True)
class QbloxSequencerConfiguration:
    """Sparse configuration supplied for one physical sequencer.

    :ivar index: Physical sequencer index within its module.
    :ivar connection: Routing supplied for the sequencer, when present.
    :ivar values: Remaining decoded sequencer values keyed by source field name.
    """

    index: int
    connection: SequencerConnection | None = None
    values: frozendict[str, ConfigValue] = field(default_factory=frozendict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
        ):
            raise ValueError("A Qblox sequencer index must be a non-negative integer")
        object.__setattr__(
            self,
            "values",
            _immutable_config_mapping(self.values, f"sequencers[{self.index}]"),
        )


@dataclass(frozen=True, slots=True)
class QbloxSuppliedConfiguration:
    """Configuration one canonical port supplies for its physical module.

    A source may describe the same installed configuration once and reference it from the
    other ports sharing the module. Such a port carries ``is_reference`` instead of a copy,
    leaving the compiler to resolve which fragment it refers to.

    :ivar module_configuration: Module-wide configuration supplied through this port, when
        present.
    :ivar sequencers: Sequencer bank supplied for this port, ordered by index.
    :ivar is_reference: Whether the source referenced another port's configuration.
    """

    module_configuration: QbloxModuleConfiguration | None = None
    sequencers: tuple[QbloxSequencerConfiguration, ...] = ()
    is_reference: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequencers", tuple(self.sequencers))
        indices = [sequencer.index for sequencer in self.sequencers]
        if len(set(indices)) != len(indices):
            raise ValueError("A Qblox sequencer bank must not repeat a sequencer index")
        if indices != sorted(indices):
            raise ValueError("A Qblox sequencer bank must be ordered by sequencer index")
        if self.is_reference and (self.module_configuration is not None or self.sequencers):
            raise ValueError("A Qblox configuration reference carries no values")


def supplied_configurations(
    canonical_data: CanonicalSystemData,
) -> frozendict[str, QbloxSuppliedConfiguration]:
    """Read the typed Qblox configuration extension from canonical system data.

    :param canonical_data: Canonical system data produced by a validating materialiser.
    :returns: Supplied configurations keyed by external-resource identifier.
    :raises ValueError: If a resource carries a duplicate or malformed extension.
    """

    return frozendict(
        (resource.id, configuration)
        for resource in canonical_data.external_resources
        if (configuration := _supplied_configuration(resource)) is not None
    )


def _supplied_configuration(
    resource: ExternalResourceData,
) -> QbloxSuppliedConfiguration | None:
    """Read the unique typed Qblox configuration attached to one external resource.

    :param resource: Canonical external resource joined to a port.
    :returns: The supplied configuration, or ``None`` when the resource carries none.
    :raises ValueError: If the extension is duplicated or is not a typed configuration.
    """

    matching_values = [
        attribute.value
        for attribute in resource.attributes
        if attribute.key == QBLOX_CONFIGURATION_ATTRIBUTE
    ]
    if not matching_values:
        return None
    if len(matching_values) > 1:
        raise ValueError(
            f"External resource {resource.id!r} has duplicate "
            f"{QBLOX_CONFIGURATION_ATTRIBUTE!r} attributes"
        )
    if not isinstance(matching_values[0], QbloxSuppliedConfiguration):
        raise ValueError(
            f"External resource {resource.id!r} has an invalid materialised Qblox "
            "configuration"
        )
    return matching_values[0]
