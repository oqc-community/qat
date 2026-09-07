# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Read-only Qblox projection of canonical installed hardware.

PuRR signal-path materialisation attaches a typed :class:`PortReference` to each Qblox
port resource while preserving the source payload. This module consumes that identity
as authoritative; it does not infer hardware from names or interpret configuration.
The projection groups canonical ports, local oscillators, and calibrated channels by
physical module for later compiler layers.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from frozendict import frozendict

from qat.experimental.system_data.canonical.schema import (
    CanonicalSystemData,
    ChannelData,
    ExternalResourceData,
    OscillatorData,
    PortData,
)
from qat.experimental.system_data.derived.interface import DerivedViewInterface
from qat.experimental.system_data.hardware.qblox.models import (
    PortReference,
    QbloxAddress,
    QbloxChannelBinding,
    QbloxModuleView,
    QbloxOscillatorBinding,
    QbloxPortBinding,
)


@dataclass(frozen=True, slots=True)
class _PortProjection:
    """Canonical port joined to its external resource and typed physical identity."""

    port: PortData
    resource: ExternalResourceData
    reference: PortReference


@dataclass(frozen=True, init=False)
class QbloxHardwareView(DerivedViewInterface[CanonicalSystemData]):
    """Installed modules and canonical bindings used by Q1 configuration resolution.

    The view owns immutable records and does not retain its canonical parent. Its module-
    oriented shape is the stable input expected by later allocation and Q1 configuration
    layers.

    :param acquire_limit: Canonical runtime acquisition limit.
    :param modules: Installed modules indexed by physical address.
    """

    acquire_limit: int
    modules: frozendict[QbloxAddress, QbloxModuleView] = field(default_factory=frozendict)

    def __init__(
        self,
        acquire_limit: int,
        modules: Mapping[QbloxAddress, QbloxModuleView] = frozendict(),
    ) -> None:
        """Create an immutable view from already-projected module records."""

        object.__setattr__(self, "acquire_limit", acquire_limit)
        object.__setattr__(self, "modules", frozendict(modules))

    @cached_property
    def port_bindings(self) -> Mapping[str, QbloxPortBinding]:
        """Return projected ports indexed by canonical identifier."""

        return frozendict(
            (binding.port_id, binding)
            for module in self.modules.values()
            for binding in module.ports
        )

    @cached_property
    def oscillator_bindings(self) -> Mapping[str, QbloxOscillatorBinding]:
        """Return projected local oscillators indexed by canonical identifier."""

        return frozendict(
            (binding.oscillator_id, binding)
            for module in self.modules.values()
            for binding in module.oscillators
        )

    @cached_property
    def channel_bindings(self) -> Mapping[str, QbloxChannelBinding]:
        """Return projected calibrated channels indexed by canonical identifier."""

        return frozendict(
            (binding.channel_id, binding)
            for module in self.modules.values()
            for binding in module.channel_bindings
        )

    @classmethod
    def derive(cls, parent: CanonicalSystemData, **kwargs: Any) -> QbloxHardwareView:
        """Project typed Qblox extensions from validated canonical system data.

        :param parent: Canonical system data produced by a validating materialiser.
        :param kwargs: Unused derived-view compatibility arguments.
        :returns: An immutable module-oriented projection.
        """

        del kwargs
        return _derive_hardware_view(parent)


def _derive_hardware_view(parent: CanonicalSystemData) -> QbloxHardwareView:
    """Group typed canonical bindings into physical modules.

    :param parent: Validated canonical system data.
    :returns: A detached immutable hardware view.
    """

    resources = {resource.id: resource for resource in parent.external_resources}
    oscillators = {oscillator.id: oscillator for oscillator in parent.oscillators}
    projections: list[_PortProjection] = []
    for port in parent.ports:
        if port.external_resource_id is None:
            continue
        resource = resources.get(port.external_resource_id)
        if resource is None:
            continue
        reference = _port_reference(resource)
        if reference is not None:
            projections.append(
                _PortProjection(
                    port=port,
                    resource=resource,
                    reference=reference,
                )
            )
    projection_by_port = {projection.port.id: projection for projection in projections}

    oscillator_modules: dict[str, QbloxAddress] = {}
    for projection in projections:
        if oscillator_id := projection.reference.oscillator_id:
            if oscillator_id not in oscillators:
                raise ValueError(
                    f"Qblox port references missing oscillator {oscillator_id!r}"
                )
            _register_oscillator_module(
                oscillator_modules,
                oscillator_id,
                projection.reference.module_address,
            )

    channels_by_module: dict[QbloxAddress, list[QbloxChannelBinding]] = defaultdict(list)
    for channel in parent.channels:
        projection = projection_by_port.get(channel.port_id)
        if projection is None:
            continue
        oscillator = None
        if channel.oscillator_reference is not None:
            oscillator = oscillators.get(channel.oscillator_reference)
            if oscillator is None:
                raise ValueError(
                    "Qblox channel references missing oscillator "
                    f"{channel.oscillator_reference!r}"
                )
        address = projection.reference.module_address
        if oscillator is not None:
            _register_oscillator_module(
                oscillator_modules,
                oscillator.id,
                address,
            )
        channels_by_module[address].append(
            _channel_binding(
                channel,
                projection,
                oscillator,
                resources,
            )
        )

    ports_by_module: dict[QbloxAddress, list[_PortProjection]] = defaultdict(list)
    for projection in projections:
        ports_by_module[projection.reference.module_address].append(projection)

    modules = frozendict(
        (
            address,
            _build_module(
                address,
                module_ports,
                oscillators,
                resources,
                channels_by_module[address],
            ),
        )
        for address, module_ports in ports_by_module.items()
    )
    return QbloxHardwareView(parent.acquire_limit, modules)


def _port_reference(resource: ExternalResourceData) -> PortReference | None:
    """Read the unique typed Qblox extension from an external resource.

    :param resource: Canonical external resource joined to a port.
    :returns: Its Qblox physical identity, or ``None`` for another target.
    :raises ValueError: If the explicit ``qblox`` extension is duplicate or malformed.
    """

    values = [
        attribute.value for attribute in resource.attributes if attribute.key == "qblox"
    ]
    if not values:
        return None
    if len(values) > 1:
        raise ValueError(
            f"External resource {resource.id!r} has duplicate 'qblox' attributes"
        )
    if not isinstance(values[0], PortReference):
        raise ValueError(
            f"External resource {resource.id!r} has an invalid materialised Qblox reference"
        )
    return values[0]


def _register_oscillator_module(
    oscillator_modules: dict[str, QbloxAddress],
    oscillator_id: str,
    address: QbloxAddress,
) -> None:
    """Record module ownership and reject one oscillator spanning multiple modules.

    :param oscillator_modules: Accumulated oscillator-to-module ownership.
    :param oscillator_id: Canonical local-oscillator identifier.
    :param address: Module using the oscillator.
    :raises ValueError: If another module already owns the oscillator.
    """

    previous = oscillator_modules.setdefault(oscillator_id, address)
    if previous != address:
        raise ValueError(
            f"Oscillator {oscillator_id!r} is used across inconsistent modules"
        )


def _build_module(
    address: QbloxAddress,
    projections: list[_PortProjection],
    oscillators: Mapping[str, OscillatorData],
    resources: Mapping[str, ExternalResourceData],
    channel_bindings: list[QbloxChannelBinding],
) -> QbloxModuleView:
    """Build one module from ports sharing a physical address.

    :param address: Shared physical module address.
    :param projections: Canonical ports carrying that address.
    :param oscillators: Canonical oscillators indexed by identifier.
    :param resources: Canonical external resources indexed by identifier.
    :param channel_bindings: Calibrated channels routed through the module.
    :returns: An immutable module projection.
    :raises ValueError: If grouped ports disagree on module kind or oscillator binding.
    """

    projections = sorted(projections, key=lambda projection: projection.port.id)
    kinds = {projection.reference.kind for projection in projections}
    if len(kinds) != 1:
        raise ValueError(f"Ports at {address!r} have inconsistent module kinds")

    port_bindings: list[QbloxPortBinding] = []
    oscillator_ids: list[str] = []
    for projection in projections:
        joined = [
            binding.oscillator_id
            for binding in channel_bindings
            if binding.port_id == projection.port.id and binding.oscillator_id is not None
        ]
        reference_oscillator = projection.reference.oscillator_id
        if reference_oscillator is not None:
            if any(oscillator_id != reference_oscillator for oscillator_id in joined):
                raise ValueError(
                    f"Port {projection.port.id!r} local oscillator conflicts with channels"
                )
            joined.append(reference_oscillator)
        unique_joined = tuple(dict.fromkeys(joined))
        oscillator_ids.extend(unique_joined)
        port_bindings.append(
            QbloxPortBinding(
                port_id=projection.port.id,
                resource_id=projection.resource.id,
                sample_time=projection.port.sample_time,
                block_size=projection.port.block_size,
                min_blocks=projection.port.min_blocks,
                max_blocks=projection.port.max_blocks,
                acquire_allowed=projection.port.acquire_allowed,
                oscillator_ids=unique_joined,
            )
        )

    return QbloxModuleView(
        kind=kinds.pop(),
        address=address,
        ports=tuple(port_bindings),
        oscillators=tuple(
            _oscillator_binding(oscillators[oscillator_id], resources)
            for oscillator_id in dict.fromkeys(oscillator_ids)
        ),
        channel_bindings=tuple(channel_bindings),
    )


def _channel_binding(
    channel: ChannelData,
    projection: _PortProjection,
    oscillator: OscillatorData | None,
    resources: Mapping[str, ExternalResourceData],
) -> QbloxChannelBinding:
    """Copy one canonical channel into a detached module binding.

    :param channel: Canonical calibrated channel.
    :param projection: Port and module identity joined by the channel.
    :param oscillator: Referenced local oscillator, when present.
    :param resources: Canonical external resources indexed by identifier.
    :returns: A dependency-neutral channel binding.
    """

    oscillator_binding = (
        _oscillator_binding(oscillator, resources) if oscillator is not None else None
    )
    return QbloxChannelBinding(
        channel_id=channel.id,
        port_id=channel.port_id,
        port_resource_id=projection.resource.id,
        carrier_frequency=channel.frequency,
        oscillator_id=(
            oscillator_binding.oscillator_id if oscillator_binding is not None else None
        ),
        oscillator_frequency=(
            oscillator_binding.frequency if oscillator_binding is not None else None
        ),
        oscillator_resource_id=(
            oscillator_binding.resource_id if oscillator_binding is not None else None
        ),
        scale=channel.scale,
        imbalance=channel.imbalance,
        phase_offset=channel.phase_offset,
        module_address=projection.reference.module_address,
    )


def _oscillator_binding(
    oscillator: OscillatorData,
    resources: Mapping[str, ExternalResourceData],
) -> QbloxOscillatorBinding:
    """Copy a canonical oscillator and its external-resource identity.

    :param oscillator: Canonical local oscillator.
    :param resources: Canonical external resources indexed by identifier.
    :returns: A detached local-oscillator binding.
    :raises ValueError: If the oscillator has no valid external-resource join.
    """

    if oscillator.external_resource_id is None:
        raise ValueError(f"Oscillator {oscillator.id!r} has no external-resource join")
    resource = resources.get(oscillator.external_resource_id)
    if resource is None:
        raise ValueError(
            f"Oscillator {oscillator.id!r} references missing external resource "
            f"{oscillator.external_resource_id!r}"
        )
    return QbloxOscillatorBinding(
        oscillator_id=oscillator.id,
        resource_id=resource.id,
        frequency=oscillator.frequency,
    )
