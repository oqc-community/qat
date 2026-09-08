# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Builders producing canonical system data carrying typed Qblox configuration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from frozendict import frozendict

from qat.experimental.system_data.canonical.schema import (
    AttributeEntry,
    CanonicalSystemData,
    ChannelData,
    ExternalResourceData,
    OscillatorData,
    PortData,
)
from qat.experimental.system_data.hardware.qblox.configuration import (
    QBLOX_CONFIGURATION_ATTRIBUTE,
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
)

ACQUISITION_KINDS = frozenset(
    {QbloxModuleKind.qrm, QbloxModuleKind.qrm_rf, QbloxModuleKind.qrc}
)


def sequencer(
    index: int,
    outputs: Sequence[int] = (0,),
    inputs: Sequence[int] = (),
    values: Mapping[str, object] | None = None,
) -> QbloxSequencerConfiguration:
    """Build a supplied sequencer bound to the given physical lanes."""

    connections = []
    if outputs:
        connections.append(
            PortConnection(direction=DirectionKind.output, port_ids=tuple(outputs))
        )
    connections.extend(
        PortConnection(direction=DirectionKind.input, port_ids=(input_id,))
        for input_id in inputs
    )
    return QbloxSequencerConfiguration(
        index=index,
        connection=SequencerConnection(connections=tuple(connections)),
        values=frozendict(values or {}),
    )


def supplied(
    sequencers: Iterable[QbloxSequencerConfiguration] = (),
    module_values: Mapping[str, object] | None = None,
) -> QbloxSuppliedConfiguration:
    """Build the configuration one canonical port supplies for its module."""

    return QbloxSuppliedConfiguration(
        module_configuration=(
            QbloxModuleConfiguration(values=frozendict(module_values))
            if module_values is not None
            else None
        ),
        sequencers=tuple(sequencers),
    )


def canonical_data(
    kind: QbloxModuleKind = QbloxModuleKind.qcm_rf,
    configurations: Sequence[QbloxSuppliedConfiguration | None] = (),
    channels_per_port: int = 1,
    slot: int = 2,
    instrument_id: str = "cluster",
    carrier_frequency: int = 4_200_000_000,
    oscillator_frequency: int | None = 4_000_000_000,
) -> CanonicalSystemData:
    """Build canonical system data for one module exposed by several canonical ports.

    :param kind: Installed module kind of the single module.
    :param configurations: Configuration each canonical port supplies, one entry per port.
    :param channels_per_port: Calibrated channels routed through each port.
    :param slot: Physical Cluster slot the module occupies.
    :param instrument_id: Physical Cluster containing the module.
    :param carrier_frequency: Carrier frequency of the first channel of each port, in Hz.
    :param oscillator_frequency: Local oscillator frequency in Hz, or ``None`` for a port
        with no oscillator.
    :returns: Canonical system data carrying the typed Qblox extensions.
    """

    resources: list[ExternalResourceData] = []
    ports: list[PortData] = []
    oscillators: list[OscillatorData] = []
    channels: list[ChannelData] = []
    for index, configuration in enumerate(configurations):
        port_id = f"port-{index}"
        oscillator_id = f"lo-{index}" if oscillator_frequency is not None else None
        attributes = [
            AttributeEntry(
                key="qblox",
                value=PortReference(
                    kind=kind,
                    module_location=QbloxModuleLocation(instrument_id, slot),
                    oscillator_id=oscillator_id,
                ),
            )
        ]
        if configuration is not None:
            attributes.append(
                AttributeEntry(key=QBLOX_CONFIGURATION_ATTRIBUTE, value=configuration)
            )
        resources.append(
            ExternalResourceData(
                id=f"{port_id}-resource",
                object_type="port",
                attributes=tuple(attributes),
            )
        )
        ports.append(
            PortData(
                id=port_id,
                sample_time=1000,
                acquire_allowed=kind in ACQUISITION_KINDS,
                external_resource_id=f"{port_id}-resource",
            )
        )
        if oscillator_id is not None:
            resources.append(
                ExternalResourceData(
                    id=f"{oscillator_id}-resource", object_type="oscillator"
                )
            )
            oscillators.append(
                OscillatorData(
                    id=oscillator_id,
                    frequency=oscillator_frequency,
                    external_resource_id=f"{oscillator_id}-resource",
                )
            )
        channels.extend(
            ChannelData(
                id=f"{port_id}-channel-{channel_index}",
                port_id=port_id,
                frequency=carrier_frequency + channel_index * 100_000_000,
                oscillator_reference=oscillator_id,
                imbalance=0.9,
                phase_offset=0.0,
            )
            for channel_index in range(channels_per_port)
        )
    return CanonicalSystemData(
        acquire_limit=100,
        external_resources=tuple(resources),
        ports=tuple(ports),
        oscillators=tuple(oscillators),
        channels=tuple(channels),
    )
