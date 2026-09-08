# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Dependency-neutral values used by the Qblox hardware projection.

The records describe physical identity and canonical bindings only. They do not decode PuRR
payloads, supply Qblox defaults, allocate sequencers, or construct Q1 attributes. All
records are immutable so a derived view can be shared safely between compiler passes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class QbloxModuleKind(str, Enum):
    """Qblox module kinds represented by canonical system data."""

    qcm = "qcm"
    qcm_rf = "qcm_rf"
    qrm = "qrm"
    qrm_rf = "qrm_rf"
    qrc = "qrc"

    @classmethod
    def from_qblox_name(cls, value: str) -> QbloxModuleKind:
        """Parse an official Qblox module name.

        :param value: Module name using hyphens or underscores in any case.
        :returns: The parsed module kind.
        """

        if not isinstance(value, str):
            raise ValueError(
                f"Qblox module name must be a string, got {type(value).__name__}"
            )
        return cls(value.strip().lower().replace("-", "_"))

    @classmethod
    def from_qblox_identifier(cls, value: str | None) -> QbloxModuleKind | None:
        """Parse a module kind embedded in a canonical Qblox identifier.

        :param value: Canonical resource identifier.
        :returns: The longest matching kind, or ``None``.
        """

        if not isinstance(value, str) or not value:
            return None
        parts = tuple(value.lower().replace("-", "_").split("_"))
        return max(
            (
                kind
                for kind in cls
                if any(
                    parts[index : index + len(kind.value.split("_"))]
                    == tuple(kind.value.split("_"))
                    for index in range(len(parts))
                )
            ),
            key=lambda kind: len(kind.value),
            default=None,
        )


class SignalPath(str, Enum):
    """I, Q, or combined IQ signal path exposed by a Q1 sequencer."""

    i = "I"
    q = "Q"
    iq = "IQ"


class DirectionKind(str, Enum):
    """Direction prefix accepted by the public sequencer connection API."""

    output = "out"
    input = "in"
    io = "io"


def connection_output_ids(
    direction: DirectionKind, port_ids: Sequence[int]
) -> tuple[int, ...]:
    """Return the physical outputs a single sequencer connection drives.

    The Qblox driver decodes ``<direction><I>[_<Q>]`` by applying the direction to *every*
    listed lane: ``out`` binds the waveform generator, ``in`` binds the acquisition path,
    and ``io`` binds both. A single lane puts the sequencer in real mode and two lanes put
    it in complex mode, so ``io0_1`` drives outputs 0 and 1 *and* acquires on inputs 0 and
    1. See ``qblox_instruments.qcodes_drivers.sequencer.Sequencer.validate_connections``.

    :param direction: Direction prefix of the connection.
    :param port_ids: Ordered I/O ports the connection binds.
    :returns: The physical outputs the connection drives, in the supplied order.
    """

    if direction is DirectionKind.input:
        return ()
    return tuple(port_ids)


def connection_input_ids(
    direction: DirectionKind, port_ids: Sequence[int]
) -> tuple[int, ...]:
    """Return the physical inputs a single sequencer connection acquires from.

    Mirrors :func:`connection_output_ids`: ``in`` and ``io`` bind the acquisition path of
    every listed lane, while ``out`` binds none.

    :param direction: Direction prefix of the connection.
    :param port_ids: Ordered I/O ports the connection binds.
    :returns: The physical inputs the connection acquires from, in the supplied order.
    """

    if direction is DirectionKind.output:
        return ()
    return tuple(port_ids)


@dataclass(frozen=True, slots=True)
class QbloxModuleLocation:
    """Physical location of a module within a Qblox instrument.

    :param instrument_id: Canonical identifier of the containing Cluster.
    :param slot: One-based physical Cluster slot. Target-specific range validation is
        performed by :class:`QbloxTargetDescription`.
    """

    instrument_id: str
    slot: int

    def __post_init__(self) -> None:
        if not isinstance(self.instrument_id, str) or not self.instrument_id.strip():
            raise ValueError("Qblox instrument id must be non-empty")
        if isinstance(self.slot, bool) or not isinstance(self.slot, int) or self.slot < 1:
            raise ValueError("Qblox module slot must be a positive integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class PortReference:
    """Materialised physical identity attached to a canonical port.

    This target-specific extension gives the hardware view the stable identity needed to
    group canonical ports without interpreting legacy configuration payloads.

    :param kind: Installed module kind.
    :param module_location: Physical instrument and module slot.
    :param oscillator_id: Canonical oscillator referenced by the source port, if any.
    """

    kind: QbloxModuleKind
    module_location: QbloxModuleLocation
    oscillator_id: str | None = None

    def __post_init__(self) -> None:
        if self.oscillator_id is not None and not self.oscillator_id:
            raise ValueError("Qblox oscillator id must be non-empty when present")


@dataclass(frozen=True, slots=True, kw_only=True)
class QbloxPortBinding:
    """Canonical port projected onto one installed module.

    Timing and acquisition fields are copied from canonical system data so later compiler
    layers need not retain the canonical parent.

    :ivar port_id: Canonical port identifier.
    :ivar resource_id: External-resource identifier for the physical channel.
    :ivar sample_time: Sample period in picoseconds.
    :ivar block_size: Number of samples in one valid waveform block.
    :ivar min_blocks: Minimum waveform length in blocks.
    :ivar max_blocks: Maximum waveform length in blocks, or ``-1`` when unbounded.
    :ivar acquire_allowed: Whether the port can carry acquisition operations.
    :ivar oscillator_ids: Canonical local oscillators used by the port's channels.
    """

    port_id: str
    resource_id: str
    sample_time: int
    block_size: int
    min_blocks: int
    max_blocks: int
    acquire_allowed: bool
    oscillator_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class QbloxOscillatorBinding:
    """Canonical local oscillator used by one installed module.

    :ivar oscillator_id: Canonical oscillator identifier.
    :ivar resource_id: External-resource identifier for the physical oscillator.
    :ivar frequency: Oscillator frequency in Hz.
    """

    oscillator_id: str
    resource_id: str
    frequency: int


@dataclass(frozen=True, slots=True, kw_only=True)
class QbloxChannelBinding:
    """Calibrated canonical channel bound to a Qblox module.

    The record contains every value needed to resolve a channel after derivation, without
    retaining or rereading the canonical parent.

    :ivar channel_id: Canonical channel identifier.
    :ivar port_id: Canonical port identifier.
    :ivar port_resource_id: External resource joined to the port.
    :ivar carrier_frequency: Calibrated channel carrier frequency in Hz.
    :ivar oscillator_id: Referenced canonical oscillator identifier, when present.
    :ivar oscillator_frequency: Referenced oscillator frequency in Hz, when present.
    :ivar oscillator_resource_id: External resource joined to the oscillator, when present.
    :ivar scale: Calibrated complex channel scale.
    :ivar imbalance: Calibrated IQ gain imbalance.
    :ivar phase_offset: Calibrated IQ phase offset in radians.
    :ivar module_location: Physical module location resolved from the port.
    """

    channel_id: str
    port_id: str
    port_resource_id: str
    carrier_frequency: int
    oscillator_id: str | None
    oscillator_frequency: int | None
    oscillator_resource_id: str | None
    scale: complex
    imbalance: float
    phase_offset: float
    module_location: QbloxModuleLocation


@dataclass(frozen=True, slots=True, kw_only=True)
class QbloxModuleView:
    """Read-only module projection consumed by configuration resolution.

    The view groups canonical hardware without parsing source configuration, applying
    defaults, or selecting a sequencer. Those concerns belong to later stack layers.

    :ivar kind: Installed module kind.
    :ivar location: Instrument and slot occupied by the module.
    :ivar ports: Canonical ports attached to the module.
    :ivar oscillators: Local oscillators used by those ports and channels.
    :ivar channel_bindings: Calibrated logical channels routed through the module.
    """

    kind: QbloxModuleKind
    location: QbloxModuleLocation
    ports: tuple[QbloxPortBinding, ...] = ()
    oscillators: tuple[QbloxOscillatorBinding, ...] = ()
    channel_bindings: tuple[QbloxChannelBinding, ...] = ()
