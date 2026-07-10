# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
""":class:`PulseLevelConstraints` is the layer that contains any constraints on pulse types,
which are used to validate and legalise the pulse IR.

Some expected use cases are:

* Validating that the pulse duration is within the allowed limits of the port it plays on,
  and breaking it down into smaller pulses if possible.
* Ensuring all durations match the timing granularity of the hardware.
* Sampling waveforms according to the feature set of the hardware.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from types import MappingProxyType

from qat.experimental.dialect.pulse.ir import (
    ANALYTICAL_WAVEFORM_OPS,
    IsAnalyticalWaveformInterface,
)
from qat.experimental.system_data.canonical.schema import CanonicalSystemData, PortData
from qat.experimental.system_data.derived.interface import DerivedViewInterface

_PICOSECONDS_CONVERSION = 1e-12
_NAME_TO_WAVEFORM_OP_MAPPING = {
    waveform_op.WAVEFORM_NAME: waveform_op for waveform_op in ANALYTICAL_WAVEFORM_OPS
}


@dataclass(frozen=True)
class PortConstraints:
    """Information about the constraints of a given port.

    A given type of control hardware can have specific properties, e.g., ports that are for
    control, ports for readout, or even different specialised cards. These different classes
    of ports can carry different properties. This class contains the properties for the
    pulse types required at the pulse level.

    :ivar sample_time_ps: The sample time of the port in picoseconds.
    :ivar min_duration_ps: The minimum pulse and acquire duration times in picoseconds.
    :ivar max_duration_ps: The maximum pulse and acquire duration times in picoseconds.
    :ivar native_waveform_shapes: The native waveform shapes supported by that port.
    :ivar acquire_allowed: Whether acquisitions are supported via that port.
    """

    sample_time_ps: int
    min_duration_ps: int
    max_duration_ps: int | None
    native_waveform_shapes: tuple[type[IsAnalyticalWaveformInterface], ...]
    acquire_allowed: bool

    @cached_property
    def sample_time_s(self) -> float:
        """The sample time of the port in seconds."""
        return self.sample_time_ps * _PICOSECONDS_CONVERSION

    @cached_property
    def min_pulse_duration_s(self) -> float:
        """The minimum pulse duration of the port in seconds."""
        return self.min_duration_ps * _PICOSECONDS_CONVERSION

    @cached_property
    def max_pulse_duration_s(self) -> float | None:
        """The maximum pulse duration of the port in seconds."""
        if self.max_duration_ps is None:
            return None
        return self.max_duration_ps * _PICOSECONDS_CONVERSION

    def supports_waveform_shape(
        self, waveform_shape: type[IsAnalyticalWaveformInterface]
    ) -> bool:
        """Whether the given waveform shape is supported by the port."""
        return waveform_shape in self.native_waveform_shapes


def _build_constraints_from_port_data(port_data: PortData) -> tuple[PortConstraints, int]:
    """Builds the pulse-level constraints for a given port.

    :param port_data: The canonical port data to build the constraints from.
    :return: The pulse-level constraints for the given port and the granularity.
    """

    min_pulse_duration_ps = (
        port_data.min_blocks * port_data.block_size * port_data.sample_time
    )

    # Note that if max_blocks is -1, it means there is no limit on the duration of the
    # pulse.
    max_duration_ps = (
        port_data.max_blocks * port_data.block_size * port_data.sample_time
        if port_data.max_blocks != -1
        else None
    )

    native_shapes = []
    for shape_name in port_data.native_waveform_shapes:
        if shape_name not in _NAME_TO_WAVEFORM_OP_MAPPING:
            raise ValueError(
                f"The port {port_data.id} has an unknown waveform shape '{shape_name}' "
                f"which is not supported."
            )
        native_shapes.append(_NAME_TO_WAVEFORM_OP_MAPPING[shape_name])

    constraints = PortConstraints(
        sample_time_ps=port_data.sample_time,
        min_duration_ps=min_pulse_duration_ps,
        max_duration_ps=max_duration_ps,
        acquire_allowed=port_data.acquire_allowed,
        native_waveform_shapes=tuple(native_shapes),
    )

    granularity = port_data.sample_time * port_data.block_size

    return constraints, granularity


@dataclass(frozen=True)
class PulseLevelConstraints(DerivedViewInterface):
    """The ports available for a given hardware are static, at least at compile time, and we
    can calculate the properties of each port and the constraints they place at the pulse-
    level.

    The constraints can be accessed through the port identifier directly; how this is used
    will depend on where in the compiler the constraints are applied. For example, when
    checking if the duration of a pulse is legal, we might not know the exact frame and
    hence the exact port it plays on without traversing the entire IR. When defining a
    frame, we can look directly using the port identifier from the frame.

    :ivar ports: Maps a port by its identifier to its constraints.
    :ivar granularity_ps: The timing granularity supported by the hardware in picoseconds.
        All operations with a duration must be integer multiples of this. Currently, this is
        assumed to be constant across all ports, but later, this might be generalised to be
        specific to a given port type.
    """

    ports: Mapping[str, PortConstraints]
    granularity_ps: int

    @cached_property
    def granularity_s(self) -> float:
        """The timing granularity of the hardware in seconds."""
        return self.granularity_ps * _PICOSECONDS_CONVERSION

    @classmethod
    def from_canonical(cls, canonical_data: CanonicalSystemData) -> "PulseLevelConstraints":
        """Builds the pulse-level constraints for a given hardware model.

        :param canonical_data: The canonical hardware model to build the constraints from.
        :return: The pulse-level constraints for the given hardware model.
        """

        port_data = {}
        granularity = None
        for port in canonical_data.ports:
            if port.id in port_data:
                raise ValueError(
                    f"The port {port.id} is defined multiple times in the canonical data, "
                    f"which is not supported."
                )

            constraints, port_granularity = _build_constraints_from_port_data(port)
            if granularity is None:
                granularity = port_granularity
            elif granularity != port_granularity:
                raise ValueError(
                    f"The port {port.id} has a different granularity ({port_granularity}) "
                    f"than the other ports ({granularity}), which is not supported."
                )
            port_data[port.id] = constraints

        if granularity is None:
            raise ValueError(
                "No ports were found in the canonical data, so no granularity could "
                "be determined."
            )

        return PulseLevelConstraints(
            ports=MappingProxyType(port_data),
            granularity_ps=granularity,
        )
