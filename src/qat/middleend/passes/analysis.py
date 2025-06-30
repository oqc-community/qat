# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from dataclasses import dataclass, field

from qat.core.result_base import ResultInfoMixin
from qat.model.device import PulseChannel, Qubit


@dataclass
class ActivePulseChannelResults(ResultInfoMixin):
    """Stores the active pulse channels in a task, which is defined by pulse channels that
    are acted on by a pulse or acquisition.

    Results are stored as a map between pulse channels and the qubit they belong to. Rogue
    pulse channels are stored in the `unassigned` attribute. Various helper properties
    and methods can be used to fetch the complete lists of active pulse channels and qubits.
    """

    target_map: dict[str, Qubit] = field(default_factory=lambda: dict())
    unassigned: list[str] = field(default_factory=lambda: [])

    @property
    def physical_qubit_indices(self) -> set[int]:
        """Returns a list of all active physical qubit indices."""
        raise NotImplementedError("This method is not implemented yet.")

    @property
    def targets(self) -> list[str]:
        """Returns a dictionary of all pulse channels with their full id as a key."""
        return list(self.target_map.keys()) + self.unassigned

    @property
    def qubits(self) -> list[Qubit]:
        """Returns a list of all active qubits."""
        return list(set(self.target_map.values()))

    def from_qubit(self, qubit: Qubit) -> list[PulseChannel]:
        """Returns the list of pulse channels that belong to a qubit."""
        pulse_channels = []
        for pulse_ch in qubit.all_pulse_channels + qubit.resonator.all_pulse_channels:
            if pulse_ch.uuid in self.target_map:
                pulse_channels.append(pulse_ch)
        return pulse_channels
