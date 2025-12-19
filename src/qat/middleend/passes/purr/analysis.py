# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from dataclasses import dataclass, field

from qat.core.metrics_base import MetricsManager, MetricsType
from qat.core.pass_base import AnalysisPass, ResultManager
from qat.core.result_base import ResultInfoMixin
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel, Qubit, Resonator
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Acquire, CustomPulse, Pulse
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


@dataclass
class ActiveChannelResults(ResultInfoMixin):
    """Stores the active pulse channels in a task, which is defined by pulse channels that
    are acted on by a pulse or acquisition.

    Results are stored as a map between pulse channels and the qubit they belong to. Various
    helper properties and methods can be used to fetch the complete lists of active pulse
    channels and qubits.
    """

    target_map: dict[PulseChannel, Qubit] = field(default_factory=lambda: dict())

    @property
    def physical_qubit_indices(self) -> set[int]:
        """Returns a list of all active physical qubit indices."""
        return set([qubit.index for qubit in self.target_map.values()])

    @property
    def targets(self) -> list[PulseChannel]:
        """Returns a dictionary of all pulse channels with their full id as a key."""
        return list(self.target_map.keys())

    @property
    def qubits(self) -> list[Qubit]:
        """Returns a list of all active qubits."""
        return list(set(self.target_map.values()))

    def from_qubit(self, qubit: Qubit) -> list[PulseChannel]:
        """Returns the list of pulse channels that belong to a qubit."""
        pulse_channels = []
        for key, val in self.target_map.items():
            if val == qubit:
                pulse_channels.append(key)
        return pulse_channels


class ActivePulseChannelAnalysis(AnalysisPass):
    """Determines the set of pulse channels which are targeted by quantum instructions.

    A pulse channel that has a pulse played at any time, or an acquisition is defined to be
    active. This pass is used to determine which pulse channels are required in compilation,
    and is used in subsequent passes to easily extract pulse channel properties, and is
    useful for not performing extra analysis on rogue channels picked up by
    :class:`Synchronize` instructions.
    """

    # TODO: PydActivePulseChannelAnalysis: this will be even more useful for pydantic
    # instructions (COMPILER-393)

    def __init__(self, model: QuantumHardwareModel):
        self.model = model

    def run(
        self,
        ir: InstructionBuilder,
        res_mgr: ResultManager,
        met_mgr: MetricsManager,
        *args,
        **kwargs,
    ) -> InstructionBuilder:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to store the analysis results.
        """

        targets: set[PulseChannel] = set()
        for inst in ir.instructions:
            if isinstance(inst, (Acquire, Pulse, CustomPulse)):
                targets.add(next(iter(inst.quantum_targets)))

        result = ActiveChannelResults()
        for target in targets:
            devices = self.model.get_devices_from_pulse_channel(target)

            if len(devices) == 0:
                devices = self.model.get_devices_from_physical_channel(
                    target.physical_channel
                )

            if len(devices) > 1:
                log.warning(
                    f"Multiple targets found with pulse channel {target}: "
                    + ", ".join([str(device) for device in devices])
                    + f". Defaulting to the first quantum device found, {devices[0]}."
                )

            device = devices[0]
            if isinstance(device, Resonator):
                device = self.model._map_resonator_to_qubit(device)

            if device:
                result.target_map[target] = device
            else:
                result.unassigned.append(target)

        phys_q_indices = sorted(list(result.physical_qubit_indices))
        log.info(f"Physical qubits used in this circuit: {phys_q_indices}")
        met_mgr.record_metric(MetricsType.PhysicalQubitIndices, phys_q_indices)

        res_mgr.add(result)
        return ir
