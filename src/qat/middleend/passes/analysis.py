# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from collections import defaultdict
from dataclasses import dataclass, field

from qat.core.metrics_base import MetricsManager, MetricsType
from qat.core.pass_base import AnalysisPass, ResultManager
from qat.core.result_base import ResultInfoMixin
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.measure import Acquire
from qat.ir.waveforms import Pulse
from qat.model.device import PulseChannel, Qubit
from qat.model.hardware_model import PhysicalHardwareModel
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


@dataclass
class ActivePulseChannelResults(ResultInfoMixin):
    """Stores the active pulse channels in a task, which is defined by pulse channels that
    are acted on by a pulse or acquisition.

    Results are stored as a map between pulse channels and the qubit they belong to. Various
    helper properties and methods can be used to fetch the complete lists of active pulse
    channels and qubits.
    """

    pulse_channel_to_qubit_map: dict[str, Qubit] = field(default_factory=dict)
    qubit_to_pulse_channel_map: dict[Qubit, list[PulseChannel]] = field(
        default_factory=lambda: defaultdict(list)
    )

    @property
    def targets(self) -> set[str]:
        """Returns a dictionary of all pulse channels with their full id as a key."""
        return set(self.pulse_channel_to_qubit_map.keys())

    @property
    def qubits(self) -> set[Qubit]:
        """Returns a list of all active qubits."""
        return set(self.qubit_to_pulse_channel_map.keys())

    def add_target(self, pulse_channel: PulseChannel, qubit: Qubit):
        """Adds a pulse channel and its associated qubit to the results."""
        self.pulse_channel_to_qubit_map[pulse_channel.uuid] = qubit
        self.qubit_to_pulse_channel_map[qubit].append(pulse_channel)

    def from_qubit(self, qubit: Qubit) -> list[PulseChannel]:
        """Returns the list of pulse channels that belong to a qubit."""
        return self.qubit_to_pulse_channel_map.get(qubit)


class ActivePulseChannelAnalysis(AnalysisPass):
    """Determines the set of pulse channels which are targeted by quantum instructions.

    A pulse channel that has a pulse played at any time, or an acquisition is defined to be
    active. This pass is used to determine which pulse channels are required in compilation,
    and is used in subsequent passes to easily extract pulse channel properties, and is
    useful for not performing extra analysis on rogue channels picked up by
    :class:`Synchronize` instructions.

    Also records the physical qubit indices used in the circuit as a metric.
    """

    def __init__(self, model: PhysicalHardwareModel):
        self.model = model

    def run(
        self,
        ir: QuantumInstructionBuilder,
        res_mgr: ResultManager,
        met_mgr: MetricsManager,
        *args,
        **kwargs,
    ) -> QuantumInstructionBuilder:
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to store the analysis results.
        """

        targets: set[str] = set()
        for inst in ir.instructions:
            if isinstance(inst, (Acquire, Pulse)):
                targets.add(inst.target)

        result = ActivePulseChannelResults()
        for target in targets:
            pulse_channel = ir.get_pulse_channel(target)
            device = self.model.qubit_for_physical_channel_id(
                pulse_channel.physical_channel_id
            )

            if device is None:
                raise ValueError(
                    f"Pulse channel with id {target} cannot be mapped to a device in the "
                    "hardware model."
                )
            result.add_target(pulse_channel, device)

        phys_q_indices = sorted(
            [self.model.index_of_qubit(qubit) for qubit in result.qubits]
        )
        log.info(f"Physical qubits used in this circuit: {phys_q_indices}")
        met_mgr.record_metric(MetricsType.PhysicalQubitIndices, phys_q_indices)

        res_mgr.add(result)
        return ir


PydActivePulseChannelAnalysis = ActivePulseChannelAnalysis
