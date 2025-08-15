# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from dataclasses import dataclass, field

from qat.core.metrics_base import MetricsManager, MetricsType
from qat.core.pass_base import AnalysisPass, ResultManager
from qat.core.result_base import ResultInfoMixin
from qat.ir.instruction_builder import InstructionBuilder
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

    target_map: dict[str, Qubit] = field(default_factory=lambda: dict())
    qubit_to_id_map: dict[Qubit, int] = field(default_factory=lambda: dict())

    @property
    def physical_qubit_indices(self) -> set[int]:
        """Returns a list of all active physical qubit indices."""
        return set([self.qubit_to_id_map[qubit] for qubit in self.target_map.values()])

    @property
    def targets(self) -> list[str]:
        """Returns a dictionary of all pulse channels with their full id as a key."""
        return list(self.target_map.keys())

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


class ActivePulseChannelAnalysis(AnalysisPass):
    """Determines the set of pulse channels which are targeted by quantum instructions.

    A pulse channel that has a pulse played at any time, or an acquisition is defined to be
    active. This pass is used to determine which pulse channels are required in compilation,
    and is used in subsequent passes to easily extract pulse channel properties, and is
    useful for not performing extra analysis on rogue channels picked up by
    :class:`Synchronize` instructions.
    """

    def __init__(self, model: PhysicalHardwareModel):
        self.model = model
        self.channel_to_qubit_mapping = self._create_channel_to_qubit_mapping(model)

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
            if isinstance(inst, (Acquire, Pulse)):
                targets.add(inst.target)

        result = ActivePulseChannelResults()
        for target in targets:
            qubit = self.channel_to_qubit_mapping.get(target, None)

            if qubit is None:
                # TODO: add resilience to custom pulse channels, which are supported in
                # QASM3 (COMPILER-700)
                raise ValueError(
                    f"Pulse channel {target.uuid} is not found in the hardware model."
                )

            result.target_map[target] = qubit
            result.qubit_to_id_map[qubit] = self.model.index_of_qubit(qubit)

        phys_q_indices = sorted(list(result.physical_qubit_indices))
        log.info(f"Physical qubits used in this circuit: {phys_q_indices}")
        met_mgr.record_metric(MetricsType.PhysicalQubitIndices, phys_q_indices)

        res_mgr.add(result)
        return ir

    @staticmethod
    def _create_channel_to_qubit_mapping(
        model: PhysicalHardwareModel,
    ) -> dict[PulseChannel, Qubit]:
        """Returns a mapping between pulse channels and the qubit they belong to.

        For channels that belong to a resonator, it finds the qubit that the resonator
        belongs to and uses that.
        """
        pulse_channel_to_qubit_map = {}
        for qubit in model.qubits.values():
            for pulse_channel in (
                qubit.all_pulse_channels + qubit.resonator.all_pulse_channels
            ):
                pulse_channel_to_qubit_map[pulse_channel.uuid] = qubit
        return pulse_channel_to_qubit_map


PydActivePulseChannelAnalysis = ActivePulseChannelAnalysis
