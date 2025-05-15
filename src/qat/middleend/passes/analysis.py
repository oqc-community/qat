# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd

from dataclasses import dataclass, field

import numpy as np

from qat.core.pass_base import AnalysisPass, ResultManager
from qat.core.result_base import ResultInfoMixin
from qat.model.target_data import TargetData
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel, Qubit, Resonator
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Acquire, CustomPulse, Pulse, Repeat
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


@dataclass
class BatchedShotsResult(ResultInfoMixin):
    total_shots: int
    batched_shots: int


class BatchedShots(AnalysisPass):
    """Determines how shots should be grouped when the total number exceeds that maximum
    allowed.

    The target machine might have an allowed number of shots that can be executed by a
    single execution call. To execute a number of shots greater than this value, shots can
    be batched, with each batch executed by its own "execute" call on the target machine. For
    example, if the maximum number of shots for a target machine is 2000, but you required 4000
    shots, then this could be done as [2000, 2000] shots.

    Now consider the more complex scenario where  4001 shots are required. Clearly this can
    be done in three batches. While it is tempting to do this in batches of [2000, 2000, 1],
    for some target machines, specification of the number of shots can only be achieved at
    compilation (as opposed to runtime). Batching as described above would result in us
    needing to compile two separate programs. Instead, it makes more sense to batch the
    shots as three lots of 1334 shots, which gives a total of 4002 shots. The extra two
    shots can just be discarded at run time.
    """

    def __init__(self, model: QuantumHardwareModel, target_data: TargetData):
        """Instantiate the pass with a hardware model.

        :param model: The hardware model that contains the total number of shots.
        :param target_data: Target-related information.
        """
        # TODO: replace the hardware model with whatever structures will contain the allowed
        # number of shots in the future.
        # TODO: determine if this should be fused with `RepeatSanitisation`.
        self.model = model
        self.target_data = target_data

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to store the analysis results.
        """

        repeats = [inst for inst in ir.instructions if isinstance(inst, Repeat)]
        if len(repeats) > 0:
            shots = repeats[0].repeat_count
        else:
            shots = self.target_data.default_shots

        if shots < 0 or not isinstance(shots, int):
            raise ValueError("The number of shots must be a non-negative integer.")

        max_shots = self.target_data.max_shots
        num_batches = int(np.ceil(shots / max_shots))
        if num_batches == 0:
            shots_per_batch = 0
        else:
            shots_per_batch = int(np.ceil(shots / num_batches))
        res_mgr.add(BatchedShotsResult(total_shots=shots, batched_shots=shots_per_batch))
        return ir


@dataclass
class ActiveChannelResults(ResultInfoMixin):
    """Stores the active pulse channels in a task, which is defined by pulse channels that
    are acted on by a pulse or acquisition.

    Results are stored as a map between pulse channels and the qubit they belong to. Rogue
    pulse channels are stored in the `unassigned` attribute. Various helper properties
    and methods can be used to fetch the complete lists of active pulse channels and qubits.
    """

    target_map: dict[PulseChannel, Qubit] = field(default_factory=lambda: dict())
    unassigned: list[PulseChannel] = field(default_factory=lambda: [])

    @property
    def targets(self) -> list[PulseChannel]:
        """Returns a dictionary of all pulse channels with their full id as a key."""
        return list(self.target_map.keys()) + self.unassigned

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
        self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs
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
                device = None
            else:
                if len(devices) > 1:
                    log.warning(
                        f"Multiple targets found with pulse channel {target}: "
                        + ", ".join([str(device) for device in devices])
                        + f". Defaulting to the first pulse channel found, {devices[0]}."
                    )

                device = devices[0]
                if isinstance(device, Resonator):
                    qubits = [
                        qubit
                        for qubit in self.model.qubits
                        if qubit.measure_device == device
                    ]
                    if len(qubits) == 0:
                        device = None
                    else:
                        device = qubits[0]

            if device:
                result.target_map[target] = device
            else:
                result.unassigned.append(target)

        res_mgr.add(result)
        return ir
