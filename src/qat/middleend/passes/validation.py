# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from numbers import Number
from typing import List

import numpy as np

from qat import qatconfig
from qat.core.pass_base import ValidationPass
from qat.ir.measure import Acquire as PydAcquire
from qat.ir.measure import Pulse as PydPulse
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.purr.backends.live import LiveHardwareModel
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import MaxPulseLength, PulseChannel, Qubit
from qat.purr.compiler.execution import QuantumExecutionEngine
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    CustomPulse,
    PostProcessing,
    ProcessAxis,
    Pulse,
    Sweep,
    Variable,
)


class InstructionValidation(ValidationPass):
    """Validates instructions against the hardware.

    Extracted from :mod:`qat.purr.compiler.execution.QuantumExecutionEngine.validate`.
    """

    def __init__(self, engine: QuantumExecutionEngine, *args, **kwargs):
        """
        :param engine: The engine contains information about the hardware, such as
            instruction limits.
        """
        self.engine = engine

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        instruction_length = len(ir.instructions)
        if instruction_length > self.engine.max_instruction_len:
            raise ValueError(
                f"Program too large to be run in a single block on current hardware. "
                f"{instruction_length} instructions."
            )

        for inst in ir.instructions:
            if isinstance(inst, Acquire) and not inst.channel.acquire_allowed:
                raise ValueError(
                    f"Cannot perform an acquire on the physical channel with id "
                    f"{inst.channel.physical_channel}"
                )
            if isinstance(inst, (Pulse, CustomPulse)):
                duration = inst.duration
                if isinstance(duration, Number) and duration > MaxPulseLength:
                    if (
                        not qatconfig.DISABLE_PULSE_DURATION_LIMITS
                    ):  # Do not throw error if we specifically disabled the limit checks.
                        # TODO: Add a lower bound for the pulse duration limits as well in a later PR,
                        # which is specific to each hardware model and can be stored as a member variables there.
                        raise ValueError(
                            f"Max Waveform width is {MaxPulseLength} s "
                            f"given: {inst.duration} s"
                        )
                elif isinstance(duration, Variable):
                    values = next(
                        iter(
                            [
                                sw.variables[duration.name]
                                for sw in ir.instructions
                                if isinstance(sw, Sweep)
                                and duration.name in sw.variables.keys()
                            ]
                        )
                    )
                    if np.max(values) > MaxPulseLength:
                        if not qatconfig.DISABLE_PULSE_DURATION_LIMITS:
                            raise ValueError(
                                f"Max Waveform width is {MaxPulseLength} s "
                                f"given: {values} s"
                            )
        return ir


class ReadoutValidation(ValidationPass):
    """Validates that there are no mid-circuit measurements, and that the post-processing
    instructions do not have an invalid sequence.

    Extracted from :meth:`qat.purr.backends.live.LiveDeviceEngine.validate`.
    """

    # TODO: break this down into smaller passes!

    def __init__(
        self,
        hardware: QuantumHardwareModel,
        *args,
        **kwargs,
    ):
        """
        :param hardware: The hardware model is needed to check for mid-circuit measurments.
        """
        self.hardware = hardware

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""

        model = self.hardware

        if not isinstance(model, LiveHardwareModel):
            return ir

        consumed_qubits: List[str] = []
        chanbits_map = {}
        for inst in ir.instructions:
            if isinstance(inst, PostProcessing):
                if (
                    inst.acquire.mode == AcquireMode.SCOPE
                    and ProcessAxis.SEQUENCE in inst.axes
                ):
                    raise ValueError(
                        "Invalid post-processing! Post-processing over SEQUENCE is "
                        "not possible after the result is returned from hardware "
                        "in SCOPE mode!"
                    )
                elif (
                    inst.acquire.mode == AcquireMode.INTEGRATOR
                    and ProcessAxis.TIME in inst.axes
                ):
                    raise ValueError(
                        "Invalid post-processing! Post-processing over TIME is not "
                        "possible after the result is returned from hardware in "
                        "INTEGRATOR mode!"
                    )
                elif inst.acquire.mode == AcquireMode.RAW:
                    raise ValueError(
                        "Invalid acquire mode! The live hardware doesn't support "
                        "RAW acquire mode!"
                    )

            # Check if we've got a measure in the middle of the circuit somewhere.
            elif qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT:
                if isinstance(inst, Acquire):
                    for qbit in model.qubits:
                        if qbit.get_acquire_channel() == inst.channel:
                            consumed_qubits.append(qbit)
                elif isinstance(inst, Pulse):
                    # Find target qubit from instruction and check whether it's been
                    # measured already.
                    acquired_qubits = [
                        (
                            (
                                chanbits_map[chanbit]
                                if chanbit in chanbits_map
                                else chanbits_map.setdefault(
                                    chanbit,
                                    model._resolve_qb_pulse_channel(chanbit)[0],
                                )
                            )
                            in consumed_qubits
                        )
                        for chanbit in inst.quantum_targets
                        if isinstance(chanbit, (Qubit, PulseChannel))
                    ]

                    if any(acquired_qubits):
                        raise ValueError(
                            "Mid-circuit measurements currently unable to be used."
                        )
        return ir


class PydNoMidCircuitMeasurementValidation(ValidationPass):
    """
    Validates that there are no mid-circuit measurements by checking that no qubit
    has an acquire instruction that is later followed by a pulse instruction.
    """

    def __init__(
        self,
        model: PydHardwareModel,
        *args,
        **kwargs,
    ):
        """
        :param model: The hardware model.
        """
        self.model = model

    def run(self, ir: InstructionBuilder, *args, **kwargs):
        """
        :param ir: The intermediate representation (IR) :class:`InstructionBuilder`.
        """
        consumed_acquire_pc: set[str] = set()

        if not qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT:
            return ir

        drive_acq_pc_map = {
            qubit.drive_pulse_channel.uuid: qubit.acquire_pulse_channel.uuid
            for qubit in self.model.qubits.values()
        }

        for instr in ir:
            if isinstance(instr, PydAcquire):
                consumed_acquire_pc.add(instr.target)

            # Check if we have a measure in the middle of the circuit somewhere.
            elif isinstance(instr, PydPulse):
                acq_pc = drive_acq_pc_map.get(instr.target, None)

                if acq_pc and acq_pc in consumed_acquire_pc:
                    raise ValueError(
                        "Mid-circuit measurements currently unable to be used."
                    )
        return ir
