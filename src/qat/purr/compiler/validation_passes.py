# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from numbers import Number
from typing import List

import numpy as np

from qat.core.config.configure import get_config
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
from qat.purr.core.pass_base import ValidationPass
from qat.purr.core.result_base import ResultManager


class InstructionValidation(ValidationPass):
    """
    Extracted from QuantumExecutionEngine.validate()
    """

    def __init__(
        self,
        engine: QuantumExecutionEngine,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.engine = engine

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        qatconfig = get_config()

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
                        qatconfig.INSTRUCTION_VALIDATION.PULSE_DURATION_LIMITS
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
                        if qatconfig.INSTRUCTION_VALIDATION.PULSE_DURATION_LIMITS:
                            raise ValueError(
                                f"Max Waveform width is {MaxPulseLength} s "
                                f"given: {values} s"
                            )


class ReadoutValidation(ValidationPass):
    """
    Extracted from LiveDeviceEngine.validate()
    """

    def __init__(
        self,
        model: QuantumHardwareModel,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        qatconfig = get_config()

        if not isinstance(self.model, LiveHardwareModel):
            return

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
                    for qbit in self.model.qubits:
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
                                    self.model._resolve_qb_pulse_channel(chanbit)[0],
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
