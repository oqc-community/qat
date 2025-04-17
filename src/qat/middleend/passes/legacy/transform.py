# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from collections import defaultdict
from numbers import Number

import numpy as np
from compiler_config.config import MetricsType

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import TransformPass
from qat.core.result_base import ResultManager
from qat.purr.backends.qiskit_simulator import QiskitBuilder, QiskitBuilderWrapper
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    CustomPulse,
    Instruction,
    PhaseReset,
    PhaseShift,
    Pulse,
)


class IntegratorAcquireSanitisation(TransformPass):
    """Changes `AcquireMode.INTEGRATOR` acquisitions to `AcquireMode.RAW`.

    The legacy echo/RTCS engines expect the acquisition mode to be either `RAW` or `SCOPE`.
    While the actual execution can process `INTEGRATOR` by treating it as `RAW`, they are
    typically santitised the runtime using :meth:`EchoEngine.optimize()`. If not done in the
    new pipelines, it will conflict with :class:`PostProcessingSantisiation`, and return the
    wrong results. The new echo engine supports all acquisition modes, so this is not a
    problem here.
    """

    def run(
        self,
        ir: InstructionBuilder,
        *args,
        **kwargs,
    ):
        """:param ir: The list of instructions stored in an :class:`InstructionBuilder`."""
        for inst in [instr for instr in ir.instructions if isinstance(instr, Acquire)]:
            if inst.mode == AcquireMode.INTEGRATOR:
                inst.mode = AcquireMode.RAW
        return ir


class QiskitInstructionsWrapper(TransformPass):
    """Wraps the Qiskit builder in a wrapper to match the pipelines API.

    A really silly pass needed to wrap the :class:`QiskitBuilder` in an object that allows
    `QiskitBuilderWrapper.instructions` to be called, allowing the builder to be used in the
    the :class:`LegacyRuntime`. This is needed because the qiskit engine has a different API
    to other `purr` engines, requiring the whole builder to be passed (as opposed to
    `builder.instructions`).
    """

    def run(self, ir: QiskitBuilder, *args, **kwargs) -> QiskitBuilderWrapper:
        """:param ir: The Qiskit instructions"""
        return QiskitBuilderWrapper(ir)


class PhaseOptimisation(TransformPass):
    """Iterates through the list of instructions and compresses contiguous
    :class:`PhaseShift` instructions.
    """

    def run(
        self,
        ir: InstructionBuilder,
        res_mgr: ResultManager,
        met_mgr: MetricsManager,
        *args,
        **kwargs,
    ):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to save the analysis results.
        :param met_mgr: The metrics manager to store the number of instructions after
            optimisation.
        """

        previous_instruction = None
        accum_phaseshifts: dict[PulseChannel, float] = defaultdict(float)
        optimized_instructions: list[Instruction] = []
        for instruction in ir.instructions:
            if isinstance(instruction, PhaseShift) and isinstance(
                instruction.phase, Number
            ):
                accum_phaseshifts[instruction.channel] += instruction.phase
            elif isinstance(instruction, (Pulse, CustomPulse, PhaseShift)):
                quantum_targets = getattr(instruction, "quantum_targets", [])
                if not isinstance(quantum_targets, list):
                    quantum_targets = [quantum_targets]
                for quantum_target in quantum_targets:
                    if not np.isclose(accum_phaseshifts[quantum_target] % (2 * np.pi), 0.0):
                        optimized_instructions.append(
                            PhaseShift(
                                quantum_target, accum_phaseshifts.pop(quantum_target)
                            )
                        )
                optimized_instructions.append(instruction)

            elif isinstance(instruction, PhaseReset):
                for channel in instruction.quantum_targets:
                    accum_phaseshifts.pop(channel, None)

                if isinstance(previous_instruction, PhaseReset):
                    unseen_targets = list(
                        set(instruction.quantum_targets)
                        - set(previous_instruction.quantum_targets)
                    )
                    previous_instruction.quantum_targets.extend(unseen_targets)
                else:
                    optimized_instructions.append(instruction)

            else:
                optimized_instructions.append(instruction)

            previous_instruction = instruction

        ir.instructions = optimized_instructions
        met_mgr.record_metric(
            MetricsType.OptimizedInstructionCount, len(optimized_instructions)
        )
        return ir
