from numbers import Number
from typing import Dict, List

from qat.ir.pass_base import TransformPass
from qat.ir.result_base import ResultManager
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.instructions import (
    AcquireMode,
    CustomPulse,
    Instruction,
    PhaseReset,
    PhaseShift,
    PostProcessing,
    PostProcessType,
    ProcessAxis,
    Pulse,
)


class PhaseOptimisation(TransformPass):
    """
    Extracted from QuantumExecutionEngine.optimize()
    """

    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        accum_phaseshifts: Dict[PulseChannel, PhaseShift] = {}
        optimized_instructions: List[Instruction] = []
        for instruction in builder.instructions:
            if isinstance(instruction, PhaseShift) and isinstance(
                instruction.phase, Number
            ):
                if accum_phaseshift := accum_phaseshifts.get(instruction.channel, None):
                    accum_phaseshift.phase += instruction.phase
                else:
                    accum_phaseshifts[instruction.channel] = PhaseShift(
                        instruction.channel, instruction.phase
                    )
            elif isinstance(instruction, (Pulse, CustomPulse)):
                quantum_targets = getattr(instruction, "quantum_targets", [])
                if not isinstance(quantum_targets, List):
                    quantum_targets = [quantum_targets]
                for quantum_target in quantum_targets:
                    if quantum_target in accum_phaseshifts:
                        optimized_instructions.append(accum_phaseshifts.pop(quantum_target))
                optimized_instructions.append(instruction)
            elif isinstance(instruction, PhaseReset):
                for channel in instruction.quantum_targets:
                    accum_phaseshifts.pop(channel, None)
                optimized_instructions.append(instruction)
            else:
                optimized_instructions.append(instruction)

        builder.instructions = optimized_instructions


class PostProcessingOptimisation(TransformPass):
    """
    Extracted from LiveDeviceEngine.optimize()
    Better pass name/id ?
    """

    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        pp_insts = [val for val in builder.instructions if isinstance(val, PostProcessing)]
        discarded = []
        for pp in pp_insts:
            if pp.acquire.mode == AcquireMode.SCOPE:
                if (
                    pp.process == PostProcessType.MEAN
                    and ProcessAxis.SEQUENCE in pp.axes
                    and len(pp.axes) <= 1
                ):
                    discarded.append(pp)

            elif pp.acquire.mode == AcquireMode.INTEGRATOR:
                if (
                    pp.process == PostProcessType.DOWN_CONVERT
                    and ProcessAxis.TIME in pp.axes
                    and len(pp.axes) <= 1
                ):
                    discarded.append(pp)
                if (
                    pp.process == PostProcessType.MEAN
                    and ProcessAxis.TIME in pp.axes
                    and len(pp.axes) <= 1
                ):
                    discarded.append(pp)
        builder.instructions = [val for val in builder.instructions if val not in discarded]
