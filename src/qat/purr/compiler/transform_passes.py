# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from collections import defaultdict
from numbers import Number
from typing import Dict, List

from compiler_config.config import MetricsType

from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.instructions import (
    AcquireMode,
    CustomPulse,
    DeviceUpdate,
    Instruction,
    PhaseReset,
    PhaseShift,
    PostProcessing,
    PostProcessType,
    ProcessAxis,
    Pulse,
    Variable,
)
from qat.purr.core.metrics_base import MetricsManager
from qat.purr.core.pass_base import TransformPass
from qat.purr.core.result_base import ResultManager
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class PhaseOptimisation(TransformPass):
    """
    Extracted from QuantumExecutionEngine.optimize()
    """

    def run(
        self,
        ir: InstructionBuilder,
        res_mgr: ResultManager,
        met_mgr: MetricsManager,
        *args,
        **kwargs,
    ):
        accum_phaseshifts: Dict[PulseChannel, PhaseShift] = {}
        optimized_instructions: List[Instruction] = []
        for instruction in ir.instructions:
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

        ir.instructions = optimized_instructions
        met_mgr.record_metric(
            MetricsType.OptimizedInstructionCount, len(optimized_instructions)
        )


class PostProcessingSanitisation(TransformPass):
    """
    Extracted from LiveDeviceEngine.optimize()
    Better pass name/id ?
    """

    def run(
        self,
        ir: InstructionBuilder,
        res_mgr: ResultManager,
        met_mgr: MetricsManager,
        *args,
        **kwargs,
    ):
        pp_insts = [val for val in ir.instructions if isinstance(val, PostProcessing)]
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
        ir.instructions = [val for val in ir.instructions if val not in discarded]
        met_mgr.record_metric(MetricsType.OptimizedInstructionCount, len(ir.instructions))


class DeviceUpdateSanitisation(TransformPass):
    """
    Duplicate DeviceUpdate instructions upsets the device injection mechanism, which causes corruption
    of the HW model.

    In fact, a DeviceInjector is currently 1-1 associated with a DeviceUpdate instruction. When multiple
    DeviceUpdate instructions (sequentially) inject the same "target", the first DeviceInjector assigns the
    (correct) value of the attribute (on the target) to the revert_value. At this point the HW model (or any
    other target kind) is dirty, and any subsequent DeviceInjector updater would surely assign
    the (wrong, usually a placeholder `Variable`) to its revert_value. This results in a corrupt HW model whereby
    reversion wouldn't have the desired effect.

    This pass is a (lazy) fix, which is to analyse when such cases happen and eliminate duplicate DeviceUpdate
    instructions that target THE SAME "attribute" on THE SAME "target" with THE SAME variable.
    """

    def run(
        self,
        ir: InstructionBuilder,
        res_mgr: ResultManager,
        met_mgr: MetricsManager,
        *args,
        **kwargs,
    ):
        target_attr2value = defaultdict(list)
        for inst in ir.instructions:
            if isinstance(inst, DeviceUpdate):
                # target already validated to be a QuantumComponent during initialisation
                if not hasattr(inst.target, inst.attribute):
                    raise ValueError(
                        f"Attempting to assign {inst.value} to non existing attribute {inst.attribute}"
                    )

                if isinstance(inst.value, Variable):
                    target_attr2value[(inst.target, inst.attribute)].append(inst.value)

        for (target, attr), values in target_attr2value.items():
            if len(values) > 1:
                log.warning(
                    f"Multiple DeviceUpdate instructions attempting to update the same attribute '{attr}' on {target}"
                )

            unique_values = set(values)
            if len(unique_values) > 1:
                raise ValueError(
                    f"Cannot update the same attribute '{attr}' on {target} with distinct values {unique_values}"
                )

        new_instructions = []
        for inst in ir.instructions:
            if isinstance(inst, DeviceUpdate) and isinstance(inst.value, Variable):
                if next(iter(target_attr2value[(inst.target, inst.attribute)]), None):
                    target_attr2value[(inst.target, inst.attribute)].clear()
                    new_instructions.append(inst)
            else:
                new_instructions.append(inst)

        ir.instructions = new_instructions
