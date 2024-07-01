from builtins import isinstance
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Dict, List

from qat.purr.backends.qblox.ir import Opcode, Sequence
from qat.purr.backends.qblox.rnd.context import (
    AbstractContext,
    QbloxContext,
    RepeatContext,
    SweepContext,
)
from qat.purr.backends.qblox.rnd.control_flow import BasicBlock, CatGraph
from qat.purr.backends.qblox.rnd.instructions import EndRepeat, EndSweep
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.instructions import (
    Acquire,
    Delay,
    DeviceUpdate,
    FrequencyShift,
    Id,
    Instruction,
    MeasurePulse,
    PhaseReset,
    PhaseShift,
    PostProcessing,
    QuantumInstruction,
    Repeat,
    Sweep,
    Synchronize,
    Waveform,
)


@dataclass
class QbloxPackage:
    target: PulseChannel
    sequence: Sequence

    def is_empty(self):
        return not (any(self.sequence.waveforms) or any(self.sequence.acquisitions))


class QatEmitter:
    def emit_cfg(self, instructions: List[Instruction]) -> CatGraph:
        catg = CatGraph()
        self.build_catg(instructions, catg)
        return catg

    def build_catg(
        self, instructions: List[Instruction], catg: CatGraph
    ) -> List[BasicBlock]:
        i = None
        for j, inst in enumerate(instructions):
            if isinstance(inst, (Sweep, Repeat)):
                i = j
                break

        if i is None:
            return [BasicBlock(instructions)]

        end_type = EndRepeat if isinstance(instructions[i], Repeat) else EndSweep

        j = None
        for k, inst in enumerate(instructions[i + 1 :]):
            if isinstance(inst, end_type):
                if (isinstance(inst, EndRepeat) and end_type != EndRepeat) or (
                    isinstance(inst, EndSweep) and end_type != EndSweep
                ):
                    raise ValueError("Scoping semantics violated")
                j = k + i + 1
                break

        if j is None:
            raise ValueError("Missing scope end")

        before_i = self.build_catg(instructions[:i], catg)
        ith = BasicBlock(instructions[i])
        i2j = self.build_catg(instructions[i + 1 : j], catg)
        jth = BasicBlock(instructions[i])
        after_j = self.build_catg(instructions[j + 1 :], catg)

        nodes = before_i + [ith] + i2j + [jth] + after_j
        edges = [
            (before_i[-1] if before_i else None, ith),
            (ith, i2j[0] if i2j else None),
            (i2j[-1] if i2j else None, jth),
            (jth, after_j[0] if after_j else None),
        ]

        for n in nodes:
            catg.add_node(n)

        for src, dest in edges:
            catg.add_edge(src, dest)

        return nodes


class QbloxEmitter:
    def __init__(self, qat_file: QatFile):
        super().__init__()
        self.repeats = [
            inst for inst in qat_file.meta_instructions if isinstance(inst, Repeat)
        ]
        self.sweeps = [
            inst for inst in qat_file.meta_instructions if isinstance(inst, Sweep)
        ]

    def _init_target_contexts(
        self,
        targets: List[PulseChannel],
        global_stack: ExitStack,
        stacks: Dict[PulseChannel, ExitStack],
        contexts: Dict[PulseChannel, AbstractContext],
    ):
        for target in targets:
            if not isinstance(target, PulseChannel):
                raise ValueError(f"{target} is not a PulseChannel")
            if not target in stacks:
                stack: ExitStack = global_stack.enter_context(ExitStack())
                context: AbstractContext = stack.enter_context(QbloxContext())
                for sweep in self.sweeps:
                    context.nested = stack.enter_context(
                        SweepContext(sweep, context.builder, context.manager)
                    )
                    context = context.nested
                for repeat in self.repeats:
                    context.nested = stack.enter_context(
                        RepeatContext(repeat, context.builder, context.manager)
                    )
                    context = context.nested
                contexts[target] = context
                stacks[target] = stack

    def emit(self, qat_file: QatFile) -> List[QbloxPackage]:
        target_stacks: Dict[PulseChannel, ExitStack] = {}
        target_contexts: Dict[PulseChannel, AbstractContext] = {}

        with ExitStack() as global_stack:
            inst_iter = iter(qat_file.instructions)
            while (inst := next(inst_iter, None)) is not None:
                if not isinstance(inst, QuantumInstruction):
                    continue  # Ignore classical instructions

                if isinstance(inst, PostProcessing):
                    continue

                if isinstance(inst, DeviceUpdate):
                    continue

                if isinstance(inst, Synchronize):
                    self._init_target_contexts(
                        inst.quantum_targets,
                        global_stack,
                        target_stacks,
                        target_contexts,
                    )
                    AbstractContext.synchronize(inst, target_contexts)
                    continue

                if isinstance(inst, PhaseReset):
                    self._init_target_contexts(
                        inst.quantum_targets,
                        global_stack,
                        target_stacks,
                        target_contexts,
                    )
                    AbstractContext.reset_phase(inst, target_contexts)
                    continue

                self._init_target_contexts(
                    inst.quantum_targets,
                    global_stack,
                    target_stacks,
                    target_contexts,
                )

                for target in inst.quantum_targets:
                    context = target_contexts[target]

                    if isinstance(inst, MeasurePulse):
                        acquire = next(inst_iter, None)
                        if acquire is None or not isinstance(acquire, Acquire):
                            raise ValueError(
                                "Found a MeasurePulse but no Acquire instruction followed"
                            )
                        context.measure_acquire(inst, acquire, target)
                    elif isinstance(inst, Waveform):
                        context.waveform(inst, target)
                    elif isinstance(inst, Delay):
                        context.delay(inst)
                    elif isinstance(inst, PhaseShift):
                        context.shift_phase(inst)
                    elif isinstance(inst, FrequencyShift):
                        context.shift_frequency(inst, target)
                    elif isinstance(inst, Id):
                        context.id()

        target_contexts = self.optimize(target_contexts)

        return [
            self.create_package(target, context)
            for target, context in target_contexts.items()
        ]

    def optimize(self, qblox_contexts: Dict) -> Dict:
        # Remove empty contexts
        qblox_contexts = {
            target: context
            for target, context in qblox_contexts.items()
            if not context.is_empty()
        }

        # Remove Opcode.WAIT_SYNC instructions when the experiment contains only a singleton context
        if len(qblox_contexts) == 1:
            context = list(qblox_contexts.values())[0]
            context.builder.q1asm_instructions = [
                inst
                for inst in context.builder.q1asm_instructions
                if not inst.opcode == Opcode.WAIT_SYNC
            ]

        return qblox_contexts

    def create_package(self, target: PulseChannel, context: AbstractContext):
        sequence = context.builder.build()
        return QbloxPackage(target, sequence)
