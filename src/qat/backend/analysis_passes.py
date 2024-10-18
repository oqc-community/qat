import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Union

import numpy as np
from compiler_config.config import InlineResultsProcessing

from qat.backend.graph import ControlFlowGraph
from qat.ir.pass_base import AnalysisPass, ResultManager
from qat.ir.result_base import ResultInfoMixin
from qat.purr.backends.qblox.constants import Constants
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.instructions import (
    Acquire,
    Assign,
    DeviceUpdate,
    EndRepeat,
    EndSweep,
    Instruction,
    PostProcessing,
    QuantumInstruction,
    Repeat,
    ResultsProcessing,
    Return,
    Sweep,
    Variable,
)


@dataclass
class TriageResult(ResultInfoMixin):
    sweeps: List[Sweep] = field(default_factory=list)
    returns: List[Return] = field(default_factory=list)
    assigns: List[Assign] = field(default_factory=list)
    target_map: Dict[PulseChannel, List[Instruction]] = field(
        default_factory=lambda: defaultdict(list)
    )
    acquire_map: Dict[PulseChannel, List[Acquire]] = field(
        default_factory=lambda: defaultdict(list)
    )
    pp_map: Dict[str, List[PostProcessing]] = field(
        default_factory=lambda: defaultdict(list)
    )
    rp_map: Dict[str, ResultsProcessing] = field(default_factory=dict)


class TriagePass(AnalysisPass):
    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        Builds a view of instructions per quantum target AOT.
        Builds selections of instructions useful for subsequent analysis/transform passes,
        for code generation, and post-playback steps.

        This is equivalent to the QatFile and simplifies the duration timeline creation in
        legacy code.
        """

        targets = set()
        for inst in builder.instructions:
            if isinstance(inst, QuantumInstruction):
                if isinstance(inst, PostProcessing):
                    for qt in inst.quantum_targets:
                        if isinstance(qt, Acquire):
                            targets.update(qt.quantum_targets)
                        else:
                            targets.add(qt)
                else:
                    targets.update(inst.quantum_targets)

        result = TriageResult()
        for inst in builder.instructions:
            # View instructions by target
            if isinstance(inst, QuantumInstruction):
                for qt in inst.quantum_targets:
                    if isinstance(qt, Acquire):
                        for aqt in qt.quantum_targets:
                            result.target_map[aqt].append(inst)
                    else:
                        result.target_map[qt].append(inst)
            else:
                for t in targets:
                    result.target_map[t].append(inst)

            # Sweeps
            if isinstance(inst, Sweep):
                result.sweeps.append(inst)

            # Returns
            elif isinstance(inst, Return):
                result.returns.append(inst)

            # Assigns
            elif isinstance(inst, Assign):
                result.assigns.append(inst)

            # Acquisition by target
            elif isinstance(inst, Acquire):
                for t in inst.quantum_targets:
                    result.acquire_map[t].append(inst)

            # Post-processing by output variable
            elif isinstance(inst, PostProcessing):
                result.pp_map[inst.output_variable].append(inst)

            # Results-processing by output variable
            elif isinstance(inst, ResultsProcessing):
                result.rp_map[inst.variable] = inst

        # Assume that raw acquisitions are experiment results.
        acquires = list(itertools.chain(*result.acquire_map.values()))
        missing_results = {
            acq.output_variable
            for acq in acquires
            if acq.output_variable not in result.rp_map
        }
        for missing_var in missing_results:
            result.rp_map[missing_var] = ResultsProcessing(
                missing_var, InlineResultsProcessing.Experiment
            )

        res_mgr.add(result)


@dataclass(frozen=True)
class IterBound:
    start: Union[int, float] = 0
    step: Union[int, float] = 0
    end: Union[int, float] = 0
    count: int = 0


@dataclass
class BindingResult(ResultInfoMixin):
    binding_map: Dict[Variable, List[Instruction]] = field(
        default_factory=lambda: defaultdict(list)
    )
    iter_bounds_map: Dict[PulseChannel, Dict[str, IterBound]] = field(
        default_factory=lambda: defaultdict(dict)
    )


class BindingPass(AnalysisPass):
    """
    Naive static binding analysis of variables, instructions, and scopes.
    """

    @staticmethod
    def get_if_freq(target):
        if target.fixed_if:
            if_freq = target.baseband_if_frequency
        else:
            if_freq = target.frequency - target.baseband_frequency
        return if_freq

    @staticmethod
    def get_baseband_freq(target):
        if target.fixed_if:
            bb_freq = target.frequency - target.baseband_if_frequency
        else:
            bb_freq = target.baseband_frequency
        return bb_freq

    @staticmethod
    def extract_iter_bound(value: Union[List, np.ndarray]):
        """
        Given a sequence of numbers (typically having been generated from np.linspace()), figure out
        if the numbers are linearly/evenly spaced, in which case returns an IterBound instance holding
        the start, step, end, and count of the numbers in the array, or else fail.

        In the future, we might be interested in relaxing this condition and return "interpolated"
        evenly spaced approximation of the input sequence.
        """

        if value is None:
            raise ValueError(f"Cannot process value {value}")

        if isinstance(value, np.ndarray):
            value = value.tolist()

        if not value:
            raise ValueError(f"Cannot process value {value}")

        start = value[0]
        step = 0
        end = value[-1]
        count = len(value)

        if count < 2:
            return IterBound(start, step, end, count)

        step = value[1] - value[0]

        if not np.isclose(step, (end - start) / (count - 1)):
            raise ValueError(f"Not a regularly partitioned space {value}")

        return IterBound(start, step, end, count)

    @staticmethod
    def phase_as_steps(phase_rad: float) -> int:
        phase_deg = np.rad2deg(phase_rad)
        phase_deg %= 360
        return round(phase_deg * Constants.NCO_PHASE_STEPS_PER_DEG)

    @staticmethod
    def freq_as_steps(freq_hz: float) -> int:
        steps = int(round(freq_hz * Constants.NCO_FREQ_STEPS_PER_HZ))

        if (
            steps < -Constants.NCO_FREQ_LIMIT_STEPS
            or steps > Constants.NCO_FREQ_LIMIT_STEPS
        ):
            min_max_frequency_in_hz = (
                Constants.NCO_FREQ_LIMIT_STEPS / Constants.NCO_FREQ_STEPS_PER_HZ
            )
            raise ValueError(
                f"IF frequency must be in [-{min_max_frequency_in_hz:e}, {min_max_frequency_in_hz:e}] Hz. "
                f"Got {freq_hz:e} Hz"
            )

        return steps

    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        instructions_by_target = triage_result.target_map

        result = BindingResult()
        bounds_map = result.iter_bounds_map
        materialised_bounds = {t: defaultdict(list) for t in instructions_by_target}
        for inst in builder.instructions:
            if isinstance(inst, Sweep):
                name, value = next(iter(inst.variables.items()))
                for t in instructions_by_target:
                    bounds_map[t][name] = self.extract_iter_bound(value)
            elif isinstance(inst, DeviceUpdate) and isinstance(inst.value, Variable):
                if inst.value.name not in bounds_map[inst.target]:
                    raise ValueError(
                        f"Variable {inst.value.name} referenced but no prior declaration found"
                    )
                bound = bounds_map[inst.target][inst.value.name]
                if inst.attribute == "frequency":
                    bb_freq = self.get_baseband_freq(inst.target)
                    materialised_bounds[inst.target][inst.value.name].append(
                        IterBound(
                            self.freq_as_steps(bound.start - bb_freq),
                            self.freq_as_steps(bound.step),
                            self.freq_as_steps(bound.end - bb_freq),
                            bound.count,
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"Unsupported processing of attribute {inst.attribute}"
                    )

        for t in instructions_by_target:
            for name, bounds in materialised_bounds[t].items():
                if len(set(bounds)) != 1:
                    raise ValueError(
                        f"Found multiple different uses for variable {name} on target {t}"
                    )
                bounds_map[t][name] = bounds[0]

        res_mgr.add(result)


@dataclass
class CFGResult(ResultInfoMixin):
    cfg: ControlFlowGraph = field(default_factory=lambda: ControlFlowGraph())


class CFGPass(AnalysisPass):
    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        result = CFGResult()
        self._build_cfg(builder, result.cfg)
        res_mgr.add(result)

    def _build_cfg(self, builder: InstructionBuilder, cfg: ControlFlowGraph):
        """
        Recursively (re)discovers (new) header nodes and flow information.

        Supports Repeat, Sweep, EndRepeat, and EndSweep. More control flow and branching instructions
        will follow in the future once these foundations sink in and get stabilised in the codebase
        """

        flow = [(e.src.head(), e.dest.head()) for e in cfg.edges]
        headers = sorted([n.head() for n in cfg.nodes]) or [
            i
            for i, inst in enumerate(builder.instructions)
            if isinstance(inst, (Sweep, Repeat, EndSweep, EndRepeat))
        ]
        if headers[0] != 0:
            headers.insert(0, 0)

        next_flow = set(flow)
        next_headers = set(headers)
        for i, h in enumerate(headers):
            inst_at_h = builder.instructions[h]
            src = cfg.get_or_create_node(h)
            if isinstance(inst_at_h, Repeat):
                next_headers.add(h + 1)
                next_flow.add((h, h + 1))
                dest = cfg.get_or_create_node(h + 1)
                cfg.get_or_create_edge(src, dest)
            elif isinstance(inst_at_h, Sweep):
                next_headers.add(h + 1)
                next_flow.add((h, h + 1))
                dest = cfg.get_or_create_node(h + 1)
                cfg.get_or_create_edge(src, dest)
            elif isinstance(inst_at_h, (EndSweep, EndRepeat)):
                if h < len(builder.instructions) - 1:
                    next_headers.add(h + 1)
                    next_flow.add((h, h + 1))
                    dest = cfg.get_or_create_node(h + 1)
                    cfg.get_or_create_edge(src, dest)
                delimiter_type = Sweep if isinstance(inst_at_h, EndSweep) else Repeat
                p = next(
                    (
                        p
                        for p in headers[i::-1]
                        if isinstance(builder.instructions[p], delimiter_type)
                    )
                )
                next_headers.add(p)
                next_flow.add((h, p))
                dest = cfg.get_or_create_node(p)
                cfg.get_or_create_edge(src, dest)
            else:
                k = next(
                    (
                        s
                        for s, inst in enumerate(builder.instructions[h + 1 :])
                        if isinstance(inst, (Sweep, Repeat, EndSweep, EndRepeat))
                    ),
                    None,
                )
                if k:
                    next_headers.add(k + h + 1)
                    next_flow.add((h, k + h + 1))
                    src.elements.extend(range(src.tail() + 1, k + h + 1))
                    dest = cfg.get_or_create_node(k + h + 1)
                    cfg.get_or_create_edge(src, dest)

        if next_headers == set(headers) and next_flow == set(flow):
            # TODO - In-place use instructions instead of indices directly
            for n in cfg.nodes:
                n.elements = [builder.instructions[i] for i in n.elements]
            return

        self._build_cfg(builder, cfg)


class LegalisationPass(AnalysisPass):
    """
    The sole purpose of this pass is to allow us to understand what can and what cannot be run on the control stack.

    An instruction is legal if it has a direct equivalent in the programming model implemented by the control stack.
    The notion of "legal" is highly determined by the hardware features of the control stack as well as its
    programming model.

    Most control stacks such as Qblox have a direct ISA-level representation for basic quantum instructions such as
    frequency and phase manipulation instructions, arithmetic instructions such as add, branching instructions
    such as jump.

    Particularly:
    1) A sweep instruction is illegal because it specifies unclear iteration semantics.
    2) A repeat instruction with a very high repetition count is illegal because acquisition memory on a sequencer
    is limited. This requires optimal batching of the repeat instruction into maximally supported batches of smaller
    repeat counts.
    3) Device updates/assigns in general are illegal because they are bound to a sweep instruction via a variable
    name. In fact, a sweep variable remains obscure and opaque until a device update bound to it is encountered
    to reveal what it intends to do with the variable (usually on the instruction builder or on the hardware model).
    We say that a device update carries meaning for the sweep variable and materialises its intention.

    A brute force implementation would increase the depth of loop nests via repeat batching as well as bubble
    outwards any illegal instructions.
    """

    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        pass


class LifetimePass(AnalysisPass):
    """
    The end goal of this pass is to facilitate sequencer allocation on the control hardware. Much like classical
    register allocation techniques, this pass is intended to perform "channel liveness" analysis.

    A logical channel is alive at some point P1 in the builder (analogically to a classical program) if it is
    targeted by some quantum operation at some point P2 > P1 in the future relative to P1.

    Example:

    P1:     |-- builder.pulse(pulse_channel1)
            |   ...
    P2:     |   builder.pulse(pulse_channel2) --|
            |   ...                             |
    P3:     |-- builder.pulse(pulse_channel1)   |
                ...                             |
    P4:         builder.pulse(pulse_channel2) --|

    pulse_channel1 is alive at P1 and P2, pulse_channel2 is alive at P2 and P3. Notice how the lifetimes of
    the channels overlap and "interfere".

    Knowledge of channel/target liveness (with full awareness of control flow) is invaluable for understanding
    physical allocation requirements on the control stack. This is achieved via an interference graph which
    allows allocation to be represented as a graph coloring.

    With this in mind, this pass spits out a colored interference graph that will be used by the code generator.
    """

    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        pass
