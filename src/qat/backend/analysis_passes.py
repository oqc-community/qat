import itertools
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from compiler_config.config import InlineResultsProcessing

from qat.backend.graph import ControlFlowGraph
from qat.ir.pass_base import AnalysisPass, ResultManager
from qat.ir.result_base import ResultInfoMixin
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel, PulseShapeType
from qat.purr.compiler.instructions import (
    Acquire,
    Assign,
    DeviceUpdate,
    EndRepeat,
    EndSweep,
    Instruction,
    PostProcessing,
    Pulse,
    QuantumInstruction,
    Repeat,
    ResultsProcessing,
    Return,
    Sweep,
    SweepValue,
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
        reads: Dict[PulseChannel, Set[str]] = defaultdict(set)
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

            if isinstance(inst, DeviceUpdate):
                reads[inst.target].add(inst.value.name)

        result = TriageResult()
        for inst in builder.instructions:
            # Dissect by target
            if isinstance(inst, QuantumInstruction):
                for qt in inst.quantum_targets:
                    if isinstance(qt, Acquire):
                        for aqt in qt.quantum_targets:
                            result.target_map[aqt].append(inst)
                    else:
                        result.target_map[qt].append(inst)
            elif isinstance(inst, Sweep):
                count = len(next(iter(inst.variables.values())))
                for t in targets:
                    variables = [
                        SweepValue(name, value)
                        for name, value in inst.variables.items()
                        if name in reads[t]
                    ] or [
                        SweepValue(
                            f"sweep_{hash(inst)}", np.linspace(1, count, count, dtype=int)
                        )
                    ]
                    result.target_map[t].append(Sweep(variables))
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
    scope2symbols: Dict[Tuple[Instruction, Optional[Instruction]], Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    symbol2scopes: Dict[str, List[Tuple[Instruction, Optional[Instruction]]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    iter_bounds: Dict[PulseChannel, Dict[str, IterBound]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    reads: Dict[str, List[QuantumInstruction]] = field(
        default_factory=lambda: defaultdict(list)
    )
    writes: Dict[str, List[Instruction]] = field(default_factory=lambda: defaultdict(list))


class BindingPass(AnalysisPass):
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

    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        Builds binding of variables, instructions, and a view of variables from/to scopes.

        Variables are implicitly declared in sweep instructions and are ultimately read from quantum
        instructions. Thus, every iteration variable is associated to all the scopes it is declared in.

        Values of the iteration variable are abstract and don't mean anything. In this pass, we only extract
        the bound and associate it to the name variable. Further analysis is required on read sites to make
        sure their usage is consistent and meaningful.
        """

        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        result = BindingResult()
        stack = []
        for inst in builder.instructions:
            parent_scopes = [(s, e) for (s, e) in result.scope2symbols if s in stack]
            if isinstance(inst, Sweep):
                stack.append(inst)
                scope = (inst, None)
                if p := next(iter(parent_scopes[::-1]), None):
                    for name in result.scope2symbols[p]:
                        if name not in result.scope2symbols[scope]:
                            result.scope2symbols[scope].add(name)
                            result.symbol2scopes[name].append(scope)

                for name, value in inst.variables.items():
                    result.scope2symbols[scope].add(name)
                    result.symbol2scopes[name].append(scope)

                    for t in triage_result.target_map:
                        result.iter_bounds[t][name] = self.extract_iter_bound(value)

                    result.writes[name].append(inst)
            elif isinstance(inst, Repeat):
                stack.append(inst)
                scope = (inst, None)
                if p := next(iter(parent_scopes[::-1]), None):
                    for name in result.scope2symbols[p]:
                        if name not in result.scope2symbols[scope]:
                            result.scope2symbols[scope].add(name)
                            result.symbol2scopes[name].append(scope)

                # TODO - Desugaring could be much better when the IR is redesigned
                name = f"repeat_{inst.repeat_count}_{hash(inst)}"
                result.scope2symbols[scope].add(name)
                result.symbol2scopes[name].append(scope)

                for t in triage_result.target_map:
                    result.iter_bounds[t][name] = IterBound(
                        start=1, step=1, end=inst.repeat_count, count=inst.repeat_count
                    )

                result.writes[name].append(inst)
            elif isinstance(inst, (EndSweep, EndRepeat)):
                delimiter_type = Sweep if isinstance(inst, EndSweep) else Repeat
                try:
                    delimiter = stack.pop()
                except IndexError:
                    raise ValueError(f"Unbalanced scope. Found orphan {inst}")

                if not isinstance(delimiter, delimiter_type):
                    raise ValueError(f"Unbalanced scope. Found orphan {inst}")

                scope = next(
                    ((s, e) for (s, e) in result.scope2symbols.keys() if s == delimiter)
                )
                symbols = result.scope2symbols[scope]
                del result.scope2symbols[scope]
                result.scope2symbols[(delimiter, inst)] = symbols

                for name, scopes in result.symbol2scopes.items():
                    result.symbol2scopes[name] = [
                        (delimiter, inst) if s == scope else s for s in scopes
                    ]
            elif isinstance(inst, Acquire):
                result.writes[inst.output_variable].append(inst)
            elif isinstance(inst, DeviceUpdate):
                if not (
                    inst.value.name in result.symbol2scopes
                    and [
                        (s, e)
                        for (s, e) in result.symbol2scopes[inst.value.name]
                        if s in stack
                    ]
                ):
                    raise ValueError(
                        f"Variable {inst.value} referenced but no prior declaration found"
                    )
                result.reads[inst.value.name].append(inst)
        if stack:
            raise ValueError(f"Unbalanced scopes. Found orphans {stack}")

        res_mgr.add(result)


class TILegalisationPass(AnalysisPass):
    """
    An instruction is legal if it has a direct equivalent in the programming model implemented by the control
    stack. The notion of "legal" is highly determined by the hardware features of the control stack as well
    as its programming model. Control stacks such as Qblox have a direct ISA-level representation for basic
    RF instructions such as frequency and phase manipulation, arithmetic instructions such as add,
    and branching instructions such as jump.

    This pass performs target-independent legalisation. The goal here is to understand how variables
    are used and legalise their bounds. Furthermore, analysis in this pass is fundamentally based on QAT
    semantics and must be kept target-agnostic so that it can be reused among backends.

    Particularly in QAT:
        1) A sweep instruction is illegal because it specifies unclear iteration semantics.
        2) Device updates/assigns in general are illegal because they are bound to a sweep instruction
    via a variable. In fact, a variable (implicitly defined by a Sweep instruction) remains obscure
    until a "read" (usually on the instruction builder or on the hardware model) (typically from
    a DeviceUpdate instruction) is encountered where its intent becomes clear. We say that a DeviceUpdate
    carries meaning for the variable and materialises its intention.
    """

    @staticmethod
    def decompose_freq(frequency: float, target: PulseChannel):
        if target.fixed_if:  # NCO freq constant
            nco_freq = target.baseband_if_frequency
            lo_freq = frequency - nco_freq
        else:  # LO freq constant
            lo_freq = target.baseband_frequency
            nco_freq = frequency - lo_freq

        return lo_freq, nco_freq

    def _legalise_bound(self, name: str, bound: IterBound, inst: Instruction):
        if isinstance(inst, DeviceUpdate):
            if inst.attribute == "frequency":
                if inst.target.fixed_if:
                    raise ValueError(
                        f"fixed_if must be False on target {inst.target} to sweep over frequencies"
                    )
                return IterBound(
                    start=self.decompose_freq(bound.start, inst.target)[1],
                    step=bound.step,
                    end=self.decompose_freq(bound.end, inst.target)[1],
                    count=bound.count,
                )
            elif inst.attribute == "phase":
                return bound
            else:
                raise NotImplementedError(
                    f"Unsupported processing of attribute {inst.attribute}"
                )
        elif isinstance(inst, Pulse):
            if inst.shape != PulseShapeType.SQUARE:
                raise ValueError("Cannot process non-trivial pulses")

            attribute = next(
                (
                    attr
                    for attr, var in inst.__dict__.items()
                    if isinstance(var, Variable) and var.name == name
                )
            )
            if attribute == "width":
                return bound
            elif attribute == "amp":
                return bound
            else:
                raise NotImplementedError(
                    f"Unsupported processing of {attribute} for instruction {inst}"
                )
        else:
            raise NotImplementedError(
                f"Legalisation only supports DeviceUpdate and Pulse. Got {type(inst)} instead"
            )

    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        binding_result: BindingResult = res_mgr.lookup_by_type(BindingResult)
        legal_iter_bounds: Dict[PulseChannel, Dict[str, IterBound]] = deepcopy(
            binding_result.iter_bounds
        )

        read_iter_bounds: Dict[PulseChannel, Dict[str, Set[IterBound]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for name, instructions in binding_result.reads.items():
            for inst in instructions:
                for target in inst.quantum_targets:
                    bound = binding_result.iter_bounds[target][name]
                    legal_bound = self._legalise_bound(name, bound, inst)
                    read_iter_bounds[target][name].add(legal_bound)

        for target, bounds in binding_result.iter_bounds.items():
            if target in read_iter_bounds:
                for name, bound_set in read_iter_bounds[target].items():
                    if len(bound_set) > 1:
                        raise ValueError(
                            f"Found multiple different uses for variable {name}"
                        )
                    legal_iter_bounds[target][name] = next(iter(bound_set))
            else:
                legal_iter_bounds[target] = {
                    name: IterBound(start=1, step=1, end=bound.count, count=bound.count)
                    for name, bound in bounds.items()
                }

        # TODO: the proper way is to produce a new result and invalidate the old one
        binding_result.iter_bounds = legal_iter_bounds


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
