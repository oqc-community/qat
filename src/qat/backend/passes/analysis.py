# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import itertools
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from compiler_config.config import InlineResultsProcessing

from qat.backend.graph import ControlFlowGraph
from qat.core.pass_base import AnalysisPass, ResultManager
from qat.core.result_base import ResultInfoMixin
from qat.purr.backends.utilities import UPCONVERT_SIGN
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PhysicalChannel, PulseChannel, PulseShapeType
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    Acquire,
    Assign,
    Delay,
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
    Synchronize,
    Variable,
    calculate_duration,
)


@dataclass
class TriageResult(ResultInfoMixin):
    sweeps: List[Sweep] = field(default_factory=list)
    returns: List[Return] = field(default_factory=list)
    assigns: List[Assign] = field(default_factory=list)
    syncs: List[Dict[PulseChannel, int]] = field(default_factory=list)
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
    """
    Builds a view of instructions per quantum target AOT.

    Builds selections of instructions useful for subsequent analysis/transform passes, for
    code generation, and post-playback steps.

    This is equivalent to the :class:`QatFile` and simplifies the duration timeline creation
    in the legacy code.
    """

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to save the analysis results.
        """
        targets = set()
        for inst in ir.instructions:
            if isinstance(inst, QuantumInstruction):
                if isinstance(inst, PostProcessing):
                    for qt in inst.quantum_targets:
                        if isinstance(qt, Acquire):
                            targets.update(qt.quantum_targets)
                        else:
                            targets.add(qt)
                else:
                    targets.update(inst.quantum_targets)

        active_targets = set()
        result = TriageResult()
        for inst in ir.instructions:
            # Dissect by target
            if isinstance(inst, QuantumInstruction):
                if isinstance(inst, Synchronize):
                    result.syncs.append(
                        {qt: len(result.target_map[qt]) for qt in inst.quantum_targets}
                    )
                for qt in inst.quantum_targets:
                    if isinstance(qt, Acquire):
                        for aqt in qt.quantum_targets:
                            result.target_map[aqt].append(inst)
                    else:
                        result.target_map[qt].append(inst)
            elif isinstance(inst, Sweep | EndSweep | Repeat | EndRepeat):
                # TODO: Find a better way to handle scopes with the new instructions
                # Would a `ControlFlowInstruction` class be useful?
                # Are `Label` and `Jump` instructions needed here as well?
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
                    active_targets.add(t)
                    result.acquire_map[t].append(inst)

            elif isinstance(inst, Pulse):
                active_targets.add(inst.channel)

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
        result.target_map = {target: result.target_map[target] for target in active_targets}

        res_mgr.add(result)

        return ir


@dataclass
class ScopingResult:
    scope2symbols: Dict[Tuple[Instruction, Optional[Instruction]], Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    symbol2scopes: Dict[str, List[Tuple[Instruction, Optional[Instruction]]]] = field(
        default_factory=lambda: defaultdict(list)
    )


@dataclass
class ReadWriteResult:
    reads: Dict[str, List[QuantumInstruction]] = field(
        default_factory=lambda: defaultdict(list)
    )
    writes: Dict[str, List[Instruction]] = field(default_factory=lambda: defaultdict(list))


@dataclass(frozen=True)
class IterBound:
    start: Union[int, float] = 0
    step: Union[int, float] = 0
    end: Union[int, float] = 0
    count: int = 0


@dataclass
class BindingResult(ResultInfoMixin):
    scoping_results: Dict[PulseChannel, ScopingResult] = field(
        default_factory=lambda: defaultdict(lambda: ScopingResult())
    )
    rw_results: Dict[PulseChannel, ReadWriteResult] = field(
        default_factory=lambda: defaultdict(lambda: ReadWriteResult())
    )
    iter_bound_results: Dict[PulseChannel, Dict[str, IterBound]] = field(
        default_factory=lambda: defaultdict(dict)
    )


class BindingPass(AnalysisPass):
    """
    Builds binding of variables, instructions, and a view of variables from/to scopes.

    Variables are implicitly declared in sweep instructions and are ultimately read from
    quantum instructions. Thus, every iteration variable is associated to all the scopes it
    is declared in.

    Values of the iteration variable are abstract and don't mean anything. In this pass, we
    only extract  the bound and associate it to the name variable. Further analysis is
    required on read sites to make sure their usage is consistent and meaningful.
    """

    @staticmethod
    def extract_iter_bound(value: Union[List, np.ndarray]):
        """
        Given a sequence of numbers (typically having been generated from
        :code:`np.linspace()`), figure out if the numbers are linearly/evenly spaced,
        in which case returns an IterBound instance holding the start, step, end, and count
        of the numbers in the array, or else fail.

        In the future, we might be interested in relaxing this condition and return
        "interpolated" evenly spaced approximation of the input sequence.
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

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to save the analysis results.
        """

        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        result = BindingResult()
        for target, instructions in triage_result.target_map.items():
            scoping_result = result.scoping_results[target]
            rw_result = result.rw_results[target]
            iter_bound_result = result.iter_bound_results[target]

            stack = []
            for inst in instructions:
                parent_scopes = [
                    (s, e) for (s, e) in scoping_result.scope2symbols if s in stack
                ]
                if isinstance(inst, Sweep):
                    stack.append(inst)
                    scope = (inst, None)
                    if p := next(iter(parent_scopes[::-1]), None):
                        for name in scoping_result.scope2symbols[p]:
                            if name not in scoping_result.scope2symbols[scope]:
                                scoping_result.scope2symbols[scope].add(name)
                                scoping_result.symbol2scopes[name].append(scope)

                    iter_name = f"sweep_{hash(inst)}"
                    count = len(next(iter(inst.variables.values())))
                    scoping_result.scope2symbols[scope].add(iter_name)
                    scoping_result.symbol2scopes[iter_name].append(scope)
                    iter_bound_result[iter_name] = IterBound(
                        start=1, step=1, end=count, count=count
                    )
                    rw_result.writes[iter_name].append(inst)

                    for name, value in inst.variables.items():
                        scoping_result.scope2symbols[scope].add(name)
                        scoping_result.symbol2scopes[name].append(scope)
                        iter_bound_result[name] = self.extract_iter_bound(value)
                elif isinstance(inst, Repeat):
                    stack.append(inst)
                    scope = (inst, None)
                    if p := next(iter(parent_scopes[::-1]), None):
                        for name in scoping_result.scope2symbols[p]:
                            if name not in scoping_result.scope2symbols[scope]:
                                scoping_result.scope2symbols[scope].add(name)
                                scoping_result.symbol2scopes[name].append(scope)

                    iter_name = f"repeat_{hash(inst)}"
                    count = inst.repeat_count
                    scoping_result.scope2symbols[scope].add(iter_name)
                    scoping_result.symbol2scopes[iter_name].append(scope)
                    iter_bound_result[iter_name] = IterBound(
                        start=1, step=1, end=count, count=count
                    )
                    rw_result.writes[iter_name].append(inst)
                elif isinstance(inst, (EndSweep, EndRepeat)):
                    delimiter_type = Sweep if isinstance(inst, EndSweep) else Repeat
                    try:
                        delimiter = stack.pop()
                    except IndexError:
                        raise ValueError(f"Unbalanced scope. Found orphan {inst}")

                    if not isinstance(delimiter, delimiter_type):
                        raise ValueError(f"Unbalanced scope. Found orphan {inst}")

                    scope = next(
                        (
                            (s, e)
                            for (s, e) in scoping_result.scope2symbols.keys()
                            if s == delimiter
                        )
                    )
                    symbols = scoping_result.scope2symbols[scope]
                    del scoping_result.scope2symbols[scope]
                    scoping_result.scope2symbols[(delimiter, inst)] = symbols

                    for name, scopes in scoping_result.symbol2scopes.items():
                        scoping_result.symbol2scopes[name] = [
                            (delimiter, inst) if s == scope else s for s in scopes
                        ]
                elif isinstance(inst, Acquire):
                    sweeps = [
                        (s, iter_bound_result[next(iter(s.variables.keys()))].count)
                        for s in stack
                        if isinstance(s, Sweep)
                    ]

                    # Acquisition mem addressing
                    iter_name = f"acquire_{hash(inst)}"

                    # An acquire reads the "memory index"
                    rw_result.reads[iter_name].append(inst)

                    if sweeps:
                        loop_nest_size = reduce(mul, [c for (_, c) in sweeps], 1)
                        iter_bound_result[iter_name] = IterBound(
                            start=0, step=1, end=loop_nest_size, count=loop_nest_size
                        )

                        # Innermost sweep writes to the "memory index"
                        innermost_sweep, _ = sweeps[-1]
                        rw_result.writes[iter_name].append(innermost_sweep)
                elif isinstance(inst, DeviceUpdate) and isinstance(inst.value, Variable):
                    defining_sweep = next(
                        (
                            s
                            for s in stack[::-1]
                            if isinstance(s, Sweep) and inst.value.name in s.variables
                        ),
                        None,
                    )
                    if not defining_sweep:
                        raise ValueError(
                            f"Variable {inst.value} referenced but no prior declaration found in target {target}"
                        )

                    # Only the defining sweep writes to variable, but only on the target in question !!
                    rw_result.writes[inst.value.name].append(defining_sweep)

                    # A DeviceUpdate reads the variable named "inst.value.name"
                    rw_result.reads[inst.value.name].append(inst)
                elif isinstance(inst, Delay) and isinstance(inst.duration, Variable):
                    defining_sweep = next(
                        (
                            s
                            for s in stack[::-1]
                            if isinstance(s, Sweep) and inst.duration.name in s.variables
                        ),
                        None,
                    )
                    if not defining_sweep:
                        raise ValueError(
                            f"Variable {inst.duration} referenced but no prior declaration found in target {target}"
                        )

                    # Only the defining sweep writes to variable, but only on the target in question !!
                    rw_result.writes[inst.duration.name].append(defining_sweep)

                    # A Delay reads the variable named "inst.duration.name"
                    rw_result.reads[inst.duration.name].append(inst)

            if stack:
                raise ValueError(
                    f"Unbalanced scopes. Found orphans {stack} in target {target}"
                )

        res_mgr.add(result)
        return ir


class TILegalisationPass(AnalysisPass):
    """
    An instruction is legal if it has a direct equivalent in the programming model
    implemented by the control stack. The notion of "legal" is highly determined by the
    hardware features of the control stack as well as its programming model. Control stacks
    such as Qblox have a direct ISA-level representation for basic RF instructions such as
    frequency and phase manipulation, arithmetic instructions such as add,  and branching
    instructions such as jump.

    This pass performs target-independent legalisation. The goal here is to understand how
    variables are used and legalise their bounds. Furthermore, analysis in this pass is
    fundamentally based on QAT semantics and must be kept target-agnostic so that it can be
    reused among backends.

    Particularly in QAT:
    #. A sweep instruction is illegal because it specifies unclear iteration semantics.
    #. Device updates/assigns in general are illegal because they are bound to a sweep
       instruction via a variable. In fact, a variable (implicitly defined by a Sweep
       instruction) remains obscure until a "read" (usually on the instruction builder or on
       the hardware model) (typically from a DeviceUpdate instruction) is encountered where
       its intent becomes clear. We say that a DeviceUpdate carries meaning for the variable
       and materialises its intention.
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
        if isinstance(inst, DeviceUpdate) and isinstance(inst.value, Variable):
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
        elif isinstance(inst, Acquire):
            return bound
        elif isinstance(inst, Delay) and isinstance(inst.duration, Variable):
            return IterBound(
                start=calculate_duration(Delay(inst.quantum_targets, bound.start)),
                step=calculate_duration(Delay(inst.quantum_targets, bound.step)),
                end=calculate_duration(Delay(inst.quantum_targets, bound.end)),
                count=bound.count,
            )
        else:
            raise NotImplementedError(
                f"Legalisation only supports DeviceUpdate and Pulse. Got {type(inst)} instead"
            )

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to save the analysis results.
        """

        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        binding_result: BindingResult = res_mgr.lookup_by_type(BindingResult)

        for target in triage_result.target_map:
            rw_result = binding_result.rw_results[target]
            bound_result = binding_result.iter_bound_results[target]
            legal_bound_result: Dict[str, IterBound] = deepcopy(bound_result)

            read_bounds: Dict[str, Set[IterBound]] = defaultdict(set)
            for name, instructions in rw_result.reads.items():
                for inst in instructions:
                    legal_bound = self._legalise_bound(name, bound_result[name], inst)
                    read_bounds[name].add(legal_bound)

            for name, bound_set in read_bounds.items():
                if len(bound_set) > 1:
                    raise ValueError(
                        f"Ambiguous bounds for variable {name} in target {target}"
                    )
                legal_bound_result[name] = next(iter(bound_set))

            # TODO: the proper way is to produce a new result and invalidate the old one
            binding_result.iter_bound_results[target] = legal_bound_result
        return ir


@dataclass
class CFGResult(ResultInfoMixin):
    cfg: ControlFlowGraph = field(default_factory=lambda: ControlFlowGraph())


class CFGPass(AnalysisPass):
    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to save the analysis results.
        """

        result = CFGResult()
        self._build_cfg(ir, result.cfg)
        res_mgr.add(result)
        return ir

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

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        pass


@dataclass
class TimelineAnalysisResult(ResultInfoMixin):
    durations: Dict[PulseChannel, List[int]] = field(
        default_factory=lambda: defaultdict(list)
    )


class TimelineAnalysis(AnalysisPass):
    """Analyses the durations and timings of instructions for backends that do not natively
    support synchronization across channels.

    During code generation for a given backend, it is often the case that we need to know the
    exactly clock cycles that instructions start, and the duration they happen for. For
    most instructions, this is trivial and is often part of the instruction's definition.
    However, this is less clear for the :class:`Synchronize` instruction, which aims to bring
    all targeted channels to the same clock cycle. Calculating this requires knowledge of
    how much time has elapsed up until this point. Once this is known, delays can be added
    to all other channels to bring them in sync.

    This pass calculates the timings of each instruction by first calculating the timings of
    non-synchronize instructions, and then iteratively adapting for each :class:`Synchronize`.
    """

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to store the analysis results.
        """

        triage_results: TriageResult = res_mgr.lookup_by_type(TriageResult)
        durations = self.calculate_non_sync_durations(triage_results)
        durations = self.calculate_sync_durations(triage_results, durations)

        result = TimelineAnalysisResult()
        result.durations = {
            chan: self._durations_as_samples(chan, dur) for chan, dur in durations.items()
        }

        res_mgr.add(result)
        return ir

    def calculate_non_sync_durations(self, triage_results: TriageResult):
        """Calculates the durations for each instruction that is not a synchronize.
        Durations are rounded to be integer multiples of pulse channel sample times."""

        target_map = triage_results.target_map
        durations = {}
        for channel in target_map:
            channel_durations = np.array(
                [
                    (
                        inst.duration
                        if isinstance(inst, QuantumInstruction)
                        and not isinstance(inst, Synchronize)
                        else 0
                    )
                    for inst in target_map[channel]
                ]
            )
            durations[channel] = self._round_durations_with_samples(
                channel.physical_channel, channel_durations
            )
        return durations

    def calculate_sync_durations(
        self, triage_results: TriageResult, durations: Dict[PulseChannel, List[float]]
    ):
        """Using the durations from all instructions that are not synchronize, this will
        incrementely determine the time for each sweep for each targeted pulse channel."""
        syncs = triage_results.syncs
        for sync in syncs:
            current_durations = {
                chan: np.sum(durations[chan][: sync[chan]])
                for chan in sync
                if chan in durations
            }
            sync_time = max(current_durations.values())
            for chan, current_duration in current_durations.items():
                durations[chan][sync[chan]] = sync_time - current_duration
        return durations

    @staticmethod
    def _round_durations_with_samples(channel: PhysicalChannel, durations: List[float]):
        """The time given by an instruction must be in integer multiples of pulse channel
        sample times. This function returns durations as the next largest multiple."""

        block_numbers = np.ceil(np.round(durations / channel.block_time, decimals=4))
        return block_numbers * channel.block_time

    @staticmethod
    def _durations_as_samples(channel: PhysicalChannel, durations: List[float]):
        """
        Converts a list of durations into a number of samples.
        """

        block_numbers = np.ceil(
            np.round(durations / channel.block_time, decimals=4)
        ).astype(np.int64)
        return block_numbers * channel.block_size


@dataclass
class IntermediateFrequencyResult(ResultInfoMixin):
    frequencies: Dict[PhysicalChannel, float]


class IntermediateFrequencyAnalysis(AnalysisPass):
    """
    Adapted from :meth:`qat.purr.backends.live.LiveDeviceEngine.build_baseband_frequencies`.

    Retrieves intermediate frequencies for all physical channels if they exist,
    and validates that pulse channels that share the same physical channel cannot
    have differing fixed frequencies. This pass should always follow a :class:`TriagePass`,
    as information of pulse channels are needed.
    """

    def __init__(self, model: QuantumHardwareModel):
        """
        Instantiate the pass with a hardware model.

        :param model: The hardware model.
        """

        # TODO: determine if this pass should be split into an analysis and validation
        # pass.
        self.model = model

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to store the analysis results.
        """

        triage_results: TriageResult = res_mgr.lookup_by_type(TriageResult)
        baseband_freqs = {}
        baseband_freqs_fixed_if = {}
        for pulse_channel in triage_results.target_map:
            if pulse_channel.fixed_if:
                baseband_freq = (
                    pulse_channel.frequency
                    - UPCONVERT_SIGN * pulse_channel.baseband_if_frequency
                )
                if (
                    baseband_freqs_fixed_if.get(pulse_channel.physical_channel, False)
                    and baseband_freqs[pulse_channel.physical_channel] != baseband_freq
                ):
                    raise ValueError(
                        "Cannot fix the frequency for two pulse channels of different "
                        "frequencies on the same physical channel!"
                    )
                baseband_freqs[pulse_channel.physical_channel] = baseband_freq
                baseband_freqs_fixed_if[pulse_channel.physical_channel] = (
                    pulse_channel.fixed_if
                )
            else:
                if (
                    pulse_channel.physical_channel not in baseband_freqs_fixed_if
                    or not baseband_freqs_fixed_if[pulse_channel.physical_channel]
                ):
                    baseband_freqs_fixed_if[pulse_channel.physical_channel] = (
                        pulse_channel.fixed_if
                    )

        res_mgr.add(IntermediateFrequencyResult(frequencies=baseband_freqs))
        return ir
