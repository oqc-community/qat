# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import itertools
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from compiler_config.config import InlineResultsProcessing

from qat.purr.backends.qblox.constants import Constants
from qat.purr.backends.qblox.graph import ControlFlowGraph
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel, PulseShapeType
from qat.purr.compiler.instructions import (
    Acquire,
    Assign,
    Delay,
    DeviceUpdate,
    EndRepeat,
    EndSweep,
    Instruction,
    PostProcessing,
    PostProcessType,
    ProcessAxis,
    Pulse,
    QuantumInstruction,
    Repeat,
    ResultsProcessing,
    Return,
    Sweep,
    Variable,
    calculate_duration,
)
from qat.purr.core.pass_base import AnalysisPass
from qat.purr.core.result_base import ResultInfoMixin, ResultManager
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


@dataclass
class TriageResult(ResultInfoMixin):
    sweeps: List[Sweep] = field(default_factory=list)
    repeats: List[Repeat] = field(default_factory=list)
    returns: List[Return] = field(default_factory=list)
    assigns: List[Assign] = field(default_factory=list)
    target_map: Dict[PulseChannel, List[Instruction]] = field(
        default_factory=lambda: defaultdict(list)
    )
    device_updates: List[DeviceUpdate] = field(default_factory=list)
    quantum_instructions: List[QuantumInstruction] = field(default_factory=list)
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

        result = TriageResult()
        for inst in ir.instructions:
            # Dissect by target
            if isinstance(inst, QuantumInstruction):
                result.quantum_instructions.append(inst)
                for qt in inst.quantum_targets:
                    if isinstance(qt, Acquire):
                        for aqt in qt.quantum_targets:
                            result.target_map[aqt].append(inst)
                    else:
                        result.target_map[qt].append(inst)
            elif isinstance(inst, Sweep | EndSweep | Repeat | EndRepeat):
                for t in targets:
                    result.target_map[t].append(inst)

            # Sweeps
            if isinstance(inst, Sweep):
                result.sweeps.append(inst)

            # Repeats
            elif isinstance(inst, Repeat):
                result.repeats.append(inst)

            # DeviceUpdates
            elif isinstance(inst, DeviceUpdate):
                result.device_updates.append(inst)

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
    inits: Dict[str, List[Instruction]] = field(default_factory=lambda: defaultdict(list))
    reads: Dict[str, List[Instruction]] = field(default_factory=lambda: defaultdict(list))
    writes: Dict[str, List[Instruction]] = field(default_factory=lambda: defaultdict(list))


@dataclass(frozen=True)
class IterBound:
    start: Union[int, float, complex] = None
    step: Union[int, float, complex] = None
    end: Union[int, float, complex] = None
    count: int = None

    def astype(self, ty: type):
        return IterBound(
            start=np.array([self.start], dtype=type(self.start)).view(ty)[0],
            step=np.array([self.step], dtype=type(self.step)).view(ty)[0],
            end=np.array([self.end], dtype=type(self.end)).view(ty)[0],
            count=self.count,
        )


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

        triage_result = res_mgr.lookup_by_type(TriageResult)
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

                    for name, value in inst.variables.items():
                        scoping_result.scope2symbols[scope].add(name)
                        scoping_result.symbol2scopes[name].append(scope)
                        iter_bound_result[name] = self.extract_iter_bound(value)

                    iter_name = f"{hash(inst)}"
                    rw_result.inits[iter_name].append(inst)
                    rw_result.reads[iter_name].append(inst)
                    rw_result.writes[iter_name].append(inst)
                elif isinstance(inst, Repeat):
                    stack.append(inst)
                    scope = (inst, None)
                    if p := next(iter(parent_scopes[::-1]), None):
                        for name in scoping_result.scope2symbols[p]:
                            if name not in scoping_result.scope2symbols[scope]:
                                scoping_result.scope2symbols[scope].add(name)
                                scoping_result.symbol2scopes[name].append(scope)

                    iter_name = f"{hash(inst)}"
                    scoping_result.scope2symbols[scope].add(iter_name)
                    scoping_result.symbol2scopes[iter_name].append(scope)
                    iter_bound_result[iter_name] = IterBound(
                        start=1, step=1, end=inst.repeat_count, count=inst.repeat_count
                    )

                    iter_name = f"{hash(inst)}"
                    rw_result.inits[iter_name].append(inst)
                    rw_result.reads[iter_name].append(inst)
                    rw_result.writes[iter_name].append(inst)
                elif isinstance(inst, (EndSweep, EndRepeat)):
                    try:
                        delimiter = stack.pop()
                    except IndexError:
                        raise ValueError(f"Unbalanced scope. Found orphan {inst}")

                    delimiter_type = Sweep if isinstance(inst, EndSweep) else Repeat
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
                    if any(
                        pp.process == PostProcessType.MEAN
                        and ProcessAxis.SEQUENCE in pp.axes
                        for pp in triage_result.pp_map[inst.output_variable]
                    ):
                        # Discard scopes defined by Repeat instructions
                        parent_scopes = [
                            (s, e) for (s, e) in parent_scopes if not isinstance(s, Repeat)
                        ]

                    iter_name = f"{hash(inst)}"
                    shape = tuple(
                        iter_bound_result[f"{hash(s)}"].count for (s, e) in parent_scopes
                    )
                    loop_nest_size = reduce(mul, shape, 1)
                    iter_bound_result[iter_name] = IterBound(
                        start=0, step=1, end=loop_nest_size, count=loop_nest_size
                    )

                    iter_name = f"{hash(inst)}"
                    rw_result.reads[iter_name].append(inst)
                    if (
                        innermost := next((s for (s, e) in parent_scopes[::-1]), None)
                    ) is not None:
                        rw_result.writes[iter_name].append(innermost)
                else:
                    attr2var = {
                        attr: var
                        for attr, var in vars(inst).items()
                        if isinstance(var, Variable)
                    }

                    for attr, var in attr2var.items():
                        defining_sweep = next(
                            (
                                s
                                for s in stack[::-1]
                                if isinstance(s, Sweep) and var.name in s.variables
                            ),
                            None,
                        )
                        if defining_sweep is None:
                            raise ValueError(
                                f"Variable {var} referenced but no prior declaration found in target {target}"
                            )

                        rw_result.inits[var.name].append(defining_sweep)
                        rw_result.reads[var.name].append(inst)
                        rw_result.writes[var.name].append(defining_sweep)

            if len(stack) > 0:
                raise ValueError(
                    f"Unbalanced scopes. Found orphans {stack} in target {target}"
                )

        res_mgr.add(result)


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

    @staticmethod
    def transform_amp(amp: float, scale_factor: float, ignore_scale, target: PulseChannel):
        bias = target.bias
        scale = (1.0 if ignore_scale else target.scale) + 0.0j
        pulse_amp = scale * (scale_factor * amp) + bias

        if abs(pulse_amp.real) > 1 or abs(pulse_amp.imag) > 1:
            raise ValueError("Illegal DAC/ADC ratio. It must be within range [-1, 1]")

        if pulse_amp.imag != 0:
            raise NotImplementedError("Unsupported processing of complex amplitudes")

        return pulse_amp.real

    def _legalise_bound(self, name: str, bound: IterBound, inst: Instruction):
        legal_bound = bound

        if isinstance(inst, Delay):
            legal_bound = IterBound(
                start=int(calculate_duration(Delay(inst.quantum_targets, bound.start))),
                step=int(calculate_duration(Delay(inst.quantum_targets, bound.step))),
                end=int(calculate_duration(Delay(inst.quantum_targets, bound.end))),
                count=bound.count,
            )
        elif isinstance(inst, DeviceUpdate):
            if inst.attribute not in ["frequency", "scale", "amp"]:
                raise NotImplementedError(
                    f"Unsupported processing of attribute {inst.attribute} for instruction {inst}"
                )

            if inst.attribute == "frequency":
                if inst.target.fixed_if:
                    raise ValueError(
                        f"fixed_if must be False on target {inst.target} to sweep over frequencies"
                    )
                legal_bound = IterBound(
                    start=self.decompose_freq(bound.start, inst.target)[1],
                    step=bound.step,
                    end=self.decompose_freq(bound.end, inst.target)[1],
                    count=bound.count,
                )
        else:
            attr2var = {
                attr: var
                for attr, var in vars(inst).items()
                if isinstance(var, Variable) and var.name == name
            }

            if not attr2var:
                return legal_bound
            if len(attr2var) > 1:
                raise ValueError(
                    f"Unsafe analysis. Distinct attributes expecting the same variable bound {attr2var}"
                )
            if isinstance(inst, Pulse) and inst.shape != PulseShapeType.SQUARE:
                raise ValueError("Cannot process non-trivial pulses")
            attr, var = next(iter(attr2var.items()))
            if attr not in ["width", "amp", "phase"]:
                raise NotImplementedError(
                    f"Unsupported processing of attribute {attr} for instruction {inst}"
                )

            target: PulseChannel = next(iter(inst.quantum_targets))
            if attr == "width":
                legal_bound = IterBound(
                    start=int(calculate_duration(Delay(target, bound.start))),
                    step=int(calculate_duration(Delay(target, bound.step))),
                    end=int(calculate_duration(Delay(target, bound.end))),
                    count=bound.count,
                )
            elif attr == "amp" and isinstance(inst, Pulse):
                legal_bound = IterBound(
                    start=self.transform_amp(
                        bound.start,
                        inst.scale_factor,
                        inst.ignore_channel_scale,
                        target,
                    ),
                    step=self.transform_amp(
                        bound.step, inst.scale_factor, inst.ignore_channel_scale, target
                    ),
                    end=self.transform_amp(
                        bound.end, inst.scale_factor, inst.ignore_channel_scale, target
                    ),
                    count=bound.count,
                )

        return legal_bound

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to save the analysis results.
        """

        triage_result = res_mgr.lookup_by_type(TriageResult)
        binding_result = res_mgr.lookup_by_type(BindingResult)

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


class QbloxLegalisationPass(AnalysisPass):
    @staticmethod
    def phase_as_steps(phase_rad: float) -> int:
        """
        The instruction `set_ph_delta` expects the phase shift as a (potentially signed) integer operand.
        """

        phase_deg = np.rad2deg(phase_rad)
        phase_deg %= 360
        steps = int(round(phase_deg * Constants.NCO_PHASE_STEPS_PER_DEG))
        return steps

    @staticmethod
    def freq_as_steps(freq_hz: float) -> int:
        """
        The instruction `set_freq` expects the frequency as a (potentially signed) integer operand.
        """

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

    def amp_as_steps(self, amp: float) -> int:
        """
        The instruction `set_awg_offs` expects DAC ratio as a (potentially signed) integer operand.
        """

        amp_steps = int(amp.real * Constants.MAX_OFFSET)
        if amp_steps < Constants.MIN_OFFSET or amp_steps > Constants.MAX_OFFSET:
            raise ValueError(
                f"""
                Expected offset to be in range [{Constants.MIN_OFFSET}, {Constants.MAX_OFFSET}].
                Got {amp_steps} instead
                """
            )

        return amp_steps

    def _legalise_bound(self, name: str, bound: IterBound, inst: Instruction):
        legal_bound = bound

        if isinstance(inst, Acquire):
            num_bins = bound.count
            if num_bins > Constants.MAX_012_BINNED_ACQUISITIONS:
                raise ValueError(
                    f"""
                        Loop nest size would require {num_bins} acquisition memory bins which exceeds the maximum {Constants.MAX_012_BINNED_ACQUISITIONS}.
                        Please reduce number of points
                        """
                )
        elif isinstance(inst, Delay):
            if bound.start < Constants.GRID_TIME:
                log.warning(
                    f"Undefined runtime behaviour. Variable {name} has illegal lower bound"
                )
            if bound.end > Constants.MAX_WAIT_TIME:
                log.warning(
                    f"Undefined runtime behaviour. Will be batching variable {name} at runtime"
                )
        elif isinstance(inst, DeviceUpdate):
            if inst.attribute not in ["frequency", "phase"]:
                raise NotImplementedError(
                    f"Unsupported processing of attribute {inst.attribute} for instruction {inst}"
                )

            if inst.attribute == "frequency":
                legal_bound = IterBound(
                    start=self.freq_as_steps(bound.start),
                    step=self.freq_as_steps(bound.step),
                    end=self.freq_as_steps(bound.end),
                    count=bound.count,
                )
            elif inst.attribute == "phase":
                legal_bound = IterBound(
                    start=self.phase_as_steps(bound.start),
                    step=self.phase_as_steps(bound.step),
                    end=self.phase_as_steps(bound.end),
                    count=bound.count,
                )
        else:
            attr2var = {
                attr: var
                for attr, var in vars(inst).items()
                if isinstance(var, Variable) and var.name == name
            }

            if attr2var and isinstance(inst, Pulse) and inst.shape != PulseShapeType.SQUARE:
                raise ValueError("Cannot process non-trivial pulses")
            if not attr2var:
                return legal_bound
            if len(attr2var) > 1:
                raise ValueError(
                    f"Unsafe analysis. Distinct attributes expecting the same variable bound {attr2var}"
                )
            attr, var = next(iter(attr2var.items()), (None, None))
            if attr not in ["width", "amp", "phase"]:
                raise NotImplementedError(
                    f"Unsupported processing of attribute {attr} for instruction {inst}"
                )

            if attr == "width":
                if bound.start < Constants.GRID_TIME:
                    log.warning(
                        f"Undefined runtime behaviour. Variable {name} has illegal lower bound"
                    )
                if bound.end > Constants.MAX_WAIT_TIME:
                    log.warning(
                        f"Undefined runtime behaviour. Will be batching variable {name} at runtime"
                    )
            elif attr == "amp" and isinstance(inst, Pulse):
                legal_bound = IterBound(
                    start=self.amp_as_steps(bound.start),
                    step=self.amp_as_steps(bound.step),
                    end=self.amp_as_steps(bound.end),
                    count=bound.count,
                )
            elif attr == "phase":
                legal_bound = IterBound(
                    start=self.phase_as_steps(bound.start),
                    step=self.phase_as_steps(bound.step),
                    end=self.phase_as_steps(bound.end),
                    count=bound.count,
                )

        for attr, val in vars(legal_bound).items():
            if not isinstance(val, int):
                raise ValueError(
                    f"Illegal value {val} for attribute {attr}. Expected value to be an integer"
                )

        # Qblox registers are unsigned 32bit integers.
        return legal_bound.astype(np.uint32)

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        Performs target-dependent legalisation for QBlox.

            A) A repeat instruction with a very high repetition count is illegal because acquisition memory
        on a QBlox sequencer is limited. This requires optimal batching of the repeat instruction into maximally
        supported batches of smaller repeat counts.

        This pass does not do any batching. More features and adjustments will follow in future iterations.

            B) Previously processed variables such as frequencies, phases, and amplitudes still need digital conversion
        to a representation that's required by the QBlox ISA.

        + NCO's 1GHz frequency range by 4e9 steps:
            + [-500, 500] Mhz <=> [-2e9, 2e9] steps
            + 1 Hz            <=> 4 steps
        + NCO's 360° phase range by 1e9 steps:
            + 1e9    steps    <=> 2*pi rad
            + 125e6  steps    <=> pi/4 rad
        + Time and samples are quantified in nanoseconds
        + Amplitude depends on the type of the module:
            + [-1, 1]         <=> [-2.5, 2.5] V (for QCM)
            + [-1, 1]         <=> [-0.5, 0.5] V (for QRM)
        + AWG offset:
            + [-1, 1]         <=> [-32 768, 32 767]

        The last point is interesting as it requires knowledge of physical configuration of qubits and the modules
        they are wired to. This knowledge is typically found during execution and involving it early on would upset
        the rest of the compilation flow. In fact, it complicates this pass in particular, involves allocation
        concepts that should not be treated here, and promotes a monolithic compilation style. A temporary workaround
        is to simply assume the legality of amplitudes from the start whereby users are required to convert
        the desired voltages to the equivalent ratio AOT.

        This pass performs target-dependent conversion as described in part (B). More features and adjustments
        will follow in future iterations.
        """

        triage_result = res_mgr.lookup_by_type(TriageResult)
        binding_result = res_mgr.lookup_by_type(BindingResult)

        for target in triage_result.target_map:
            rw_result = binding_result.rw_results[target]
            bound_result = binding_result.iter_bound_results[target]
            legal_bound_result: Dict[str, IterBound] = deepcopy(bound_result)

            qblox_bounds: Dict[str, Set[IterBound]] = defaultdict(set)
            for name, instructions in rw_result.reads.items():
                for inst in instructions:
                    legal_bound = self._legalise_bound(name, bound_result[name], inst)
                    qblox_bounds[name].add(legal_bound)

            for name, bound in bound_result.items():
                if name in qblox_bounds:
                    bound_set = qblox_bounds[name]
                    if len(bound_set) > 1:
                        raise ValueError(
                            f"Ambiguous Qblox bounds for variable {name} in target {target}"
                        )
                    legal_bound_result[name] = next(iter(bound_set))

            # TODO: the proper way is to produce a new result and invalidate the old one
            binding_result.iter_bound_results[target] = legal_bound_result


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
