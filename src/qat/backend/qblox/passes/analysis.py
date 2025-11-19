# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from bisect import insort
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain

import numpy as np

from qat.backend.passes.purr.analysis import BindingResult, IterBound, TriageResult
from qat.backend.qblox.config.constants import Constants
from qat.core.pass_base import AnalysisPass
from qat.core.result_base import ResultInfoMixin, ResultManager
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import PulseChannel, PulseShapeType
from qat.purr.compiler.instructions import (
    Acquire,
    Delay,
    DeviceUpdate,
    Instruction,
    Pulse,
    Variable,
)
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


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
            legal_bound_result: dict[str, IterBound] = deepcopy(bound_result)

            qblox_bounds: dict[str, set[IterBound]] = defaultdict(set)
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
        return ir


@dataclass
class AllocationManager:
    _reg_pool: list[str] = field(
        default_factory=lambda: sorted(
            f"R{index}" for index in range(Constants.NUMBER_OF_REGISTERS)
        )
    )
    _lbl_counters: dict[str, int] = field(default_factory=dict)

    registers: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    def reg_alloc(self, name: str) -> str:
        if name in self.registers:
            log.warning(f"Returning a register already allocated for Variable {name}")
            register = self.registers[name]
        elif len(self._reg_pool) < 1:
            raise IndexError(
                "Out of registers. Attempting to use more registers than available in the Q1 sequence processor"
            )
        else:
            register = self._reg_pool.pop(0)
            self.registers[name] = register

        return register

    def reg_free(self, register: str) -> None:
        if register in self._reg_pool:
            raise RuntimeError(f"Cannot free register '{register}' as it's not in use")
        insort(self._reg_pool, register)
        self.registers = {
            var: reg for var, reg in self.registers.items() if reg != register
        }

    @contextmanager
    def reg_borrow(self, name: str):
        """
        Short-lived register allocation
        """

        register = self.reg_alloc(name)
        yield register
        self.reg_free(register)

    def label_gen(self, name: str):
        counter = self._lbl_counters.setdefault(name, 0)
        self._lbl_counters[name] += 1
        label = f"block_{name}_{counter}"
        self.labels[name] = label
        return label


@dataclass
class PreCodegenResult(ResultInfoMixin):
    alloc_mgrs: dict[PulseChannel, AllocationManager] = field(
        default_factory=lambda: defaultdict(lambda: AllocationManager())
    )


class PreCodegenPass(AnalysisPass):
    """
    Precedes code generation. The context-based emitter needs registers pre-allocated for every
    live variable in program. This pass performs a naive register allocation through a manager object.
    """

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        triage_result = res_mgr.lookup_by_type(TriageResult)
        binding_result = res_mgr.lookup_by_type(BindingResult)
        result = PreCodegenResult()

        for target in triage_result.target_map:
            alloc_mgr = result.alloc_mgrs[target]
            iter_bound_result = binding_result.iter_bound_results[target]
            reads = binding_result.rw_results[target].reads
            writes = binding_result.rw_results[target].writes

            alloc_mgr.reg_alloc("zero")
            names = set(chain(*[iter_bound_result.keys(), reads.keys(), writes.keys()]))
            for name in names:
                alloc_mgr.reg_alloc(name)
                alloc_mgr.label_gen(name)

        res_mgr.add(result)

        return ir
