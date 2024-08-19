import itertools
from collections import defaultdict
from typing import List, Union

import numpy as np

from qat.ir.pass_base import AnalysisPass, PassResultSet
from qat.purr.backends.codegen import CodegenResultType
from qat.purr.backends.graph import ControlFlowGraph
from qat.purr.backends.qblox.constants import Constants
from qat.purr.backends.qblox.instructions import EndRepeat, EndSweep
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.config import InlineResultsProcessing
from qat.purr.compiler.instructions import (
    Acquire,
    Assign,
    DeviceUpdate,
    PostProcessing,
    QuantumInstruction,
    Repeat,
    ResultsProcessing,
    Return,
    Sweep,
    Variable,
)


class TriagePass(AnalysisPass):
    def run(self, builder: InstructionBuilder, *args, **kwargs):
        """
        Builds a view of instructions per quantum targets AOT.
        Builds selections of instructions useful for subsequent analysis/transform passes,
        for code generation and post-playback steps.
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

        sweeps = []
        returns = []
        assigns = []
        target_view = defaultdict(list)
        acquire_map = defaultdict(list)
        pp_map = defaultdict(list)
        rp_map = dict()
        for inst in builder.instructions:
            # Sweeps
            if isinstance(inst, Sweep):
                sweeps.append(inst)

            # Returns
            if isinstance(inst, Return):
                returns.append(inst)

            # Assigns
            if isinstance(inst, Assign):
                assigns.append(inst)

            # Instructions by target
            if isinstance(inst, QuantumInstruction):
                for qt in inst.quantum_targets:
                    if isinstance(qt, Acquire):
                        for aqt in qt.quantum_targets:
                            target_view[aqt].append(inst)
                    else:
                        target_view[qt].append(inst)
            else:
                for t in targets:
                    target_view[t].append(inst)

            # Acquisition by target
            if isinstance(inst, Acquire):
                for t in inst.quantum_targets:
                    acquire_map[t].append(inst)

            # Post-processing by output variable
            if isinstance(inst, PostProcessing):
                pp_map[inst.acquire.output_variable].append(inst)

            # Results-processing by output variable
            if isinstance(inst, ResultsProcessing):
                rp_map[inst.variable] = inst

        # Assume that raw acquisitions are experiment results.
        acquires = list(itertools.chain(*acquire_map.values()))
        missing_results = {
            acq.output_variable for acq in acquires if acq.output_variable not in rp_map
        }
        for missing_var in missing_results:
            rp_map[missing_var] = ResultsProcessing(
                missing_var, InlineResultsProcessing.Experiment
            )

        analyses: PassResultSet = args[0] if args else PassResultSet()
        analyses.update(
            PassResultSet(
                (hash(builder), self.id(), CodegenResultType.SWEEPS, sweeps),
                (
                    hash(builder),
                    self.id(),
                    CodegenResultType.RETURN,
                    returns[
                        0
                    ],  # See ReturnSanitisation and ReturnSanitisationValidation passes
                ),
                (hash(builder), self.id(), CodegenResultType.ASSIGNS, assigns),
                (hash(builder), self.id(), CodegenResultType.TARGET_VIEW, target_view),
                (hash(builder), self.id(), CodegenResultType.ACQUIRE_MAP, acquire_map),
                (hash(builder), self.id(), CodegenResultType.PP_MAP, pp_map),
                (hash(builder), self.id(), CodegenResultType.RP_MAP, rp_map),
            )
        )
        return analyses


class VariableBoundsPass(AnalysisPass):

    @staticmethod
    def get_if_freq(target):
        if target.fixed_if:
            if_freq = target.baseband_if_frequency
        else:
            if_freq = target.frequency - target.baseband_frequency
        return if_freq

    @staticmethod
    def get_baseband_freq(target):
        return target.frequency - VariableBoundsPass.get_if_freq(target)

    @staticmethod
    def extract_variable_bounds(value: Union[List, np.ndarray]):
        """
        Returns a tuple (start, step, end, count) if the value is linearly/evenly spaced or fails otherwise.
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

        if count >= 2:
            step = value[1] - value[0]

        if not np.isclose(step, (end - start) / (count - 1)):
            raise ValueError(f"Not a regularly partitioned space {value}")

        return start, step, end, count

    @staticmethod
    def phase_as_steps(phase_rad: float) -> int:
        phase_deg = np.rad2deg(phase_rad)
        phase_deg %= 360
        return round(phase_deg * Constants.NCO_PHASE_STEPS_PER_DEG)

    @staticmethod
    def freq_as_steps(if_freq: float) -> int:
        steps = round(if_freq * Constants.NCO_FREQ_STEPS_PER_HZ)

        if (
            steps < -Constants.NCO_FREQ_LIMIT_STEPS
            or steps > Constants.NCO_FREQ_LIMIT_STEPS
        ):
            min_max_frequency_in_hz = (
                Constants.NCO_FREQ_LIMIT_STEPS / Constants.NCO_FREQ_STEPS_PER_HZ
            )
            raise ValueError(
                f"IF frequency must be in [-{min_max_frequency_in_hz:e}, {min_max_frequency_in_hz:e}] Hz. "
                f"Got {if_freq:e} Hz"
            )

        return steps

    def run(self, builder: InstructionBuilder, *args, **kwargs):
        analyses: PassResultSet = args[0]
        target_view = analyses.get_result(CodegenResultType.TARGET_VIEW)

        variable_bounds = {t: dict() for t in target_view}
        materialised_bounds = {t: defaultdict(list) for t in target_view}
        for inst in builder.instructions:
            if isinstance(inst, Sweep):
                name, value = next(iter(inst.variables.items()))
                start, step, end, count = self.extract_variable_bounds(value)
                for t in target_view:
                    variable_bounds[t][name] = (start, step, end, count)
            elif isinstance(inst, DeviceUpdate) and isinstance(inst.value, Variable):
                if inst.value.name not in variable_bounds[inst.target]:
                    raise ValueError(
                        f"Iterator variable {inst.value.name} referenced but no prior declaration found"
                    )
                bounds = variable_bounds[inst.target][inst.value.name]
                if inst.attribute == "frequency":
                    bb_freq = self.get_baseband_freq(inst.target)
                    materialised_bounds[inst.target][inst.value.name].append(
                        (
                            self.freq_as_steps(bounds[0] - bb_freq),
                            self.freq_as_steps(bounds[1]),
                            self.freq_as_steps(bounds[2] - bb_freq),
                            bounds[3],
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"Unsupported processing of attribute {inst.attribute}"
                    )

        for t in target_view:
            for name, bounds in materialised_bounds[t].items():
                if len(set(bounds)) != 1:
                    raise ValueError(f"Inconsistent use of variable {name} for target {t}")
                variable_bounds[t][name] = bounds[0]

        analyses.update(
            PassResultSet(
                (
                    hash(builder),
                    self.id(),
                    CodegenResultType.VARIABLE_BOUNDS,
                    variable_bounds,
                )
            )
        )
        return analyses


class CFGPass(AnalysisPass):

    def run(self, builder: InstructionBuilder, *args, **kwargs):
        cfg = ControlFlowGraph()
        self._build_cfg(builder, cfg)
        analyses: PassResultSet = args[0] if args else PassResultSet()
        analyses.update(
            PassResultSet((hash(builder), self.id(), CodegenResultType.CFG, cfg))
        )
        return analyses

    def _build_cfg(self, builder: InstructionBuilder, cfg: ControlFlowGraph):
        """
        Recursively (re)discovers (new) header nodes and flow information.
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


class CtrlHwPass(AnalysisPass):
    """
    Perform analyses such as:
    - Lowerability: What can be run on a control hardware stack
    - Batching: After all loop analysis, how many levels or a loop nest can run on the control hardware
    """

    def run(self, builder: InstructionBuilder, *args, **kwargs):
        pass


class TimelinePass(AnalysisPass):
    """
    Performs analyses necessary for dynamic allocation of control hardware resources to logical channels.
    Loosely speaking, it aims at understanding when exactly instructions are invoked (with **full awareness**
    of control flow (especially loops)) and whether prior resources could be freed up.
    """

    def run(self, builder: InstructionBuilder, *args, **kwargs):
        pass
