import numpy as np

from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.instructions import (
    Acquire,
    PostProcessing,
    QuantumInstruction, Sweep, Repeat, DeviceUpdate,
)
from qat.purr.compiler.ir.graph import ControlFlowGraph
from qat.purr.compiler.ir.instructions import EndSweep, EndRepeat


class AnalysisPass:
    def run(self, ir: InstructionBuilder, *args):
        pass


class CtrlHwAnalysis(AnalysisPass):
    def run(self, ir: InstructionBuilder, *args):
        pass


class TargetAnalysis(AnalysisPass):
    def run(self, ir: InstructionBuilder, *args):
        """
        Collects quantum targets AOT. Useful for subsequent analysis/transform passes
        as well as for backend code generator.
        """
        result = set()
        for inst in ir.instructions:
            if isinstance(inst, QuantumInstruction):
                if isinstance(inst, PostProcessing):
                    for qt in inst.quantum_targets:
                        if isinstance(qt, Acquire):
                            result.update(qt.quantum_targets)
                        else:
                            result.add(qt)
                else:
                    result.update(inst.quantum_targets)
        return result


class BoundsAnalysis(AnalysisPass):
    def run(self, ir: InstructionBuilder, *args):
        """
        Analyses loop bounds from given value if it's linearly and evenly
        spaced or fails otherwise.
        """
        result = {}
        for inst in ir.instructions:
            if isinstance(inst, Sweep):
                name, sweep_value = next(iter(inst.variables.items()))

                value = sweep_value.value
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

                result[inst] = start, step, end, count
            elif isinstance(inst, Repeat):
                start = 0
                step = 1
                end = inst.repeat_count
                count = inst.repeat_count
                result[inst] = start, step, end, count

            return result


class CFGAnalysis(AnalysisPass):
    def run(self, ir: InstructionBuilder, *args):
        cfg = ControlFlowGraph()
        self._build_cfg(ir, cfg)
        return cfg

    def _build_cfg(self, ir: InstructionBuilder, cfg: ControlFlowGraph):
        """
        Recursively (re)discovers (new) header nodes and flow information.
        """

        flow = [(e.src.head(), e.dest.head()) for e in cfg.edges]
        headers = sorted([n.head() for n in cfg.nodes]) or [
            i
            for i, inst in enumerate(ir.instructions)
            if isinstance(inst, (Sweep, Repeat, EndSweep, EndRepeat))
        ]
        if headers[0] != 0:
            headers.insert(0, 0)

        next_flow = set(flow)
        next_headers = set(headers)
        for i, h in enumerate(headers):
            inst_at_h = ir.instructions[h]
            src = cfg.get_or_create_node(h)
            if isinstance(inst_at_h, Repeat):
                next_headers.add(h + 1)
                next_flow.add((h, h + 1))
                dest = cfg.get_or_create_node(h + 1)
                cfg.get_or_create_edge(src, dest)
            elif isinstance(inst_at_h, Sweep):
                s = next(
                    (
                        s
                        for s, inst in enumerate(ir.instructions[h + 1:])
                        if not isinstance(inst, DeviceUpdate)
                    )
                )
                next_headers.add(s + h + 1)
                next_flow.add((h, s + h + 1))
                src.indices.extend(range(src.tail() + 1, s + h + 1))
                dest = cfg.get_or_create_node(s + h + 1)
                cfg.get_or_create_edge(src, dest)
            elif isinstance(inst_at_h, (EndSweep, EndRepeat)):
                if h < len(ir.instructions) - 1:
                    next_headers.add(h + 1)
                    next_flow.add((h, h + 1))
                    dest = cfg.get_or_create_node(h + 1)
                    cfg.get_or_create_edge(src, dest)
                type = Sweep if isinstance(inst_at_h, EndSweep) else Repeat
                p = next(
                    (p for p in headers[i::-1] if isinstance(ir.instructions[p], type))
                )
                next_headers.add(p)
                next_flow.add((h, p))
                dest = cfg.get_or_create_node(p)
                cfg.get_or_create_edge(src, dest)
            else:
                k = next(
                    (
                        s
                        for s, inst in enumerate(ir.instructions[h + 1:])
                        if isinstance(inst, (Sweep, Repeat, EndSweep, EndRepeat))
                    ),
                    None,
                )
                if k:
                    next_headers.add(k + h + 1)
                    next_flow.add((h, k + h + 1))
                    src.indices.extend(range(src.tail() + 1, k + h + 1))
                    dest = cfg.get_or_create_node(k + h + 1)
                    cfg.get_or_create_edge(src, dest)

        if next_headers == set(headers) and next_flow == set(flow):
            return

        self._build_cfg(ir, cfg)


class TimelineAnalysis(AnalysisPass):
    pass
