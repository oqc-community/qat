from typing import List

from qat.ir.pass_base import ResultType
from qat.purr.backends.graph import BasicBlock, ControlFlowGraph


class CodegenResultType(ResultType):
    ACQUIRE_MAP = "ACQUIRE_MAP"
    ASSIGNS = "ASSIGNS"
    CFG = "CFG"
    CONTEXTS = "CONTEXTS"
    PP_MAP = "OUTPUT_VAR_PP_VIEW"
    RP_MAP = "OUTPUT_VAR_RP_VIEW"
    RETURN = "RETURN"
    SWEEPS = "SWEEPS"
    VARIABLE_BOUNDS = "VARIABLE_BOUNDS"
    TARGET_VIEW = "TARGET_VIEW"
    TIMELINE = "TIMELINE"


class DfsTraversal:
    def __init__(self):
        self._entered: List[BasicBlock] = []

    def clear(self):
        self._entered.clear()

    def run(self, graph: ControlFlowGraph):
        self.clear()
        self._visit(graph.entry, graph)

    def _visit(self, node: BasicBlock, graph: ControlFlowGraph):
        self.enter(node)
        self._entered.append(node)
        for neighbour in graph.out_nbrs(node):
            if neighbour not in self._entered:
                self._visit(neighbour, graph)
        self.exit(node)

    def enter(self, node: BasicBlock):
        pass

    def exit(self, node: BasicBlock):
        pass
