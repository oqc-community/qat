from abc import ABC, abstractmethod
from typing import List

from qat.backend.graph import BasicBlock, ControlFlowGraph


class DfsTraversal(ABC):
    """
    Base Depth-first Search algorithm on the control flow graph
    """

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

    @abstractmethod
    def enter(self, node: BasicBlock):
        pass

    @abstractmethod
    def exit(self, node: BasicBlock):
        pass
