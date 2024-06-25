from typing import Tuple

from qat.purr.compiler.instructions import Instruction
from typing import List, Dict, Any


class BasicBlock:
    def __init__(self, instructions=None):
        self.instructions: List[Instruction] = instructions or []
        self.values: Dict[str, Any] = {}

    def entry(self):
        if self.instructions:
            return self.instructions[0]
        return None

    def exit(self):
        if self.instructions:
            return self.instructions[-1]
        return None


class CatGraph:
    def __init__(self, nodes=None, edges=None):
        self.nodes: List[BasicBlock] = nodes or []
        self.edges: List[Tuple[BasicBlock, BasicBlock]] = edges or []

        self.entry = None
        self.exit = None
