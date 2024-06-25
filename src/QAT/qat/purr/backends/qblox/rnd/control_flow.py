from typing import Tuple

from qat.purr.compiler.instructions import Instruction
from typing import List, Dict, Any


class BasicBlock:
    def __init__(self, instructions=None):
        self.instructions: List[Instruction] = instructions or []
        self.values: Dict[str, Any] = {}


class CatGraph:
    def __init__(self, nodes=None, edges=None):
        self.nodes: List[BasicBlock] = nodes or []
        self.edges: List[Tuple[BasicBlock, BasicBlock]] = edges or []

