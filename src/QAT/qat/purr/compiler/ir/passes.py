from abc import abstractmethod, ABC
from typing import List


class Pass(ABC):
    @abstractmethod
    def run(self, instructions: List):
        pass

    @abstractmethod
    def verify(self, instructions: List):
        pass


class PassManager(Pass):
    def __init__(self):
        self._passes: List[Pass] = []

    def add_pass(self, p: Pass):
        self._passes.append(p)

    def run(self, instructions):
        for p in self._passes:
            instructions = p.run(instructions)
            p.verify(instructions)
        return instructions
