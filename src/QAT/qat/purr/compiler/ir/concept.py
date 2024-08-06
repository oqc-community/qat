from abc import ABC, abstractmethod
from typing import List


class PassConcept(ABC):
    @abstractmethod
    def run(self, ir):
        pass


class PassModel(PassConcept):
    def __init__(self, pass_obj):
        self._pass = pass_obj

    def run(self, ir):
        self._pass.run(ir)


class ResultConcept(ABC):
    @abstractmethod
    def invalidate(self, ir):
        pass


class ResultModel(ResultConcept):
    def __init__(self, result_obj):
        self._result = result_obj

    def invalidate(self, ir):
        self._result.invalidate(ir)


class PassInfoMixin:
    def id(self):
        pass

    def name(self):
        pass


class PassManager(PassInfoMixin):
    def __init__(self):
        self.passes: List = []

    def run(self, ir, *args):
        for p in self.passes:
            result = p.run(ir, args)

    def add_pass(self, pass_obj):
        self.passes.append(pass_obj)
