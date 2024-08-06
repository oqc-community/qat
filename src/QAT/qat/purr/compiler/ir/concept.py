from abc import ABC, abstractmethod


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
