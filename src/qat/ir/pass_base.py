from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class ResultType(Enum):
    pass


@dataclass(frozen=True)
class ResultInfo:
    ir_id: int
    pass_id: str
    result_type: ResultType


class PassResultSet:
    """
    Models a collection of pass results with caching and aggregation capabilities. Passes that merely compute analyses
    on the IR must not invalidate prior results. Passes that mutate any IR units are likely to invalidate previous
    results.

    These result caching complexities could be closely modelled around lazy execution style of the pass pipeline
    as well as lazy construction and evaluation of the pass dependency graph.

    For now, it's just a set of pass results because today's needs are very simple. I will be adding features
    as time goes on
    """

    def __init__(self, *tuples):
        self._data: Dict[ResultInfo, Any] = {}
        for t in tuples:
            self.add_result(t[0], t[1], t[2], t[3])

    @property
    def data(self):
        return self._data

    def update(self, other_rs):
        if not isinstance(other_rs, PassResultSet):
            raise ValueError(
                f"Invalid type, expected {PassResultSet}, but got {type(other_rs)}"
            )
        self._data.update(other_rs._data)

    def add_result(self, ir_id, pass_id, result_type, value):
        key = ResultInfo(ir_id, pass_id, result_type)
        if key in self._data:
            log.warning(f"Writing over existing entry {key}")
        return self._data.setdefault(key, value)

    def get_result(self, result_type):
        keys = [info for info in self._data if info.result_type == result_type]
        if not keys:
            raise ValueError(f"Could not find result for {result_type}")

        if len(keys) > 1:
            raise ValueError(f"Found multiple results for {result_type}")

        return self._data[keys[0]]


class PassConcept(ABC):
    """
    Base class describing the abstraction of a pass.
    """

    @abstractmethod
    def run(self, ir, *args, **kwargs) -> PassResultSet:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class PassModel(PassConcept):
    """
    Implement the polymorphic pass API.
    A wrapper for any object providing a run() method that accepts some unit of IR.
    """

    def __init__(self, pass_obj):
        self._pass = pass_obj

    def run(self, ir, *args, **kwargs):
        return self._pass.run(ir, *args, **kwargs)

    def name(self):
        return self._pass.name()


class PassInfoMixin(ABC):
    def id(self):
        return type(self).__name__

    def name(self):
        return self.id()


class AnalysisPass(PassInfoMixin):
    """
    A base pass to compute some form of analysis on the IR and return it.
    The IR is imperatively left intact.
    """

    def run(self, ir, *args, **kwargs):
        pass


class TransformPass(PassInfoMixin):
    """
    A base pass that mutates the IR in-place.
    """

    def run(self, ir, *args, **kwargs):
        if args:
            analyses = args[0]
        else:
            analyses = PassResultSet()
            args = tuple([analyses] + list(args))
        self.do_run(ir, *args, **kwargs)
        return analyses

    @abstractmethod
    def do_run(self, ir, *args, **kwargs):
        pass


class ValidationPass(PassInfoMixin):
    """
    A base pass to validates semantics and enforce legalisation on the IR.
    """

    def run(self, ir, *args, **kwargs):
        if args:
            analyses = args[0]
        else:
            analyses = PassResultSet()
            args = tuple([analyses] + list(args))
        self.do_run(ir, *args, **kwargs)
        return analyses

    @abstractmethod
    def do_run(self, ir, *args, **kwargs):
        pass


class PassManager(PassInfoMixin):
    """
    A base pass representing a sequential composite of passes, which can be themselves composites.
    The pass manager acts as a pass itself. In doing so, it runs a sequence of passes over
    some unit of IR and aggregates results from them.

    Result aggregation could potentially involve caching invalidation concepts as described in
    PassResultSet. For today's needs, it just accumulates and returns a set of results from the passes.

    TODO - investigate parallel execution of passes for speedup gains. Will be needed
    TODO - idea: Prove that the graph of passes is a DAG. Lazy mode allows for "realtime" and safe
    TODO - pass dependency evaluation
    """

    def __init__(self):
        self.passes: List[PassModel] = []

    def run(self, ir, *args, **kwargs):
        if args:
            analyses = args[0]
        else:
            analyses = PassResultSet()
            args = tuple([analyses] + list(args))
        for p in self.passes:
            p.run(ir, *args, **kwargs)
        return analyses

    def add(self, pass_obj):
        self.passes.append(PassModel(pass_obj))


class InvokerMixin(ABC):
    """
    Indicates stages that are (module/global design-wise) meant to invoke pass pipelines.
    This means an invoker object is responsible to build and run its own pipeline, validating its
    analysis results, and using those results for its purpose.
    """

    @abstractmethod
    def build_pass_pipeline(self, *args, **kwargs) -> PassManager:
        pass

    def run_pass_pipeline(self, ir, *args, **kwargs):
        model = kwargs.get("model", None)
        return self.build_pass_pipeline(model=model).run(ir, *args, **kwargs)
