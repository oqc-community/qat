from abc import ABC, abstractmethod
from typing import List

from qat.ir.result_base import ResultManager
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class PassConcept(ABC):
    """
    Base class describing the abstraction of a pass.

    See PassManager.
    """

    @abstractmethod
    def run(self, ir, res_mgr: ResultManager, *args, **kwargs):
        pass


class PassModel(PassConcept):
    """
    Implements the polymorphic pass API. A wrapper for any object providing a run()
    method that accepts some unit of IR as well as a ResultManager.

    See PassManager.
    """

    def __init__(self, pass_obj):
        self._pass = pass_obj

    def run(self, ir, res_mgr: ResultManager, *args, **kwargs):
        return self._pass.run(ir, res_mgr, *args, **kwargs)


class PassInfoMixin(ABC):
    """
    Base mixin specifying pass identification mechanism. A pass has an id and name.

    Such identification is loosely specified as it related closely to how passes and their results
    are managed throughout their lifecycle.

    See PassManager.
    """

    def id(self):
        pass

    def name(self):
        pass


class AnalysisPass(PassInfoMixin):
    """
    Base class of all passes that compute some form of analysis on the IR.
    The IR is imperatively left intact.
    """

    def run(self, ir, res_mgr: ResultManager, *args, **kwargs):
        pass


class TransformPass(PassInfoMixin):
    """
    Base class for all passes that mutate the IR in-place.
    """

    def run(self, ir, res_mgr: ResultManager, *args, **kwargs):
        pass


class ValidationPass(PassInfoMixin):
    """
    Base class for all passes that verify or run some form of check (i.e semantics, legalisation, ...)
    on the IR.

    Behaviour is loose as to what the result of the validation would be: either a result or an error raised.
    It can change according to circumstances.
    """

    def run(self, ir, res_mgr: ResultManager, *args, **kwargs):
        pass


class PassManager(PassInfoMixin):
    """
    Represents a sequential composite of passes, which can be composites themselves. Although not explicitly
    specified, the pass manager is best modelled as a DAG that acts as a pass itself. In doing so, it runs
    a sequence of (composite or leaf) passes over some unit of IR and aggregates results from them.

    Result aggregation (in general) can be achieved via side effects or simply return values. We adopted
    the latter approach where a ResultManager is passed in as argument. The rationale behind this choice
    is two-fold:

    1) Allow passes to peek, use, or potentially invalidate results via the ResultManager without burdening
    their parent PassManager.

    2) Let the passes and results define their dependency graphs on the fly which saves us from explicitly
    building and maintaining them. Here it's also a possible direction to let passes lazily declare, create,
    and run their dependencies as they get discovered.

    Result aggregation would also potentially involve cache handling as described in the ResultManager.
    This calls for proper pass identification, cache invalidation, and pass/result cycle detection techniques.

    Example of a DAG:

    (PM1 A B (PM2 C D ) E F) can be thought of as:

            <---PM2--->

               --> C
    A --> B --|    ↓
               --> D --> E --> F

    <-----------PM1------------>

    Although the notation of a given (sub-) DAG looks sequential, passes within can run sequentially
    or in parallel depending on their inter-dependencies.

    Notes thus far describe an ideal fully-fledged PassManager. However, today's needs are very simple as we barely
    have visible features of modular quantum compilation workflow. It is therefore wise to keep the implementation
    light and basic where component passes run and register their own results within the ResultManager passed
    in as argument. Another reason is the possibility in the near future to adopt and tap into existing and mature
    pass pipeline infrastructure which saves us from reinventing or rediscovering the same concepts already covered
    by the opensource community.
    """

    def __init__(self):
        self.passes: List[PassModel] = []

    def run(self, ir, res_mgr: ResultManager, *args, **kwargs):
        for p in self.passes:
            p.run(ir, res_mgr, *args, **kwargs)

    def add(self, pass_obj):
        self.passes.append(PassModel(pass_obj))
        return self

    def __or__(self, pass_obj):
        return self.add(pass_obj)


class InvokerMixin(ABC):
    """
    Useful for compilation (global design-wise) stages/phases that are meant to invoke some
    arbitrary formation of passes in the form of a pipeline.

    Organisation, registering, visibility, and discovery of pipelines w.r.t the global quantum compilation
    workflow is hard to pin down early on in this design partly because today's needs are trivial and
    the traditional "hand-me-down" cascade-like compilation stages is likely to shift as QAT matures.

    However, we'll start by specifying that an invoker builds and runs its own pipeline, validates its
    analyses results, and uses those results for its purpose.
    """

    @abstractmethod
    def build_pass_pipeline(self, *args, **kwargs) -> PassManager:
        pass

    def run_pass_pipeline(self, ir, res_mgr: ResultManager, *args, **kwargs):
        return self.build_pass_pipeline(*args, **kwargs).run(ir, res_mgr, *args, **kwargs)
