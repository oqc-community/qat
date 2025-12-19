# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
from abc import ABC, abstractmethod

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class PassConcept(ABC):
    """Base class describing the abstraction of a pass.

    See :class:`PassManager`.
    """

    @abstractmethod
    def run(self, ir, res_mgr: ResultManager, met_mgr: MetricsManager, *args, **kwargs):
        pass


class PassModel(PassConcept):
    """Implements the polymorphic pass API. A wrapper for any object providing a :meth:`run`
    method that accepts an argument as well as a :class:`ResultManager` and
    :class:`MetricsManager`.

    See :class:`PassManager`.
    """

    def __init__(self, pass_obj):
        self._pass = pass_obj

    def run(self, ir, res_mgr: ResultManager, met_mgr: MetricsManager, *args, **kwargs):
        """
        :param ir: Argument to pass to the object.
        :param res_mgr:
        :param met_mgr:
        """
        return self._pass.run(ir, res_mgr, met_mgr, *args, **kwargs)


class PassInfoMixin(ABC):
    """Base mixin specifying pass identification mechanism. A pass has an :attr:`id` and
    :attr:`name`.

    Such identification is loosely specified as it related closely to how passes and their
    results are managed throughout their lifecycle.

    See :class:`PassManager`.
    """

    def id(self):
        pass

    def name(self):
        pass


class AnalysisPass(PassInfoMixin):
    """Base class of all passes that compute some form of analysis on the input argument,
    with the input IR left intact."""

    def run(self, ir, res_mgr: ResultManager, met_mgr: MetricsManager, *args, **kwargs):
        pass


class TransformPass(PassInfoMixin):
    """Base class for all passes that mutates the IR.

    It is expected that child classes do not alter the type of the IR, but only alter its
    contents.
    """

    def run(self, ir, res_mgr: ResultManager, met_mgr: MetricsManager, *args, **kwargs):
        pass


class ValidationPass(PassInfoMixin):
    """Base class for all passes that verify or run some form of check (i.e semantics,
    legalisation, ...) on the IR.

    Behaviour is loose as to what the result of the validation would be: either a result or
    an error raised. It can change according to circumstances.
    """

    def run(self, ir, res_mgr: ResultManager, met_mgr: MetricsManager, *args, **kwargs):
        pass


class LoweringPass(PassInfoMixin):
    """Base case for all passes that modify the IR, instrinsically changing its type and
    structure.

    Acts as insulation between passes that that expect to see the IR in some
    given format.
    """

    def run(self, ir, res_mgr: ResultManager, met_mgr: MetricsManager, *args, **kwargs):
        pass


class PassManager(PassInfoMixin):
    """Contains a sequence of passes.

    Represents a sequential composite of passes, which can be composites themselves.
    Although not explicitly specified, the pass manager is best modelled as a DAG that acts
    as a pass itself. In doing so, it runs a sequence of (composite or leaf) passes over the
    input and aggregates results from them.

    Result aggregation (in general) can be achieved via return values or side effects. We
    adopted the latter approach where a :class:`ResultManager` is passed in as argument. The
    rationale behind this choice is two-fold:

    #. Allow passes to peek, use, or potentially invalidate results via the ResultManager
       without burdening their parent :class:`PassManager`.
    #. Let the passes and results define their dependency graphs on the fly which saves us
       from explicitly building and maintaining them. Here it's also a possible direction to
       let passes lazily declare, create, and run their dependencies as they get discovered.

    Result aggregation would also potentially involve cache handling as described in the
    :class:`ResultManager`. This calls for proper pass identification, cache invalidation,
    and pass/result cycle detection techniques.

    Example of a DAG:

    (PM1 A B (PM2 C D ) E F) can be thought of as:

    .. code-block::

                <---PM2--->

                   --> C
        A --> B --|    ↓
                   --> D --> E --> F

        <-----------PM1------------>

    Although the notation of a given (sub-) DAG looks sequential, passes within can run
    sequentially or in parallel depending on their inter-dependencies.

    Notes thus far describe an ideal fully-fledged PassManager. However, today's needs are
    very simple as we barely have visible features of modular quantum compilation workflow.
    It is therefore wise to keep the implementation light and basic where component passes
    run and register their own results within the :class:`ResultManager` passed in as
    argument. Another reason is the possibility in the near future to adopt and tap into
    existing and mature pass pipeline infrastructure which saves us from reinventing or
    rediscovering the same concepts already covered by the opensource community.
    """

    def __init__(self):
        self.passes: list[PassModel] = []

    def run(self, ir, res_mgr: ResultManager, met_mgr: MetricsManager, *args, **kwargs):
        """
        :param arg: Argument to pass to the object.
        :param res_mgr:
        :param met_mgr:
        """
        for p in self.passes:
            ir = p.run(ir, res_mgr, met_mgr, *args, **kwargs)
        return ir

    def add(self, pass_obj):
        """
        Add a pass to the pass manager.

        This can be achieved by either using :code:`pass_mgr.add(pass)`, or
        :code:`pass_mgr | pass`.

        :param pass_obj:
        """
        self.passes.append(PassModel(pass_obj))
        return self

    def __or__(self, pass_obj):
        return self.add(pass_obj)


class InvokerMixin(ABC):
    """Useful for compilation (global design-wise) stages/phases that are meant to invoke
    some arbitrary formation of passes in the form of a pipeline.

    Organisation, registering, visibility, and discovery of pipelines w.r.t the global
    quantum compilation workflow is hard to pin down early on in this design partly because
    today's needs are trivial and the traditional "hand-me-down" cascade-like compilation
    stages is likely to shift as QAT matures.

    However, we'll start by specifying that an invoker builds and runs its own pipeline,
    validates its analyses results, and uses those results for its purpose.
    """

    @abstractmethod
    def build_pass_pipeline(self, *args, **kwargs) -> PassManager:
        pass

    def run_pass_pipeline(
        self, arg, res_mgr: ResultManager, met_mgr: MetricsManager, *args, **kwargs
    ):
        return self.build_pass_pipeline(*args, **kwargs).run(
            arg, res_mgr, met_mgr, *args, **kwargs
        )
