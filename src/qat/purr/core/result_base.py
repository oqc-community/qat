# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import uuid
from abc import ABC
from typing import Set, TypeVar


class ResultConcept(ABC):
    """
    Base class describing the abstraction of an analysis result.

    See :class:`ResultManager`.
    """

    pass


class ResultModel(ResultConcept):
    """
    Wrapper for any result object typically produced by an analysis pass.

    See :class:`ResultManager`.
    """

    def __init__(self, res_obj):
        self._result = res_obj
        self._uuid = uuid.uuid4()

        self.is_valid = True

    @property
    def value(self):
        return self._result

    @property
    def id(self):
        return self._uuid

    def __hash__(self):
        return hash(self._uuid)


class ResultInfoMixin(ABC):
    """
    Base mixin specifying result identification mechanism. A result has an :attr:`id`,
    :attr:`name`, and :attr:`value`.

    See :class:`ResultManager`.
    """

    def id(self):
        pass

    def name(self):
        pass

    def value(self):
        pass


ResultType = TypeVar("ResultType", bound=ResultInfoMixin)


class ResultManager:
    """
    Represents a collection of analysis results with caching and aggregation
    capabilities.

    Passes that merely compute analyses on the IR must not invalidate prior results. Passes
    that mutate any IR units are likely to invalidate predecessor results.

    An analysis pass can produce 1 or more result objects. There is in theory a duality
    between passes and the results they produce.

    To keep things simple, the ResultManager is just a set of analysis results. Result
    identification is also kept trivial where a UUID is used internally. Proper
    identification mechanism will be called upon once we feel the need for a more
    sophisticated :class:`PassManager`.

    See :class:`PassManager`.
    """

    def __init__(self):
        self._results: Set[ResultModel] = set()

    @property
    def results(self):
        return self._results

    def _find(self, res_obj):
        found = [res for res in self._results if res.value == res_obj]
        if not found:
            raise ValueError(f"Could not find result {res_obj}")

        if len(found) > 1:
            raise ValueError("Found multiple results")
        return found[0]

    def _remove(self, *res_objs):
        for res_obj in res_objs:
            found = self._find(res_obj)
            self._results.remove(found)

    def cleanup(self):
        self._results = set(res for res in self._results if res.is_valid)

    def update(self, other_res_mgr):
        if not isinstance(other_res_mgr, ResultManager):
            raise ValueError(
                f"Invalid type, expected {ResultManager}, but got {type(other_res_mgr)}"
            )
        self._results.update(other_res_mgr._results)

    def add(self, res_obj):
        self._results.add(ResultModel(res_obj))

    def mark_as_dirty(self, *res_objs):
        results = [self._find(res_obj) for res_obj in res_objs]
        for result in results:
            result.is_valid = False

    def lookup_by_type(self, ty: type[ResultType]) -> ResultType:
        found = [res.value for res in self._results if isinstance(res.value, ty)]
        if not found:
            raise ValueError(f"Could not find any results instances of {ty}")

        if len(found) > 1:
            raise ValueError(f"Found multiple results instances of {ty}")

        return found[0]

    def remove_by_type(self, ty: type[ResultType]):
        found = [res for res in self._results if isinstance(res.value, ty)]
        if not found:
            raise ValueError(f"Could not find any results instances of {ty}")

        if len(found) > 1:
            raise ValueError(f"Found multiple results instances of {ty}")

        self._results.remove(found[0])


class PreservedResults:
    """
    A mechanism for result invalidation and preservation. Similar to LLVM's new PassManager, we state
    that each transform pass must declare what analysis results it preserves. In this case, the pass returns
    a PreservedResults instance which is then used by the :class:`PassManager` for cache housekeeping.
    """

    @staticmethod
    def all():
        """
        Indicates that **all** of the previously computed results remain valid. The pass manager will do nothing
        as analyses results are still safe and correct after the pass in question has finished running.
        """

        # TODO - Implement PreservedResults [COMPILER-843]
        pass

    @staticmethod
    def none():
        """
        Indicates that **none** of the previously computed results are valid. This will cause the pass manager
        to evict cached results after the (transform) pass in question has finished running.
        """

        # TODO - Implement PreservedResults [COMPILER-843]
        pass

    @staticmethod
    def preserve(*res_obj):
        """
        Selective preservation of the results or indeed result sets indicated by the argument. This will
        cause the pass manager to filter through the result cache and preserve **only** the ones that
        the argument points to.
        """

        # TODO - Implement PreservedResults [COMPILER-843]
        pass

    @staticmethod
    def discard(*res_obj):
        """
        Convenient inverse selector for the :meth:`preserve()` API. This will cause the pass manager
        to discard and evict **only** results pointed to by the argument.
        """

        # TODO - Implement PreservedResults [COMPILER-843]
        pass
