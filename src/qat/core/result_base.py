# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import uuid
from abc import ABC
from typing import Set

from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class ResultConcept(ABC):
    """Base class describing the abstraction of an analysis result.

    See :class:`ResultManager`.
    """

    pass


class ResultModel(ResultConcept):
    """Wrapper for any result object typically produced by an analysis pass.

    See :class:`ResultManager`.
    """

    def __init__(self, res_obj):
        self._result = res_obj
        self._uuid = uuid.uuid4()

    @property
    def value(self):
        return self._result

    def __hash__(self):
        return hash(self._uuid)


class ResultInfoMixin(ABC):
    """Base mixin specifying result identification mechanism. A result has an :attr:`id`,
    :attr:`name`, and :attr:`value`.

    See :class:`ResultManager`.
    """

    def id(self):
        pass

    def name(self):
        pass

    def value(self):
        pass


class ResultManager:
    """Represents a collection of analysis results with caching and aggregation
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

    def update(self, other_res_mgr):
        """Add the results from another results manager.

        :param ResultManager other_res_mgr:
        """
        if not isinstance(other_res_mgr, ResultManager):
            raise ValueError(
                f"Invalid type, expected {ResultManager}, but got {type(other_res_mgr)}"
            )
        self._results.update(other_res_mgr._results)

    def add(self, res_obj: ResultInfoMixin):
        """Add a results object to the manager.

        :param res_obj: Results from a pass, typically an analysis pass.
        """
        self._results.add(ResultModel(res_obj))

    def lookup_by_type(self, ty: type):
        """Find a result by its type.

        :param ty: The results type.
        """
        found = [res.value for res in self._results if isinstance(res.value, ty)]
        if not found:
            raise ValueError(f"Could not find any results instances of {ty}")

        if len(found) > 1:
            raise ValueError(f"Found multiple results instances of {ty}")

        return found[0]
