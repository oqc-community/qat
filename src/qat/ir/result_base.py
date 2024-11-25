import uuid
from abc import ABC
from typing import Set

from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class ResultConcept(ABC):
    """
    Base class describing the abstraction of an analysis result.

    See ResultManager.
    """

    pass


class ResultModel(ResultConcept):
    """
    Wrapper for any result object typically produced by an analysis pass.

    See ResultManager.
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
    """
    Base mixin specifying result identification mechanism. A result has an id, name, and value.

    See ResultManager.
    """

    def id(self):
        pass

    def name(self):
        pass

    def value(self):
        pass


class ResultManager:
    """
    Represents a collection of analysis results with caching and aggregation capabilities.

    Passes that merely compute analyses on the IR must not invalidate prior results. Passes that mutate
    any IR units are likely to invalidate predecessor results.

    An analysis pass can produce 1 or more result objects. There is in theory a duality between passes
    and the results they produce.

    To keep things simple, the ResultManager is just a set of analysis results. Result identification is also
    kept trivial where a UUID is used internally. Proper identification mechanism will be called upon
    once we feel the need for a more sophisticated PassManager.

    See PassManager.
    """

    def __init__(self):
        self._results: Set[ResultModel] = set()

    @property
    def results(self):
        return self._results

    def update(self, other_res_mgr):
        if not isinstance(other_res_mgr, ResultManager):
            raise ValueError(
                f"Invalid type, expected {ResultManager}, but got {type(other_res_mgr)}"
            )
        self._results.update(other_res_mgr._results)

    def add(self, res_obj):
        self._results.add(ResultModel(res_obj))

    def lookup_by_type(self, ty: type):
        found = [res.value for res in self._results if isinstance(res.value, ty)]
        if not found:
            raise ValueError(f"Could not find any results instances of {ty}")

        if len(found) > 1:
            raise ValueError(f"Found multiple results instances of {ty}")

        return found[0]
