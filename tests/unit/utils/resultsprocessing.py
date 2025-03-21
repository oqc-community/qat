from pydantic import validate_call

from qat.core.pass_base import AnalysisPass, PassManager


class DummyAnalysisPass(AnalysisPass):
    """Just a dummy testing pass that takes some values"""

    @validate_call
    def __init__(self, model, some_int: int = 3):
        self.model = model
        self.some_int = some_int


@validate_call
def get_pipeline(model, some_int: int = 3) -> PassManager:
    return PassManager() | DummyAnalysisPass(model, some_int=some_int)
