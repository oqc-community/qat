import pytest

from qat.ir.pass_base import PassManager
from qat.ir.result_base import ResultManager
from qat.purr.backends.analysis_passes import BindingPass, TriagePass, TriageResult
from qat.purr.backends.qblox.codegen import PreCodegenPass, PreCodegenResult
from qat.purr.backends.transform_passes import ReturnSanitisation
from qat.purr.backends.validation_passes import NCOFrequencyVariability
from tests.qat.utils.builder_nuggets import resonator_spect


@pytest.mark.parametrize("model", [None], indirect=True)
class TestAnalysisPasses:
    def test_precodegen_pass(self, model):
        res_mgr = ResultManager()
        builder = resonator_spect(model)

        pipeline = (
            PassManager()
            | ReturnSanitisation()
            | TriagePass()
            | NCOFrequencyVariability()
            | BindingPass()
            | PreCodegenPass()
        )
        pipeline.run(builder, res_mgr, model)

        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
        target_map = triage_result.target_map
        precodegen_result: PreCodegenResult = res_mgr.lookup_by_type(PreCodegenResult)

        assert precodegen_result.contexts.keys() == target_map.keys()
