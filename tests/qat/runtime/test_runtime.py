from qat.ir.pass_base import AnalysisPass, TransformPass, ValidationPass
from qat.ir.result_base import ResultManager
from qat.purr.backends.echo import get_default_echo_hardware
from qat.runtime.runtime import QuantumRuntime

from tests.qat.utils.builder_nuggets import resonator_spect


def test_new_quantum_runtime():
    model = get_default_echo_hardware()
    engine = model.create_engine()
    runtime = QuantumRuntime(engine)

    builder = resonator_spect(model)
    res_mgr = ResultManager()
    pipeline = runtime.build_pass_pipeline()
    assert pipeline.passes
    assert len(pipeline.passes) == 4
    assert not any([m for m in pipeline.passes if isinstance(m._pass, AnalysisPass)])
    assert any([m for m in pipeline.passes if isinstance(m._pass, TransformPass)])
    assert any([m for m in pipeline.passes if isinstance(m._pass, ValidationPass)])
    runtime.run_pass_pipeline(builder, res_mgr, model, engine)
