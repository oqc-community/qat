# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

from qat.model.loaders.purr import EchoModelLoader
from qat.purr.backends.qblox.metrics_base import MetricsManager
from qat.purr.backends.qblox.pass_base import (
    AnalysisPass,
    QatIR,
    TransformPass,
    ValidationPass,
)
from qat.purr.backends.qblox.result_base import ResultManager
from qat.purr.compiler.runtime import NewQuantumRuntime

from tests.unit.utils.builder_nuggets import resonator_spect


def test_new_quantum_runtime():
    model = EchoModelLoader().load()
    engine = model.create_engine()
    runtime = NewQuantumRuntime(engine)

    builder = resonator_spect(model)
    res_mgr = ResultManager()
    met_mgr = MetricsManager()
    pipeline = runtime.build_pass_pipeline()
    assert pipeline.passes
    assert len(pipeline.passes) == 5
    assert not any([m for m in pipeline.passes if isinstance(m._pass, AnalysisPass)])
    assert any([m for m in pipeline.passes if isinstance(m._pass, TransformPass)])
    assert any([m for m in pipeline.passes if isinstance(m._pass, ValidationPass)])
    runtime.run_pass_pipeline(QatIR(builder), res_mgr, met_mgr, model, engine)
