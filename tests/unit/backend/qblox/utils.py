# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from compiler_config.config import CompilerConfig

from qat.backend.qblox.codegen import QbloxBackend1, QbloxBackend2
from qat.backend.qblox.config.constants import QbloxTargetData
from qat.backend.qblox.execution import QbloxProgram
from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.engines.qblox.execution import QbloxEngine
from qat.executables import Executable
from qat.pipelines.purr.qblox.compile import (
    backend_pipeline1,
    backend_pipeline2,
    middleend_pipeline1,
    middleend_pipeline2,
)
from qat.pipelines.purr.qblox.execute import get_results_pipeline
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel
from qat.runtime import SimpleRuntime
from qat.runtime.aggregator import QBloxAggregator


def do_emit(model: QbloxLiveHardwareModel, backend_type: type, builder, ignore_empty=True):
    if backend_type == QbloxBackend1:
        middleend_pipeline = middleend_pipeline1(model, QbloxTargetData.default())
        backend_pipeline = backend_pipeline1()
    elif backend_type == QbloxBackend2:
        middleend_pipeline = middleend_pipeline2(model, QbloxTargetData.default())
        backend_pipeline = backend_pipeline2()
    else:
        raise ValueError(f"Expected QbloxBackend1 or QbloxBackend2, got {backend_type}")

    res_mgr = ResultManager()
    met_mgr = MetricsManager()
    middleend_pipeline.run(builder, res_mgr, met_mgr, enable_hw_averaging=True)
    backend = backend_type(model=model, pipeline=backend_pipeline)
    executable = backend.emit(builder, res_mgr, met_mgr, ignore_empty)
    return executable


def do_execute(model, instrument, executable: Executable[QbloxProgram]):
    engine = QbloxEngine(instrument)
    runtime = SimpleRuntime(
        engine=engine,
        aggregator=QBloxAggregator(),
        results_pipeline=get_results_pipeline(model),
    )
    assert isinstance(runtime.aggregator, QBloxAggregator)
    results = runtime.execute(executable=executable, compiler_config=CompilerConfig())
    return results
