# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import pytest

from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.frontends import QASMFrontend

from benchmarks.utils.helpers import load_experiments, load_qasm

experiments = load_experiments(return_builder=False)


@pytest.mark.benchmark(disable_gc=True, max_time=2, min_rounds=10)
@pytest.mark.parametrize("key", experiments.keys())
def test_benchmarks_qasm(benchmark, key):
    # Create the hw model
    circuit = load_qasm(experiments[key]["circuit"])
    hw = experiments[key]["hardware"]
    engine = hw.create_engine()

    # Create a wrapper for the pipeline
    def run():
        frontend = QASMFrontend()
        builder, _ = frontend.parse(circuit, hw)
        builder._instructions = engine.optimize(builder.instructions)
        engine.validate(builder.instructions)
        qatfile = InstructionEmitter().emit(builder.instructions, hw)
        engine.create_duration_timeline(qatfile.instructions)

    benchmark(run)
    assert True
