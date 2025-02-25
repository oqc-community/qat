# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
import pytest

from qat.backend.waveform_v1 import EchoEngine
from qat.backend.waveform_v1.codegen import WaveformV1Backend
from qat.passes.pass_base import PassManager
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.instructions import AcquireMode
from qat.runtime import SimpleRuntime
from qat.runtime.transform_passes import (
    InlineResultsProcessingTransform,
    PostProcessingTransform,
)


class TestSimpleRuntime:

    def basic_acquire_circuit(self, shots, mode=AcquireMode.INTEGRATOR):
        model = get_default_echo_hardware()
        builder = model.create_builder()
        builder.repeat(shots)
        block, _ = builder._generate_measure_block(model.get_qubit(0), mode, "test")
        builder.add(block)
        return WaveformV1Backend(model).emit(builder)

    @pytest.mark.parametrize("shots", [0, 500, 1000, 1007, 2000])
    def test_execute_gives_correct_number_of_shots(self, shots):
        package = self.basic_acquire_circuit(shots)
        with SimpleRuntime(EchoEngine()) as runtime:
            if shots == 0:
                with pytest.warns():
                    results = runtime.execute(package)
            else:
                results = runtime.execute(package)
        assert len(results) == 1
        assert np.shape(next(iter(results.values()))) == (shots,)

    def test_execute_with_pipelines(self):
        model = get_default_echo_hardware()
        builder = model.create_builder()
        builder.repeat(254)
        builder.measure(model.get_qubit(0), output_variable="qubit0")
        builder.measure(model.get_qubit(1), output_variable="qubit1")
        builder.returns("qubit0")

        # Test with default pipeline
        package = WaveformV1Backend(model).emit(builder)
        with SimpleRuntime(EchoEngine()) as runtime:
            results = runtime.execute(package)
        assert "qubit0" in results
        assert "qubit1" not in results

        # Test with custom pipeline
        pipeline = (
            PassManager() | PostProcessingTransform() | InlineResultsProcessingTransform()
        )
        with SimpleRuntime(EchoEngine(), pipeline) as runtime:
            results = runtime.execute(package)
        assert "qubit0" in results
        assert "qubit1" in results
