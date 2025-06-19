# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
import pytest

from qat.backend.waveform_v1.codegen import WaveformV1Backend
from qat.core.pass_base import PassManager
from qat.engines.waveform_v1 import EchoEngine
from qat.middleend.passes.legacy.transform import (
    RepeatTranslation,
)
from qat.model.loaders.legacy import EchoModelLoader
from qat.model.target_data import TargetData
from qat.purr.compiler.instructions import AcquireMode, PulseShapeType
from qat.runtime import SimpleRuntime
from qat.runtime.passes.transform import (
    InlineResultsProcessingTransform,
    PostProcessingTransform,
)


class TestSimpleRuntime:
    def basic_acquire_circuit(self, shots, mode=AcquireMode.INTEGRATOR):
        model = EchoModelLoader().load()
        qubit = model.qubits[0]
        builder = model.create_builder()
        builder.repeat(shots, 100e-6)
        builder.pulse(
            qubit.get_measure_channel(), shape=PulseShapeType.SQUARE, width=800e-9
        )
        builder.acquire(
            qubit.get_acquire_channel(),
            delay=0.0,
            mode=mode,
            time=800e-9,
            output_variable="test",
        )
        builder.returns("test")
        builder = RepeatTranslation(TargetData.default()).run(builder)
        return WaveformV1Backend(model).emit(builder)

    @pytest.mark.parametrize("shots", [500, 1000, 1007, 2000])
    def test_execute_gives_correct_number_of_shots(self, shots):
        package = self.basic_acquire_circuit(shots)
        with SimpleRuntime(EchoEngine()) as runtime:
            results = runtime.execute(package)
        assert len(results) == 1
        assert np.shape(next(iter(results.values()))) == (shots,)

    def test_execute_with_pipelines(self):
        model = EchoModelLoader().load()
        builder = model.create_builder()
        builder.repeat(254, 100e-6)
        for i in range(2):
            qubit = model.qubits[i]
            builder.pulse(
                qubit.get_measure_channel(), shape=PulseShapeType.SQUARE, width=800e-9
            )
            builder.acquire(
                qubit.get_acquire_channel(),
                delay=0.0,
                mode=AcquireMode.INTEGRATOR,
                time=800e-9,
                output_variable=f"qubit{i}",
            )
        builder.returns("qubit0")
        builder = RepeatTranslation(TargetData.default()).run(builder)

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
