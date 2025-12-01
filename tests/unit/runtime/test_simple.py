# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.backend.waveform_v1.purr.codegen import WaveformV1Backend
from qat.core.pass_base import PassManager
from qat.engines.waveform import EchoEngine
from qat.engines.zero import ZeroEngine
from qat.executables import AcquireData, Executable
from qat.middleend.passes.purr.transform import (
    RepeatTranslation,
)
from qat.model.loaders.purr import EchoModelLoader
from qat.model.target_data import TargetData
from qat.purr.compiler.instructions import AcquireMode, PulseShapeType
from qat.runtime import SimpleRuntime
from qat.runtime.connection import ConnectionMode
from qat.runtime.passes.transform import (
    AcquisitionPostprocessing,
    InlineResultsProcessingTransform,
)

from tests.unit.utils.engines import MockConnectedEngine
from tests.unit.utils.executables import MockProgram


class TestSimpleRuntime:
    def test_connection_is_held_during_execute(self):
        engine = MockConnectedEngine()
        runtime = SimpleRuntime(
            engine, results_pipeline=PassManager(), connection_mode=ConnectionMode.DEFAULT
        )
        assert engine.is_connected == False
        assert engine.connections == 0

        runtime.execute(Executable(programs=[]))
        assert engine.is_connected == False
        assert engine.connections == 1

    def test_execute_with_programs_and_batched_acquisitions(self):
        engine = ZeroEngine()
        runtime = SimpleRuntime(engine, results_pipeline=PassManager())

        dummy_programs = [
            MockProgram(shapes={"a": (1000,), "b": (1000, 254)}),
            MockProgram(shapes={"a": (1000,), "b": (1000, 254)}),
            MockProgram(shapes={"a": (500,), "b": (500, 254)}),
        ]
        acquire_data = {
            "a": AcquireData(
                mode=AcquireMode.INTEGRATOR, shape=(2500,), physical_channel="ch1"
            ),
            "b": AcquireData(
                mode=AcquireMode.RAW, shape=(2500, 254), physical_channel="ch2"
            ),
        }
        package = Executable(
            programs=dummy_programs,
            acquires=acquire_data,
            assigns=[],
            returns={"a", "b"},
        )

        results = runtime.execute(package)
        assert "a" in results
        assert "b" in results
        assert results["a"].shape == (2500,)
        assert results["b"].shape == (2500, 254)

    def test_execute_with_programs_with_iterator(self):
        engine = ZeroEngine()
        runtime = SimpleRuntime(engine, results_pipeline=PassManager())

        dummy_programs = [
            MockProgram(shapes={"a": (1000,), "b": (1000, 254), "c": (254,)}),
            MockProgram(shapes={"a": (1000,), "b": (1000, 254), "c": (254,)}),
            MockProgram(shapes={"a": (1000,), "b": (1000, 254), "c": (254,)}),
        ]
        acquire_data = {
            "a": AcquireData(
                mode=AcquireMode.INTEGRATOR,
                shape=(3, 1000),
                physical_channel="ch1",
            ),
            "b": AcquireData(
                mode=AcquireMode.RAW, shape=(3, 1000, 254), physical_channel="ch2"
            ),
            "c": AcquireData(
                mode=AcquireMode.SCOPE, shape=(3, 254), physical_channel="ch3"
            ),
        }
        package = Executable(
            programs=dummy_programs,
            acquires=acquire_data,
            assigns=[],
            returns={"a", "b", "c"},
        )

        results = runtime.execute(package)
        assert "a" in results
        assert "b" in results
        assert "c" in results
        assert results["a"].shape == (3, 1000)
        assert results["b"].shape == (3, 1000, 254)
        assert results["c"].shape == (3, 254)

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
        with pytest.warns(DeprecationWarning):
            package = WaveformV1Backend(model).emit(builder)
        with SimpleRuntime(EchoEngine()) as runtime:
            results = runtime.execute(package)
        assert "qubit0" in results
        assert "qubit1" not in results

        # Test with custom pipeline
        pipeline = (
            PassManager()
            | AcquisitionPostprocessing(TargetData.default())
            | InlineResultsProcessingTransform()
        )
        with SimpleRuntime(EchoEngine(), pipeline) as runtime:
            results = runtime.execute(package)
        assert "qubit0" in results
        assert "qubit1" in results
