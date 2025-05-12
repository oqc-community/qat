# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
from compiler_config.config import InlineResultsProcessing

from qat.backend.waveform_v1 import WaveformV1Backend
from qat.engines.waveform_v1 import EchoEngine
from qat.ir.measure import PostProcessing
from qat.model.loaders.legacy import EchoModelLoader
from qat.purr.compiler.instructions import AcquireMode, PostProcessType, ProcessAxis
from qat.runtime.executables import AcquireData, ChannelData, ChannelExecutable
from qat.runtime.passes.transform import (
    AssignResultsTransform,
    InlineResultsProcessingTransform,
    PostProcessingTransform,
)


class TestPostProcessingTransform:
    def test_raw_to_bits(self):
        mock_readout = {"test": np.ones((1000, 254))}
        pp_instructions = [
            PostProcessing(
                output_variable="test",
                process_type="down_convert",
                axes=[ProcessAxis.TIME],
                args=[0.0, 1e-9],
            ),
            PostProcessing(
                output_variable="test",
                process_type="mean",
                axes=[ProcessAxis.TIME],
            ),
            PostProcessing(
                output_variable="test",
                process_type=PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
                axes=[ProcessAxis.SEQUENCE],
                args=[-2.54, 1.1],
            ),
            PostProcessing(
                output_variable="test",
                process_type=PostProcessType.DISCRIMINATE,
                axes=[ProcessAxis.SEQUENCE],
                args=[1.6],
            ),
        ]
        acquire = AcquireData(
            length=254, position=0, mode=AcquireMode.RAW, output_variable="test"
        )
        package = ChannelExecutable(
            channel_data={"CH1": ChannelData(acquires=acquire)},
            post_processing={"test": pp_instructions},
        )
        result = PostProcessingTransform().run(mock_readout, package=package)
        assert len(result) == 1
        assert "test" in result
        assert np.shape((result["test"])) == (1000,)
        assert np.allclose(result["test"], -1.0)

    def test_integrator_to_bits(self):
        mock_readout = {"test": np.ones((1000))}
        pp_instructions = [
            PostProcessing(
                output_variable="test",
                process_type=PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
                axes=[ProcessAxis.SEQUENCE],
                args=[-2.54, 1.1],
            ),
            PostProcessing(
                output_variable="test",
                process_type=PostProcessType.DISCRIMINATE,
                axes=[ProcessAxis.SEQUENCE],
                args=[1.6],
            ),
        ]
        acquire = AcquireData(
            length=254, position=0, mode=AcquireMode.RAW, output_variable="test"
        )
        package = ChannelExecutable(
            channel_data={"CH1": ChannelData(acquires=acquire)},
            post_processing={"test": pp_instructions},
        )
        result = PostProcessingTransform().run(mock_readout, package=package)
        assert len(result) == 1
        assert "test" in result
        assert np.shape((result["test"])) == (1000,)
        assert np.allclose(result["test"], -1.0)


class TestInlineResultsProcessingTransform:
    def test_run_results_processing_with_program(self):
        model = EchoModelLoader().load()
        builder = model.create_builder()
        builder.repeat(1000, 100e-6)
        builder.measure_single_shot_binned(model.get_qubit(0), output_variable="test")
        builder.results_processing("test", InlineResultsProcessing.Program)
        package = WaveformV1Backend(model).emit(builder)
        engine = EchoEngine()
        results = engine.execute(package)
        InlineResultsProcessingTransform().run(results, package=package)
        assert isinstance(results["test"], int)

    def test_run_results_processing_with_experiment(self):
        model = EchoModelLoader().load()
        builder = model.create_builder()
        builder.repeat(254, 100e-6)
        builder.measure(model.get_qubit(0), output_variable="test")
        builder.results_processing("test", InlineResultsProcessing.Experiment)
        package = WaveformV1Backend(model).emit(builder)
        engine = EchoEngine()
        results = engine.execute(package)
        InlineResultsProcessingTransform().run(results, package=package)
        assert isinstance(results["test"], np.ndarray)
        assert len(results["test"]) == 254


class TestAssignResultsTransform:
    def test_only_returns_what_is_asked(self):
        model = EchoModelLoader().load()
        builder = model.create_builder()
        builder.repeat(254, 100e-6)
        builder.measure(model.get_qubit(0), output_variable="q0")
        builder.measure(model.get_qubit(0), output_variable="q1")
        builder.returns("q0")
        package = WaveformV1Backend(model).emit(builder)
        engine = EchoEngine()
        results = engine.execute(package)
        results = AssignResultsTransform().run(results, package=package)
        assert "q0" in results
        assert "q1" not in results
