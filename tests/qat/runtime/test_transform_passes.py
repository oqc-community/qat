# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
from compiler_config.config import InlineResultsProcessing

from qat.backend.waveform_v1.codegen import WaveformV1Emitter
from qat.ir.measure import PostProcessing
from qat.ir.pass_base import QatIR
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.instructions import AcquireMode, PostProcessType, ProcessAxis
from qat.runtime import EchoEngine
from qat.runtime.executables import AcquireDataStruct, ChannelData, Executable
from qat.runtime.transform_passes import (
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
        acquire = AcquireDataStruct(
            length=254, position=0, mode=AcquireMode.RAW, output_variable="test"
        )
        package = Executable(
            channel_data={"CH1": ChannelData(acquires=acquire)},
            post_processing={"test": pp_instructions},
        )
        # TODO: remove QatIR after pipeline changes
        result = QatIR(mock_readout)
        PostProcessingTransform().run(result, package=package)
        result = result.value
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
        acquire = AcquireDataStruct(
            length=254, position=0, mode=AcquireMode.RAW, output_variable="test"
        )
        package = Executable(
            channel_data={"CH1": ChannelData(acquires=acquire)},
            post_processing={"test": pp_instructions},
        )
        # TODO: remove QatIR after pipeline changes
        result = QatIR(mock_readout)
        PostProcessingTransform().run(result, package=package)
        result = result.value
        assert len(result) == 1
        assert "test" in result
        assert np.shape((result["test"])) == (1000,)
        assert np.allclose(result["test"], -1.0)


class TestInlineResultsProcessingTransform:

    def test_run_results_processing_with_program(self):
        model = get_default_echo_hardware()
        builder = model.create_builder()
        builder.measure_single_shot_binned(model.get_qubit(0), output_variable="test")
        builder.results_processing("test", InlineResultsProcessing.Program)
        package = WaveformV1Emitter(model).emit(builder)
        engine = EchoEngine()
        results = QatIR(engine.execute(package))
        InlineResultsProcessingTransform().run(results, package=package)
        results = results.value
        assert isinstance(results["test"], int)

    def test_run_results_processing_with_experiment(self):
        model = get_default_echo_hardware()
        builder = model.create_builder()
        builder.repeat(254)
        builder.measure(model.get_qubit(0), output_variable="test")
        builder.results_processing("test", InlineResultsProcessing.Experiment)
        package = WaveformV1Emitter(model).emit(builder)
        engine = EchoEngine()
        results = QatIR(engine.execute(package))
        InlineResultsProcessingTransform().run(results, package=package)
        results = results.value
        assert isinstance(results["test"], np.ndarray)
        assert len(results["test"]) == 254


class TestAssignResultsTransform:

    def test_only_returns_what_is_asked(self):
        model = get_default_echo_hardware()
        builder = model.create_builder()
        builder.repeat(254)
        builder.measure(model.get_qubit(0), output_variable="q0")
        builder.measure(model.get_qubit(0), output_variable="q1")
        builder.returns("q0")
        package = WaveformV1Emitter(model).emit(builder)
        engine = EchoEngine()
        results = QatIR(engine.execute(package))
        AssignResultsTransform().run(results, package=package)
        results = results.value
        assert "q0" in results
        assert "q1" not in results
