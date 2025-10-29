# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest
from compiler_config.config import CompilerConfig, QuantumResultsFormat, Tket

from qat import QAT
from qat.backend.qblox.codegen import QbloxBackend1, QbloxBackend2
from qat.backend.qblox.config.constants import QbloxTargetData, TargetData
from qat.backend.qblox.execution import QbloxExecutable

# TODO: Pipelines using `QbloxEngine`: COMPILER-730
# from qat.engines.qblox.live import QbloxEngine
from qat.executables import BaseExecutable
from qat.frontend import AutoFrontend
from qat.ir.measure import AcquireMode, PostProcessing, PostProcessType
from qat.middleend import CustomMiddleend, DefaultMiddleend
from qat.model.loaders.purr import QbloxDummyModelLoader
from qat.pipelines.pipeline import CompilePipeline, ExecutePipeline, Pipeline
from qat.pipelines.purr.qblox import (
    PipelineConfig,
    QbloxCompilePipeline1,
    QbloxExecutePipeline,
    QbloxPipeline,
)
from qat.pipelines.purr.qblox.compile import QbloxCompilePipeline2
from qat.purr.backends.qblox.live import QbloxLiveEngineAdapter
from qat.runtime import SimpleRuntime

from tests.unit.utils.qasm_qir import (
    get_pipeline_tests,
)


class TestQbloxPipeline:
    def test_build_pipeline(self):
        """Test the build_pipeline method to ensure it constructs the pipeline correctly."""
        model = QbloxDummyModelLoader(qubit_count=4).load()
        pipeline = QbloxPipeline._build_pipeline(
            config=PipelineConfig(name="qblox"), model=model, target_data=None
        )
        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "qblox"
        assert isinstance(pipeline.frontend, AutoFrontend)
        assert isinstance(pipeline.middleend, DefaultMiddleend)
        assert isinstance(pipeline.backend, QbloxBackend2)
        assert isinstance(pipeline.runtime, SimpleRuntime)
        assert isinstance(pipeline.target_data, TargetData)
        assert pipeline.target_data == QbloxTargetData.default()
        # TODO: Pipelines using `QbloxEngine`: COMPILER-730
        assert isinstance(pipeline.engine, QbloxLiveEngineAdapter)

    def test_build_compile_pipeline(self):
        """Test the build_compile_pipeline method to ensure it constructs the compile pipeline correctly."""
        model = QbloxDummyModelLoader(qubit_count=4).load()
        compile_pipeline = QbloxPipeline._build_pipeline(
            config=PipelineConfig(name="qblox_compile"), model=model, target_data=None
        )
        assert isinstance(compile_pipeline, CompilePipeline)
        assert compile_pipeline.name == "qblox_compile"
        assert isinstance(compile_pipeline.frontend, AutoFrontend)
        assert isinstance(compile_pipeline.middleend, DefaultMiddleend)
        assert isinstance(compile_pipeline.backend, QbloxBackend2)
        assert isinstance(compile_pipeline.target_data, TargetData)

    def test_build_execute_pipeline(self):
        """Test the build_execute_pipeline method to ensure it constructs the execute pipeline correctly."""
        model = QbloxDummyModelLoader(qubit_count=4).load()
        execute_pipeline = QbloxPipeline._build_pipeline(
            config=PipelineConfig(name="qblox_execute"), model=model, target_data=None
        )
        assert isinstance(execute_pipeline, ExecutePipeline)
        assert execute_pipeline.name == "qblox_execute"
        assert isinstance(execute_pipeline.target_data, TargetData)
        assert isinstance(execute_pipeline.runtime, SimpleRuntime)
        # TODO: Pipelines using `QbloxEngine`: COMPILER-730
        assert isinstance(execute_pipeline.engine, QbloxLiveEngineAdapter)


class TestQbloxCompilePipeline:
    def test_build_pipeline1(self):
        """Test the build_pipeline method to ensure it constructs the pipeline correctly."""
        model = QbloxDummyModelLoader(qubit_count=4).load()
        compile_pipeline = QbloxCompilePipeline1._build_pipeline(
            config=PipelineConfig(name="qblox_compile"), model=model, target_data=None
        )
        assert isinstance(compile_pipeline, CompilePipeline)
        assert compile_pipeline.name == "qblox_compile"
        assert isinstance(compile_pipeline.frontend, AutoFrontend)
        assert isinstance(compile_pipeline.middleend, CustomMiddleend)
        assert isinstance(compile_pipeline.backend, QbloxBackend1)
        assert isinstance(compile_pipeline.target_data, TargetData)

    def test_build_pipeline2(self):
        """Test the build_pipeline method to ensure it constructs the pipeline correctly."""
        model = QbloxDummyModelLoader(qubit_count=4).load()
        compile_pipeline = QbloxCompilePipeline2._build_pipeline(
            config=PipelineConfig(name="qblox_compile"), model=model, target_data=None
        )
        assert isinstance(compile_pipeline, CompilePipeline)
        assert compile_pipeline.name == "qblox_compile"
        assert isinstance(compile_pipeline.frontend, AutoFrontend)
        assert isinstance(compile_pipeline.middleend, CustomMiddleend)
        assert isinstance(compile_pipeline.backend, QbloxBackend2)
        assert isinstance(compile_pipeline.target_data, TargetData)


class TestQbloxExecutePipeline:
    def test_build_pipeline(self):
        """Test the build_pipeline method to ensure it constructs the pipeline correctly."""
        model = QbloxDummyModelLoader(qubit_count=4).load()
        execute_pipeline = QbloxExecutePipeline._build_pipeline(
            config=PipelineConfig(name="echo_execute"), model=model, target_data=None
        )
        assert isinstance(execute_pipeline, ExecutePipeline)
        assert execute_pipeline.name == "echo_execute"
        assert isinstance(execute_pipeline.target_data, TargetData)
        assert isinstance(execute_pipeline.runtime, SimpleRuntime)
        # TODO: Pipelines using `QbloxEngine`: COMPILER-730
        assert isinstance(execute_pipeline.engine, QbloxLiveEngineAdapter)


# test_files = get_pipeline_tests(openpulse=True)
# TODO: 2Q Gate support: COMPILER-685
coupled_programs = [
    "bell_psi_plus.qasm",
    "ecr_exists.qasm",
    "example.qasm",
    "ghz.qasm",
    "logic_example.qasm",
    "random_n5_d5.qasm",
    "qft_5q.qasm",
    "primitives.qasm",
]
test_files = get_pipeline_tests(qasm3=False, qir=False, skips=coupled_programs)


# TODO: COMPILER-729
@pytest.mark.skip(reason="QbloxBackend2 not yet compatible with DefaultMiddleend.")
@pytest.mark.parametrize("shots", [1254, 10002], scope="class")
@pytest.mark.parametrize("passive_reset_time", [1e-3], scope="class")
@pytest.mark.parametrize(
    "program_file, num_readouts, num_registers",
    test_files.values(),
    ids=test_files.keys(),
    scope="class",
)
@pytest.mark.parametrize(
    "results_format",
    [
        QuantumResultsFormat().binary_count(),
        QuantumResultsFormat().binary(),
        QuantumResultsFormat().raw(),
    ],
    ids=["binary_count", "binary", "raw"],
    scope="class",
)
class TestQbloxPipelineWithCircuits:
    """A class that tests the compilation and execution of the QbloxPipeline with a
    QbloxBackend2 against circuit programs.

    It tests the expectations of the compilation pipelines, inspecting the properties of
    the executable and the results returned by the QbloxLiveEngineAdapter.
    """

    target_data = QbloxTargetData.default()
    # TODO: 32Q support: COMPILER-728
    model = QbloxDummyModelLoader(qubit_count=16).load()
    pipeline = QbloxPipeline(
        config=PipelineConfig(name="stable"), model=model, target_data=target_data
    )

    @pytest.fixture(scope="class")
    def compiler_config(self, shots, passive_reset_time, results_format) -> CompilerConfig:
        """Initialize a compiler config per program, as it is possible for the compiler
        config to be adjusted during compilation."""
        return CompilerConfig(
            repeats=shots,
            results_format=results_format,
            passive_reset_time=passive_reset_time,
            optimizations=Tket().default(),
        )

    @pytest.fixture(scope="class")
    def executable(
        self, program_file, compiler_config, num_readouts, num_registers
    ) -> QbloxExecutable:
        """Compile the program file using the stable pipeline."""
        return QAT().compile(str(program_file), compiler_config, pipeline=self.pipeline)[0]

    @pytest.fixture(scope="class")
    def results(self, executable: QbloxExecutable, compiler_config: CompilerConfig) -> dict:
        return QAT().execute(executable, compiler_config, pipeline=self.pipeline)[0]

    @pytest.fixture(scope="class")
    def returned_acquires(self, executable: QbloxExecutable):
        """Returns the acquires that are returned by the executable."""
        returned_acquires = set()
        acquires = [acq.output_variable for acq in executable.acquires]
        for return_ in executable.returns:
            if return_ in executable.acquires:
                returned_acquires.add(return_)
                continue

            for assign in executable.assigns:
                if assign.name != return_:
                    continue

                if isinstance(assign.value, str) and assign.value in acquires:
                    returned_acquires.add(assign.value)
                elif isinstance(assign.value, list):
                    for value in assign.value:
                        if isinstance(value, str) and value in acquires:
                            returned_acquires.add(value)
                break
        return returned_acquires

    def test_executable(self, executable):
        assert isinstance(executable, QbloxExecutable)

    def test_shots(self, executable, shots):
        assert executable.shots == shots
        assert executable.compiled_shots <= shots
        assert executable.compiled_shots <= self.target_data.max_shots

    def test_repetition_period(self, executable, passive_reset_time):
        """Checks that i) the repetition time accounts for both the passive reset time and
        circuit time, and ii) the buffers for each channel do not exceed the repetition
        time."""
        assert executable.repetition_time >= passive_reset_time
        for physical_channel, channel_data in executable.channel_data.items():
            sample_time = self.model.physical_channels[physical_channel].sample_time
            assert len(channel_data.buffer) * sample_time <= executable.repetition_time

    def test_channel_data(self, executable):
        """QbloxExecutables are expected to have channel data for each physical
        channel available, regardless of if they're used."""

        assert len(executable.channel_data) == len(self.model.physical_channels)
        for physical_channel in self.model.physical_channels.values():
            assert physical_channel.id in executable.channel_data
            channel_data = executable.channel_data[physical_channel.id]
            # TODO: What type is the channel_data?
            # assert isinstance(channel_data, WaveformV1ChannelData)
            assert channel_data.baseband_frequency in (
                physical_channel.baseband.frequency,
                None,
            )

    def test_executable_has_correct_number_of_acquires(
        self, returned_acquires, num_readouts
    ):
        """Check that the executable has a number of acquires that matches the provided
        value. In the future, this might need adjusting to account of active reset."""

        assert len(returned_acquires) == num_readouts

    def test_acquires_are_integrator_mode(self, executable, request):
        """Check that all acquires are in INTEGRATOR mode, as this is the expected mode for
        the EchoPipeline."""

        if "openpulse" in request.node.callspec.id:
            pytest.mark.skip("Openpulse has more expressive use of acquires.")

        for acquire in executable.acquires:
            assert acquire.mode == AcquireMode.INTEGRATOR

    def test_executable_has_correct_returns(
        self, executable: QbloxExecutable, num_registers: int
    ):
        """Check that the executable has a number of returns that matches the provided
        value. In the future, this might need adjusting to account of active reset."""

        assert len(executable.returns) == num_registers

    def test_executable_has_correct_post_processing(
        self, executable: QbloxExecutable, request
    ):
        """Each acquisition will be acquired using the INTEGRATOR mode, and will need
        correctly post-processing to be discriminated as a bit. We can assume the acquire
        mode is INTEGRATOR."""

        if "openpulse" in request.node.callspec.id:
            pytest.mark.skip("Openpulse has more expressive use of acquires.")

        for acquire in executable.acquires:
            output_variable = acquire.output_variable
            assert isinstance(output_variable, str)
            assert output_variable in executable.post_processing
            pps = executable.post_processing[output_variable]
            assert len(pps) == 1
            assert isinstance(pps[0], PostProcessing)
            assert pps[0].process_type == PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL

    def test_executable_has_correct_results_processing(
        self,
        executable: QbloxExecutable,
        compiler_config: CompilerConfig,
        returned_acquires: set[str],
    ):
        """Check that the executable has a results processing that matches the provided
        value."""

        for acquire in returned_acquires:
            assert acquire in executable.results_processing
            rp = executable.results_processing[acquire]
            assert rp == compiler_config.results_format.format

    def test_results_contain_all_returns(
        self,
        results: dict[str],
        executable: QbloxExecutable,
        shots: int,
        returned_acquires: set[str],
        request,
    ):
        """Checks that the results of the zero engine match the expected format."""

        returns = executable.returns

        if len(returns) == 0:
            assert len(results) == 0
            return

        if not (len(returns) == 1 and next(iter(returns)).startswith("generated")):
            # checking for multiple registers basically; generated registers get stripped
            # away
            assert isinstance(results, dict)
            assert len(returns) == len(results)

        total_length = 0
        for output_variable in returns:
            if output_variable.startswith("generated"):
                result = results
            else:
                assert output_variable in results
                result = results[output_variable]
            if "binary_count" in request.node.callspec.id:
                # Binary count stores as a dictionary {bitstring: counts}
                assert isinstance(result, dict)
                assert all(isinstance(key, str) for key in result)
                lengths = [len(key) for key in result]
                assert all(length == lengths[0] for length in lengths)
                total_length += lengths[0]
                assert sum(result.values()) == shots
            elif "binary" in request.node.callspec.id:
                # Binary stores a list of most seen bits: [0, 1, 0, ...]
                assert isinstance(result, list)
                assert all(val in (0, 1) for val in result)
                total_length += len(result)
            else:
                # Raw is a list of list of all readouts: [[2.54, 2.54, ...], ...]
                assert isinstance(result, list)
                total_length += len(result)
                assert all(len(val) == shots for val in result if isinstance(val, list))

        # returned acquires dont account for all classical bits if a bigger classical
        # register is used.
        assert total_length >= len(returned_acquires)

    @pytest.mark.parametrize("cls", [BaseExecutable, QbloxExecutable])
    def test_serialization(self, executable: QbloxExecutable, cls):
        """Check that the executable can be serialized and deserialized correctly."""
        json_blob = executable.serialize()
        new_package = cls.deserialize(json_blob)
        assert isinstance(new_package, QbloxExecutable)
        assert new_package == executable
