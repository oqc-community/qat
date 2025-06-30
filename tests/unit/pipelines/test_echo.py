# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
import pytest
from compiler_config.config import CompilerConfig, QuantumResultsFormat, Tket

from qat import QAT
from qat.backend.waveform_v1.codegen import PydWaveformV1Backend, WaveformV1Backend
from qat.backend.waveform_v1.executable import WaveformV1ChannelData, WaveformV1Executable
from qat.engines.waveform_v1 import EchoEngine
from qat.frontend import AutoFrontend
from qat.ir.instructions import Variable as PydVariable
from qat.ir.measure import AcquireMode, PostProcessing, PostProcessType
from qat.middleend import DefaultMiddleend
from qat.middleend.middleends import ExperimentalDefaultMiddleend
from qat.model.loaders.legacy import EchoModelLoader
from qat.model.target_data import TargetData
from qat.pipelines.echo import EchoPipeline, EchoPipelineConfig, ExperimentalEchoPipeline
from qat.pipelines.pipeline import Pipeline
from qat.runtime import SimpleRuntime
from qat.runtime.executables import Executable

from tests.unit.utils.qasm_qir import get_qasm2, get_qir, qasm2_files, short_file_name


class TestEchoPipeline:
    def test_build_pipeline(self):
        """Test the build_pipeline method to ensure it constructs the pipeline correctly."""
        model = EchoModelLoader(qubit_count=4).load()
        pipeline = EchoPipeline._build_pipeline(
            config=EchoPipelineConfig(), model=model, target_data=None
        )
        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "echo"
        assert isinstance(pipeline.frontend, AutoFrontend)
        assert isinstance(pipeline.middleend, DefaultMiddleend)
        assert isinstance(pipeline.backend, WaveformV1Backend)
        assert isinstance(pipeline.runtime, SimpleRuntime)
        assert isinstance(pipeline.target_data, TargetData)
        assert pipeline.target_data == TargetData.default()
        assert isinstance(pipeline.engine, EchoEngine)


@pytest.mark.experimental
class TestExperimentalEchoPipeline:
    def test_build_pipeline(self):
        """Test the build_pipeline method to ensure it constructs the pipeline correctly."""
        model = EchoModelLoader(qubit_count=4).load()
        pipeline = ExperimentalEchoPipeline._build_pipeline(
            config=EchoPipelineConfig(), model=model, target_data=None
        )
        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "echo"
        assert isinstance(pipeline.frontend, AutoFrontend)
        assert isinstance(pipeline.middleend, ExperimentalDefaultMiddleend)
        assert isinstance(pipeline.backend, PydWaveformV1Backend)
        assert isinstance(pipeline.runtime, SimpleRuntime)
        assert isinstance(pipeline.target_data, TargetData)
        assert pipeline.target_data == TargetData.default()
        assert isinstance(pipeline.engine, EchoEngine)


# Signature is circuit_name: (loader_function, num_readouts, num_classical_registers)
test_files = {
    "basic.qasm": (lambda: get_qasm2("basic.qasm"), 2, 1),
    "ecr.qasm": (lambda: get_qasm2("ecr.qasm"), 2, 1),
    "basic_single_measures.qasm": (lambda: get_qasm2("basic_single_measures.qasm"), 2, 1),
    "random_n5_d5.qasm": (lambda: get_qasm2("random_n5_d5.qasm"), 5, 1),
    "bell_psi_plus.qasm": (lambda: get_qasm2("bell_psi_plus.qasm"), 2, 1),
    "decoupled.qasm": (lambda: get_qasm2("decoupled.qasm"), 2, 2),
    "bell_psi_plus.ll": (lambda: get_qir("bell_psi_plus.ll"), 2, 1),
    "basic_cudaq.ll": (lambda: get_qir("basic_cudaq.ll"), 1, 1),
    "cudaq-ghz.ll": (lambda: get_qir("cudaq-ghz.ll"), 3, 1),
}


# TODO: update to use mixed files, QASM3 is missing and maybe some QIR use cases
# (COMPILER-608)
@pytest.mark.parametrize("shots", [1254, 10002], scope="class")
@pytest.mark.parametrize("passive_reset_time", [1e-3], scope="class")
@pytest.mark.parametrize(
    "file_loader, num_readouts, num_registers",
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
class TestEchoPipelineWithCircuits:
    """A class that tests the compilation and execution of the EchoPipeline with a
    WaveformV1Backend against circuit programs.

    It tests the expectations of the compilation pipelines, inspecting the properties of
    the executable and the results returned by the EchoEngine.
    """

    target_data = TargetData.default()
    model = EchoModelLoader(qubit_count=32).load()
    pipeline = EchoPipeline(
        config=EchoPipelineConfig(name="stable"), model=model, target_data=target_data
    )

    @pytest.fixture(scope="class")
    def program_file(self, file_loader, num_readouts, num_registers) -> str:
        return file_loader()

    @pytest.fixture(scope="class")
    def compiler_config(
        self, shots, passive_reset_time, results_format, file_loader
    ) -> CompilerConfig:
        """Initialize a compiler config per program, as it is possible for the compiler
        config to be adjusted during compilation."""
        return CompilerConfig(
            repeats=shots,
            results_format=results_format,
            passive_reset_time=passive_reset_time,
            optimizations=Tket().default(),
        )

    @pytest.fixture(scope="class")
    def executable(self, program_file, compiler_config) -> WaveformV1Executable:
        """Compile the program file using the stable pipeline."""
        return QAT().compile(str(program_file), compiler_config, pipeline=self.pipeline)[0]

    @pytest.fixture(scope="class")
    def results(
        self, executable: WaveformV1Executable, compiler_config: CompilerConfig
    ) -> dict:
        return QAT().execute(executable, compiler_config, pipeline=self.pipeline)[0]

    def test_executable(self, executable):
        assert isinstance(executable, WaveformV1Executable)

    def test_shots(self, executable, shots):
        assert executable.shots == shots
        assert executable.compiled_shots <= shots
        assert executable.compiled_shots <= self.target_data.max_shots

    def test_repetition_period(self, executable, passive_reset_time):
        """Checks that i) the repetition time accounts for both the passive reset time and
        circuit time, and ii) the buffers for each channel do not exceed the repetition
        time."""
        assert executable.repetition_time > passive_reset_time
        for physical_channel, channel_data in executable.channel_data.items():
            sample_time = self.model.physical_channels[physical_channel].sample_time
            assert len(channel_data.buffer) * sample_time <= executable.repetition_time

    def test_channel_data(self, executable):
        """WaveformV1Executables are expected to have channel data for each physical
        channel available, regardless of if they're used."""

        assert len(executable.channel_data) == len(self.model.physical_channels)
        for physical_channel in self.model.physical_channels.values():
            assert physical_channel.id in executable.channel_data
            channel_data = executable.channel_data[physical_channel.id]
            assert isinstance(channel_data, WaveformV1ChannelData)
            assert channel_data.baseband_frequency in (
                physical_channel.baseband.frequency,
                None,
            )

    def test_executable_has_correct_number_of_acquires(self, executable, num_readouts):
        """Check that the executable has a number of acquires that matches the provided
        value. In the future, this might need adjusting to account of active reset."""

        assert len(executable.acquires) == num_readouts

    def test_acquires_are_integrator_mode(self, executable):
        """Check that all acquires are in INTEGRATOR mode, as this is the expected mode for
        the EchoPipeline."""

        for acquire in executable.acquires:
            assert acquire.mode == AcquireMode.INTEGRATOR

    def test_executable_has_correct_returns(
        self, executable: WaveformV1Executable, num_registers: int
    ):
        """Check that the executable has a number of returns that matches the provided
        value. In the future, this might need adjusting to account of active reset."""

        assert len(executable.returns) == num_registers

    def test_executable_has_correct_post_processing(self, executable: WaveformV1Executable):
        """Each acquisition will be acquired using the INTEGRATOR mode, and will need
        correctly post-processing to be discriminated as a bit. We can assume the acquire
        mode is INTEGRATOR."""

        for acquire in executable.acquires:
            output_variable = acquire.output_variable
            assert isinstance(output_variable, str)
            assert output_variable in executable.post_processing
            pps = executable.post_processing[output_variable]
            assert len(pps) == 1
            assert isinstance(pps[0], PostProcessing)
            assert pps[0].process_type == PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL

    def test_executable_has_correct_results_processing(
        self, executable: WaveformV1Executable, compiler_config: CompilerConfig
    ):
        """Check that the executable has a results processing that matches the provided
        value."""
        for acquire in executable.acquires:
            output_variable = acquire.output_variable
            assert isinstance(output_variable, str)
            assert output_variable in executable.results_processing
            rp = executable.results_processing[output_variable]
            assert rp == compiler_config.results_format.format

    def test_results_contain_all_returns(
        self,
        results: dict[str],
        executable: WaveformV1Executable,
        shots: int,
        num_readouts: int,
        request,
    ):
        """Checks that the results of the zero engine match the expected format."""

        returns = executable.returns
        total_length = 0
        for output_variable in returns:
            if output_variable.startswith("generated"):
                result = results
            else:
                assert isinstance(results, dict)
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
                assert all(len(val) == shots for val in result)

        assert total_length == num_readouts

    def test_serialization(self, executable: WaveformV1Executable):
        """Check that the executable can be serialized and deserialized correctly."""
        json_blob = executable.serialize()
        new_package = Executable.deserialize(json_blob)
        assert isinstance(new_package, WaveformV1Executable)
        assert new_package == executable


class MockEchoModelLoader(EchoModelLoader):
    """A model loader with a fixed calibration ID and sample times for testing purposes."""

    def __init__(self, qubit_count, target_data):
        super().__init__(qubit_count=qubit_count)
        self.target_data = target_data

    def load(self):
        model = super().load()
        model.calibration_id = "default_id"
        sample_time_qubit = self.target_data.QUBIT_DATA.sample_time
        sample_time_resonator = self.target_data.RESONATOR_DATA.sample_time
        for qubit in model.qubits:
            qubit.measure_acquire["delay"] = 0.0
            qubit.physical_channel.sample_time = sample_time_qubit
        for resonator in model.resonators:
            resonator.physical_channel.sample_time = sample_time_resonator
        return model


@pytest.mark.experimental
@pytest.mark.parametrize(
    "program_file",
    qasm2_files,  # TODO: update to mixed_files COMPILER-608
    ids=short_file_name,
    scope="class",
)
class TestExperimentalEchoPipelineParity:
    """
    Parity tests specifically targeted at the echo engine in new QAT and the
    experimental stack that incrementally uses the Pydantic stack. Aims to test:

    - New echo pipelines: Does compile and execute work in the way we expect using the new
      QAT infrastructure?
    - Code generation to a "waveform" executable: do the buffers we send to the target
      machine match those generated by legacy execution engines?
    - Execution and results processing: are the results outputted in consistent formats?
    """

    target_data = TargetData.default()
    model = MockEchoModelLoader(qubit_count=32, target_data=target_data).load()
    stable_pipeline = EchoPipeline(
        config=EchoPipelineConfig(name="stable"), model=model, target_data=target_data
    )
    experimental_pipeline = ExperimentalEchoPipeline(
        config=EchoPipelineConfig(name="experimental"), model=model, target_data=target_data
    )

    def compiler_config(self):
        return CompilerConfig(
            results_format=QuantumResultsFormat().binary_count(),
            repeats=TargetData.default().default_shots,
            passive_reset_time=2e-8,
        )

    @pytest.fixture(scope="class")
    def stable_config(self, program_file):
        # This is intentionally parametrized with program_file so a new instance is created
        # per program.
        return self.compiler_config()

    @pytest.fixture(scope="class")
    def experimental_config(self, program_file):
        # This is intentionally parametrized with program_file so a new instance is created
        # per program.
        return self.compiler_config()

    @pytest.fixture(scope="class")
    def physical_channel_map(self):
        pyd_model = self.experimental_pipeline.backend.pyd_model
        physical_channel_map = {}
        for index, qubit in pyd_model.qubits.items():
            legacy_qubit = self.model.get_qubit(index)
            physical_channel_map[legacy_qubit.physical_channel.id] = (
                qubit.physical_channel.uuid
            )
            physical_channel_map[legacy_qubit.measure_device.physical_channel.id] = (
                qubit.resonator.physical_channel.uuid
            )
        return physical_channel_map

    @pytest.fixture(scope="class")
    def stable_executable(self, program_file, stable_config):
        return QAT().compile(
            str(program_file), stable_config, pipeline=self.stable_pipeline
        )[0]

    @pytest.fixture(scope="class")
    def experimental_executable(self, program_file, experimental_config):
        return QAT().compile(
            str(program_file), experimental_config, pipeline=self.experimental_pipeline
        )[0]

    @pytest.fixture(scope="class")
    def stable_results(self, stable_executable, stable_config):
        return QAT().execute(
            stable_executable, stable_config, pipeline=self.stable_pipeline
        )[0]

    @pytest.fixture(scope="class")
    def experimental_results(self, experimental_executable, experimental_config):
        return QAT().execute(
            experimental_executable,
            experimental_config,
            pipeline=self.experimental_pipeline,
        )[0]

    def test_package_metadata(self, stable_executable, experimental_executable):
        assert stable_executable.calibration_id == experimental_executable.calibration_id
        assert stable_executable.compiled_shots == experimental_executable.compiled_shots
        assert stable_executable.post_processing == experimental_executable.post_processing
        assert np.isclose(
            stable_executable.repetition_time, experimental_executable.repetition_time
        )
        assert stable_executable.shots == experimental_executable.shots

    def test_postprocessing_data(self, stable_executable, experimental_executable):
        assert (
            stable_executable.results_processing
            == experimental_executable.results_processing
        )
        assert stable_executable.post_processing == experimental_executable.post_processing
        assert stable_executable.acquires == experimental_executable.acquires
        for stable, experimental in zip(
            stable_executable.assigns, experimental_executable.assigns
        ):
            assert stable.name == experimental.name
            if isinstance(stable.value, list):
                for sv, ev in zip(stable.value, experimental.value):
                    assert sv == (ev.name if isinstance(ev, PydVariable) else ev)
            else:
                assert stable.value == experimental.value.name

    def test_subpackages(
        self, stable_executable, experimental_executable, physical_channel_map
    ):
        for key, stable_data in stable_executable.channel_data.items():
            new_key = physical_channel_map[key]
            assert new_key in experimental_executable.channel_data
            assert isinstance(stable_data, WaveformV1ChannelData)
            experimental_data = experimental_executable.channel_data[new_key]
            assert np.allclose(stable_data.buffer, experimental_data.buffer)
            assert (
                stable_data.baseband_frequency is None
                and experimental_data.baseband_frequency is None
            ) or np.isclose(
                stable_data.baseband_frequency, experimental_data.baseband_frequency
            )
            assert stable_data.acquires == experimental_data.acquires

    def test_results_formatting(self, stable_results, experimental_results):
        assert isinstance(stable_results, dict)
        assert isinstance(experimental_results, dict)
        assert stable_results.keys() == experimental_results.keys()
        for key in stable_results.keys():
            assert stable_results[key] == experimental_results[key]

    def test_serialization(self, experimental_executable):
        json_blob = experimental_executable.serialize()
        new_package = Executable.deserialize(json_blob)
        assert experimental_executable == new_package
