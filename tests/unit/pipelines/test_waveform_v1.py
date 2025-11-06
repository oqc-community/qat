# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import random

import numpy as np
import pytest
from compiler_config.config import CompilerConfig, QuantumResultsFormat, Tket

from qat import QAT
from qat.backend.waveform_v1.codegen import PydWaveformV1Backend
from qat.backend.waveform_v1.executable import WaveformV1ChannelData, WaveformV1Executable
from qat.engines.waveform_v1 import EchoEngine
from qat.executables import BaseExecutable, Executable
from qat.frontend import AutoFrontend
from qat.ir.instructions import Variable as PydVariable
from qat.ir.measure import AcquireMode, PostProcessing, PostProcessType
from qat.middleend import PydDefaultMiddleend
from qat.model.convert_purr import convert_purr_echo_hw_to_pydantic
from qat.model.loaders.lucy import LucyModelLoader
from qat.model.loaders.purr import EchoModelLoader
from qat.model.target_data import TargetData
from qat.pipelines.pipeline import CompilePipeline, ExecutePipeline, Pipeline
from qat.pipelines.purr.waveform_v1 import EchoPipeline
from qat.pipelines.waveform_v1 import (
    EchoExecutePipeline as ExperimentalEchoExecutePipeline,
)
from qat.pipelines.waveform_v1 import EchoPipeline as ExperimentalEchoPipeline
from qat.pipelines.waveform_v1 import PipelineConfig
from qat.pipelines.waveform_v1 import (
    WaveformV1CompilePipeline as ExperimentalWaveformV1CompilePipeline,
)
from qat.runtime import SimpleRuntime

from tests.unit.utils.qasm_qir import get_pipeline_tests

pytestmark = pytest.mark.experimental


class TestExperimentalEchoPipeline:
    def test_build_pipeline(self):
        """Test the build_pipeline method to ensure it constructs the pipeline correctly."""
        model = LucyModelLoader(qubit_count=4).load()
        pipeline = ExperimentalEchoPipeline._build_pipeline(
            config=PipelineConfig(name="echo"), model=model, target_data=None
        )
        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "echo"
        assert isinstance(pipeline.frontend, AutoFrontend)
        assert isinstance(pipeline.middleend, PydDefaultMiddleend)
        assert isinstance(pipeline.backend, PydWaveformV1Backend)
        assert isinstance(pipeline.runtime, SimpleRuntime)
        assert isinstance(pipeline.target_data, TargetData)
        assert pipeline.target_data == TargetData.default()
        assert isinstance(pipeline.engine, EchoEngine)

    def test_build_compile_pipeline(self):
        """Test the build_compile_pipeline method to ensure it constructs the compile pipeline correctly."""
        model = LucyModelLoader(qubit_count=4).load()
        compile_pipeline = ExperimentalWaveformV1CompilePipeline._build_pipeline(
            config=PipelineConfig(name="compile"),
            model=model,
            target_data=None,
        )
        assert isinstance(compile_pipeline, CompilePipeline)
        assert compile_pipeline.name == "compile"
        assert isinstance(compile_pipeline.frontend, AutoFrontend)
        assert isinstance(compile_pipeline.middleend, PydDefaultMiddleend)
        assert isinstance(compile_pipeline.backend, PydWaveformV1Backend)

    def test_build_execute_pipeline(self):
        """Test the build_execute_pipeline method to ensure it constructs the execute pipeline correctly."""
        model = LucyModelLoader(qubit_count=4).load()
        execute_pipeline = ExperimentalEchoExecutePipeline._build_pipeline(
            config=PipelineConfig(name="execute"),
            model=model,
            target_data=None,
        )
        assert isinstance(execute_pipeline, ExecutePipeline)
        assert execute_pipeline.name == "execute"
        assert isinstance(execute_pipeline.runtime, SimpleRuntime)
        assert isinstance(execute_pipeline.engine, EchoEngine)


test_files = get_pipeline_tests(
    openpulse=True,
    skips=[
        "cudaq-ghz.ll",
    ],
)


@pytest.mark.parametrize("shots", [1254, 10002], scope="class")
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
class TestExperimentalEchoPipelineWithCircuits:
    """A class that tests the compilation and execution of the EchoPipeline with a
    WaveformV1Backend against circuit programs.

    It tests the expectations of the compilation pipelines, inspecting the properties of
    the executable and the results returned by the EchoEngine.
    """

    target_data = TargetData.default()
    model = LucyModelLoader(qubit_count=32).load()
    pipeline = ExperimentalEchoPipeline(
        config=PipelineConfig(name="stable"), model=model, target_data=target_data
    )

    @pytest.fixture(scope="class")
    def compiler_config(self, shots, results_format) -> CompilerConfig:
        """Initialize a compiler config per program, as it is possible for the compiler
        config to be adjusted during compilation."""
        return CompilerConfig(
            repeats=shots,
            results_format=results_format,
            optimizations=Tket().default(),
        )

    @pytest.fixture(scope="class")
    def executable(
        self, program_file, compiler_config, num_readouts, num_registers
    ) -> WaveformV1Executable:
        """Compile the program file using the stable pipeline."""
        return QAT().compile(str(program_file), compiler_config, pipeline=self.pipeline)[0]

    @pytest.fixture(scope="class")
    def results(
        self, executable: WaveformV1Executable, compiler_config: CompilerConfig
    ) -> dict:
        return QAT().execute(executable, compiler_config, pipeline=self.pipeline)[0]

    @pytest.fixture(scope="class")
    def returned_acquires(self, executable: WaveformV1Executable):
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
                elif (
                    isinstance(assign.value, PydVariable) and assign.value.name in acquires
                ):
                    returned_acquires.add(assign.value.name)
                elif isinstance(assign.value, list):
                    for value in assign.value:
                        if isinstance(value, str) and value in acquires:
                            returned_acquires.add(value)
                        elif isinstance(value, PydVariable) and value.name in acquires:
                            returned_acquires.add(value.name)
                break
        return returned_acquires

    def test_executable(self, executable):
        assert isinstance(executable, WaveformV1Executable)

    def test_shots(self, executable, shots):
        assert executable.shots == shots
        assert executable.compiled_shots <= shots
        assert executable.compiled_shots <= self.target_data.max_shots

    def test_shot_period(self, executable):
        """Checks that i) the repetition time accounts for both the passive reset time and
        circuit time, and ii) the buffers for each channel do not exceed the repetition
        time."""
        passive_reset_time = self.target_data.QUBIT_DATA.passive_reset_time
        assert executable.repetition_time >= passive_reset_time
        for physical_channel, channel_data in executable.channel_data.items():
            for qubit in self.pipeline.backend.model.qubits.values():
                if physical_channel == qubit.physical_channel.uuid:
                    sample_time = self.target_data.QUBIT_DATA.sample_time
                    break
                elif physical_channel == qubit.resonator.physical_channel.uuid:
                    sample_time = self.target_data.RESONATOR_DATA.sample_time
                    break
            assert len(channel_data.buffer) * sample_time <= executable.repetition_time

    def test_channel_data(self, executable):
        """WaveformV1Executables are expected to have channel data for each physical
        channel available, regardless of if they're used."""

        assert len(executable.channel_data) == 2 * len(self.model.qubits)
        for qubit in self.model.qubits.values():
            for physical_channel in [
                qubit.physical_channel,
                qubit.resonator.physical_channel,
            ]:
                assert physical_channel.uuid in executable.channel_data
                channel_data = executable.channel_data[physical_channel.uuid]
                assert isinstance(channel_data, WaveformV1ChannelData)
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
        self, executable: WaveformV1Executable, num_registers: int
    ):
        """Check that the executable has a number of returns that matches the provided
        value. In the future, this might need adjusting to account of active reset."""

        assert len(executable.returns) == num_registers

    def test_executable_has_correct_post_processing(
        self, executable: WaveformV1Executable, request
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
        executable: WaveformV1Executable,
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
        executable: WaveformV1Executable,
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

    @pytest.mark.parametrize("cls", [BaseExecutable, Executable])
    def test_serialization(self, executable: WaveformV1Executable, cls):
        """Check that the executable can be serialized and deserialized correctly."""
        json_blob = executable.serialize()
        new_package = cls.deserialize(json_blob)
        assert isinstance(new_package, WaveformV1Executable)
        assert new_package == executable


class MockEchoModelLoader(EchoModelLoader):
    """A model loader with a fixed calibration ID and sample times for testing purposes."""

    def __init__(self, qubit_count, target_data, random_seed=None):
        super().__init__(qubit_count=qubit_count)
        self.target_data = target_data
        self._random_seed = random_seed

    def load(self):
        try:
            random.seed(self._random_seed)
            model = super().load()
            model.calibration_id = "default_id"
            sample_time_qubit = self.target_data.QUBIT_DATA.sample_time
            sample_time_resonator = self.target_data.RESONATOR_DATA.sample_time
            for qubit in model.qubits:
                qubit.measure_acquire["delay"] = 0.0
                qubit.physical_channel.sample_time = sample_time_qubit
            for resonator in model.resonators:
                resonator.physical_channel.sample_time = sample_time_resonator
        finally:
            random.seed()
        return model


parity_test_files = get_pipeline_tests(openpulse=True)


@pytest.mark.parametrize(
    "program_file",
    [val[0] for val in parity_test_files.values()],
    ids=parity_test_files.keys(),
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
    model = MockEchoModelLoader(
        qubit_count=32, target_data=target_data, random_seed=42
    ).load()
    pydantic_model = convert_purr_echo_hw_to_pydantic(model)
    stable_pipeline = EchoPipeline(
        config=PipelineConfig(name="stable"), model=model, target_data=target_data
    )
    experimental_pipeline = ExperimentalEchoPipeline(
        config=PipelineConfig(name="experimental"),
        model=pydantic_model,
        target_data=target_data,
    )

    def compiler_config(self):
        return CompilerConfig(
            results_format=QuantumResultsFormat().binary_count(),
            repeats=TargetData.default().default_shots,
            passive_reset_time=2e-8,
            optimizations=Tket().default(),
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
        pyd_model = self.experimental_pipeline.backend.model
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

    def test_results_processing(self, stable_executable, experimental_executable, request):
        if "qir-select" in request.node.callspec.id:
            pytest.xfail(
                "This is a known issue with the legacy workflow for doing TKET opts on QIR."
                " The parser will add results processing only on results that are returned,"
                " but it should add to to all measures. So there is a divergence here "
                "because the new stack handles it slightly differently to give the correct "
                "result."
            )
        assert (
            stable_executable.results_processing
            == experimental_executable.results_processing
        )

    def test_postprocessing_data(self, stable_executable, experimental_executable):
        assert stable_executable.post_processing == experimental_executable.post_processing
        # the ordering of packages, and different times when measures are done mean we
        # no longer have complete parity here.
        assert len(stable_executable.assigns) == len(experimental_executable.assigns)
        for stable, experimental in zip(
            stable_executable.assigns, experimental_executable.assigns
        ):
            if stable.name.startswith("generated_name_"):
                assert experimental.name.startswith("generated_name_")
            else:
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
            assert (
                stable_data.baseband_frequency is None
                and experimental_data.baseband_frequency is None
            ) or np.isclose(
                stable_data.baseband_frequency, experimental_data.baseband_frequency
            )

            # Since how we sync qubits before measurement has changed, we have diverged
            # from complete feature parity here. The buffers and acquires might look
            # different: check what we can, but otherwise, skip
            assert len(stable_data.acquires) == len(experimental_data.acquires)
            if len(stable_data.acquires) > 0:
                for acq1, acq2 in zip(stable_data.acquires, experimental_data.acquires):
                    assert acq1.length == acq2.length
                    assert np.allclose(
                        stable_data.buffer[acq1.position : acq1.position + acq1.length],
                        experimental_data.buffer[
                            acq2.position : acq2.position + acq2.length
                        ],
                    )
            else:
                assert np.allclose(stable_data.buffer, experimental_data.buffer)

    def test_results_formatting(self, stable_results, experimental_results):
        assert stable_results == experimental_results

    @pytest.mark.parametrize("cls", [BaseExecutable, Executable])
    def test_serialization(self, experimental_executable, cls):
        json_blob = experimental_executable.serialize()
        new_package = cls.deserialize(json_blob)
        assert experimental_executable == new_package
