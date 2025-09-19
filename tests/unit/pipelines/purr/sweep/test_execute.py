# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from itertools import product
from random import randint

import numpy as np
import pytest

from qat.engines.zero import ZeroEngine
from qat.executables import (
    AcquireData,
    BatchedExecutable,
    Executable,
    ParameterizedExecutable,
)
from qat.ir.measure import AcquireMode, PostProcessing, PostProcessType, ProcessAxis
from qat.model.loaders.purr.echo import EchoModelLoader
from qat.pipelines.pipeline import ExecutePipeline
from qat.pipelines.purr.sweep.execute import ExecuteSweepPipeline
from qat.pipelines.purr.waveform_v1 import WaveformV1CompilePipeline
from qat.runtime import SimpleRuntime


class MockExecutable(Executable):
    acquire_data: list[AcquireData] = []

    @property
    def acquires(self) -> list[AcquireData]:
        return self.acquire_data

    @classmethod
    def create_with_mean_signals(cls, *output_variables: str):
        """A factory for creating a mock executable with the correct post-processing
        instructions to return the mean signal for each given result."""

        acquire_data = [
            AcquireData(
                length=800, position=0, mode=AcquireMode.INTEGRATOR, output_variable=var
            )
            for var in output_variables
        ]
        post_processing = {
            var: [
                PostProcessing(
                    output_variable=var,
                    process_type=PostProcessType.MEAN,
                    axes=[ProcessAxis.SEQUENCE],
                )
            ]
            for var in output_variables
        }
        returns = set(output_variables)
        return cls(
            acquire_data=acquire_data,
            post_processing=post_processing,
            returns=returns,
            shots=1000,
            compiled_shots=1000,
        )

    @classmethod
    def create_with_single_shot_z(cls, *output_variables: str):
        """A factory for creating a mock executable with the correct post-processing
        instructions to return the single shot z-measurements for each given result."""

        acquire_data = [
            AcquireData(
                length=800, position=0, mode=AcquireMode.INTEGRATOR, output_variable=var
            )
            for var in output_variables
        ]

        post_processing = {
            var: [
                PostProcessing(
                    output_variable=var,
                    process_type=PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
                    axes=[],
                    args=[0.5, 1.0],
                )
            ]
            for var in output_variables
        }
        returns = set(output_variables)
        return cls(
            acquire_data=acquire_data,
            post_processing=post_processing,
            returns=returns,
            shots=1000,
            compiled_shots=1000,
        )

    @classmethod
    def create_with_mean_z(cls, *output_variables: str):
        """A factory for creating a mock executable with the correct post-processing
        instructions to return the mean z-measurement for each given result."""

        acquire_data = [
            AcquireData(
                length=800, position=0, mode=AcquireMode.INTEGRATOR, output_variable=var
            )
            for var in output_variables
        ]

        post_processing = {
            var: [
                PostProcessing(
                    output_variable=var,
                    process_type=PostProcessType.MEAN,
                    axes=[ProcessAxis.SEQUENCE],
                ),
                PostProcessing(
                    output_variable=var,
                    process_type=PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
                    axes=[],
                    args=[0.5, 1.0],
                ),
            ]
            for var in output_variables
        }
        returns = set(output_variables)
        return cls(
            acquire_data=acquire_data,
            post_processing=post_processing,
            returns=returns,
            shots=1000,
            compiled_shots=1000,
        )


class TestExecuteSweepPipeline:
    model = EchoModelLoader(qubit_count=4).load()
    pipeline = ExecutePipeline(
        name="zero",
        model=model,
        runtime=SimpleRuntime(engine=ZeroEngine()),
    )
    sweep_pipeline = ExecuteSweepPipeline(base_pipeline=pipeline)

    def create_sweep_parameters(self, num_sweeps: int):
        param_sizes = tuple([randint(2, 10) for _ in range(num_sweeps)])
        params = {f"param_{j}": list(range(param_sizes[j])) for j in range(num_sweeps)}
        return params, param_sizes

    def create_batch_executable(self, params, executable):
        param_combinations = [
            dict(zip(params.keys(), values)) for values in product(*params.values())
        ]
        executable_list = [
            ParameterizedExecutable(executable=executable, param=param)
            for param in param_combinations
        ]
        batched_executable = BatchedExecutable(
            executables=executable_list, shape=tuple([len(v) for v in params.values()])
        )
        return batched_executable

    def test_init_with_non_execute_pipeline_raises(self):
        with pytest.raises(
            TypeError, match="The base pipeline must be an ExecutePipeline."
        ):
            ExecuteSweepPipeline(
                base_pipeline=WaveformV1CompilePipeline(
                    config=dict(name="test"), model=self.model
                )
            )

    def test_execute_rejects_non_batched_executable(self):
        with pytest.raises(
            TypeError, match="ExecuteSweepPipeline expects a BatchedExecutable."
        ):
            self.sweep_pipeline.execute(executable="not a batched executable")

    @pytest.mark.parametrize("num_qubits", [1, 2])
    @pytest.mark.parametrize("num_sweeps", [1, 2, 3])
    def test_with_mock_mean_signal(self, num_qubits, num_sweeps):
        executable = MockExecutable.create_with_mean_signals(
            *[f"Q{i}" for i in range(num_qubits)]
        )
        params, param_sizes = self.create_sweep_parameters(num_sweeps)
        batched_executable = self.create_batch_executable(params, executable)
        results, _ = self.sweep_pipeline.execute(batched_executable)

        assert isinstance(results, dict)
        assert len(results) == num_qubits
        for i in range(num_qubits):
            assert f"Q{i}" in results
            sub_results = results[f"Q{i}"]
            np_results = np.array(sub_results)
            assert np.shape(np_results) == param_sizes

    @pytest.mark.parametrize("num_qubits", [1, 2])
    @pytest.mark.parametrize("num_sweeps", [1, 2, 3])
    def test_with_mock_single_shot_z(self, num_qubits, num_sweeps):
        executable = MockExecutable.create_with_single_shot_z(
            *[f"Q{i}" for i in range(num_qubits)]
        )
        params, param_sizes = self.create_sweep_parameters(num_sweeps)
        batched_executable = self.create_batch_executable(params, executable)
        results, _ = self.sweep_pipeline.execute(batched_executable)

        assert isinstance(results, dict)
        assert len(results) == num_qubits
        for i in range(num_qubits):
            assert f"Q{i}" in results
            sub_results = results[f"Q{i}"]
            np_results = np.array(sub_results)
            assert np.shape(np_results) == tuple(list(param_sizes) + [1000])

    @pytest.mark.parametrize("num_qubits", [1, 2])
    @pytest.mark.parametrize("num_sweeps", [1, 2, 3])
    def test_with_mock_mean_z(self, num_qubits, num_sweeps):
        executable = MockExecutable.create_with_mean_z(
            *[f"Q{i}" for i in range(num_qubits)]
        )
        params, param_sizes = self.create_sweep_parameters(num_sweeps)
        batched_executable = self.create_batch_executable(params, executable)
        results, _ = self.sweep_pipeline.execute(batched_executable)

        assert isinstance(results, dict)
        assert len(results) == num_qubits
        for i in range(num_qubits):
            assert f"Q{i}" in results
            sub_results = results[f"Q{i}"]
            np_results = np.array(sub_results)
            assert np.shape(np_results) == param_sizes
