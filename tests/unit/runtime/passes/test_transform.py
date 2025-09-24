# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np
from compiler_config.config import InlineResultsProcessing

from qat.executables import AcquireData, ChannelData, ChannelExecutable
from qat.ir.instructions import Assign as PydAssign
from qat.ir.instructions import Variable as PydVariable
from qat.ir.measure import PostProcessing
from qat.model.target_data import TargetData
from qat.purr.compiler.instructions import (
    AcquireMode,
    PostProcessType,
    ProcessAxis,
)
from qat.runtime.passes.transform import (
    AssignResultsTransform,
    InlineResultsProcessingTransform,
    PostProcessingTransform,
)


class TestPostProcessingTransform:
    target_data = TargetData.default()

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
        result = PostProcessingTransform(self.target_data).run(
            mock_readout, package=package
        )
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
        result = PostProcessingTransform(self.target_data).run(
            mock_readout, package=package
        )
        assert len(result) == 1
        assert "test" in result
        assert np.shape((result["test"])) == (1000,)
        assert np.allclose(result["test"], -1.0)


class TestInlineResultsProcessingTransform:
    def test_run_results_processing_with_program(self):
        results = {"test": np.random.rand(254, 100)}
        package = ChannelExecutable(
            results_processing={"test": InlineResultsProcessing.Program}
        )
        results = InlineResultsProcessingTransform().run(results, package=package)
        assert isinstance(results["test"], int)

    def test_run_results_processing_with_experiment(self):
        results = {
            "test": np.random.rand(
                254,
            )
        }
        package = ChannelExecutable(
            results_processing={"test": InlineResultsProcessing.Experiment}
        )
        InlineResultsProcessingTransform().run(results, package=package)
        assert isinstance(results["test"], np.ndarray)
        assert len(results["test"]) == 254


class TestAssignResultsTransform:
    def test_only_returns_what_is_asked(self):
        results = {"q0": np.random.rand(100), "q1": np.random.rand(100)}
        package = ChannelExecutable(returns=set(["q0"]))
        results = AssignResultsTransform().run(results, package=package)
        assert "q0" in results
        assert "q1" not in results

    def test_assigns_with_variables(self):
        results = {
            "q0": np.asarray([1] * 100),
            "q1": np.asarray([2] * 100),
            "q2": np.asarray([3] * 100),
        }
        package = ChannelExecutable(
            returns=set(["c"]),
            assigns=[
                PydAssign(
                    name="c",
                    value=[
                        PydVariable(name="q0"),
                        PydVariable(name="q1"),
                        [PydVariable(name="q2")],
                    ],
                )
            ],
            channel_data={
                "CH1": ChannelData(
                    acquires=AcquireData(
                        output_variable="q0",
                        length=100,
                        position=0,
                        mode=AcquireMode.INTEGRATOR,
                    )
                ),
                "CH2": ChannelData(
                    acquires=AcquireData(
                        output_variable="q1",
                        length=100,
                        position=0,
                        mode=AcquireMode.INTEGRATOR,
                    )
                ),
                "CH3": ChannelData(
                    acquires=AcquireData(
                        output_variable="q2",
                        length=100,
                        position=0,
                        mode=AcquireMode.INTEGRATOR,
                    )
                ),
            },
        )
        results = AssignResultsTransform().run(results, package=package)
        assert len(results) == 1
        assert "c" in results
        results = results["c"]
        assert len(results) == 3
        assert isinstance(results[0], np.ndarray)
        assert np.allclose(results[0], 1)
        assert len(results[0]) == 100
        assert isinstance(results[1], np.ndarray)
        assert np.allclose(results[1], 2)
        assert len(results[1]) == 100
        assert isinstance(results[2], list)
        assert isinstance(results[2][0], np.ndarray)
        assert np.allclose(results[2][0], 3)
        assert len(results[2][0]) == 100
