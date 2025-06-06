# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.backend.passes.legacy.lowering import PartitionByPulseChannel, PartitionedIR
from qat.model.loaders.legacy import EchoModelLoader
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    Assign,
    Delay,
    InlineResultsProcessing,
    PostProcessing,
    PostProcessType,
    ProcessAxis,
    Pulse,
    PulseShapeType,
    ResultsProcessing,
    Return,
)


class TestPartitionByPulseChannel:
    model = EchoModelLoader().load()

    @pytest.fixture(scope="class")
    def builder(self):
        builder = self.model.create_builder()
        drive_chan = self.model.qubits[0].get_drive_channel()
        acq_chan = self.model.qubits[0].get_acquire_channel()
        measure_chan = self.model.qubits[0].get_measure_channel()

        builder.repeat(1000)
        builder.delay(drive_chan, 80e-9)
        builder.pulse(measure_chan, shape=PulseShapeType.SQUARE, width=240e-9)
        builder.delay(acq_chan, 40e-9)
        builder.acquire(
            acq_chan,
            time=200e-9,
            mode=AcquireMode.INTEGRATOR,
            output_variable="test",
            delay=0.0,
        )
        acq = builder.instructions[-1]
        builder.post_processing(
            acq,
            process=PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
            axes=ProcessAxis.SEQUENCE,
            args=[0.0, 1.0],
        )
        builder.post_processing(
            acq,
            process=PostProcessType.DISCRIMINATE,
            axes=ProcessAxis.SEQUENCE,
            args=[0.0, 1.0],
        )
        builder.results_processing("test", InlineResultsProcessing.Binary)
        builder.assign("test2", "test")
        builder.returns("test")
        builder.returns("test2")
        return builder

    @pytest.fixture(scope="class")
    def partitioned_ir(self, builder):
        return PartitionByPulseChannel().run(builder)

    def test_is_partitioned_ir(self, partitioned_ir):
        assert isinstance(partitioned_ir, PartitionedIR)

    def test_repeats(self, partitioned_ir):
        assert isinstance(partitioned_ir.shots, int)
        assert partitioned_ir.shots == 1000
        assert partitioned_ir.compiled_shots == 1000
        assert partitioned_ir.repetition_period is None

    def test_assigns(self, partitioned_ir):
        assert isinstance(partitioned_ir.assigns, list)
        assert len(partitioned_ir.assigns) == 1
        assert isinstance(partitioned_ir.assigns[0], Assign)

    def test_returns(self, partitioned_ir):
        assert isinstance(partitioned_ir.returns, list)
        assert len(partitioned_ir.returns) == 2
        assert isinstance(partitioned_ir.returns[0], Return)
        assert isinstance(partitioned_ir.returns[1], Return)

    def test_results_processing(self, partitioned_ir):
        assert isinstance(partitioned_ir.rp_map, dict)
        assert len(partitioned_ir.rp_map) == 1
        assert "test" in partitioned_ir.rp_map
        assert isinstance(partitioned_ir.rp_map["test"], ResultsProcessing)

    def test_post_processing(self, partitioned_ir):
        assert isinstance(partitioned_ir.pp_map, dict)
        assert len(partitioned_ir.pp_map) == 1
        assert "test" in partitioned_ir.pp_map
        assert isinstance(partitioned_ir.pp_map["test"], list)
        assert len(partitioned_ir.pp_map["test"]) == 2
        assert all([isinstance(pp, PostProcessing) for pp in partitioned_ir.pp_map["test"]])

    def test_acquire_map(self, partitioned_ir):
        acq_chan = self.model.qubits[0].get_acquire_channel()
        assert isinstance(partitioned_ir.acquire_map, dict)
        assert len(partitioned_ir.acquire_map) == 1
        assert acq_chan in partitioned_ir.acquire_map
        assert len(partitioned_ir.acquire_map[acq_chan]) == 1
        assert partitioned_ir.acquire_map[acq_chan][0].quantum_targets[0] == acq_chan

    def test_target_map(self, partitioned_ir):
        drive_chan = self.model.qubits[0].get_drive_channel()
        measure_chan = self.model.qubits[0].get_measure_channel()
        acq_chan = self.model.qubits[0].get_acquire_channel()
        assert isinstance(partitioned_ir.target_map, dict)
        assert len(partitioned_ir.target_map) == 3
        assert drive_chan in partitioned_ir.target_map
        assert len(partitioned_ir.target_map[drive_chan]) == 1
        assert isinstance(partitioned_ir.target_map[drive_chan][0], Delay)
        assert len(partitioned_ir.target_map[measure_chan]) == 1
        assert isinstance(partitioned_ir.target_map[measure_chan][0], Pulse)
        assert len(partitioned_ir.target_map[acq_chan]) == 2
        assert isinstance(partitioned_ir.target_map[acq_chan][0], Delay)
        assert isinstance(partitioned_ir.target_map[acq_chan][1], Acquire)

    def test_repeats_on_ir(self):
        builder = self.model.create_builder()
        builder.repeat(1000)
        builder.shots = 2000
        builder.compiled_shots = 1000

        ir = PartitionByPulseChannel().run(builder)
        assert ir.shots == 2000
        assert ir.compiled_shots == 1000
