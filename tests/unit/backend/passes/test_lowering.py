# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.backend.passes.lowering import PartitionByPulseChannel, PartitionedIR
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
    Repeat,
    ResultsProcessing,
    Return,
)


class TestPartitionByPulseChannel:
    def create_triaged_instructions(self):
        model = EchoModelLoader().load()
        builder = model.create_builder()
        drive_chan = model.qubits[0].get_drive_channel()
        acq_chan = model.qubits[0].get_acquire_channel()
        measure_chan = model.qubits[0].get_measure_channel()

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

        return model, PartitionByPulseChannel().run(builder)

    def test_is_partitioned_ir(self):
        model, ir = self.create_triaged_instructions()
        assert isinstance(ir, PartitionedIR)

    def test_repeats(self):
        model, ir = self.create_triaged_instructions()
        assert isinstance(ir.shots, Repeat)

    def test_assigns(self):
        model, ir = self.create_triaged_instructions()
        assert isinstance(ir.assigns, list)
        assert len(ir.assigns) == 1
        assert isinstance(ir.assigns[0], Assign)

    def test_returns(self):
        model, ir = self.create_triaged_instructions()
        assert isinstance(ir.returns, list)
        assert len(ir.returns) == 2
        assert isinstance(ir.returns[0], Return)
        assert isinstance(ir.returns[1], Return)

    def test_results_processing(self):
        model, ir = self.create_triaged_instructions()
        assert isinstance(ir.rp_map, dict)
        assert len(ir.rp_map) == 1
        assert "test" in ir.rp_map
        assert isinstance(ir.rp_map["test"], ResultsProcessing)

    def test_post_processing(self):
        model, ir = self.create_triaged_instructions()
        assert isinstance(ir.pp_map, dict)
        assert len(ir.pp_map) == 1
        assert "test" in ir.pp_map
        assert isinstance(ir.pp_map["test"], list)
        assert len(ir.pp_map["test"]) == 2
        assert all([isinstance(pp, PostProcessing) for pp in ir.pp_map["test"]])

    def test_acquire_map(self):
        model, ir = self.create_triaged_instructions()
        acq_chan = model.qubits[0].get_acquire_channel()
        assert isinstance(ir.acquire_map, dict)
        assert len(ir.acquire_map) == 1
        assert acq_chan in ir.acquire_map
        assert len(ir.acquire_map[acq_chan]) == 1
        assert ir.acquire_map[acq_chan][0].quantum_targets[0] == acq_chan

    def test_target_map(self):
        model, ir = self.create_triaged_instructions()
        drive_chan = model.qubits[0].get_drive_channel()
        measure_chan = model.qubits[0].get_measure_channel()
        acq_chan = model.qubits[0].get_acquire_channel()
        assert isinstance(ir.target_map, dict)
        assert len(ir.target_map) == 3
        assert drive_chan in ir.target_map
        assert len(ir.target_map[drive_chan]) == 1
        assert isinstance(ir.target_map[drive_chan][0], Delay)
        assert len(ir.target_map[measure_chan]) == 1
        assert isinstance(ir.target_map[measure_chan][0], Pulse)
        assert len(ir.target_map[acq_chan]) == 2
        assert isinstance(ir.target_map[acq_chan][0], Delay)
        assert isinstance(ir.target_map[acq_chan][1], Acquire)
