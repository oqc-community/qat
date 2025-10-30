# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest
from compiler_config.config import InlineResultsProcessing

from qat.backend.passes.lowering import PydPartitionByPulseChannel
from qat.backend.passes.purr.lowering import (
    PartitionByPulseChannel as LegPartitionByPulseChannel,
)
from qat.ir.instruction_builder import PydQuantumInstructionBuilder
from qat.ir.instructions import Assign, Delay, ResultsProcessing, Return
from qat.ir.lowered import PartitionedIR
from qat.ir.measure import (
    Acquire,
    AcquireMode,
    PostProcessing,
    PostProcessType,
    ProcessAxis,
)
from qat.ir.waveforms import Pulse, SquareWaveform
from qat.model.loaders.lucy import LucyModelLoader
from qat.model.loaders.purr import EchoModelLoader as LegEchoModelLoader
from qat.purr.compiler.instructions import PulseShapeType

pytestmark = pytest.mark.experimental


class TestPydPartitionByPulseChannel:
    hw = LucyModelLoader().load()

    @pytest.fixture(scope="class")
    def builder(self):
        qubit = self.hw.qubit_with_index(0)
        drive_chan = qubit.drive_pulse_channel
        measure_chan = qubit.measure_pulse_channel
        acq_chan = qubit.acquire_pulse_channel

        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.repeat(1000)
        builder.delay(drive_chan, 80e-9)
        builder.pulse(targets=measure_chan.uuid, waveform=SquareWaveform(width=240e-9))
        builder.delay(acq_chan, 40e-9)
        builder.acquire(
            qubit,
            duration=200e-9,
            mode=AcquireMode.INTEGRATOR,
            output_variable="test",
            delay=0.0,
        )
        acq = builder._ir.tail
        builder.post_processing(
            qubit,
            process_type=PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
            axes=ProcessAxis.SEQUENCE,
            output_variable=acq.output_variable,
            args=[0.0, 1.0],
        )
        builder.post_processing(
            qubit,
            process_type=PostProcessType.DISCRIMINATE,
            axes=ProcessAxis.SEQUENCE,
            output_variable=acq.output_variable,
            args=[0.0, 1.0],
        )
        builder.results_processing("test", InlineResultsProcessing.Binary)
        builder.assign("test2", "test")
        builder.returns("test")
        builder.returns("test2")

        return builder

    @pytest.fixture(scope="class")
    def partitioned_ir(self, builder):
        return PydPartitionByPulseChannel().run(builder)

    def test_is_partitioned_ir(self, partitioned_ir):
        assert isinstance(partitioned_ir, PartitionedIR)

    def test_repeats(self, partitioned_ir):
        assert isinstance(partitioned_ir.shots, int)
        assert partitioned_ir.shots == 1000
        assert partitioned_ir.compiled_shots == 1000

    def test_acquire_map(self, partitioned_ir):
        acq_chan = self.hw.qubit_with_index(0).acquire_pulse_channel
        assert isinstance(partitioned_ir.acquire_map, dict)
        assert len(partitioned_ir.acquire_map) == 1
        assert acq_chan.uuid in partitioned_ir.acquire_map
        assert len(partitioned_ir.acquire_map[acq_chan.uuid]) == 1
        assert partitioned_ir.acquire_map[acq_chan.uuid][0].target == acq_chan.uuid

    def test_assigns(self, partitioned_ir):
        assert isinstance(partitioned_ir.assigns, list)
        assert len(partitioned_ir.assigns) == 1
        assert isinstance(partitioned_ir.assigns[0], Assign)

    def test_post_processing(self, partitioned_ir):
        assert isinstance(partitioned_ir.pp_map, dict)
        assert len(partitioned_ir.pp_map) == 1
        assert "test" in partitioned_ir.pp_map
        assert isinstance(partitioned_ir.pp_map["test"], list)
        assert len(partitioned_ir.pp_map["test"]) == 2
        assert all([isinstance(pp, PostProcessing) for pp in partitioned_ir.pp_map["test"]])

    def test_results_processing_missing_var(self, builder):
        incomplete_builder = PydQuantumInstructionBuilder(
            hardware_model=self.hw, instructions=builder.instructions[:-2]
        )
        partitioned_ir = PydPartitionByPulseChannel().run(incomplete_builder)
        assert "test" in partitioned_ir.rp_map

    def test_results_processing(self, partitioned_ir):
        assert isinstance(partitioned_ir.rp_map, dict)
        assert len(partitioned_ir.rp_map) == 1
        assert "test" in partitioned_ir.rp_map
        assert isinstance(partitioned_ir.rp_map["test"], ResultsProcessing)

    def test_returns(self, partitioned_ir):
        assert isinstance(partitioned_ir.returns, list)
        assert len(partitioned_ir.returns) == 2
        assert isinstance(partitioned_ir.returns[0], Return)
        assert isinstance(partitioned_ir.returns[1], Return)

    def test_target_map(self, partitioned_ir):
        drive_chan_id = self.hw.qubit_with_index(0).drive_pulse_channel.uuid
        measure_chan_id = self.hw.qubit_with_index(0).measure_pulse_channel.uuid
        acq_chan_id = self.hw.qubit_with_index(0).acquire_pulse_channel.uuid
        assert isinstance(partitioned_ir.target_map, dict)
        assert len(partitioned_ir.target_map) == 3
        assert drive_chan_id in partitioned_ir.target_map
        assert len(partitioned_ir.target_map[drive_chan_id]) == 1
        assert isinstance(partitioned_ir.target_map[drive_chan_id][0], Delay)
        assert len(partitioned_ir.target_map[measure_chan_id]) == 1
        assert isinstance(partitioned_ir.target_map[measure_chan_id][0], Pulse)
        assert len(partitioned_ir.target_map[acq_chan_id]) == 2
        assert isinstance(partitioned_ir.target_map[acq_chan_id][0], Delay)
        assert isinstance(partitioned_ir.target_map[acq_chan_id][1], Acquire)

    def test_sync_throws_error(self, builder):
        qubit = self.hw.qubit_with_index(0)
        builder.synchronize(qubit)
        with pytest.raises(
            ValueError, match="is not supported by the PartitionByPulseChannel pass"
        ):
            PydPartitionByPulseChannel().run(builder)

    def test_pulse_channels(self, partitioned_ir):
        qubit = self.hw.qubit_with_index(0)
        drive_chan = qubit.drive_pulse_channel
        measure_chan = qubit.measure_pulse_channel
        acq_chan = qubit.acquire_pulse_channel

        assert isinstance(partitioned_ir.pulse_channels, dict)
        assert len(partitioned_ir.pulse_channels) == 3
        assert partitioned_ir.get_pulse_channel(drive_chan.uuid) is not None
        assert partitioned_ir.get_pulse_channel(measure_chan.uuid) is not None
        assert partitioned_ir.get_pulse_channel(acq_chan.uuid) is not None

    @pytest.fixture(scope="class")
    def leg_partitioned_ir(self):
        model = LegEchoModelLoader().load()
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
        return LegPartitionByPulseChannel().run(builder)

    def test_parity_legacy_vs_pydantic(self, leg_partitioned_ir, partitioned_ir):
        # TODO: Remove once we have fully migrated to Pydantic 581
        assert len(leg_partitioned_ir.target_map) == len(partitioned_ir.target_map)
        assert leg_partitioned_ir.shots == partitioned_ir.shots
        assert leg_partitioned_ir.compiled_shots == partitioned_ir.compiled_shots
        assert len(leg_partitioned_ir.returns) == len(partitioned_ir.returns)
        assert len(leg_partitioned_ir.assigns) == len(partitioned_ir.assigns)
        assert len(leg_partitioned_ir.acquire_map) == len(partitioned_ir.acquire_map)
        assert len(leg_partitioned_ir.pp_map) == len(partitioned_ir.pp_map)
        assert len(leg_partitioned_ir.rp_map) == len(partitioned_ir.rp_map)
