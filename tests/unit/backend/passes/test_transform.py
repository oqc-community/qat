# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025-2025 Oxford Quantum Circuits Ltd

import math
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pytest

from qat.backend.passes.analysis import TriagePass, TriageResult
from qat.backend.passes.lowering import PartitionByPulseChannel, PartitionedIR
from qat.backend.passes.transform import (
    DesugaringPass,
    FreqShiftSanitisation,
    LowerSyncsToDelays,
    SquashDelaysOptimisation,
)
from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.model.loaders.legacy import EchoModelLoader
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.devices import ChannelType, FreqShiftPulseChannel
from qat.purr.compiler.instructions import Delay, Pulse, PulseShapeType

from tests.unit.utils.builder_nuggets import resonator_spect


class TestTransformPasses:
    def test_desugaring_pass(self):
        model = EchoModelLoader().load()
        builder = resonator_spect(model)
        res_mgr = ResultManager()

        TriagePass().run(builder, res_mgr)
        triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)

        assert len(triage_result.sweeps) == 1
        sweep = next(iter(triage_result.sweeps))
        assert len(sweep.variables) == 1

        DesugaringPass().run(builder, res_mgr)
        assert len(sweep.variables) == 2
        assert f"sweep_{hash(sweep)}" in sweep.variables


class TestLowerSyncsToDelays:

    def test_sync_with_two_channels(self):
        model = EchoModelLoader().load()
        chan1 = model.qubits[0].get_drive_channel()
        chan2 = model.qubits[1].get_drive_channel()

        builder = model.create_builder()
        builder.pulse(chan1, shape=PulseShapeType.SQUARE, width=120e-9)
        builder.delay(chan1, 48e-9)
        builder.delay(chan2, 72e-9)
        builder.pulse(chan2, shape=PulseShapeType.SQUARE, width=168e-9)
        builder.synchronize([chan1, chan2])

        LowerSyncsToDelays().run(builder)
        assert len(builder.instructions) == 5
        assert [type(inst) for inst in builder.instructions] == [
            Pulse,
            Delay,
            Delay,
            Pulse,
            Delay,
        ]
        times = [inst.duration for inst in builder.instructions]
        assert times[0] + times[1] + times[4] == times[2] + times[3]


class TestSquashDelaysOptimisation:

    @pytest.mark.parametrize("num_delays", [1, 2, 3, 4])
    @pytest.mark.parametrize("with_phase", [True, False])
    def test_multiple_delays_on_one_channel(self, num_delays, with_phase):
        delay_times = np.random.rand(num_delays)
        hw = EchoModelLoader().load()
        chan = hw.qubits[0].get_drive_channel()
        builder = hw.create_builder()
        for delay in delay_times:
            builder.delay(chan, delay)
            if with_phase:
                builder.phase_shift(chan, np.random.rand())

        builder = SquashDelaysOptimisation().run(builder, ResultManager(), MetricsManager())
        assert len(builder.instructions) == 1 + with_phase * num_delays
        delay_instructions = [
            inst for inst in builder.instructions if isinstance(inst, Delay)
        ]
        assert len(delay_instructions) == 1
        assert np.isclose(delay_instructions[0].time, sum(delay_times))

    @pytest.mark.parametrize("num_delays", [1, 2, 3, 4])
    @pytest.mark.parametrize("num_channels", [1, 2, 3, 4])
    @pytest.mark.parametrize("with_phase", [True, False])
    def test_multiple_delays_on_multiple_channels(
        self, num_delays, num_channels, with_phase
    ):
        hw = EchoModelLoader(num_channels).load()
        chans = [qubit.get_drive_channel() for qubit in hw.qubits]
        builder = hw.create_builder()
        accumulated_delays = defaultdict(float)
        for _ in range(num_delays):
            random.shuffle(chans)
            for chan in chans:
                delay = np.random.rand()
                accumulated_delays[chan] += delay
                builder.delay(chan, delay)
                if with_phase:
                    builder.phase_shift(chan, np.random.rand())

        builder = SquashDelaysOptimisation().run(builder, ResultManager(), MetricsManager())
        assert len(builder.instructions) == (1 + with_phase * num_delays) * num_channels
        delay_instructions = [
            inst for inst in builder.instructions if isinstance(inst, Delay)
        ]
        assert len(delay_instructions) == num_channels
        for delay in delay_instructions:
            assert delay.time == accumulated_delays[delay.quantum_targets[0]]

    def test_optimize_with_pulse(self):
        delay_times = np.random.rand(5)
        hw = EchoModelLoader().load()
        chan = hw.qubits[0].get_drive_channel()
        builder = hw.create_builder()
        builder.delay(chan, delay_times[0])
        builder.delay(chan, delay_times[1])
        builder.pulse(chan, width=80e-9, shape=PulseShapeType.SQUARE)
        builder.delay(chan, delay_times[2])
        builder.delay(chan, delay_times[3])
        builder.delay(chan, delay_times[4])
        builder = SquashDelaysOptimisation().run(builder, ResultManager(), MetricsManager())
        assert len(builder.instructions) == 3
        assert isinstance(builder.instructions[0], Delay)
        assert np.isclose(builder.instructions[0].time, np.sum(delay_times[0:2]))
        assert isinstance(builder.instructions[1], Pulse)
        assert isinstance(builder.instructions[2], Delay)
        assert np.isclose(builder.instructions[2].time, np.sum(delay_times[2:5]))


class TestFreqShiftSanitisation:

    def test_no_freq_shift_pulse_channel(self):
        hw = EchoModelLoader(8).load()

        builder = QuantumInstructionBuilder(hw)
        res_mgr = ResultManager()

        for qubit in hw.qubits:
            builder.X(qubit)

        partitioned_ir = PartitionByPulseChannel().run(builder, res_mgr=res_mgr)
        ref_partitioned_ir = deepcopy(partitioned_ir)

        FreqShiftSanitisation(hw).run(partitioned_ir, res_mgr=res_mgr)

        # No instructions added since we do not have freq shift pulse channels.
        assert len(partitioned_ir.target_map) == len(ref_partitioned_ir.target_map)
        for target, target_ref in zip(
            partitioned_ir.target_map, ref_partitioned_ir.target_map
        ):
            assert target == target_ref

    @pytest.mark.parametrize("total_duration", [1e-02, 1e-01, 3, 5])
    def test_freq_shift_pulse_channel(self, total_duration):
        hw = EchoModelLoader(8).load()

        builder = QuantumInstructionBuilder(hw)
        res_mgr = ResultManager()

        # Add frequency shift pulse channel to first qubit.
        qubit0 = hw.qubits[0]
        freq_shift_pulse_ch = FreqShiftPulseChannel(
            id_="pulse_ch_Q0", physical_channel=qubit0.physical_channel
        )
        qubit0.add_pulse_channel(freq_shift_pulse_ch, channel_type=ChannelType.freq_shift)

        for qubit in hw.qubits:
            builder.X(qubit)

        # Total duration of the freq shift pulse should be equal to the length of this pulse.
        builder.pulse(
            qubit0.get_drive_channel(),
            shape=PulseShapeType.GAUSSIAN,
            amp=0.1,
            width=total_duration,
        )

        partitioned_ir = PartitionByPulseChannel().run(builder, res_mgr=res_mgr)
        ref_partitioned_ir = deepcopy(partitioned_ir)

        FreqShiftSanitisation(hw).run(partitioned_ir, res_mgr=res_mgr)

        # One pulse on the freq shift pulse channel of the first qubit.
        assert len(partitioned_ir.target_map) == len(ref_partitioned_ir.target_map) + 1

        assert len(partitioned_ir.target_map[freq_shift_pulse_ch]) == 1
        freq_shift_pulse = partitioned_ir.target_map[freq_shift_pulse_ch][0]
        assert isinstance(freq_shift_pulse, Pulse)
        assert (
            freq_shift_pulse_ch in freq_shift_pulse.quantum_targets
            and len(freq_shift_pulse.quantum_targets) == 1
        )
        assert freq_shift_pulse.shape == PulseShapeType.SQUARE
        assert freq_shift_pulse.amp == freq_shift_pulse_ch.amp
        # TO DO: Change the timings to integers of nanoseconds to improve accuracy.
        assert math.isclose(freq_shift_pulse.width, total_duration, abs_tol=1e-06)

    def test_freq_shift_empty_target(self):
        hw = EchoModelLoader(8).load()
        res_mgr = ResultManager()
        partitioned_ir = PartitionedIR()
        ir = FreqShiftSanitisation(hw).run(partitioned_ir, res_mgr=res_mgr)
        assert len(ir.target_map) == 0
        assert ir is partitioned_ir
