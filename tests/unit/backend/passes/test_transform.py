# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025-2025 Oxford Quantum Circuits Ltd

import random
from collections import defaultdict

import numpy as np
import pytest

from qat.backend.passes.analysis import TriagePass, TriageResult
from qat.backend.passes.transform import (
    DesugaringPass,
    LowerSyncsToDelays,
    SquashDelaysOptimisation,
)
from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.model.loaders.legacy import EchoModelLoader
from qat.purr.compiler.instructions import (
    Delay,
    Pulse,
    PulseShapeType,
)

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

    def test_delay_with_multiple_channels(self):
        delay_times = np.random.rand(2)
        hw = EchoModelLoader().load()
        chan1 = hw.qubits[0].get_drive_channel()
        chan2 = hw.qubits[0].get_measure_channel()
        builder = hw.create_builder()
        builder.add(Delay([chan1, chan2], 5))
        builder.delay(chan1, delay_times[0])
        builder.delay(chan2, delay_times[1])
        builder = SquashDelaysOptimisation().run(builder, ResultManager(), MetricsManager())
        assert len(builder.instructions) == 2
        assert builder.instructions[0].time == 5 + delay_times[0]
        assert builder.instructions[1].time == 5 + delay_times[1]
