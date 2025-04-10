# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import numpy as np
import pytest

from qat.core.result_base import ResultManager
from qat.middleend.passes.analysis import (
    ActiveChannelResults,
    ActivePulseChannelAnalysis,
    BatchedShots,
    BatchedShotsResult,
)
from qat.model.loaders.legacy import EchoModelLoader
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.instructions import CustomPulse, PulseShapeType


class TestBatchedShots:

    @pytest.mark.parametrize("shots", [0, 1000, 10000])
    def test_shots_less_than_equal_to_max_gives_expected(self, shots):
        model = EchoModelLoader().load()
        model.repeat_limit = 10000
        builder = model.create_builder()
        builder.repeat(shots)
        batch_pass = BatchedShots(model)
        res_mgr = ResultManager()
        batch_pass.run(builder, res_mgr)
        batch_res = res_mgr.lookup_by_type(BatchedShotsResult)
        assert batch_res.total_shots == shots
        assert batch_res.batched_shots == shots

    def test_no_repeat_instruction(self):
        model = EchoModelLoader().load()
        builder = model.create_builder()
        batch_pass = BatchedShots(model)
        res_mgr = ResultManager()
        batch_pass.run(builder, res_mgr)
        batch_res = res_mgr.lookup_by_type(BatchedShotsResult)
        assert batch_res.total_shots == model.default_repeat_count
        assert batch_res.batched_shots == model.default_repeat_count

    @pytest.mark.parametrize("shots", [10001, 20000, 29999])
    def test_shots_greater_than_max_gives_appropiate_batches(self, shots):
        model = EchoModelLoader().load()
        model.repeat_limit = 10000
        builder = model.create_builder()
        builder.repeat(shots)
        batch_pass = BatchedShots(model)
        res_mgr = ResultManager()
        batch_pass.run(builder, res_mgr)
        batch_res = res_mgr.lookup_by_type(BatchedShotsResult)
        assert batch_res.total_shots == shots
        assert batch_res.batched_shots <= shots
        assert batch_res.batched_shots * np.ceil(shots / model.repeat_limit) >= shots


class TestActivePulseChannelAnalysis:

    def test_valid_instructions(self):
        model = EchoModelLoader().load()
        builder = model.create_builder()
        qubit = model.qubits[0]
        drive_chan = qubit.get_drive_channel()
        measure_chan = qubit.get_measure_channel()
        acquire_chan = qubit.get_acquire_channel()

        builder.pulse(drive_chan, width=80e-9, shape=PulseShapeType.SQUARE)
        builder.add(CustomPulse(measure_chan, [1.0] * 80))
        builder.acquire(acquire_chan, time=80e-9, delay=0.0)

        res_mgr = ResultManager()
        builder = ActivePulseChannelAnalysis(model).run(builder, res_mgr)
        res: ActiveChannelResults = res_mgr.lookup_by_type(ActiveChannelResults)
        assert len(res.targets) == 3
        assert set(res.targets) == set([drive_chan, measure_chan, acquire_chan])
        assert len(res.target_map) == 3
        assert all([val == qubit for val in res.target_map.values()])
        assert set(res.target_map.keys()) == set([drive_chan, measure_chan, acquire_chan])
        assert set(res.from_qubit(qubit)) == set([drive_chan, measure_chan, acquire_chan])

    def test_syncs_dont_add_extra_channels(self):
        model = EchoModelLoader().load()
        builder = model.create_builder()
        qubit = model.qubits[0]
        drive_chan = qubit.get_drive_channel()

        builder.pulse(drive_chan, width=80e-9, shape=PulseShapeType.SQUARE)
        builder.synchronize(qubit)
        assert len(builder.instructions[-1].quantum_targets) > 1

        res_mgr = ResultManager()
        builder = ActivePulseChannelAnalysis(model).run(builder, res_mgr)
        res = res_mgr.lookup_by_type(ActiveChannelResults)
        assert len(res.targets) == 1
        assert set(res.targets) == set([drive_chan])

    def test_rogue_pulse_channel(self):
        model = EchoModelLoader().load()
        phys_chan = next(iter(model.physical_channels.values()))
        pulse_chan = PulseChannel("test", phys_chan)
        builder = model.create_builder()

        builder.pulse(pulse_chan, width=80e-9, shape=PulseShapeType.SQUARE)
        res_mgr = ResultManager()
        builder = ActivePulseChannelAnalysis(model).run(builder, res_mgr)
        res = res_mgr.lookup_by_type(ActiveChannelResults)
        assert len(res.targets) == 1
        assert len(res.unassigned) == 1
        assert res.targets == res.unassigned
