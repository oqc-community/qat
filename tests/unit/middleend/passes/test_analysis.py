# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import numpy as np
import pytest

from qat.core.metrics_base import MetricsManager, MetricsType
from qat.core.result_base import ResultManager
from qat.middleend.passes.analysis import (
    ActiveChannelResults,
    ActivePulseChannelAnalysis,
    BatchedShots,
    BatchedShotsResult,
)
from qat.model.loaders.legacy import EchoModelLoader
from qat.model.target_data import AbstractTargetData
from qat.purr.compiler.devices import PulseChannel
from qat.purr.compiler.instructions import CustomPulse, PulseShapeType
from qat.purr.utils.logger import LoggerLevel


class TestBatchedShots:
    model = EchoModelLoader().load()

    @pytest.mark.parametrize("shots", [0, 1000, 10000])
    def test_shots_less_than_equal_to_max_gives_expected(self, shots):
        target_data = AbstractTargetData(max_shots=10000, default_shots=100)

        builder = self.model.create_builder()
        builder.repeat(shots)
        batch_pass = BatchedShots(self.model, target_data)
        res_mgr = ResultManager()
        batch_pass.run(builder, res_mgr)
        batch_res = res_mgr.lookup_by_type(BatchedShotsResult)
        assert batch_res.total_shots == shots
        assert batch_res.batched_shots == shots

    def test_no_repeat_instruction(self):
        target_data = AbstractTargetData(max_shots=10000, default_shots=100)

        builder = self.model.create_builder()
        batch_pass = BatchedShots(self.model, target_data)
        res_mgr = ResultManager()
        batch_pass.run(builder, res_mgr)
        batch_res = res_mgr.lookup_by_type(BatchedShotsResult)
        assert batch_res.total_shots == target_data.default_shots
        assert batch_res.batched_shots == target_data.default_shots

    @pytest.mark.parametrize("shots", [10001, 20000, 29999])
    def test_shots_greater_than_max_gives_appropiate_batches(self, shots):
        target_data = AbstractTargetData(max_shots=10000, default_shots=100)

        builder = self.model.create_builder()
        builder.repeat(shots)
        batch_pass = BatchedShots(self.model, target_data)
        res_mgr = ResultManager()
        batch_pass.run(builder, res_mgr)
        batch_res = res_mgr.lookup_by_type(BatchedShotsResult)
        assert batch_res.total_shots == shots
        assert batch_res.batched_shots <= shots
        assert batch_res.batched_shots * np.ceil(shots / target_data.max_shots) >= shots


class TestActivePulseChannelAnalysis:
    model = EchoModelLoader().load()

    def test_valid_instructions(self):
        builder = self.model.create_builder()
        qubit = self.model.qubits[0]
        drive_chan = qubit.get_drive_channel()
        measure_chan = qubit.get_measure_channel()
        acquire_chan = qubit.get_acquire_channel()

        builder.pulse(drive_chan, width=80e-9, shape=PulseShapeType.SQUARE)
        builder.add(CustomPulse(measure_chan, [1.0] * 80))
        builder.acquire(acquire_chan, time=80e-9, delay=0.0)

        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        builder = ActivePulseChannelAnalysis(self.model).run(builder, res_mgr, met_mgr)
        res: ActiveChannelResults = res_mgr.lookup_by_type(ActiveChannelResults)
        assert len(res.targets) == 3
        assert set(res.targets) == set([drive_chan, measure_chan, acquire_chan])
        assert len(res.target_map) == 3
        assert all([val == qubit for val in res.target_map.values()])
        assert set(res.target_map.keys()) == set([drive_chan, measure_chan, acquire_chan])
        assert set(res.from_qubit(qubit)) == set([drive_chan, measure_chan, acquire_chan])

    def test_syncs_dont_add_extra_channels(self):
        builder = self.model.create_builder()
        qubit = self.model.qubits[0]
        drive_chan = qubit.get_drive_channel()

        builder.pulse(drive_chan, width=80e-9, shape=PulseShapeType.SQUARE)
        builder.synchronize(qubit)
        assert len(builder.instructions[-1].quantum_targets) > 1

        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        builder = ActivePulseChannelAnalysis(self.model).run(builder, res_mgr, met_mgr)
        res = res_mgr.lookup_by_type(ActiveChannelResults)
        assert len(res.targets) == 1
        assert set(res.targets) == set([drive_chan])

    def test_rogue_pulse_channel(self):
        phys_chan = next(iter(self.model.physical_channels.values()))
        pulse_chan = PulseChannel("test", phys_chan)
        builder = self.model.create_builder()

        builder.pulse(pulse_chan, width=80e-9, shape=PulseShapeType.SQUARE)
        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        builder = ActivePulseChannelAnalysis(self.model).run(builder, res_mgr, met_mgr)
        res = res_mgr.lookup_by_type(ActiveChannelResults)
        assert len(res.targets) == 1
        assert len(res.unassigned) == 1
        assert res.targets == res.unassigned

    def test_no_active_qubits(self):
        builder = self.model.create_builder()
        builder.repeat(100)
        builder.synchronize(self.model.qubits)

        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        ActivePulseChannelAnalysis(self.model).run(builder, res_mgr, met_mgr)

        res = res_mgr.lookup_by_type(ActiveChannelResults)
        assert len(res.physical_qubit_indices) == 0

        physical_qubit_indices = met_mgr.get_metric(MetricsType.PhysicalQubitIndices)
        assert len(physical_qubit_indices) == 0

    def test_logs_info(self, caplog):
        builder = self.model.create_builder()
        qubit = self.model.qubits[0]
        builder.X(qubit)

        with caplog.at_level(LoggerLevel.INFO.value):
            ActivePulseChannelAnalysis(self.model).run(
                builder, ResultManager(), MetricsManager()
            )
            assert "Physical qubits used in this circuit" in caplog.text
            assert str(qubit.index) in caplog.text

    @pytest.mark.parametrize("active_qubits", [[0], [2, 3], [0, 1, 2, 3]])
    # Do not test Z since it is a virtual gate.
    @pytest.mark.parametrize("gate", ["X", "Y", "had"])
    def test_1Q_gate(self, active_qubits, gate):
        builder = self.model.create_builder()
        for qubit_idx in active_qubits:
            getattr(builder, gate)(self.model.get_qubit(qubit_idx))

        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        ActivePulseChannelAnalysis(self.model).run(builder, res_mgr, met_mgr)

        res = res_mgr.lookup_by_type(ActiveChannelResults)
        assert res.physical_qubit_indices == set(active_qubits)

        physical_qubit_indices = met_mgr.get_metric(MetricsType.PhysicalQubitIndices)
        assert physical_qubit_indices == active_qubits

    @pytest.mark.parametrize("active_qubits", [[0, 1], [2, 3], [0, 1, 2, 3]])
    @pytest.mark.parametrize("gate", ["cnot", "ECR"])
    def test_2Q_gate(self, active_qubits, gate):
        builder = self.model.create_builder()
        for c_qubit_idx, t_qubit_idx in zip(active_qubits[::2], active_qubits[1::2]):
            getattr(builder, gate)(
                self.model.get_qubit(c_qubit_idx), self.model.get_qubit(t_qubit_idx)
            )

        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        ActivePulseChannelAnalysis(self.model).run(builder, res_mgr, met_mgr)

        res = res_mgr.lookup_by_type(ActiveChannelResults)
        assert res.physical_qubit_indices == set(active_qubits)

        physical_qubit_indices = met_mgr.get_metric(MetricsType.PhysicalQubitIndices)
        assert physical_qubit_indices == active_qubits
