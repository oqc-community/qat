# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import pytest

from qat.core.metrics_base import MetricsManager, MetricsType
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.waveforms import SquareWaveform
from qat.middleend.passes.analysis import (
    ActivePulseChannelAnalysis,
    ActivePulseChannelResults,
)
from qat.model.loaders.lucy import LucyModelLoader
from qat.purr.utils.logger import LoggerLevel

pytestmark = pytest.mark.experimental


class TestActivePulseChannelAnalysis:
    model = LucyModelLoader().load()

    def test_valid_instructions(self):
        builder = QuantumInstructionBuilder(self.model)
        qubit = self.model.qubit_with_index(0)
        drive_chan = qubit.drive_pulse_channel
        measure_chan = qubit.measure_pulse_channel
        acquire_chan = qubit.acquire_pulse_channel

        builder.pulse(target=drive_chan.uuid, waveform=SquareWaveform(width=80e-9, amp=1.0))
        builder.pulse(
            target=measure_chan.uuid, waveform=SquareWaveform(width=80e-9, amp=1.0)
        )
        builder.acquire(qubit, time=80e-9, delay=0.0)

        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        builder = ActivePulseChannelAnalysis(self.model).run(builder, res_mgr, met_mgr)
        res = res_mgr.lookup_by_type(ActivePulseChannelResults)
        assert len(res.targets) == 3
        channel_ids = set([drive_chan.uuid, measure_chan.uuid, acquire_chan.uuid])
        assert res.targets == channel_ids
        assert len(res.targets) == 3
        assert all([val == qubit for val in res.pulse_channel_to_qubit_map.values()])
        assert set([channel.uuid for channel in res.from_qubit(qubit)]) == channel_ids
        assert res.qubits == set([qubit])

        physical_qubit_indices = met_mgr.get_metric(MetricsType.PhysicalQubitIndices)
        assert physical_qubit_indices == [0]

    def test_syncs_dont_add_extra_channels(self):
        builder = QuantumInstructionBuilder(self.model)
        qubit = self.model.qubit_with_index(0)
        drive_chan = qubit.drive_pulse_channel

        builder.pulse(target=drive_chan.uuid, waveform=SquareWaveform(width=80e-9, amp=1.0))
        builder.synchronize(qubit)
        assert len(builder.instructions[-1].targets) > 1

        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        builder = ActivePulseChannelAnalysis(self.model).run(builder, res_mgr, met_mgr)
        res = res_mgr.lookup_by_type(ActivePulseChannelResults)
        assert len(res.targets) == 1
        assert res.targets == set([drive_chan.uuid])

    def test_no_active_qubits(self):
        builder = QuantumInstructionBuilder(self.model)
        builder.repeat(100)
        builder.synchronize(list(self.model.qubits.values()))

        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        ActivePulseChannelAnalysis(self.model).run(builder, res_mgr, met_mgr)

        physical_qubit_indices = met_mgr.get_metric(MetricsType.PhysicalQubitIndices)
        assert len(physical_qubit_indices) == 0

    def test_logs_info(self, caplog):
        builder = QuantumInstructionBuilder(self.model)
        qubit = self.model.qubit_with_index(0)
        builder.X(qubit)

        with caplog.at_level(LoggerLevel.INFO.value):
            ActivePulseChannelAnalysis(self.model).run(
                builder, ResultManager(), MetricsManager()
            )
            assert "Physical qubits used in this circuit" in caplog.text
            assert str(0) in caplog.text

    @pytest.mark.parametrize("active_qubits", [[0], [2, 3], [0, 1, 2, 3]])
    # Do not test Z since it is a virtual gate.
    @pytest.mark.parametrize("gate", ["X", "Y", "had"])
    def test_1Q_gate(self, active_qubits, gate):
        builder = QuantumInstructionBuilder(self.model)
        for qubit_idx in active_qubits:
            getattr(builder, gate)(self.model.qubit_with_index(qubit_idx))

        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        ActivePulseChannelAnalysis(self.model).run(builder, res_mgr, met_mgr)

        physical_qubit_indices = met_mgr.get_metric(MetricsType.PhysicalQubitIndices)
        assert physical_qubit_indices == active_qubits

    @pytest.mark.parametrize("active_qubits", [[0, 1], [2, 3], [0, 1, 2, 3]])
    @pytest.mark.parametrize("gate", ["cnot", "ECR"])
    def test_2Q_gate(self, active_qubits, gate):
        builder = QuantumInstructionBuilder(self.model)
        for c_qubit_idx, t_qubit_idx in zip(active_qubits[::2], active_qubits[1::2]):
            getattr(builder, gate)(
                self.model.qubit_with_index(c_qubit_idx),
                self.model.qubit_with_index(t_qubit_idx),
            )

        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        ActivePulseChannelAnalysis(self.model).run(builder, res_mgr, met_mgr)

        physical_qubit_indices = met_mgr.get_metric(MetricsType.PhysicalQubitIndices)
        assert physical_qubit_indices == active_qubits

    def test_with_custom_pulse_channel(self):
        qubit = self.model.qubit_with_index(0)
        physical_channel = qubit.physical_channel

        builder = QuantumInstructionBuilder(self.model)
        drive_channel = qubit.drive_pulse_channel
        measure_channel = qubit.measure_pulse_channel
        pulse_channel = builder.create_pulse_channel(
            drive_channel.frequency, physical_channel
        )
        builder.pulse(
            target=pulse_channel.uuid, waveform=SquareWaveform(width=80e-9, amp=1.0)
        )
        builder.pulse(
            target=drive_channel.uuid, waveform=SquareWaveform(width=160e-9, amp=1.0)
        )
        builder.synchronize([drive_channel, pulse_channel, measure_channel])

        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        builder = ActivePulseChannelAnalysis(self.model).run(builder, res_mgr, met_mgr)
        res = res_mgr.lookup_by_type(ActivePulseChannelResults)
        assert len(res.targets) == 2
        channel_ids = set([drive_channel.uuid, pulse_channel.uuid])
        assert res.targets == channel_ids
        assert len(res.targets) == 2
        assert all([val == qubit for val in res.pulse_channel_to_qubit_map.values()])
        assert set([channel.uuid for channel in res.from_qubit(qubit)]) == channel_ids
        assert res.qubits == set([qubit])

        physical_qubit_indices = met_mgr.get_metric(MetricsType.PhysicalQubitIndices)
        assert physical_qubit_indices == [0]
