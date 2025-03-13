# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import math
import random

import pytest

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.instructions import PhaseShift
from qat.ir.waveforms import GaussianWaveform, SquareWaveform
from qat.middleend.passes.transform import PydPhaseOptimisation
from qat.utils.hardware_model import generate_hw_model


class TestPhaseOptimisation:
    def test_empty_constructor(self):
        hw = generate_hw_model(8)
        builder = QuantumInstructionBuilder(hardware_model=hw)

        builder_optimised = PydPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        assert builder_optimised.number_of_instructions == 0

    def test_zero_phase(self):
        hw = generate_hw_model(8)
        builder = QuantumInstructionBuilder(hardware_model=hw)
        for qubit in hw.qubits.values():
            builder.phase_shift(target=qubit, theta=0)

        assert builder.number_of_instructions == 0

    @pytest.mark.parametrize("phase", [0.15, 1.0, 3.14])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_single_phase(self, phase, pulse_enabled):
        hw = generate_hw_model(8)
        builder = QuantumInstructionBuilder(hardware_model=hw)
        for qubit in hw.qubits.values():
            builder.phase_shift(target=qubit, theta=phase)
            if pulse_enabled:
                builder.pulse(
                    targets=qubit.pulse_channels.drive.uuid,
                    waveform=GaussianWaveform(),
                )

        builder_optimised = PydPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        phase_shifts = [
            instr for instr in builder_optimised if isinstance(instr, PhaseShift)
        ]

        if pulse_enabled:
            assert len(phase_shifts) == hw.number_of_qubits
        else:
            assert (
                len(phase_shifts) == 0
            )  # Phase shifts without a pulse/reset afterwards are removed.

    @pytest.mark.parametrize("phase", [0.5, 0.73, 2.75])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_accumulate_phases(self, phase, pulse_enabled):
        hw = generate_hw_model(8)
        builder = QuantumInstructionBuilder(hardware_model=hw)
        qubits = list(hw.qubits.values())

        for qubit in qubits:
            builder.phase_shift(target=qubit, theta=phase)

        random.shuffle(qubits)
        for qubit in qubits:
            builder.phase_shift(target=qubit, theta=phase + 0.3)
            if pulse_enabled:
                builder.pulse(
                    targets=qubit.pulse_channels.drive.uuid,
                    waveform=SquareWaveform(),
                )

        builder_optimised = PydPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        phase_shifts = [
            instr for instr in builder_optimised if isinstance(instr, PhaseShift)
        ]

        if pulse_enabled:
            assert len(phase_shifts) == hw.number_of_qubits
            for phase_shift in phase_shifts:
                assert math.isclose(phase_shift.phase, 2 * phase + 0.3)
        else:
            assert (
                len(phase_shifts) == 0
            )  # Phase shifts without a pulse/reset afterwards are removed.

    def test_phase_reset(self):
        hw = generate_hw_model(2)
        builder = QuantumInstructionBuilder(hardware_model=hw)
        qubits = list(hw.qubits.values())

        for qubit in qubits:
            builder.phase_shift(target=qubit, theta=0.5)
            builder.reset(qubit)

        builder_optimised = PydPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )

        phase_shifts = [
            instr for instr in builder_optimised if isinstance(instr, PhaseShift)
        ]
        assert len(phase_shifts) == 0
