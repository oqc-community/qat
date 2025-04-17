# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import math
import random

import numpy as np
import pytest

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.middleend.passes.legacy.transform import PhaseOptimisation
from qat.model.loaders.legacy import EchoModelLoader
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.instructions import PhaseReset, PhaseShift, PulseShapeType


class TestPhaseOptimisation:
    hw = EchoModelLoader(qubit_count=4).load()

    def test_merged_identical_phase_resets(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        phase_reset = PhaseReset(self.hw.qubits)
        builder.add(phase_reset)
        builder.add(phase_reset)
        assert len(builder.instructions) == 2

        PhaseOptimisation().run(builder, res_mgr=ResultManager(), met_mgr=MetricsManager())
        # The two phase resets should be merged to one.
        assert len(builder.instructions) == 1
        assert set(builder.instructions[0].quantum_targets) == set(
            phase_reset.quantum_targets
        )

    def test_merged_phase_resets(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        targets_q1 = self.hw.qubits[0].get_all_channels()
        targets_q2 = self.hw.qubits[1].get_all_channels()
        builder.add(PhaseReset(targets_q1))
        builder.add(PhaseReset(targets_q2))

        PhaseOptimisation().run(builder, res_mgr=ResultManager(), met_mgr=MetricsManager())
        # The two phase resets should be merged to one, and the targets of both phase resets should be merged.
        assert len(builder.instructions) == 1
        merged_targets = set(targets_q1) | set(targets_q2)
        assert set(builder.instructions[0].quantum_targets) == merged_targets

    def test_empty_constructor(self):
        hw = EchoModelLoader(8).load()
        builder = hw.create_builder()

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        assert len(builder_optimised.instructions) == 0

    @pytest.mark.parametrize("phase", [-4 * np.pi, -2 * np.pi, 0.0, 2 * np.pi, 4 * np.pi])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_zero_phase(self, phase, pulse_enabled):
        hw = EchoModelLoader(8).load()
        builder = hw.create_builder()
        for qubit in hw.qubits:
            builder.add(PhaseShift(qubit.get_drive_channel(), phase))
            if pulse_enabled:
                builder.pulse(
                    qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9
                )

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )

        if pulse_enabled:
            assert len(builder_optimised.instructions) == len(hw.qubits)
        else:
            assert len(builder_optimised.instructions) == 0

    @pytest.mark.parametrize("phase", [0.15, 1.0, 3.14])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_single_phase(self, phase, pulse_enabled):
        hw = EchoModelLoader(8).load()
        builder = hw.create_builder()
        for qubit in hw.qubits:
            builder.phase_shift(qubit, phase)
            if pulse_enabled:
                builder.pulse(
                    qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9
                )

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        phase_shifts = [
            instr
            for instr in builder_optimised.instructions
            if isinstance(instr, PhaseShift)
        ]

        if pulse_enabled:
            assert len(phase_shifts) == len(hw.qubits)
        else:
            assert len(phase_shifts) == 0

    @pytest.mark.parametrize("phase", [0.5, 0.73, 2.75])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_accumulate_phases(self, phase, pulse_enabled):
        hw = EchoModelLoader(8).load()
        builder = hw.create_builder()
        qubits = hw.qubits
        for qubit in qubits:
            builder.phase_shift(qubit, phase)

        random.shuffle(hw.qubits)
        for qubit in qubits:
            builder.phase_shift(qubit, phase + 0.3)
            if pulse_enabled:
                builder.pulse(
                    qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9
                )

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        phase_shifts = [
            instr
            for instr in builder_optimised.instructions
            if isinstance(instr, PhaseShift)
        ]

        if pulse_enabled:
            assert len(phase_shifts) == len(hw.qubits)
            for phase_shift in phase_shifts:
                assert math.isclose(phase_shift.phase, 2 * phase + 0.3)
        else:
            assert (
                len(phase_shifts) == 0
            )  # Phase shifts without a pulse/reset afterwards are removed.

    def test_phase_reset(self):
        hw = EchoModelLoader(2).load()
        builder = hw.create_builder()
        qubits = list(hw.qubits)

        for qubit in qubits:
            builder.phase_shift(qubit, 0.5)
            builder.reset(qubit)

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )

        phase_shifts = [
            instr
            for instr in builder_optimised.instructions
            if isinstance(instr, PhaseShift)
        ]
        assert len(phase_shifts) == 0
