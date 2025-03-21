# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import math
import random

import pytest

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import (
    QuantumInstructionBuilder as PydQuantumInstructionBuilder,
)
from qat.ir.instructions import PhaseShift as PydPhaseShift
from qat.ir.measure import AcquireMode
from qat.ir.measure import MeasureBlock as PydMeasureBlock
from qat.ir.measure import PostProcessing as PydPostProcessing
from qat.ir.measure import PostProcessType, ProcessAxis
from qat.ir.waveforms import GaussianWaveform, SquareWaveform
from qat.middleend.passes.transform import (
    PydPhaseOptimisation,
    PydPostProcessingSanitisation,
)
from qat.utils.hardware_model import generate_hw_model


class TestPydPhaseOptimisation:
    def test_empty_constructor(self):
        hw = generate_hw_model(8)
        builder = PydQuantumInstructionBuilder(hardware_model=hw)

        builder_optimised = PydPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        assert builder_optimised.number_of_instructions == 0

    def test_zero_phase(self):
        hw = generate_hw_model(8)
        builder = PydQuantumInstructionBuilder(hardware_model=hw)
        for qubit in hw.qubits.values():
            builder.phase_shift(target=qubit, theta=0)

        assert builder.number_of_instructions == 0

    @pytest.mark.parametrize("phase", [0.15, 1.0, 3.14])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_single_phase(self, phase, pulse_enabled):
        hw = generate_hw_model(8)
        builder = PydQuantumInstructionBuilder(hardware_model=hw)
        for qubit in hw.qubits.values():
            builder.phase_shift(target=qubit, theta=phase)
            if pulse_enabled:
                builder.pulse(
                    targets=qubit.drive_pulse_channel.uuid,
                    waveform=GaussianWaveform(),
                )

        builder_optimised = PydPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        phase_shifts = [
            instr for instr in builder_optimised if isinstance(instr, PydPhaseShift)
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
        builder = PydQuantumInstructionBuilder(hardware_model=hw)
        qubits = list(hw.qubits.values())

        for qubit in qubits:
            builder.phase_shift(target=qubit, theta=phase)

        random.shuffle(qubits)
        for qubit in qubits:
            builder.phase_shift(target=qubit, theta=phase + 0.3)
            if pulse_enabled:
                builder.pulse(
                    targets=qubit.drive_pulse_channel.uuid,
                    waveform=SquareWaveform(),
                )

        builder_optimised = PydPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        phase_shifts = [
            instr for instr in builder_optimised if isinstance(instr, PydPhaseShift)
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
        builder = PydQuantumInstructionBuilder(hardware_model=hw)
        qubits = list(hw.qubits.values())

        for qubit in qubits:
            builder.phase_shift(target=qubit, theta=0.5)
            builder.reset(qubit)

        builder_optimised = PydPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )

        phase_shifts = [
            instr for instr in builder_optimised if isinstance(instr, PydPhaseShift)
        ]
        assert len(phase_shifts) == 0


class TestPydPostProcessingSanitisation:
    hw = generate_hw_model(32)

    def test_meas_acq_with_pp(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        for qubit in self.hw.qubits.values():
            builder.measure_single_shot_z(target=qubit)

        PydPostProcessingSanitisation().run(builder, ResultManager(), MetricsManager())

    @pytest.mark.parametrize(
        "acq_mode,pp_type,pp_axes",
        [
            (AcquireMode.SCOPE, PostProcessType.MEAN, [ProcessAxis.SEQUENCE]),
            (AcquireMode.INTEGRATOR, PostProcessType.DOWN_CONVERT, [ProcessAxis.TIME]),
            (AcquireMode.INTEGRATOR, PostProcessType.MEAN, [ProcessAxis.TIME]),
        ],
    )
    def test_invalid_acq_pp(self, acq_mode, pp_type, pp_axes):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(0)

        builder.measure(targets=qubit, mode=acq_mode, output_variable="test")
        builder.post_processing(
            target=qubit, process_type=pp_type, axes=pp_axes, output_variable="test"
        )
        assert isinstance(builder._ir.tail, PydPostProcessing)

        # Pass should remove the invalid post-processing instruction from the IR.
        assert not PydPostProcessingSanitisation()._valid_pp(acq_mode, builder._ir.tail)

        PydPostProcessingSanitisation().run(builder, ResultManager(), MetricsManager())
        assert not isinstance(builder._ir.tail, PydPostProcessing)

    def test_invalid_raw_acq(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(0)

        builder.measure(targets=qubit, mode=AcquireMode.RAW, output_variable="test")

        with pytest.raises(ValueError):
            PydPostProcessingSanitisation().run(builder, ResultManager(), MetricsManager())

    def test_mid_circuit_measurement_two_diff_post_processing(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(2)

        # Mid-circuit measurement with some manual (different) post-processing options.
        builder.measure(targets=qubit, mode=AcquireMode.SCOPE)
        assert isinstance(builder._ir.tail, PydMeasureBlock)
        builder.post_processing(
            target=qubit,
            output_variable=builder._ir.tail.output_variables[0],
            process_type=PostProcessType.DOWN_CONVERT,
        )
        builder.X(target=qubit)
        builder.measure(targets=qubit, mode=AcquireMode.INTEGRATOR)
        assert isinstance(builder._ir.tail, PydMeasureBlock)
        builder.post_processing(
            target=qubit,
            output_variable=builder._ir.tail.output_variables[0],
            process_type=PostProcessType.MEAN,
        )

        PydPostProcessingSanitisation().run(builder, ResultManager(), MetricsManager())

        # Make sure no instructions get discarded in the post-processing sanitisation for a mid-circuit measurement.
        pp = [instr for instr in builder if isinstance(instr, PydPostProcessing)]
        assert len(pp) == 2
        assert pp[0].output_variable != pp[1].output_variable
