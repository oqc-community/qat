# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import math
import random

import numpy as np
import pytest
from compiler_config.config import MetricsType

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import (
    QuantumInstructionBuilder as PydQuantumInstructionBuilder,
)
from qat.ir.instructions import PhaseReset as PydPhaseReset
from qat.ir.instructions import PhaseShift as PydPhaseShift
from qat.ir.instructions import Return as PydReturn
from qat.ir.measure import Acquire as PydAcquire
from qat.ir.measure import AcquireMode, PostProcessType, ProcessAxis
from qat.ir.measure import MeasureBlock as PydMeasureBlock
from qat.ir.measure import PostProcessing as PydPostProcessing
from qat.ir.waveforms import GaussianWaveform, SquareWaveform
from qat.middleend.passes.transform import (
    PydPhaseOptimisation,
    PydPostProcessingSanitisation,
    PydReturnSanitisation,
)
from qat.middleend.passes.validation import (
    PydReturnSanitisationValidation,
)
from qat.model.loaders.converted import EchoModelLoader as PydEchoModelLoader


class TestPydPhaseOptimisation:
    hw = PydEchoModelLoader(8).load()

    def test_empty_constructor(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)

        builder_optimised = PydPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        assert builder_optimised.number_of_instructions == 0

    @pytest.mark.parametrize("phase", [-4 * np.pi, -2 * np.pi, 0.0, 2 * np.pi, 4 * np.pi])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_zero_phase(self, phase, pulse_enabled):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        for qubit in self.hw.qubits.values():
            builder.add(PydPhaseShift(targets=qubit.drive_pulse_channel.uuid, phase=phase))
            if pulse_enabled:
                builder.pulse(
                    targets=qubit.drive_pulse_channel.uuid,
                    waveform=GaussianWaveform(),
                )

        builder_optimised = PydPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )

        if pulse_enabled:
            assert builder_optimised.number_of_instructions == self.hw.number_of_qubits
        else:
            assert builder_optimised.number_of_instructions == 0

    @pytest.mark.parametrize("phase", [0.15, 1.0, 3.14])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_single_phase(self, phase, pulse_enabled):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        for qubit in self.hw.qubits.values():
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
            assert len(phase_shifts) == self.hw.number_of_qubits
        else:
            assert (
                len(phase_shifts) == 0
            )  # Phase shifts without a pulse/reset afterwards are removed.

    @pytest.mark.parametrize("phase", [0.5, 0.73, 2.75])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_accumulate_phases(self, phase, pulse_enabled):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        qubits = list(self.hw.qubits.values())

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
            assert len(phase_shifts) == self.hw.number_of_qubits
            for phase_shift in phase_shifts:
                assert math.isclose(phase_shift.phase, 2 * phase + 0.3)
        else:
            assert (
                len(phase_shifts) == 0
            )  # Phase shifts without a pulse/reset afterwards are removed.

    def test_phase_reset(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        qubits = list(self.hw.qubits.values())

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

    def test_merged_identical_phase_resets(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        qubit = self.hw.qubit_with_index(0)
        target = qubit.drive_pulse_channel.uuid

        phase_reset = PydPhaseReset(targets=target)
        builder.add(phase_reset)
        builder.add(phase_reset)
        assert builder.number_of_instructions == 2

        PydPhaseOptimisation().run(
            builder, res_mgr=ResultManager(), met_mgr=MetricsManager()
        )
        # The two phase resets should be merged to one.
        assert builder.number_of_instructions == 1
        # assert set(builder.instructions[0].quantum_targets) == set(phase_reset.quantum_targets)


class TestPydPostProcessingSanitisation:
    hw = PydEchoModelLoader(32).load()

    def test_meas_acq_with_pp(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        for qubit in self.hw.qubits.values():
            builder.measure(targets=qubit, mode=AcquireMode.SCOPE, output_variable="test")
            builder.post_processing(
                target=qubit, process_type=PostProcessType.MEAN, output_variable="test"
            )
        n_instr_before = builder.number_of_instructions

        met_mgr = MetricsManager()
        PydPostProcessingSanitisation().run(builder, ResultManager(), met_mgr)

        assert builder.number_of_instructions == met_mgr.get_metric(
            MetricsType.OptimizedInstructionCount
        )
        assert builder.number_of_instructions == n_instr_before

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


class TestPydReturnSanitisation:
    hw = PydEchoModelLoader(8).load()

    def test_empty_builder(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        res_mgr = ResultManager()

        with pytest.raises(ValueError):
            PydReturnSanitisationValidation().run(builder, res_mgr)

        PydReturnSanitisation().run(builder, res_mgr)
        PydReturnSanitisationValidation().run(builder, res_mgr)

        return_instr: PydReturn = builder._ir.tail
        assert len(return_instr.variables) == 0

    def test_single_return(self):
        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.returns(variables=["test"])
        ref_nr_instructions = builder.number_of_instructions

        res_mgr = ResultManager()
        PydReturnSanitisationValidation().run(builder, res_mgr)
        PydReturnSanitisation().run(builder, res_mgr)

        assert builder.number_of_instructions == ref_nr_instructions
        assert builder.instructions[0].variables == ["test"]

    def test_multiple_returns_squashed(self):
        q0 = self.hw.qubit_with_index(0)
        q1 = self.hw.qubit_with_index(1)

        builder = PydQuantumInstructionBuilder(hardware_model=self.hw)
        builder.measure_single_shot_z(target=q0, output_variable="out_q0")
        builder.measure_single_shot_z(target=q1, output_variable="out_q1")

        output_vars = [
            instr.output_variable for instr in builder if isinstance(instr, PydAcquire)
        ]
        assert len(output_vars) == 2

        builder.returns(variables=[output_vars[0]])
        builder.returns(variables=[output_vars[1]])

        res_mgr = ResultManager()
        # Two returns in a single IR should raise an error.
        with pytest.raises(ValueError):
            PydReturnSanitisationValidation().run(builder, res_mgr)

        # Compress the two returns to a single return and validate.
        PydReturnSanitisation().run(builder, res_mgr)
        PydReturnSanitisationValidation().run(builder, res_mgr)

        return_instr = builder._ir.tail
        assert isinstance(return_instr, PydReturn)
        for var in return_instr.variables:
            assert var in output_vars
