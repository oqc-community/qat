# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import random

import numpy as np
import pytest

from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.instructions import PhaseShift, Synchronize
from qat.ir.measure import (
    Acquire,
    AcquireMode,
    MeasureBlock,
    PostProcessing,
    ProcessAxis,
    acq_mode_process_axis,
)
from qat.ir.waveforms import Pulse
from qat.model.device import (
    CrossResonanceCancellationPulseChannel,
    CrossResonancePulseChannel,
    MeasurePulseChannel,
)
from qat.utils.hardware_model import generate_hw_model
from qat.utils.pydantic import QubitId, ValidatedSet

hw_model = generate_hw_model(n_qubits=8)


class TestInstructionBuilder:
    def test_empty_builder(self):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        assert builder.number_of_instructions == 0

    def test_add_two_builders(self):
        builder1 = QuantumInstructionBuilder(hardware_model=hw_model)
        builder2 = QuantumInstructionBuilder(hardware_model=hw_model)

        builder1.X(target=hw_model.qubit_with_index(0))
        builder2.Y(target=hw_model.qubit_with_index(1))
        before1 = builder1.number_of_instructions
        before2 = builder2.number_of_instructions

        combined_builder = builder1 + builder2
        assert combined_builder.number_of_instructions == before1 + before2
        assert builder1.number_of_instructions == before1
        assert builder2.number_of_instructions == before2

    def test_radd(self):
        builder1 = QuantumInstructionBuilder(hardware_model=hw_model)
        builder2 = QuantumInstructionBuilder(hardware_model=hw_model)

        builder1.X(target=hw_model.qubit_with_index(0))
        builder2.Y(target=hw_model.qubit_with_index(1))
        before1 = builder1.number_of_instructions
        before2 = builder2.number_of_instructions

        builder1 += builder2
        assert builder1.number_of_instructions == before1 + before2
        assert builder2.number_of_instructions == before2


@pytest.mark.parametrize("qubit_index", list(range(0, hw_model.number_of_qubits)))
class TestPauliGates:
    @pytest.mark.parametrize("amp_scale", [0.5, 1.0, 2.0])
    def test_amp_scale_hw_X_pi_2(self, qubit_index, amp_scale):
        qubit = hw_model.qubit_with_index(qubit_index)

        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        pulse = builder._hw_X_pi_2(target=qubit, amp_scale=amp_scale)[0]

        assert isinstance(pulse, Pulse)
        assert pulse.target == qubit.drive_pulse_channel.uuid
        assert pulse.waveform.amp == qubit.drive_pulse_channel.pulse.amp * amp_scale

    def test_X_pi_2(self, qubit_index):
        qubit = hw_model.qubit_with_index(qubit_index)

        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        builder.X(target=qubit, theta=np.pi / 2.0)
        assert builder.number_of_instructions == 1
        assert builder._ir.instructions[0].target == qubit.drive_pulse_channel.uuid

    def test_X_min_pi_2(self, qubit_index):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        builder.X(target=hw_model.qubit_with_index(qubit_index), theta=-np.pi / 2.0)

        # 1 pulse on the drive pulse channel, two phaseshifts on the drive channel,
        # 2 phase shifts per coupled qubit for each cross resonance (cancellation) channel.
        ref_number_of_instructions = (
            1 + 1 * 2 + len(hw_model.logical_connectivity[qubit_index]) * 2 * 2
        )
        assert builder.number_of_instructions == ref_number_of_instructions

        phase_shifts = [instr for instr in builder._ir if isinstance(instr, PhaseShift)]
        x_pi_2_pulses = [instr for instr in builder._ir if (isinstance(instr, Pulse))]
        assert len(phase_shifts) == ref_number_of_instructions - 1
        assert len(x_pi_2_pulses) == 1

    @pytest.mark.parametrize(
        "theta", [np.pi / 4, -np.pi / 4, np.pi, -np.pi, 3 * np.pi / 4, -3 * np.pi / 4]
    )
    @pytest.mark.parametrize("pauli_gate", ["X", "Y"])
    def test_X_Y_arbitrary_rotation(self, qubit_index, theta, pauli_gate):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        getattr(builder, pauli_gate)(
            target=hw_model.qubit_with_index(qubit_index), theta=theta
        )

        if pauli_gate == "X":
            number_of_z = 3
        elif pauli_gate == "Y":
            number_of_z = 2

        if builder.constrain(theta):
            # 2 pulses on the drive pulse channel, (two phaseshifts on the drive channel,
            # 2 phase shifts per coupled qubit for each cross resonance (cancellation) channel) * `number_of_z``.
            ref_number_of_instructions = (
                1 * 2
                + 1 * number_of_z
                + len(hw_model.logical_connectivity[qubit_index]) * 2 * number_of_z
            )
        else:
            ref_number_of_instructions = 1
        assert builder.number_of_instructions == ref_number_of_instructions

        phase_shifts = [instr for instr in builder._ir if isinstance(instr, PhaseShift)]
        x_pi_2_pulses = [instr for instr in builder._ir if isinstance(instr, Pulse)]
        assert len(phase_shifts) == ref_number_of_instructions - 2
        assert len(x_pi_2_pulses) == 2

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_unitary_gate_single_angle(self, qubit_index, seed):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        builder.U(
            target=hw_model.qubit_with_index(qubit_index),
            theta=random.Random(seed).uniform(0.01, np.pi - 0.01),
            phi=random.Random(seed + 1).uniform(0.01, np.pi - 0.01),
            lamb=random.Random(seed + 2).uniform(0.01, np.pi - 0.01),
        )

        ref_number_of_instructions = (
            1 * 2 + 1 * 3 + len(hw_model.logical_connectivity[qubit_index]) * 2 * 3
        )  # 2 drive pulses, 3 phase shifts for drive ch + 6 phase shifts per coupling

        assert builder.number_of_instructions == ref_number_of_instructions

    @pytest.mark.parametrize("theta", [-np.pi / 2, np.pi / 2])
    def test_Y_pi_2(self, qubit_index, theta):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        builder.Y(target=hw_model.qubit_with_index(qubit_index), theta=theta)

        ref_number_of_instructions = (
            1 + 1 * 2 + len(hw_model.logical_connectivity[qubit_index]) * 2 * 2
        )
        assert builder.number_of_instructions == ref_number_of_instructions

        phase_shifts = [instr for instr in builder._ir if isinstance(instr, PhaseShift)]
        x_pi_2_pulses = [instr for instr in builder._ir if isinstance(instr, Pulse)]
        assert len(phase_shifts) == ref_number_of_instructions - 1
        assert len(x_pi_2_pulses) == 1

    @pytest.mark.parametrize(
        "theta", [np.pi / 4, np.pi / 2, np.pi, -np.pi, 3 * np.pi / 2, -3 * np.pi / 4]
    )
    def test_Z(self, qubit_index, theta):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        builder.Z(target=hw_model.qubit_with_index(qubit_index), theta=theta)

        # 1 phase shift on the drive pulse channel, 2 phase shifts per coupled qubit
        # for each cross resonance (cancellation) channel
        ref_number_of_instructions = 1 + len(hw_model.logical_connectivity[qubit_index]) * 2
        assert builder.number_of_instructions == ref_number_of_instructions

        for instruction in builder._ir:
            assert isinstance(instruction, PhaseShift)
            assert instruction.phase == QuantumInstructionBuilder.constrain(theta)


@pytest.mark.parametrize("qubit_index", list(range(0, hw_model.number_of_qubits)))
@pytest.mark.parametrize("pauli_gate", ["X", "Y", "Z"])
class TestConstrainedPauliGates:
    def test_unit_pauli(self, qubit_index, pauli_gate):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        getattr(builder, pauli_gate)(target=hw_model.qubit_with_index(qubit_index), theta=0)
        assert builder.number_of_instructions == 0

        getattr(builder, pauli_gate)(
            target=hw_model.qubit_with_index(qubit_index), theta=2 * np.pi
        )
        assert builder.number_of_instructions == 0

    @pytest.mark.parametrize("theta", [np.pi / 2, np.pi, 3 * np.pi / 2])
    def test_non_unit_pauli(self, qubit_index, pauli_gate, theta):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        getattr(builder, pauli_gate)(
            target=hw_model.qubit_with_index(qubit_index), theta=theta
        )
        assert builder.number_of_instructions > 0

    @pytest.mark.parametrize("theta", [0, np.pi / 2, np.pi, 3 * np.pi / 2])
    def test_redundant_2pi_rotation(self, qubit_index, pauli_gate, theta):
        builder1 = QuantumInstructionBuilder(hardware_model=hw_model)
        getattr(builder1, pauli_gate)(
            target=hw_model.qubit_with_index(qubit_index), theta=theta
        )

        builder2 = QuantumInstructionBuilder(hardware_model=hw_model)
        getattr(builder2, pauli_gate)(
            target=hw_model.qubit_with_index(qubit_index), theta=theta + 2 * np.pi
        )
        assert builder1.number_of_instructions == builder2.number_of_instructions

        for instr1, instr2 in zip(builder1._ir, builder2._ir):
            assert instr1 == instr2


class TestTwoQubitGates:
    def test_invalid_theta(self):
        invalid_theta = np.pi / 2
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        target1 = hw_model.qubit_with_index(0)
        target2 = hw_model.qubit_with_index(1)

        with pytest.raises(NotImplementedError):
            builder.ZX(target1, target2, theta=invalid_theta)

    def test_invalid_hw_topology(self):
        idx = 0

        target1 = hw_model.qubit_with_index(idx)
        qubit_indices = ValidatedSet[QubitId](set(hw_model.qubits.keys()))
        invalid_qubit_indices = qubit_indices - hw_model.physical_connectivity[idx]

        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        for invalid_target_idx in invalid_qubit_indices:
            with pytest.raises(ValueError):
                invalid_target2 = hw_model.qubit_with_index(invalid_target_idx)
                builder.ZX(target1, invalid_target2, theta=np.pi / 4)

    def test_ZX_pi_4(self):
        target1_id = 1
        target1 = hw_model.qubit_with_index(target1_id)

        for target2_id in hw_model.physical_connectivity[target1_id]:
            target2 = hw_model.qubit_with_index(target2_id)
            builder = QuantumInstructionBuilder(hardware_model=hw_model)
            builder.ZX(target1, target2, theta=np.pi / 4)

            synchronizes = [
                instr for instr in builder._ir if isinstance(instr, Synchronize)
            ]

            cr_pulses = [
                instr
                for instr in builder._ir
                if isinstance(instr, Pulse)
                and isinstance(
                    hw_model.pulse_channel_with_id(instr.target), CrossResonancePulseChannel
                )
            ]
            crc_pulses = [
                instr
                for instr in builder._ir
                if isinstance(instr, Pulse)
                and isinstance(
                    hw_model.pulse_channel_with_id(instr.target),
                    CrossResonanceCancellationPulseChannel,
                )
            ]

            assert len(builder._ir.instructions) == 4
            assert len(synchronizes) == 2
            assert len(cr_pulses) == 1
            assert len(crc_pulses) == 1

    def test_ZX_min_pi_4(self):
        target1_id = 5
        target1 = hw_model.qubit_with_index(target1_id)

        for target2_id in hw_model.physical_connectivity[target1_id]:
            target2 = hw_model.qubit_with_index(target2_id)
            builder = QuantumInstructionBuilder(hardware_model=hw_model)
            builder.ZX(target1, target2, theta=-np.pi / 4)

            synchronizes = [
                instr for instr in builder._ir if isinstance(instr, Synchronize)
            ]
            cr_pulses = [
                instr
                for instr in builder._ir
                if isinstance(instr, Pulse)
                and isinstance(
                    hw_model.pulse_channel_with_id(instr.target), CrossResonancePulseChannel
                )
            ]
            crc_pulses = [
                instr
                for instr in builder._ir
                if isinstance(instr, Pulse)
                and isinstance(
                    hw_model.pulse_channel_with_id(instr.target),
                    CrossResonanceCancellationPulseChannel,
                )
            ]
            phase_shifts = [instr for instr in builder._ir if isinstance(instr, PhaseShift)]

            assert len(builder._ir.instructions) == 8
            assert len(synchronizes) == 2
            assert len(cr_pulses) == 1
            assert len(crc_pulses) == 1
            assert len(phase_shifts) == 4

    def test_ECR(self):
        target1_id = 5
        target1 = hw_model.qubit_with_index(target1_id)
        target2_id = next(iter(hw_model.physical_connectivity[target1_id]))
        target2 = hw_model.qubit_with_index(target2_id)

        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        builder.ECR(target1, target2)

        synchronizes = [instr for instr in builder._ir if isinstance(instr, Synchronize)]
        cr_pulses = [
            instr
            for instr in builder._ir
            if isinstance(instr, Pulse)
            and isinstance(
                hw_model.pulse_channel_with_id(instr.target), CrossResonancePulseChannel
            )
        ]
        crc_pulses = [
            instr
            for instr in builder._ir
            if isinstance(instr, Pulse)
            and isinstance(
                hw_model.pulse_channel_with_id(instr.target),
                CrossResonanceCancellationPulseChannel,
            )
        ]

        assert len(synchronizes) == 2 * 2  # 2 synchronizes per ZX_pi_4 gate
        assert len(cr_pulses) == 2
        assert len(crc_pulses) == 2


class TestMeasure:
    @pytest.mark.parametrize("mode", list(AcquireMode))
    def test_measure_block_contains_meas_acq(self, mode):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        qubit_indices = list(range(hw_model.number_of_qubits))

        for qubit in hw_model.qubits.values():
            builder.measure(targets=qubit, mode=mode)

        number_of_measure_blocks = 0
        # Looping over `builder._instructions` would result in
        # flattening the whole composite instruction, which we
        # we do not want for testing purposes here as we want to
        # detect `MeasureBlock`s and check their properties.
        for instruction in builder._ir.instructions:
            if isinstance(instruction, MeasureBlock):
                assert len(instruction.qubit_targets) == 1
                assert list(instruction.qubit_targets)[0] in qubit_indices

                number_of_meas, number_of_acq = 0, 0
                for sub_instruction in instruction:
                    if isinstance(sub_instruction, Pulse) and isinstance(
                        hw_model.pulse_channel_with_id(sub_instruction.target),
                        MeasurePulseChannel,
                    ):
                        number_of_meas += 1
                    elif isinstance(sub_instruction, Acquire):
                        number_of_acq += 1
                        assert sub_instruction.mode == mode

                assert number_of_meas == 1
                assert number_of_acq == 1

                number_of_measure_blocks += 1

        assert number_of_measure_blocks == hw_model.number_of_qubits

    @pytest.mark.parametrize("mode", list(AcquireMode))
    @pytest.mark.parametrize("qubit_index", list(range(0, hw_model.number_of_qubits)))
    def test_single_qubit_measurement(self, qubit_index, mode):
        qubit = hw_model.qubit_with_index(qubit_index)

        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        builder.X(target=qubit)
        builder.measure(targets=qubit, mode=mode)

        measure_block = None
        for composite_instr in builder._ir.instructions:
            if isinstance(composite_instr, MeasureBlock):
                measure_block = composite_instr

        assert measure_block

        assert (
            measure_block.number_of_instructions == 4
        )  # 2 synchonises, 1 measure and 1 acquire
        assert isinstance(measure_block.instructions[0], Synchronize) and isinstance(
            measure_block.instructions[-1], Synchronize
        )
        assert isinstance(measure := measure_block.instructions[1], Pulse) and isinstance(
            hw_model.pulse_channel_with_id(measure.target), MeasurePulseChannel
        )
        assert (
            isinstance(acquire := measure_block.instructions[2], Acquire)
            and measure_block.instructions[2].mode == mode
        )

        assert measure_block.duration == max(
            measure.duration, acquire.duration + acquire.delay
        )

    @pytest.mark.parametrize("axis", list(ProcessAxis))
    @pytest.mark.parametrize(
        ("measure_method", "pp_length"),
        (
            ["measure_single_shot_z", 3],
            ["measure_single_shot_signal", 2],
            ["measure_mean_z", 4],
            ["measure_mean_signal", 3],
            ["measure_scope_mode", 2],
            ["measure_single_shot_binned", 4],
        ),
    )
    def test_single_qubit_measurement_with_pp(self, axis, measure_method, pp_length):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        qubit_indices = list(range(hw_model.number_of_qubits))

        for q_idx, q in hw_model.qubits.items():
            if measure_method == "measure_mean_signal":
                getattr(builder, measure_method)(target=q, output_variable=str(q_idx))
                ref_mode = AcquireMode.INTEGRATOR
            elif measure_method == "measure_scope_mode":
                getattr(builder, measure_method)(target=q, output_variable=str(q_idx))
                ref_mode = AcquireMode.SCOPE
            else:
                getattr(builder, measure_method)(
                    target=q, axis=axis, output_variable=str(q_idx)
                )
                ref_mode = acq_mode_process_axis[axis]

        measure_blocks_per_qubit = [0 for _ in qubit_indices]
        pps_per_qubit = [0 for _ in qubit_indices]
        for instruction in builder._ir.instructions:
            if isinstance(instruction, MeasureBlock):
                for q_idx in instruction.qubit_targets:
                    measure_blocks_per_qubit[q_idx] += 1

                for acq in instruction:
                    if isinstance(acq, Acquire):
                        assert acq.mode == ref_mode

            if isinstance(instruction, PostProcessing):
                pps_per_qubit[int(instruction.output_variable)] += 1

        assert measure_blocks_per_qubit == [1] * len(qubit_indices)
        assert pps_per_qubit == [pp_length] * len(qubit_indices)

    @pytest.mark.parametrize("mode", list(AcquireMode))
    def test_multi_qubit_measurement_no_qubit_sync(self, mode):
        qubits = list(hw_model.qubits.values())

        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        for qubit in qubits:
            builder.X(target=qubit)
        builder.measure(targets=qubits, mode=mode, sync_qubits=False)

        measure_block = None
        for composite_instr in builder._ir.instructions:
            if isinstance(composite_instr, MeasureBlock):
                measure_block = composite_instr

        assert measure_block
        assert measure_block.number_of_instructions == 4 * len(
            qubits
        )  # 2 synchonises, 1 measure and 1 acquire per qubit

        max_duration = 0.0
        for instruction in measure_block:
            if isinstance(instruction, Acquire):
                max_duration = max(max_duration, instruction.duration + instruction.delay)
            else:
                max_duration = max(max_duration, instruction.duration)

        assert measure_block.duration == max_duration

    @pytest.mark.parametrize("mode", list(AcquireMode))
    def test_multi_qubit_measurement_qubit_sync(self, mode):
        qubits = list(hw_model.qubits.values())

        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        for qubit in qubits:
            builder.X(target=qubit)
        builder.measure(targets=qubits, mode=mode, sync_qubits=True)

        measure_block = None
        for composite_instr in builder._ir.instructions:
            if isinstance(composite_instr, MeasureBlock):
                measure_block = composite_instr

        assert measure_block
        assert measure_block.number_of_instructions == 2 + 2 * len(
            qubits
        )  # 2 synchonises in total, 1 measure and 1 acquire per qubit

        max_duration = 0.0
        for instruction in measure_block:
            if isinstance(instruction, Acquire):
                max_duration = max(max_duration, instruction.duration + instruction.delay)
            else:
                max_duration = max(max_duration, instruction.duration)

        assert measure_block.duration == max_duration
