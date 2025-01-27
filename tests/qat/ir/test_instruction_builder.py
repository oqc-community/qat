# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import numpy as np
import pytest

from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.instructions import PhaseShift, Synchronize
from qat.ir.measure import Acquire, AcquireMode, MeasureBlock
from qat.ir.waveforms import Pulse
from qat.model.device import QubitId
from qat.utils.pydantic import ValidatedSet

from tests.qat.utils.hardware_models import generate_hw_model

hw_model = generate_hw_model(n_qubits=8)


class TestInstructionBuilder:
    def test_empty_builder(self):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        assert builder.number_of_instructions == 0


@pytest.mark.parametrize("qubit_index", list(range(0, hw_model.number_of_qubits)))
class TestPauliGates:
    def test_X_pi_2(self, qubit_index):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        builder.X(target=hw_model.qubit_with_index(qubit_index), theta=np.pi / 2.0)
        assert builder.number_of_instructions == 1
        assert builder._ir.instructions[0].type.value == "drive"

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
        x_pi_2_pulses = [
            instr
            for instr in builder._ir
            if (isinstance(instr, Pulse) and instr.type.value == "drive")
        ]
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

        # 2 pulses on the drive pulse channel, (two phaseshifts on the drive channel,
        # 2 phase shifts per coupled qubit for each cross resonance (cancellation) channel) * 3.
        ref_number_of_instructions = (
            1 * 2 + 1 * 3 + len(hw_model.logical_connectivity[qubit_index]) * 2 * 3
        )
        assert builder.number_of_instructions == ref_number_of_instructions

        phase_shifts = [instr for instr in builder._ir if isinstance(instr, PhaseShift)]
        x_pi_2_pulses = [
            instr
            for instr in builder._ir
            if (isinstance(instr, Pulse) and instr.type.value == "drive")
        ]
        assert len(phase_shifts) == ref_number_of_instructions - 2
        assert len(x_pi_2_pulses) == 2

    @pytest.mark.parametrize("theta", [-np.pi / 2, np.pi / 2])
    def test_Y_pi_2(self, qubit_index, theta):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        builder.Y(target=hw_model.qubit_with_index(qubit_index), theta=theta)

        ref_number_of_instructions = (
            1 + 1 * 2 + len(hw_model.logical_connectivity[qubit_index]) * 2 * 2
        )
        assert builder.number_of_instructions == ref_number_of_instructions

        phase_shifts = [instr for instr in builder._ir if isinstance(instr, PhaseShift)]
        x_pi_2_pulses = [
            instr
            for instr in builder._ir
            if (isinstance(instr, Pulse) and instr.type.value == "drive")
        ]
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
                if isinstance(instr, Pulse) and instr.type.value == "cross_resonance"
            ]
            crc_pulses = [
                instr
                for instr in builder._ir
                if isinstance(instr, Pulse) and instr.type.value == "cross_resonance_cancel"
            ]

            assert len(builder._ir.instructions) == 3
            assert len(synchronizes) == 1
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
                if isinstance(instr, Pulse) and instr.type.value == "cross_resonance"
            ]
            crc_pulses = [
                instr
                for instr in builder._ir
                if isinstance(instr, Pulse) and instr.type.value == "cross_resonance_cancel"
            ]
            phase_shifts = [instr for instr in builder._ir if isinstance(instr, PhaseShift)]

            assert len(builder._ir.instructions) == 7
            assert len(synchronizes) == 1
            assert len(cr_pulses) == 1
            assert len(crc_pulses) == 1
            assert len(phase_shifts) == 4


class TestMeasure:
    @pytest.mark.parametrize("mode", list(AcquireMode))
    def test_measure_block_contains_meas_acq(self, mode):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        qubit_indices = list(range(hw_model.number_of_qubits))

        for qubit_index in qubit_indices:
            builder.measure(targets=qubit_index, mode=mode)

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
                    if (
                        isinstance(sub_instruction, Pulse)
                        and sub_instruction.type.value == "measure"
                    ):
                        number_of_meas += 1
                    elif isinstance(sub_instruction, Acquire):
                        number_of_acq += 1
                        assert sub_instruction.mode == mode

                assert number_of_meas == 1
                assert number_of_acq == 1

                number_of_measure_blocks += 1

        assert number_of_measure_blocks == hw_model.number_of_qubits


@pytest.mark.parametrize("acquire_mode", AcquireMode)
class TestMeasure:
    @pytest.mark.parametrize("qubit_index", list(range(0, hw_model.number_of_qubits)))
    def test_single_qubit_measurement(self, qubit_index, acquire_mode):
        qubit = hw_model.qubit_with_index(qubit_index)

        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        builder.X(target=qubit)
        builder.measure(targets=qubit, mode=acquire_mode)

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
        assert (
            isinstance(measure := measure_block.instructions[1], Pulse)
            and measure_block.instructions[1].type.value == "measure"
        )
        assert (
            isinstance(acquire := measure_block.instructions[2], Acquire)
            and measure_block.instructions[2].mode == acquire_mode
        )

        assert measure_block.duration == max(
            measure.duration, acquire.duration + acquire.delay
        )

    def test_multi_qubit_measurement_no_qubit_sync(self, acquire_mode):
        qubits = list(hw_model.qubits.values())

        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        for qubit in qubits:
            builder.X(target=qubit)
        builder.measure(targets=qubits, mode=acquire_mode, sync_qubits=False)

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

    def test_multi_qubit_measurement_qubit_sync(self, acquire_mode):
        qubits = list(hw_model.qubits.values())

        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        for qubit in qubits:
            builder.X(target=qubit)
        builder.measure(targets=qubits, mode=acquire_mode, sync_qubits=True)

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
