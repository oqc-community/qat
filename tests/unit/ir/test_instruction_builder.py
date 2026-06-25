# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
"""Unit tests for the instruction builder and quantum instruction helpers.

These tests exercise high-level builder convenience methods and ensure the generated
instruction sequences match expectations for pulse, acquire and post-processing generation.
"""

import copy
import random

import numpy as np
import pytest

from qat.ir.instruction_basetypes import AcquireMode, PostProcessType, ProcessAxis
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.instructions import Delay, PhaseShift, Synchronize
from qat.ir.measure import (
    Acquire,
    Discriminate,
    Equalise,
    MeasureBlock,
    PostProcessing,
    PostSelect,
    acq_mode_process_axis,
)
from qat.ir.pulse_channel import PulseChannel
from qat.ir.waveforms import GaussianWaveform, Pulse, SampledWaveform
from qat.model.device import (
    CalibratablePulse,
    CrossResonanceCancellationPulseChannel,
    CrossResonancePulseChannel,
    MeasurePulseChannel,
    Qubit,
)
from qat.model.hardware_model import PhysicalHardwareModel
from qat.model.loaders.lucy import LucyModelLoader
from qat.model.post_processing import (
    LinearMapToRealMethod,
    MaxLikelihoodMethod,
    MLDiscriminateParams,
)
from qat.utils.hardware_model import generate_hw_model
from qat.utils.pydantic import QubitId, ValidatedSet

hw_model = generate_hw_model(n_qubits=8, seed=10)


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

    def test_flatten(self):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        qubit = hw_model.qubit_with_index(0)
        builder.X(target=qubit)
        builder.measure_single_shot_z(target=qubit)
        before = builder.number_of_instructions

        measure_block_exists = False
        measure_pulse_exists = False
        acquire_exists = False
        for instruction in builder.instructions:
            if isinstance(instruction, MeasureBlock):
                measure_block_exists = True
            elif (
                isinstance(instruction, Pulse)
                and instruction.target == qubit.measure_pulse_channel.uuid
            ):
                measure_pulse_exists = True
            elif (
                isinstance(instruction, Acquire)
                and instruction.target == qubit.acquire_pulse_channel.uuid
            ):
                acquire_exists = True
        assert measure_block_exists
        assert not measure_pulse_exists
        assert not acquire_exists

        builder.flatten()
        assert builder.number_of_instructions == before
        measure_pulse_exists = False
        acquire_exists = False
        for instruction in builder.instructions:
            assert not isinstance(instruction, MeasureBlock)
            if (
                isinstance(instruction, Pulse)
                and instruction.target == qubit.measure_pulse_channel.uuid
            ):
                measure_pulse_exists = True
            if (
                isinstance(instruction, Acquire)
                and instruction.target == qubit.acquire_pulse_channel.uuid
            ):
                acquire_exists = True
        assert measure_pulse_exists
        assert acquire_exists

    def test_synchronize(self):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        qubit = hw_model.qubit_with_index(0)
        builder.synchronize(qubit)
        assert builder.number_of_instructions == 1
        sync = builder._ir.instructions[0]
        assert isinstance(sync, Synchronize)

        # check for channels in both qubit and resonator
        assert qubit.drive_pulse_channel.uuid in sync.targets
        assert qubit.acquire_pulse_channel.uuid in sync.targets
        assert qubit.measure_pulse_channel.uuid in sync.targets

    def test_get_pulse_channel_from_hardware_model(self):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        qubit = hw_model.qubit_with_index(0)
        pulse_channel = builder.get_pulse_channel(qubit.drive_pulse_channel.uuid)
        assert pulse_channel is not qubit.drive_pulse_channel
        assert pulse_channel.uuid == qubit.drive_pulse_channel.uuid

    @pytest.mark.parametrize("string", [True, False])
    def test_create_pulse_channel(self, string: bool):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        qubit = hw_model.qubit_with_index(0)
        physical_channel = qubit.physical_channel.uuid if string else qubit.physical_channel

        pulse_channel = builder.create_pulse_channel(
            frequency=5.0e9,
            physical_channel=physical_channel,
            imbalance=0.9,
            phase_iq_offset=0.1,
            scale=0.8 + 0.1j,
            uuid="test-uuid",
        )
        assert isinstance(pulse_channel, PulseChannel)
        assert pulse_channel.uuid == "test-uuid"
        assert pulse_channel.frequency == 5.0e9
        assert pulse_channel.imbalance == 0.9
        assert pulse_channel.phase_iq_offset == 0.1
        assert pulse_channel.scale == 0.8 + 0.1j
        assert pulse_channel.physical_channel_id == qubit.physical_channel.uuid

    def test_create_pulse_channel_with_unknown_pulse_channel_raises(self):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)

        with pytest.raises(
            ValueError, match="Physical channel with id 'unknown' not found."
        ):
            builder.create_pulse_channel(
                frequency=5.0e9,
                physical_channel="unknown",
                imbalance=0.9,
                phase_iq_offset=0.1,
                scale=0.8 + 0.1j,
                uuid="test-uuid",
            )

    def test_pulse_channel_map_on_instantiation(self):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        assert len(builder._pulse_channels) == len(hw_model._ids_to_pulse_channels)
        assert builder._pulse_channels.keys() == hw_model._ids_to_pulse_channels.keys()
        for key, channel in builder._pulse_channels.items():
            hw_channel = hw_model._ids_to_pulse_channels[key]
            assert channel.scale == hw_channel.scale
            assert channel.phase_iq_offset == hw_channel.phase_iq_offset
            assert channel.imbalance == hw_channel.imbalance
            assert (
                channel.physical_channel_id
                == hw_model.physical_channel_for_pulse_channel_id(channel.uuid).uuid
            )


class TestQuantumInstructionBuilder:
    def test_blank_copy(self):
        model = LucyModelLoader(qubit_count=4).load()
        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.X(target=model.qubit_with_index(0), theta=np.pi / 2)

        new_builder = builder._create_empty_builder()
        assert new_builder.instructions == []
        assert new_builder._pulse_channels is builder._pulse_channels
        assert new_builder._qubit_index_by_uuid is builder._qubit_index_by_uuid
        assert new_builder._qubits_ordered_by_index is builder._qubits_ordered_by_index


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

    @pytest.mark.skip(
        reason="constrain theta takes theta=pi to -pi, making this test invalid"
    )
    @pytest.mark.parametrize("add_xpi_pulse", [True])
    @pytest.mark.parametrize("use_xpi_pulse", [True])
    def test_X_pi(self, qubit_index, add_xpi_pulse, use_xpi_pulse):
        model = generate_hw_model(n_qubits=8, seed=49)

        if add_xpi_pulse:
            model.qubit_with_index(
                qubit_index
            ).drive_pulse_channel.pulse_x_pi = CalibratablePulse(
                waveform_type=GaussianWaveform, width=100e-9, rise=1.0 / 3.0
            )
        qubit = model.qubit_with_index(qubit_index)
        qubit.direct_x_pi = use_xpi_pulse

        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.X(target=qubit, theta=np.pi)

        number_of_instructions = 1
        if not (use_xpi_pulse and add_xpi_pulse):
            number_of_instructions = (
                2 + 3 + len(hw_model.logical_connectivity[qubit_index]) * 6
            )

        assert builder.number_of_instructions == number_of_instructions
        assert builder._ir.instructions[0].target == qubit.drive_pulse_channel.uuid
        if number_of_instructions == 2:
            assert builder._ir.instructions[1].target == qubit.drive_pulse_channel.uuid

    @pytest.mark.parametrize("add_xpi_pulse", [False, True])
    @pytest.mark.parametrize("use_xpi_pulse", [False, True])
    def test_X_min_pi(self, qubit_index, add_xpi_pulse, use_xpi_pulse):
        model = generate_hw_model(n_qubits=8, seed=28)

        if add_xpi_pulse:
            model.qubit_with_index(
                qubit_index
            ).drive_pulse_channel.pulse_x_pi = CalibratablePulse(
                waveform_type=GaussianWaveform, width=100e-9, rise=1.0 / 3.0
            )
        qubit = model.qubit_with_index(qubit_index)
        qubit.direct_x_pi = use_xpi_pulse

        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.X(target=qubit, theta=-np.pi)

        if use_xpi_pulse and add_xpi_pulse:
            # 1 pulse on the drive pulse channel, two phaseshifts on the drive channel,
            # 2 Z pulses to rotate to -pi
            # 2 phase shifts per coupled qubit for each cross resonance (cancellation) channel) * 2 Z pulses`.
            ref_number_of_instructions = (
                1 + 2 + len(model.logical_connectivity[qubit_index]) * 2 * 2
            )
        else:
            # 2 pulses on the drive pulse channel, (two phaseshifts on the drive channel,
            # 3 Z pulses in U gate
            # 2 phase shifts per coupled qubit for each cross resonance (cancellation) channel) * 3 Z pulses`.
            ref_number_of_instructions = (
                2 + 3 + len(model.logical_connectivity[qubit_index]) * 6
            )

        assert builder.number_of_instructions == ref_number_of_instructions

        phase_shifts = [instr for instr in builder._ir if isinstance(instr, PhaseShift)]
        pulses = [instr for instr in builder._ir if (isinstance(instr, Pulse))]
        n_pulses = 1 if (use_xpi_pulse and add_xpi_pulse) else 2

        assert len(phase_shifts) == ref_number_of_instructions - n_pulses
        assert len(pulses) == n_pulses

    @pytest.mark.parametrize(
        "theta", [np.pi / 4, -np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4]
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

    @pytest.mark.parametrize("theta", [-np.pi, np.pi])
    @pytest.mark.parametrize("add_xpi_pulse", [False, True])
    @pytest.mark.parametrize("use_xpi_pulse", [False, True])
    def test_Y_pi(self, qubit_index, theta, add_xpi_pulse, use_xpi_pulse):
        model = generate_hw_model(n_qubits=8, seed=13)
        if add_xpi_pulse:
            model.qubit_with_index(
                qubit_index
            ).drive_pulse_channel.pulse_x_pi = CalibratablePulse(
                waveform_type=GaussianWaveform, width=100e-9, rise=1.0 / 3.0
            )
        qubit = model.qubit_with_index(qubit_index)
        qubit.direct_x_pi = use_xpi_pulse

        builder = QuantumInstructionBuilder(hardware_model=model)

        builder.Y(target=model.qubit_with_index(qubit_index), theta=theta)

        ref_number_of_instructions = (
            1 + 1 * 2 + len(model.logical_connectivity[qubit_index]) * 2 * 2
        )

        if not (use_xpi_pulse and add_xpi_pulse):
            ref_number_of_instructions += 1

        assert builder.number_of_instructions == ref_number_of_instructions

        phase_shifts = [instr for instr in builder._ir if isinstance(instr, PhaseShift)]
        pulses = [instr for instr in builder._ir if (isinstance(instr, Pulse))]
        n_pulses = 1 if (use_xpi_pulse and add_xpi_pulse) else 2

        assert len(phase_shifts) == ref_number_of_instructions - n_pulses
        assert len(pulses) == n_pulses

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

        for instr1, instr2 in zip(builder1._ir, builder2._ir, strict=True):
            assert instr1 == instr2


class TestTwoQubitGates:
    def test_invalid_theta(self):
        invalid_theta = np.pi / 2
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        target1 = hw_model.qubit_with_index(0)
        target2 = hw_model.qubit_with_index(1)

        with pytest.raises(NotImplementedError, match="Generic ZX gate not implemented"):
            builder.ZX(target1, target2, theta=invalid_theta)

    def test_invalid_hw_topology(self):
        idx = 0

        target1 = hw_model.qubit_with_index(idx)
        qubit_indices = ValidatedSet[QubitId](set(hw_model.qubits.keys()))
        invalid_qubit_indices = qubit_indices - hw_model.physical_connectivity[idx]

        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        for invalid_target_idx in invalid_qubit_indices:
            with pytest.raises(ValueError, match="Tried to perform cross resonance"):
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

    def test_ZX_with_no_calibrated_pulse_raises_error(self):
        model = LucyModelLoader(qubit_count=4).load()
        model_dump = model.model_dump()
        model_dump["qubits"][0]["pulse_channels"]["cross_resonance_channels"][1][
            "zx_pi_4_pulse"
        ] = None
        model = PhysicalHardwareModel.model_validate(model_dump)

        builder = QuantumInstructionBuilder(hardware_model=model)
        with pytest.raises(ValueError, match="No `zx_pi_4_pulse` available"):
            builder.ZX(
                target1=model.qubit_with_index(0),
                target2=model.qubit_with_index(1),
                theta=np.pi / 4,
            )


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
            measure_block.number_of_instructions == 5
        )  # 2 synchonises, 1 measure and 1 acquire
        assert isinstance(measure_block.instructions[0], Synchronize) and isinstance(
            measure_block.instructions[-1], Synchronize
        )
        assert isinstance(measure := measure_block.instructions[1], Pulse) and isinstance(
            hw_model.pulse_channel_with_id(measure.target), MeasurePulseChannel
        )
        assert (
            isinstance(delay := measure_block.instructions[2], Delay)
            and measure_block.instructions[3].mode == mode
        )
        assert (
            isinstance(acquire := measure_block.instructions[3], Acquire)
            and measure_block.instructions[3].mode == mode
        )

        assert measure_block.duration == max(
            measure.duration, acquire.duration + delay.duration
        )

    def test_measures(self):
        qubit = hw_model.qubit_with_index(0)

        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        builder.X(target=qubit)
        builder.measure_single_shot_z(target=qubit)

    def test_state_map_legacy_post_processing(self):
        """When a qubit has no post_process_method (legacy mean_z_map_args path),
        measure_with_granular_post_processing should emit the granular Equalise →
        Discriminate chain derived from mean_z_map_args.

        Discriminate outputs integer keys directly.
        """

        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        qubit: Qubit = copy.deepcopy(hw_model.qubit_with_index(0))
        # Simulate a legacy qubit: mean_z_map_args set, no post_process_method.
        qubit.__dict__["post_process_method"] = None
        qubit.__dict__["mean_z_map_args"] = [1 + 0j, 0j]

        builder.measure_with_granular_post_processing(target=qubit, output_variable="0")
        eq_instructions = [i for i in builder.instructions if isinstance(i, Equalise)]
        pp_instructions = [i for i in builder.instructions if isinstance(i, PostProcessing)]
        disc_instructions = [i for i in builder.instructions if isinstance(i, Discriminate)]

        assert len(pp_instructions) == 0
        assert len(eq_instructions) == 1
        assert len(disc_instructions) == 1

    def test_state_map_post_processing(self):
        """When a qubit has LinearMapToRealMethod, measure_with_granular_post_processing
        emits Equalise → Discriminate (threshold=0.0).

        Discriminate outputs integer keys directly.
        """

        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        qubit: Qubit = hw_model.qubit_with_index(0)
        qubit = copy.deepcopy(qubit)  # Makes a temporary copy for this test.
        qubit.__dict__["post_process_method"] = LinearMapToRealMethod()
        qubit.__dict__["mean_z_map_args"] = None

        builder.measure_with_granular_post_processing(target=qubit, output_variable="0")
        eq_instructions = [i for i in builder.instructions if isinstance(i, Equalise)]
        disc_instructions = [i for i in builder.instructions if isinstance(i, Discriminate)]
        assert len(eq_instructions) == 1
        assert len(disc_instructions) == 1
        assert disc_instructions[0].threshold == 0.0

    def test_legacy_post_processing_inferred_from_target(self):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        qubit: Qubit = copy.deepcopy(hw_model.qubit_with_index(0))
        qubit.__dict__["post_process_method"] = None
        qubit.__dict__["mean_z_map_args"] = [1 + 0j, 0j]

        builder.post_processing(
            target=qubit,
            output_variable="0",
            process_type=PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
        )
        pp_instructions = [i for i in builder.instructions if isinstance(i, PostProcessing)]
        assert len(pp_instructions) == 1
        pp = pp_instructions[0]
        assert pp.process_type == PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL
        assert pp.args == [1 + 0j, 0j]

    def test_measure_post_selected_linear_map(self):
        """measure_with_granular_post_processing with LinearMapToRealMethod emits
        MeasureBlock then Equalise → Discriminate.

        No PostProcessing(LINEAR_MAP) is emitted.
        """

        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        qubit: Qubit = copy.deepcopy(hw_model.qubit_with_index(0))
        qubit.__dict__["post_process_method"] = LinearMapToRealMethod()
        qubit.__dict__["mean_z_map_args"] = None

        builder.measure_with_granular_post_processing(target=qubit, output_variable="0")

        pp_instructions = [i for i in builder.instructions if isinstance(i, PostProcessing)]
        eq_instructions = [i for i in builder.instructions if isinstance(i, Equalise)]
        disc_instructions = [i for i in builder.instructions if isinstance(i, Discriminate)]

        assert len(pp_instructions) == 0
        assert len(eq_instructions) == 1
        assert len(disc_instructions) == 1
        assert disc_instructions[0].threshold == 0.0

    def test_measure_post_selected_legacy_qubit(self):
        """measure_with_granular_post_processing for a legacy qubit (no post_process_method)
        emits the granular Equalise → Discriminate chain derived from mean_z_map_args."""

        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        qubit: Qubit = copy.deepcopy(hw_model.qubit_with_index(0))
        qubit.__dict__["post_process_method"] = None
        qubit.__dict__["mean_z_map_args"] = [1 + 0j, 0j]

        builder.measure_with_granular_post_processing(target=qubit, output_variable="0")

        pp_instructions = [i for i in builder.instructions if isinstance(i, PostProcessing)]
        eq_instructions = [i for i in builder.instructions if isinstance(i, Equalise)]
        disc_instructions = [i for i in builder.instructions if isinstance(i, Discriminate)]

        assert len(pp_instructions) == 0
        assert len(eq_instructions) == 1
        assert len(disc_instructions) == 1

    @pytest.mark.parametrize("axis", list(ProcessAxis))
    @pytest.mark.parametrize(
        "measure_method",
        (
            "measure_single_shot_z",
            "measure_single_shot_signal",
            "measure_mean_z",
            "measure_mean_signal",
            "measure_scope_mode",
            "measure_single_shot_binned",
        ),
    )
    def test_single_qubit_measurement_with_pp(self, axis, measure_method):
        def _expected_pp_signatures(method: str, proc_axis: ProcessAxis):
            mean_time = (PostProcessType.MEAN, (ProcessAxis.TIME,))
            mean_sequence = (PostProcessType.MEAN, (ProcessAxis.SEQUENCE,))
            linear_map = (PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL, ())
            discriminate = (PostProcessType.DISCRIMINATE, ())

            if method == "measure_single_shot_z":
                return (
                    [linear_map]
                    if proc_axis == ProcessAxis.SEQUENCE
                    else [mean_time, linear_map]
                )
            if method == "measure_single_shot_signal":
                return [] if proc_axis == ProcessAxis.SEQUENCE else [mean_time]
            if method == "measure_mean_z":
                return (
                    [mean_sequence, linear_map]
                    if proc_axis == ProcessAxis.SEQUENCE
                    else [mean_time, linear_map]
                )
            if method == "measure_mean_signal":
                return [mean_sequence]
            if method == "measure_scope_mode":
                return []
            if method == "measure_single_shot_binned":
                return (
                    [linear_map, discriminate]
                    if proc_axis == ProcessAxis.SEQUENCE
                    else [mean_time, linear_map, discriminate]
                )

            raise AssertionError(f"Unhandled measure method '{method}'.")

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
        pp_signatures_per_qubit = [[] for _ in qubit_indices]
        for instruction in builder._ir.instructions:
            if isinstance(instruction, MeasureBlock):
                for q_idx in instruction.qubit_targets:
                    measure_blocks_per_qubit[q_idx] += 1

                for acq in instruction:
                    if isinstance(acq, Acquire):
                        assert acq.mode == ref_mode

            if isinstance(instruction, PostProcessing):
                pp_signatures_per_qubit[int(instruction.output_variable)].append(
                    (instruction.process_type, tuple(instruction.axes))
                )

        assert measure_blocks_per_qubit == [1] * len(qubit_indices)
        expected = _expected_pp_signatures(measure_method, axis)
        assert pp_signatures_per_qubit == [expected] * len(qubit_indices)

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

        assert measure_block is not None
        assert measure_block.number_of_instructions == 5 * len(
            qubits
        )  # 2 synchronises, 1 measure, 1 delay and 1 acquire per qubit

        max_duration = 0.0
        for i, instruction in enumerate(measure_block):
            if isinstance(instruction, Acquire):
                max_duration = max(
                    max_duration,
                    instruction.duration + measure_block.instructions[i - 1].duration,
                )
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

        assert measure_block is not None
        assert measure_block.number_of_instructions == 2 + 3 * len(
            qubits
        )  # 2 synchronises in total; 1 measure, 1 delay and 1 acquire per qubit

        max_duration = 0.0
        for i, instruction in enumerate(measure_block):
            if isinstance(instruction, Acquire):
                max_duration = max(
                    max_duration,
                    instruction.duration + measure_block.instructions[i - 1].duration,
                )
            else:
                max_duration = max(max_duration, instruction.duration)

        assert measure_block.duration == max_duration

    def test_measure_with_weights(self):
        model = generate_hw_model(n_qubits=1, seed=12)
        acquire = model.qubit_with_index(0).acquire_pulse_channel.acquire
        acquire.width = 800e-9
        acquire.weights = np.random.rand(800)
        acquire.use_weights = True

        builder = QuantumInstructionBuilder(hardware_model=model)
        builder.measure_single_shot_z(target=model.qubit_with_index(0))
        for instruction in builder._ir.instructions:
            if isinstance(instruction, MeasureBlock):
                for sub_instruction in instruction.instructions:
                    if isinstance(sub_instruction, Acquire):
                        filter = sub_instruction.filter
                        assert filter is not None
                        assert isinstance(filter, Pulse)
                        assert isinstance(filter.waveform, SampledWaveform)

                        acquire_channel = model.qubit_with_index(0).acquire_pulse_channel
                        assert filter.target == acquire_channel.uuid

    def test_measure_syncs_channels_on_both_resonator_and_qubit(self):
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        qubit = hw_model.qubit_with_index(0)

        builder.measure(qubit)

        syncs = [instr for instr in builder._ir if isinstance(instr, Synchronize)]
        assert len(syncs) == 2  # One before and one after the measure block

        for sync in syncs:
            assert qubit.drive_pulse_channel.uuid in sync.targets
            assert qubit.acquire_pulse_channel.uuid in sync.targets
            assert qubit.measure_pulse_channel.uuid in sync.targets


class TestPostSelectionConfig:
    """Tests post-select emission behavior for the current builder API."""

    @pytest.fixture
    def ml_qubit_with_disallowed(self):
        """Returns a qubit configured with MaxLikelihoodMethod where state 1 is
        disallowed."""
        qubit = copy.deepcopy(hw_model.qubit_with_index(0))
        qubit.__dict__["mean_z_map_args"] = None
        qubit.__dict__["post_process_method"] = MaxLikelihoodMethod(
            states={
                0: MLDiscriminateParams(location=1 + 0j),
                -1: MLDiscriminateParams(location=-1 + 0j),  # negative key = disallowed
            },
        )
        return qubit

    def test_ml_post_select_emitted_only_when_explicitly_added(
        self, ml_qubit_with_disallowed
    ):
        """In current API, PostSelect is appended explicitly by caller orchestration."""
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        builder.emit_granular_post_processing(
            target=ml_qubit_with_disallowed, output_variable="v"
        )
        assert [type(i) for i in builder.instructions] == [Discriminate]

        builder.emit_post_select("v")
        assert [type(i) for i in builder.instructions] == [Discriminate, PostSelect]

    def test_linear_map_post_select_is_noop_at_runtime(self):
        """LinearMapToRealMethod always produces non-negative keys so PostSelect is a noop.

        emit_post_select unconditionally emits a PostSelect instruction; at runtime
        apply_post_select produces an all-True mask when all keys are non-negative.
        """
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        qubit = copy.deepcopy(hw_model.qubit_with_index(0))
        qubit.__dict__["mean_z_map_args"] = None
        qubit.__dict__["post_process_method"] = LinearMapToRealMethod()
        builder.emit_granular_post_processing(target=qubit, output_variable="v")
        assert [type(i) for i in builder.instructions] == [Equalise, Discriminate]

        builder.emit_post_select("v")
        assert [type(i) for i in builder.instructions] == [
            Equalise,
            Discriminate,
            PostSelect,
        ]

    def test_measure_post_selected_does_not_emit_post_select(
        self, ml_qubit_with_disallowed
    ):
        """measure_post_selected emits granular processing but not PostSelect by itself."""
        builder = QuantumInstructionBuilder(hardware_model=hw_model)
        builder.measure_with_granular_post_processing(
            target=ml_qubit_with_disallowed, output_variable="v"
        )
        ps = [i for i in builder.instructions if isinstance(i, PostSelect)]
        assert len(ps) == 0

    def test_emit_post_select_always_emits(self):
        """emit_post_select unconditionally emits a PostSelect instruction."""
        b1 = QuantumInstructionBuilder(hardware_model=hw_model)
        before = b1.number_of_instructions
        b1.emit_post_select("v")
        assert b1.number_of_instructions == before + 1
