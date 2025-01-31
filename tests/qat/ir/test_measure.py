# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from typing import Literal

import numpy as np
import pytest
from pydantic import Field, ValidationError

from qat.ir.instructions import PhaseReset, PhaseShift, QuantumInstructionBlock, Synchronize
from qat.ir.measure import Acquire, MeasureBlock, PostProcessing
from qat.ir.waveforms import Pulse, PulseShapeType, Waveform
from qat.model.device import QubitId
from qat.purr.compiler.instructions import PostProcessType
from qat.utils.pydantic import ValidatedSet

from tests.qat.utils.hardware_models import generate_hw_model

model = generate_hw_model(4)
qubits = [qubit for qubit in model.qubits.values()]
qubits_uuid = [qubit.uuid for qubit in qubits]


class TestAcquire:
    def test_initiate(self):
        acquire_channel = qubits[0].resonator.pulse_channels.acquire
        inst = Acquire(targets=acquire_channel.uuid)
        assert inst.duration == 1e-6
        assert list(inst.targets)[0] == acquire_channel.uuid  # We only supplied one target.

    def test_filter(self):
        acquire_channel = qubits[0].resonator.pulse_channels.acquire
        filter = Pulse(
            targets=acquire_channel.uuid,
            waveform=Waveform(shape=PulseShapeType.GAUSSIAN, width=1e-6),
        )
        inst = Acquire(targets=acquire_channel.uuid, duration=1e-6, filter=filter)
        assert inst.filter == filter

    @pytest.mark.parametrize("time", [0, 5e-7, 1.01e-6, 2e-6])
    def test_filter_validation(self, time):
        acquire_channel = qubits[0].resonator.pulse_channels.acquire
        filter = Pulse(
            targets=acquire_channel.uuid,
            waveform=Waveform(shape=PulseShapeType.GAUSSIAN, width=time),
        )
        with pytest.raises(ValidationError):
            Acquire(targets=acquire_channel.uuid, duration=1e-6, filter=filter)

    @pytest.mark.parametrize("qubit_idx", list(model.qubits.keys()))
    def test_output_variable(self, qubit_idx):
        acquire_channel = qubits[qubit_idx].resonator.pulse_channels.acquire
        output_variable = f"Q{qubit_idx}"

        acq = Acquire(
            targets=acquire_channel.uuid, duration=1e-6, output_variable=output_variable
        )
        assert acq.output_variable == output_variable

        acq.output_variable = "new_output_var"
        assert acq.output_variable == "new_output_var"

    def test_invalid_output_variable(self):
        with pytest.raises(ValidationError):
            Acquire(targets="mock", output_variable=123)

        with pytest.raises(ValidationError):
            Acquire(targets="mock", output_variable=1.23)

        with pytest.raises(ValidationError):
            Acquire(targets="mock", output_variable=["output_var"])


class TestPostProcessing:
    @pytest.mark.parametrize("pp_type", PostProcessType)
    def test_change_pp(self, pp_type):
        inst = PostProcessing(process_type=pp_type)

        new_pp_type = PostProcessType.MUL
        inst.process_type = new_pp_type
        assert inst.process_type == PostProcessType.MUL

    @pytest.mark.parametrize("pp_type", PostProcessType)
    def test_enum_name_vs_value(self, pp_type):
        inst1 = PostProcessing(process_type=pp_type)
        inst2 = PostProcessing(process_type=pp_type.value)

        assert inst1.process_type == inst2.process_type

        with pytest.raises(ValidationError):
            inst1.process_type = "invalid"

        with pytest.raises(ValidationError):
            inst2.process_type = "invalid"


class TestMeasureBlock:
    def test_init_single_target(self):
        target = list(model.qubits.keys())[0]

        mb = MeasureBlock(qubit_targets=target)
        assert list(mb.qubit_targets)[0] == target

        with pytest.raises(ValidationError):
            MeasureBlock(qubit_targets=qubits_uuid)

    def test_init_multiple_targets(self):
        targets = list(model.qubits.keys())

        mb = MeasureBlock(qubit_targets=targets)
        assert mb.qubit_targets == targets

        with pytest.raises(ValidationError):
            MeasureBlock(qubit_targets=qubits_uuid)

    def test_add_instruction(self):
        qubit_id = list(model.qubits.keys())[0]
        qubit = model.qubits[qubit_id]
        measure_channel = qubit.resonator.pulse_channels.measure
        acquire_channel = qubit.resonator.pulse_channels.acquire

        mb = MeasureBlock(qubit_targets=qubit_id)
        mb.add(
            Pulse(
                waveform=Waveform(**measure_channel.measure_pulse.model_dump()),
                duration=1e-03,
                targets=measure_channel.uuid,
                type="measure",
            )
        )
        assert mb.number_of_instructions == 1

        mb.add(Acquire(targets=acquire_channel.uuid, duration=1e-09))
        assert mb.number_of_instructions == 2

    def add_invalid_instruction(self):
        qubit_id = list(model.qubits.keys())[0]
        qubit = model.qubits[qubit_id]
        measure_channel = qubit.resonator.pulse_channels.measure

        mb = MeasureBlock(qubit_targets=qubit_id)

        with pytest.raises(TypeError):
            mb.add(
                Pulse(
                    waveform=Waveform(**measure_channel.measure_pulse.model_dump()),
                    duration=1e-03,
                    targets=measure_channel.uuid,
                    type="drive",
                )
            )

        with pytest.raises(TypeError):
            mb.add(PhaseShift())

    def test_custom_measure_block(self):
        class AdditionalQuantumInstructionBlock(QuantumInstructionBlock):
            qubit_targets: ValidatedSet[QubitId] = Field(max_length=1)

            @property
            def target(self):
                return next(iter(self.targets))

        class CustomMeasureBlock(MeasureBlock):
            _valid_instructions: Literal[
                (Synchronize, Pulse, Acquire, AdditionalQuantumInstructionBlock)
            ] = (Synchronize, Pulse, Acquire, AdditionalQuantumInstructionBlock)

        qubit_indices = list(model.qubits.keys())
        custom_mb = CustomMeasureBlock(qubit_targets=qubit_indices)

        max_duration = 0.0
        for qubit_idx, qubit in model.qubits.items():
            qubit_pulse_channels = [pc.uuid for pc in qubit.all_pulse_channels]
            measure_channel = qubit.resonator.pulse_channels.measure
            acquire_channel = qubit.resonator.pulse_channels.acquire
            # Synchronise all pulse channels within a qubit.
            custom_mb.add(Synchronize(targets=qubit_pulse_channels))
            # Some dummy additional instruction block in the measure block.
            additional_block = AdditionalQuantumInstructionBlock(
                qubit_targets={qubit_idx},
                instructions=[
                    PhaseShift(targets={qubit.pulse_channels.drive.uuid}, phase=np.pi / 2),
                    PhaseReset(targets={qubit.pulse_channels.drive.uuid}),
                ],
            )

            custom_mb.add(additional_block)
            # Standard measure and acquire.
            measure = Pulse(
                waveform=Waveform(**measure_channel.measure_pulse.model_dump()),
                duration=1e-03,
                targets=measure_channel.uuid,
                type="measure",
            )
            custom_mb.add(measure)

            acquire = Acquire(targets=acquire_channel.uuid, duration=1e-09)
            custom_mb.add(acquire)
            # Synchronise all pulse channels within a qubit after measurement.
            custom_mb.add(Synchronize(targets=qubit_pulse_channels))

            duration = max(measure.duration, acquire.duration + acquire.delay)
            max_duration = max(max_duration, duration)

        assert custom_mb.duration == duration

        found_additional_block = False
        for instr in custom_mb.instructions:
            if isinstance(instr, AdditionalQuantumInstructionBlock):
                found_additional_block = True

        assert found_additional_block
