# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
from typing import Union, get_args

import numpy as np
import pytest
from pydantic import Field, ValidationError

from qat.ir.instructions import (
    Delay,
    PhaseReset,
    PhaseShift,
    QuantumInstructionBlock,
    Synchronize,
)
from qat.ir.measure import VALID_MEASURE_INSTR, Acquire, MeasureBlock, PostProcessing
from qat.ir.waveforms import GaussianWaveform, Pulse
from qat.model.loaders.purr import EchoModelLoader
from qat.purr.compiler.instructions import Acquire as LegacyAcquire
from qat.purr.compiler.instructions import PostProcessing as LegacyPostProcessing
from qat.purr.compiler.instructions import PostProcessType, ProcessAxis
from qat.utils.hardware_model import generate_hw_model
from qat.utils.pydantic import QubitId, ValidatedList, ValidatedSet

model = generate_hw_model(4)
qubits = [qubit for qubit in model.qubits.values()]
qubits_uuid = [qubit.uuid for qubit in qubits]


class TestAcquire:
    def test_initiate(self):
        acquire_channel = qubits[0].acquire_pulse_channel
        inst = Acquire(targets=acquire_channel.uuid)
        assert inst.duration == 1e-6
        assert list(inst.targets)[0] == acquire_channel.uuid  # We only supplied one target.

    def test_filter(self):
        acquire_channel = qubits[0].acquire_pulse_channel
        filter = Pulse(
            targets=acquire_channel.uuid,
            waveform=GaussianWaveform(width=1e-6),
        )
        inst = Acquire(targets=acquire_channel.uuid, duration=1e-6, filter=filter)
        assert inst.filter == filter

    @pytest.mark.parametrize("time", [0, 5e-7, 1.01e-6, 2e-6])
    def test_filter_validation(self, time):
        acquire_channel = qubits[0].acquire_pulse_channel
        filter = Pulse(
            targets=acquire_channel.uuid,
            waveform=GaussianWaveform(width=time),
        )
        with pytest.raises(ValidationError):
            Acquire(targets=acquire_channel.uuid, duration=1e-6, filter=filter)

    @pytest.mark.parametrize("qubit_idx", list(model.qubits.keys()))
    def test_output_variable(self, qubit_idx):
        acquire_channel = qubits[qubit_idx].acquire_pulse_channel
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

    @pytest.mark.parametrize(
        ["pp_type", "args"],
        [
            (PostProcessType.DISCRIMINATE, [0.0]),
            (PostProcessType.DISCRIMINATE, [0.0 + 1j]),
            (PostProcessType.DISCRIMINATE, [np.array([0.0 + 1j])[0]]),
            (PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL, [-0.1, 1]),
            (PostProcessType.MEAN, []),
            (PostProcessType.DOWN_CONVERT, [1e8, 1e-8]),
            (PostProcessType.DOWN_CONVERT, [1e8, 1e-8 + 0j]),
            (PostProcessType.DOWN_CONVERT, [1e8, np.array([1e-8 + 0j])[0]]),
        ],
    )
    def test_serialize_deserialize_roundtrip(self, pp_type, args):
        pp_inst = PostProcessing(
            output_variable="test",
            process_type=pp_type,
            axes=[ProcessAxis.SEQUENCE],
            args=args,
        )
        blob = pp_inst.model_dump()
        new_pp_inst = PostProcessing(**blob)
        assert pp_inst == new_pp_inst

    @pytest.mark.parametrize(
        "args", [[1e8, 1e-8], [1 + 1.0j, 0.5 - 0.5j], [1, 0.5j], [1 + 0.5j, -2.54]]
    )
    def test_numpy_to_list(self, args):
        args = np.asarray(args)
        args = [args[0], args[1]]
        assert isinstance(args[0], np.number)
        assert isinstance(args[1], np.number)

        pp = PostProcessing(
            output_variable="test",
            process_type=PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
            axes=[ProcessAxis.SEQUENCE],
            args=args,
        )
        assert pp.args[0] == args[0]
        assert pp.args[1] == args[1]
        assert not isinstance(pp.args[0], np.number)
        assert not isinstance(pp.args[1], np.number)

    @pytest.mark.parametrize(
        "args", [[1e8, 1e-8], [1 + 1.0j, 0.5 - 0.5j], [1, 0.5j], [1 + 0.5j, -2.54]]
    )
    def test_legacy_numpy_to_list(self, args):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_acquire_channel()

        args = np.asarray(args)
        args = [args[0], args[1]]
        assert isinstance(args[0], np.number)
        assert isinstance(args[1], np.number)

        pp = LegacyPostProcessing(
            LegacyAcquire(chan, 0.0, delay=0.0, output_variable="test"),
            PostProcessType.LINEAR_MAP_COMPLEX_TO_REAL,
            axes=[ProcessAxis.SEQUENCE],
            args=args,
        )
        new_pp = PostProcessing._from_legacy(pp)
        assert new_pp.args[0] == args[0]
        assert new_pp.args[1] == args[1]
        assert not isinstance(new_pp.args[0], np.number)
        assert not isinstance(new_pp.args[1], np.number)


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
        measure_channel = qubit.measure_pulse_channel
        acquire_channel = qubit.acquire_pulse_channel

        mb = MeasureBlock(qubit_targets=qubit_id)
        mb.add(
            Pulse(
                waveform=measure_channel.pulse.waveform_type(
                    **measure_channel.pulse.model_dump()
                ),
                duration=1e-03,
                targets=measure_channel.uuid,
            )
        )
        assert mb.number_of_instructions == 1

        mb.add(Acquire(targets=acquire_channel.uuid, duration=1e-09))
        assert mb.number_of_instructions == 2

    def test_add_instruction_with_delay(self):
        qubit_id = list(model.qubits.keys())[0]
        qubit = model.qubits[qubit_id]
        measure_channel = qubit.measure_pulse_channel
        acquire_channel = qubit.acquire_pulse_channel

        mb = MeasureBlock(qubit_targets=qubit_id)
        mb.add(
            Pulse(
                waveform=measure_channel.pulse.waveform_type(
                    **measure_channel.pulse.model_dump()
                ),
                duration=1e-03,
                targets=measure_channel.uuid,
            )
        )
        assert mb.number_of_instructions == 1

        mb.add(Acquire(targets=acquire_channel.uuid, duration=1e-09, delay=1e-09))
        assert mb.number_of_instructions == 3

    def add_invalid_instruction(self):
        qubit_id = list(model.qubits.keys())[0]
        qubit = model.qubits[qubit_id]
        measure_channel = qubit.measure_pulse_channel

        mb = MeasureBlock(qubit_targets=qubit_id)

        with pytest.raises(TypeError):
            mb.add(
                Pulse(
                    waveform=measure_channel.pulse.waveform_type(
                        **measure_channel.pulse.model_dump()
                    ),
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

        VALID_CUSTOM_INSTR = Union[
            tuple(list(get_args(VALID_MEASURE_INSTR)) + [AdditionalQuantumInstructionBlock])
        ]

        class CustomMeasureBlock(MeasureBlock):
            instructions: ValidatedList[VALID_CUSTOM_INSTR] = Field(
                default_factory=lambda: ValidatedList[VALID_CUSTOM_INSTR]()
            )

        qubit_indices = list(model.qubits.keys())
        custom_mb = CustomMeasureBlock(qubit_targets=qubit_indices)

        max_duration = 0.0
        for qubit_idx, qubit in model.qubits.items():
            qubit_pulse_channels = [pc.uuid for pc in qubit.all_pulse_channels]
            measure_channel = qubit.measure_pulse_channel
            acquire_channel = qubit.acquire_pulse_channel
            # Synchronise all pulse channels within a qubit.
            custom_mb.add(Synchronize(targets=qubit_pulse_channels))
            # Some dummy additional instruction block in the measure block.
            additional_block = AdditionalQuantumInstructionBlock(
                qubit_targets={qubit_idx},
                instructions=[
                    PhaseShift(targets={qubit.drive_pulse_channel.uuid}, phase=np.pi / 2),
                    PhaseReset(targets={qubit.drive_pulse_channel.uuid}),
                ],
            )

            custom_mb.add(additional_block)
            # Standard measure and acquire.
            measure = Pulse(
                waveform=measure_channel.pulse.waveform_type(
                    **measure_channel.pulse.model_dump()
                ),
                duration=1e-03,
                targets=measure_channel.uuid,
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

    def test_acquire_with_no_delay_is_unchanged(self):
        qubit_id, qubit = list(model.qubits.items())[0]
        acquire_chan = qubit.acquire_pulse_channel

        mb = MeasureBlock(qubit_targets=qubit_id)

        mb.add(Acquire(targets=acquire_chan.uuid, duration=10, delay=0.0))

        assert mb.number_of_instructions == 1
        assert isinstance(mb.instructions[0], Acquire)

    def test_acquire_with_delay_is_decomposed(self):
        qubit_id, qubit = list(model.qubits.items())[0]
        acquire_chan = qubit.acquire_pulse_channel

        mb = MeasureBlock(qubit_targets=qubit_id)
        mb.add(Acquire(targets=acquire_chan.uuid, duration=10, delay=10))

        assert mb.number_of_instructions == 2
        assert isinstance(mb.instructions[0], Delay)
        assert isinstance(mb.instructions[1], Acquire)

    def test_acquire_with_delay_two_chans_is_decomposed(self):
        qubit_ids = list(model.qubits.keys())[:2]
        qubits = list(model.qubits.values())[:2]

        mb = MeasureBlock(qubit_targets=qubit_ids)

        for qubit in (0, 1):
            acquire_chan = qubits[qubit].acquire_pulse_channel
            mb.add(
                Acquire(
                    targets=acquire_chan.uuid, duration=10 + qubit, delay=10 + 2 * qubit
                )
            )

        assert mb.number_of_instructions == 4
        assert isinstance(mb.instructions[0], Delay)
        assert mb.instructions[0].duration == 10
        assert isinstance(mb.instructions[1], Acquire)
        assert mb.instructions[1].delay == 0
        assert isinstance(mb.instructions[2], Delay)
        assert mb.instructions[2].duration == 12
        assert isinstance(mb.instructions[3], Acquire)
        assert mb.instructions[3].delay == 0
        assert mb.duration == 23
