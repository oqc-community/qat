# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd


from qat.model.loaders.purr.echo import EchoModelLoader
from qat.pipelines.purr.sweep.passes import FrequencyAssignSanitisation
from qat.purr.compiler.instructions import DeviceUpdate, FrequencySet, Variable


class TestFrequencyAssignSanitisation:
    model = EchoModelLoader(qubit_count=4).load()

    def test_device_assign_with_immediate_becomes_frequency_set(self):
        builder = self.model.create_builder()
        qubit = self.model.get_qubit(0)
        drive_channel = qubit.get_drive_channel()
        builder.device_assign(drive_channel, "frequency", 5e9)
        builder.X(qubit)
        builder.measure_single_shot_z(qubit, output_variable="result")

        builder = FrequencyAssignSanitisation().run(builder)

        device_assigns = [
            inst for inst in builder._instructions if isinstance(inst, DeviceUpdate)
        ]
        assert len(device_assigns) == 0
        frequency_instructions = [
            inst for inst in builder._instructions if isinstance(inst, FrequencySet)
        ]
        # we can have frequency set enter in other ways
        assert len(frequency_instructions) > 0
        freq_inst = frequency_instructions[0]
        assert freq_inst.channel == drive_channel
        assert freq_inst.frequency == 5e9

    def test_device_assign_with_variable_becomes_frequency_set(self):
        builder = self.model.create_builder()
        qubit = self.model.get_qubit(0)
        drive_channel = qubit.get_drive_channel()
        builder.device_assign(drive_channel, "frequency", Variable("freq"))
        builder.X(qubit)
        builder.measure_single_shot_z(qubit, output_variable="result")

        builder = FrequencyAssignSanitisation().run(builder)

        device_assigns = [
            inst for inst in builder._instructions if isinstance(inst, DeviceUpdate)
        ]
        assert len(device_assigns) == 0
        frequency_instructions = [
            inst for inst in builder._instructions if isinstance(inst, FrequencySet)
        ]
        # we can have frequency set enter in other ways
        assert len(frequency_instructions) > 0
        freq_inst = frequency_instructions[0]
        assert freq_inst.channel == drive_channel
        assert isinstance(freq_inst.frequency, Variable)
        assert freq_inst.frequency.name == "freq"
