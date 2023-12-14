# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from qat.purr.backends.echo import get_default_echo_hardware, EchoEngine
from qat.purr.compiler.devices import PulseShapeType, ChannelType, FreqShiftPulseChannel
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.runtime import get_builder

import numpy as np

def get_physical_buffers(hardware, package):
    position_map = hardware.create_duration_timeline(package)
    pulse_channel_buffers = hardware.build_pulse_channel_buffers(position_map, True)
    return hardware.build_physical_channel_buffers(pulse_channel_buffers)


def setup_qubit_buffers(freq_shift_1=False):
    hardware = get_default_echo_hardware(2)
    builder = get_builder(hardware)
    engine = EchoEngine(hardware)

    qubit1 = hardware.get_qubit(0)
    qubit2 = hardware.get_qubit(1)

    drive_channel_1 = qubit1.get_drive_channel()
    drive_channel_2 = qubit2.get_drive_channel()

    physical_channel_1 = qubit1.physical_channel
    physical_channel_2 = qubit2.physical_channel

    if freq_shift_1:
        freq_channel = FreqShiftPulseChannel(id_='freq_shift',
                                             physical_channel=physical_channel_1,
                                             amp=1.0,
                                             scale=1.0,
                                             frequency=8.5e9)
        qubit1.add_pulse_channel(pulse_channel=freq_channel, channel_type=ChannelType.freq_shift)
        hardware.add_pulse_channel(freq_channel,)

    builder.pulse(
        drive_channel_1, PulseShapeType.SQUARE, width=1e-6, amp=1, ignore_channel_scale=True
    ).pulse(
        drive_channel_2, PulseShapeType.SQUARE, width=1e-6, amp=1, ignore_channel_scale=True
    )

    qat_file = InstructionEmitter().emit(builder.instructions, builder.model)

    buffers = get_physical_buffers(engine, qat_file)
    return buffers[physical_channel_1.id], buffers[physical_channel_2.id]

def test_no_freq_shift():
    qubit1_buffer, qubit2_buffer = setup_qubit_buffers()

    assert np.isclose(qubit1_buffer, 1+0j).all()
    assert np.isclose(qubit2_buffer, 1+0j).all()

def test_freq_shift():
    qubit1_buffer, qubit2_buffer = setup_qubit_buffers(freq_shift_1=True)

    assert np.isclose([abs(val) for val in qubit1_buffer], 2).all()
    assert np.isclose([abs(val) for val in qubit2_buffer], 1).all()
