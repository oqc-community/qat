# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

import numpy as np

from qat.purr.backends.echo import QbloxEchoHardwareModel, apply_setup_to_hardware
from qat.purr.backends.qblox import SequenceEmitter
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.runtime import get_builder
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


def test_play_guassian():
    width = 100e-9
    rise = 1./5.
    model = QbloxEchoHardwareModel()
    apply_setup_to_hardware(model)
    builder = (
        get_builder(model)
        .pulse(model.get_qubit(0).get_drive_channel(), PulseShapeType.GAUSSIAN, width=width, rise=rise)
    )
    qat_file = InstructionEmitter().emit(builder.instructions, model)
    seq_files = SequenceEmitter().emit(qat_file)
    for _, seq_file in seq_files.items():
        assert seq_file is not None


def test_play_square():
    width = 100e-9
    amp = 1
    model = QbloxEchoHardwareModel()
    apply_setup_to_hardware(model)
    builder = (
        get_builder(model)
        .pulse(model.get_qubit(0).get_drive_channel(), PulseShapeType.SQUARE, width=width, amp=amp)
    )
    qat_file = InstructionEmitter().emit(builder.instructions, model)
    seq_files = SequenceEmitter().emit(qat_file)
    for _, seq_file in seq_files.items():
        assert seq_file is not None


def test_play_multiple_instructions():
    amp = 1
    rise = 1./3.
    model = QbloxEchoHardwareModel()
    apply_setup_to_hardware(model)
    builder = (
        get_builder(model)
        .pulse(model.get_qubit(0).get_drive_channel(), PulseShapeType.SQUARE, width=100e-9, amp=amp)
        .pulse(model.get_qubit(0).get_drive_channel(), PulseShapeType.GAUSSIAN, width=150e-9, rise=rise)
    )
    qat_file = InstructionEmitter().emit(builder.instructions, model)
    seq_files = SequenceEmitter().emit(qat_file)
    for _, seq_file in seq_files.items():
        assert seq_file is not None


def test_phase_and_frequency_shift():
    amp = 1
    rise = 1./3.
    phase = 0.72
    frequency = 500
    model = QbloxEchoHardwareModel()
    apply_setup_to_hardware(model)
    builder = (
        get_builder(model)
        .pulse(model.get_qubit(0).get_drive_channel(), PulseShapeType.SQUARE, width=100e-9, amp=amp)
        .pulse(model.get_qubit(0).get_drive_channel(), PulseShapeType.GAUSSIAN, width=150e-9, rise=rise)
        .phase_shift(model.get_qubit(0).get_drive_channel(), phase)
        .frequency_shift(model.get_qubit(0).get_drive_channel(), frequency)
    )
    qat_file = InstructionEmitter().emit(builder.instructions, model)
    seq_files = SequenceEmitter().emit(qat_file)
    for pulse_channel, seq_file in seq_files.items():
        assert seq_file is not None
        assert f"set_ph_delta {int(np.rad2deg(phase) / 360 * 1e9)}" in seq_file.instructions
        assert f"set_freq {int((pulse_channel.frequency + frequency)*1e-6*4)}" in seq_file.instructions
