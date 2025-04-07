# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

import numpy as np

from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    MeasurePulse,
    SweepValue,
    Variable,
)
from qat.purr.compiler.runtime import get_builder


def resonator_spect(model, qubit_indices=None, num_points=None):
    qubit_indices = qubit_indices if qubit_indices is not None else [0]
    num_points = num_points if num_points is not None else 10
    freq_range = 50e6

    readout_freqs = {
        f"Q{index}": model.get_qubit(index).get_measure_channel().frequency
        for index in qubit_indices
    }
    scan_freqs = {
        f"Q{index}": readout_freqs[f"Q{index}"]
        + np.linspace(-freq_range, freq_range, num_points)
        for index in qubit_indices
    }

    builder = get_builder(model)
    builder.synchronize([model.get_qubit(index) for index in qubit_indices])
    builder.sweep(
        [SweepValue(f"freq{index}", scan_freqs[f"Q{index}"]) for index in qubit_indices]
    )
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        measure_channel = qubit.get_measure_channel()
        acquire_channel = qubit.get_acquire_channel()

        builder.device_assign(measure_channel, "frequency", Variable(f"freq{qubit.index}"))
        builder.device_assign(acquire_channel, "frequency", Variable(f"freq{qubit.index}"))
        builder.measure_mean_signal(qubit, f"Q{qubit.index}")

    builder.repeat(1000, 500e-6)
    return builder


def qubit_spect(model, qubit_indices=None, num_points=None):
    qubit_indices = qubit_indices or [0]
    num_points = num_points if num_points is not None else 10
    freq_range = 50e6
    drive_amp_dbm = -40
    drive_amp_v = np.sqrt(10 ** (((drive_amp_dbm + 12) / 10) - 1))

    drive_freqs = {
        f"Q{index}": model.get_qubit(index).get_drive_channel().frequency
        for index in qubit_indices
    }
    scan_freqs = {
        f"Q{index}": drive_freqs[f"Q{index}"]
        + np.linspace(-freq_range, freq_range, num_points)
        for index in qubit_indices
    }

    builder = get_builder(model)
    builder.synchronize([model.get_qubit(index) for index in qubit_indices])
    for index in qubit_indices:
        # TODO - A skeptical usage of DeviceInjectors on static device updates
        # TODO - Figure out what they mean w/r to scopes and control flow
        builder.device_assign(model.get_qubit(index).get_drive_channel(), "scale", 1)
    builder.sweep(
        [SweepValue(f"freq{index}", scan_freqs[f"Q{index}"]) for index in qubit_indices]
    )
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        drive_channel = qubit.get_drive_channel()
        builder.device_assign(drive_channel, "frequency", Variable(f"freq{index}"))
        builder.pulse(
            drive_channel,
            PulseShapeType.SQUARE,
            width=80e-6,
            amp=drive_amp_v,
            phase=0.0,
            drag=0.0,
            rise=1.0 / 3.0,
        )
        builder.measure_mean_signal(qubit, output_variable=f"Q{index}")

    builder.repeat(1000, 500e-6)
    return builder


def delay_iteration(model, qubit_indices=None, num_points=None, width=None):
    """
    Typically found in T1 measurement where an X gate is applied followed by a variable delay.
    """

    qubit_indices = qubit_indices or [0]
    num_points = num_points or 100
    width = width or 500e-6

    time = np.linspace(0.0, width, num_points)
    var_name = "t"

    builder = get_builder(model)
    builder.synchronize([model.get_qubit(index) for index in qubit_indices])
    builder.sweep(SweepValue(var_name, time))
    for index in qubit_indices:
        qubit = model.get_qubit(index)

        # Dummy adjustment of channel scale to compensate for the Pi/2 pulse amplitude
        qubit.get_drive_channel().scale = 1.0e-8 + 0.0j
        qubit.get_second_state_channel().scale = 1.0e-8 + 0.0j

        builder.X(qubit, np.pi)
        builder.delay(qubit, Variable(var_name))
    builder.synchronize([model.get_qubit(index) for index in qubit_indices])
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        builder.measure_mean_z(qubit, output_variable=f"Q{index}")

    return builder


def scope_acq(model, qubit_indices=None, do_X=False):
    qubit_indices = qubit_indices or [0]

    builder = get_builder(model)
    builder.synchronize([model.get_qubit(index) for index in qubit_indices])
    if do_X:
        for index in qubit_indices:
            qubit = model.get_qubit(index)
            # Dummy adjustment of channel scale to compensate for the Pi/2 pulse amplitude
            qubit.get_drive_channel().scale = 1.0e-8 + 0.0j

            builder.add(model.get_gate_X(qubit, np.pi, qubit.get_drive_channel()))
    builder.synchronize([model.get_qubit(index) for index in qubit_indices])
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        measure_pulse = qubit.pulse_measure
        measure_acquire = qubit.measure_acquire
        width = (
            measure_pulse["width"] if measure_acquire["sync"] else measure_acquire["width"]
        )
        builder.add(MeasurePulse(qubit.get_measure_channel(), **measure_pulse))
        builder.add(
            Acquire(
                qubit.get_acquire_channel(),
                output_variable=f"Q{index}",
                mode=AcquireMode.SCOPE,
                filter=None,
                delay=measure_acquire["delay"],
                time=width,
            )
        )
    return builder


def pulse_width_iteration(model, qubit_indices=None, num_points=None):
    """
    A variation of Rabi where the pulse amplitude is fixed and its width is variable.
    """

    qubit_indices = qubit_indices or [0]
    num_points = num_points or 10

    drive_rate = 5e6
    pulse_shape = PulseShapeType.SQUARE
    time_start = 0
    width = 10e-6
    time = np.linspace(time_start, time_start + width, num_points)
    var_name = "t"

    builder = get_builder(model)
    builder.synchronize([model.get_qubit(index) for index in qubit_indices])
    builder.sweep(SweepValue(var_name, time))
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        drive_channel = qubit.get_drive_channel()
        drive_channel.scale = 1.0e-8 + 0.0j

        builder.pulse(
            drive_channel,
            pulse_shape,
            width=Variable(var_name),
            amp=drive_rate,
            phase=0.0,
            drag=0.0,
            rise=1.0 / 3.0,
        )

    builder.synchronize([model.get_qubit(index) for index in qubit_indices])
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        builder.measure_mean_signal(qubit, f"Q{qubit.index}")
    return builder


def pulse_amplitude_iteration(model, qubit_indices=None, num_points=None):
    """
    A variation of Rabi where the pulse width is fixed and its amplitude is variable.
    """

    qubit_indices = qubit_indices or [0]
    num_points = num_points or 10

    dr_start = 0.0
    dr_end = 5e6
    amplitude = np.linspace(dr_start, dr_end, num_points)
    width = 10e-6
    pulse_shape = PulseShapeType.SQUARE

    builder = get_builder(model)
    builder.synchronize([model.get_qubit(index) for index in qubit_indices])

    builder.sweep(SweepValue("amp", amplitude))
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        drive_channel = qubit.get_drive_channel()
        drive_channel.scale = 1.0e-8 + 0.0j

        builder.pulse(
            drive_channel,
            pulse_shape,
            width=width,
            amp=Variable("amp"),
            phase=0.0,
            drag=0.0,
            rise=1.0 / 3.0,
        )

    builder.synchronize([model.get_qubit(index) for index in qubit_indices])
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        builder.measure_mean_signal(qubit, f"Q{qubit.index}")

    builder.synchronize([model.get_qubit(index) for index in qubit_indices])
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        builder.measure_mean_signal(qubit, f"Q{qubit.index}")

    return builder


def time_and_phase_iteration(model, qubit_indices=None, num_points=None):
    """
    Typical in Ramsey measurement where both time and phase are variable.
    """

    qubit_indices = qubit_indices or [0]
    num_points = num_points or 10

    detuning = 5e6
    width = 10e-6
    time = np.linspace(0, width, num_points)
    phase = 2.0 * np.pi * detuning * time
    time_var_name = "t"
    phase_var_name = "p"

    builder = get_builder(model)
    builder.synchronize([model.get_qubit(index) for index in qubit_indices])

    builder.sweep([SweepValue(time_var_name, time), SweepValue(phase_var_name, phase)])
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        drive_channel = qubit.get_drive_channel()
        drive_channel.scale = 1.0e-8 + 0.0j

        builder.X(qubit, np.pi / 2.0)
        builder.delay(qubit, Variable(time_var_name))
        builder.phase_shift(qubit, Variable(phase_var_name))
        builder.Y(qubit, -np.pi / 2.0)

    builder.synchronize([model.get_qubit(index) for index in qubit_indices])
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        builder.measure_mean_z(qubit, output_variable=f"Q{index}")
    return builder
