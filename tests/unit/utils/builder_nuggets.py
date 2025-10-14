# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import numbers

import numpy as np

from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    MeasurePulse,
    PostProcessType,
    ProcessAxis,
    Pulse,
    SweepValue,
    Variable,
)
from qat.purr.compiler.runtime import get_builder


def direct_x(qubit, channel=None, theta=None, amp=None, drag=None, width=None, rise=None):
    pulse = Pulse(
        channel or qubit.get_default_pulse_channel(), **dict(qubit.pulse_hw_x_pi_2)
    )
    if theta is not None:
        if isinstance(theta, numbers.Number):
            pulse.amp *= theta / (0.5 * np.pi)
        else:
            pulse.amp = theta
    if amp is not None:
        pulse.amp = amp
    if drag is not None:
        pulse.drag = drag
    if width is not None:
        pulse.width = width
    if rise is not None:
        pulse.rise = rise
    return pulse


def empty(model, qubit_indices=None):
    """
    Not literally empty, just contains stalling instructions.
    """

    qubit_indices = qubit_indices if qubit_indices is not None else [0]
    builder = get_builder(model)

    for index in qubit_indices:
        qubit = model.get_qubit(index)
        drive_channel = qubit.get_drive_channel()
        measure_channel = qubit.get_measure_channel()
        acquire_channel = qubit.get_acquire_channel()

        builder.id(drive_channel)
        builder.id(measure_channel)
        builder.id(acquire_channel)

    return builder


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

    builder.repeat(1000, passive_reset_time=500e-6)
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

    builder.repeat(1000, passive_reset_time=500e-6)
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
        drive_channel = qubit.get_drive_channel()
        second_state_channel = qubit.get_second_state_channel()

        # Dummy adjustment of channel scale to compensate for the Pi/2 pulse amplitude
        drive_channel.scale = 1.0e-8 + 0.0j
        second_state_channel.scale = 1.0e-8 + 0.0j

        builder.X(qubit, np.pi)
        builder.delay(qubit, Variable(var_name))
    builder.synchronize([model.get_qubit(index) for index in qubit_indices])
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        builder.measure_mean_z(qubit, output_variable=f"Q{index}")

    return builder


def measure_acquire(model, qubit_indices=None, do_X=False, acq_mode=AcquireMode.SCOPE):
    qubit_indices = qubit_indices or [0]

    builder = get_builder(model)
    builder.synchronize([model.get_qubit(index) for index in qubit_indices])
    if do_X:
        for index in qubit_indices:
            qubit = model.get_qubit(index)
            drive_channel = qubit.get_drive_channel()
            second_state_channel = qubit.get_second_state_channel()

            # Dummy adjustment of channel scale to compensate for the Pi/2 pulse amplitude
            drive_channel.scale = 1.0e-8 + 0.0j
            second_state_channel.scale = 1.0e-8 + 0.0j

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
                mode=acq_mode,
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
        second_state_channel = qubit.get_second_state_channel()

        # Dummy adjustment of channel scale to compensate for the Pi/2 pulse amplitude
        drive_channel.scale = 1.0e-8 + 0.0j
        second_state_channel.scale = 1.0e-8 + 0.0j

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
        second_state_channel = qubit.get_second_state_channel()

        # Dummy adjustment of channel scale to compensate for the Pi/2 pulse amplitude
        drive_channel.scale = 1.0e-8 + 0.0j
        second_state_channel.scale = 1.0e-8 + 0.0j
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
        second_state_channel = qubit.get_second_state_channel()

        # Dummy adjustment of channel scale to compensate for the Pi/2 pulse amplitude
        drive_channel.scale = 1.0e-8 + 0.0j
        second_state_channel.scale = 1.0e-8 + 0.0j

        builder.X(qubit, np.pi / 2.0)
        builder.delay(qubit, Variable(time_var_name))
        builder.phase_shift(qubit, Variable(phase_var_name))
        builder.Y(qubit, -np.pi / 2.0)

    builder.synchronize([model.get_qubit(index) for index in qubit_indices])
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        builder.measure_mean_z(qubit, output_variable=f"Q{index}")
    return builder


def multi_readout(model, qubit_indices=None, do_X=False):
    qubit_indices = qubit_indices or [0]

    builder = get_builder(model)
    builder.synchronize([model.get_qubit(index) for index in qubit_indices])

    for index in qubit_indices:
        qubit = model.get_qubit(index)
        drive_channel = qubit.get_drive_channel()
        second_state_channel = qubit.get_second_state_channel()

        # Dummy adjustment of channel scale to compensate for the Pi/2 pulse amplitude
        drive_channel.scale = 1.0e-8 + 0.0j
        second_state_channel.scale = 1.0e-8 + 0.0j

        builder.measure_single_shot_binned(qubit, output_variable=f"0_Q{index}")
        builder.delay(drive_channel, 5e-6)
        if do_X:
            builder.add(direct_x(qubit, theta=np.pi / 2))
            builder.add(direct_x(qubit, theta=np.pi / 2))
        else:
            builder.delay(drive_channel, 2 * qubit.pulse_hw_x_pi_2["width"])
        builder.measure_single_shot_signal(qubit, output_variable=f"1_Q{index}")

    return builder


def xpi2amp(model, qubit_indices=None, num_points=None):
    qubit_indices = qubit_indices or [0]
    num_points = num_points or 10

    amp_range = 0.35
    N = 3

    builder = get_builder(model)
    builder.synchronize([model.get_qubit(index) for index in qubit_indices])

    rel_amps = np.linspace((1 - amp_range / N), (1 + amp_range / N), num_points)
    start_shift = np.linspace(0.0, np.pi, 2)
    amps_dict = {
        f"Q{q}": rel_amps * model.get_qubit(q).pulse_hw_x_pi_2["amp"] for q in qubit_indices
    }

    builder.sweep(SweepValue("p", start_shift))
    builder.sweep([SweepValue(f"amp{q}", amps_dict[f"Q{q}"]) for q in qubit_indices])
    for q in qubit_indices:
        qubit = model.get_qubit(q)
        drive_channel = qubit.get_drive_channel()
        second_state_channel = qubit.get_second_state_channel()

        # Dummy adjustment of channel scale to compensate for the Pi/2 pulse amplitude
        drive_channel.scale = 1.0e-8 + 0.0j
        second_state_channel.scale = 1.0e-8 + 0.0j

        builder.add([direct_x(qubit, amp=Variable(f"amp{q}")) for _ in range(4 * N)])
        builder.phase_shift(qubit, Variable("p"))
        builder.add(direct_x(qubit, amp=Variable(f"amp{q}")))

    builder.synchronize([model.get_qubit(index) for index in qubit_indices])
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        builder.measure_mean_z(qubit, output_variable=f"Q{index}")

    return builder


def readout_freq(model, qubit_indices=None, num_points=None):
    qubit_indices = qubit_indices or [0]
    num_points = num_points or 10

    old_freqs = {
        f"Q{q}": model.get_qubit(q).get_measure_channel().frequency for q in qubit_indices
    }
    freq_delta = np.linspace(-2e6, 2e6, num_points)

    builder = get_builder(model)
    builder.synchronize([model.get_qubit(index) for index in qubit_indices])

    builder.sweep(
        [
            SweepValue(f"readout_freq_Q{q}", old_freqs[f"Q{q}"] + freq_delta)
            for q in qubit_indices
        ]
    )
    builder.sweep(
        [
            SweepValue(f"amp_Q{q}", [0, model.get_qubit(q).pulse_hw_x_pi_2["amp"]])
            for q in qubit_indices
        ]
    )
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        drive_channel = qubit.get_drive_channel()
        second_state_channel = qubit.get_second_state_channel()
        measure_channel = qubit.get_measure_channel()
        acquire_channel = qubit.get_acquire_channel()

        # Dummy adjustment of channel scale to compensate for the Pi/2 pulse amplitude
        drive_channel.scale = 1.0e-8 + 0.0j
        second_state_channel.scale = 1.0e-8 + 0.0j

        drive_pulse_dict = qubit.pulse_hw_x_pi_2.copy()
        drive_pulse_dict.pop("amp")
        builder.device_assign(
            measure_channel, "frequency", Variable(f"readout_freq_Q{index}")
        )
        builder.device_assign(
            acquire_channel, "frequency", Variable(f"readout_freq_Q{index}")
        )
        builder.pulse(drive_channel, amp=Variable(f"amp_Q{index}"), **drive_pulse_dict)
        builder.pulse(drive_channel, amp=Variable(f"amp_Q{index}"), **drive_pulse_dict)
        builder.measure_single_shot_signal(qubit, output_variable=f"Q{index}")

    return builder


def zmap(model, qubit_indices=None, do_X=False):
    qubit_indices = qubit_indices or [0]

    x12 = False

    builder = get_builder(model)
    builder.synchronize([model.get_qubit(index) for index in qubit_indices])

    if x12:
        for index in qubit_indices:
            builder.X(model.get_qubit(index))
    builder.synchronize([model.get_qubit(q) for q in qubit_indices])
    if do_X:
        for index in qubit_indices:
            qubit = model.get_qubit(index)
            qubit_drive_channel = (
                qubit.get_second_state_channel() if x12 else qubit.get_drive_channel()
            )
            builder.add(model.get_gate_X(qubit, np.pi, qubit_drive_channel))
    builder.synchronize([model.get_qubit(q) for q in qubit_indices])
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        drive_channel = qubit.get_drive_channel()
        second_state_channel = qubit.get_second_state_channel()
        measure_channel = qubit.get_measure_channel()
        acquire_channel = qubit.get_acquire_channel()

        # Dummy adjustment of channel scale to compensate for the Pi/2 pulse amplitude
        drive_channel.scale = 1.0e-8 + 0.0j
        second_state_channel.scale = 1.0e-8 + 0.0j

        measure_pulse_dict = qubit.pulse_measure.copy()
        measure_pulse_dict.pop("width")
        measure_pulse_dict.pop("amp")
        measure_amp = qubit.pulse_measure["amp"]
        measure_width = qubit.pulse_measure["width"]
        builder.device_assign(
            measure_channel,
            "frequency",
            measure_channel.frequency,
        )
        width = (
            measure_width
            if qubit.measure_acquire["sync"]
            else qubit.measure_acquire["width"]
        )
        builder.add(
            MeasurePulse(
                qubit.get_measure_channel(),
                width=measure_width,
                amp=measure_amp,
                **measure_pulse_dict,
            )
        )
        acquire = Acquire(
            acquire_channel,
            time=width,
            mode=AcquireMode.INTEGRATOR,
            filter=None,
            output_variable=f"Q{qubit.index}",
            delay=qubit.measure_acquire["delay"],
        )
        builder.add(acquire)
        builder.synchronize(qubit)
        builder.post_processing(
            acquire, PostProcessType.DOWN_CONVERT, ProcessAxis.TIME, qubit
        )
        builder.post_processing(acquire, PostProcessType.MEAN, ProcessAxis.TIME, qubit)
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        builder.synchronize([qubit.get_all_channels()])
    return builder


def hidden_mode(model, qubit_indices=None, mode_indices=None, num_points=None):
    num_points = num_points or 10
    qubit_indices = qubit_indices or [0]
    mode_indices = mode_indices or [q + 1 for q in qubit_indices]

    detuning = 5e6
    width = 8e-6
    time = np.linspace(0, width, num_points)
    phase = 2.0 * np.pi * detuning * time
    dephasing_freqs = np.linspace(10.25e9 - 25e6, 10.25e9 + 25e6, 3)
    dephase_amp = 0.5
    rise = 1.0 / 3.0

    all_indices = qubit_indices + mode_indices
    for index in all_indices:
        qubit = model.get_qubit(index)
        qubit.get_drive_channel().scale = 1.0e-8 + 0.0j

    builder = get_builder(model)
    builder.synchronize([model.get_qubit(index) for index in all_indices])

    builder.sweep([SweepValue("f", dephasing_freqs)])
    builder.sweep([SweepValue("t", time), SweepValue("p", phase)])
    for index, mode_index in zip(qubit_indices, mode_indices):
        qubit = model.get_qubit(index)
        mode_channel = model.get_qubit(mode_indices[index]).get_measure_channel()
        builder.device_assign(mode_channel, "frequency", Variable("f"))
        builder.X(qubit, np.pi / 2.0)
        builder.synchronize([qubit, mode_channel])
        builder.pulse(
            mode_channel,
            PulseShapeType.GAUSSIAN,
            width=Variable("t"),
            amp=dephase_amp,
            rise=rise,
        )
        builder.synchronize([qubit, mode_channel])
        builder.phase_shift(qubit, Variable("p"))
        builder.Y(qubit, -np.pi / 2.0)

    builder.synchronize([model.get_qubit(index) for index in qubit_indices])
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        builder.measure_mean_z(qubit, output_variable=f"Q{index}")

    return builder


def discrimination(model, qubit_indices=None):
    qubit_indices = qubit_indices or [0]

    builder = get_builder(model)
    builder.synchronize([model.get_qubit(index) for index in qubit_indices])

    for index in qubit_indices:
        qubit = model.get_qubit(index)
        builder.measure_single_shot_binned(qubit)
    return builder
