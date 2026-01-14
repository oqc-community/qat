# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd


from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import PulseShapeType, SweepValue, Variable


def sweep_pulse_widths(model: QuantumHardwareModel, qubit: int, times: list[float]):
    """Factory for creating a builder that sweeps over a pulse width for a given qubit,
    and a given set of times."""

    builder = model.create_builder()
    qubit = model.get_qubit(qubit)
    amp = qubit.pulse_hw_x_pi_2["amp"]
    drive_channel = qubit.get_drive_channel()

    builder.sweep(SweepValue("t", times))
    builder.pulse(
        drive_channel,
        shape=PulseShapeType.GAUSSIAN,
        width=Variable("t"),
        amp=amp,
        phase=0.0,
        drag=0.0,
        rise=1.0 / 3.0,
    )
    builder.measure_single_shot_z(qubit, output_variable="Q0")
    return builder


def sweep_pulse_scales(model: QuantumHardwareModel, qubit: int, scales: list[float]):
    """Factory for creating a builder that sweeps over a pulse amplitude scale for a given
    qubit, and a given set of scales."""

    builder = model.create_builder()
    qubit = model.get_qubit(qubit)
    width = qubit.pulse_hw_x_pi_2["width"]
    amp = qubit.pulse_hw_x_pi_2["amp"]
    drive_channel = qubit.get_drive_channel()

    builder.sweep(SweepValue("s", scales))
    builder.device_assign(drive_channel, "scale", Variable("s"))
    builder.pulse(
        drive_channel,
        shape=PulseShapeType.GAUSSIAN,
        width=width,
        amp=amp,
        phase=0.0,
        drag=0.0,
        rise=1.0 / 3.0,
    )
    builder.measure_single_shot_z(qubit, output_variable="Q0")
    return builder


def sweep_pulse_widths_and_amps(
    model: QuantumHardwareModel, qubit: int, times: list[float], amps: list[float]
):
    """Factory for creating a builder that sweeps over both a pulse width and amplitude
    for a given qubit, and a given set of times and amplitudes."""

    builder = model.create_builder()
    qubit = model.get_qubit(qubit)
    drive_channel = qubit.get_drive_channel()

    builder.sweep(SweepValue("t", times))
    builder.sweep(SweepValue("a", amps))
    builder.pulse(
        drive_channel,
        shape=PulseShapeType.GAUSSIAN,
        width=Variable("t"),
        amp=Variable("a"),
        phase=0.0,
        drag=0.0,
        rise=1.0 / 3.0,
    )
    builder.measure_single_shot_z(qubit, output_variable="Q0")
    return builder


def sweep_sequential_pulse_widths(
    model: QuantumHardwareModel, qubit1: int, qubit2: int, times: list[float]
):
    """A factory for creating a builder that drives two qubits. The first qubit is driven
    for some time we're sweeping over, which is immediately followed by a constant pulse on
    the second.

    Obviously the same behaviour can be achieved with a Synchronize instruction inbetween
    the pulses, but the point it to test dynamic timing on multiple instructions.
    """

    builder = model.create_builder()
    qubit1 = model.get_qubit(qubit1)
    qubit2 = model.get_qubit(qubit2)
    drive_channel1 = qubit1.get_drive_channel()
    drive_channel2 = qubit2.get_drive_channel()

    builder.sweep(SweepValue("t", times))
    builder.pulse(
        drive_channel1,
        shape=PulseShapeType.GAUSSIAN,
        width=Variable("t"),
        amp=qubit1.pulse_hw_x_pi_2["amp"],
        phase=0.0,
        drag=0.0,
        rise=1.0 / 3.0,
    )
    builder.delay(drive_channel2, time=Variable("t"))
    builder.pulse(
        drive_channel2,
        shape=PulseShapeType.GAUSSIAN,
        width=qubit2.pulse_hw_x_pi_2["width"],
        amp=qubit2.pulse_hw_x_pi_2["amp"],
        phase=0.0,
        drag=0.0,
        rise=1.0 / 3.0,
    )
    builder.measure_single_shot_z(qubit1, output_variable="Q0")
    builder.measure_single_shot_z(qubit2, output_variable="Q1")
    return builder


def sweep_zipped_parameters(
    model: QuantumHardwareModel, qubit: int, times1: list[float], times2: list[float]
):
    """Sweeps over pairs of parameters, treating them as a zip."""

    builder = model.create_builder()
    qubit = model.get_qubit(qubit)
    drive_channel = qubit.get_drive_channel()

    builder.sweep([SweepValue("t1", times1), SweepValue("t2", times2)])
    builder.pulse(
        drive_channel,
        shape=PulseShapeType.SQUARE,
        width=Variable("t1"),
        amp=1.0,
    )
    builder.delay(drive_channel, time=Variable("t2"))
    builder.measure_single_shot_z(qubit, output_variable="Q0")
    return builder
