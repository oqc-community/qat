import numpy as np

from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.instructions import SweepValue, Variable
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


def qubit_spect(model, qubit_indices=None):
    qubit_indices = qubit_indices or [0]
    num_points = 10
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
        # TODO - Provide better processing of static DeviceUpdates
        model.get_qubit(index).get_drive_channel().scale = 1
        # builder.device_assign(model.get_qubit(index).get_drive_channel(), "scale", 1)
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
