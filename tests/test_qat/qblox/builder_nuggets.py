import numpy as np

from qat.purr.compiler.instructions import SweepValue, Variable
from qat.purr.compiler.runtime import get_builder


def resonator_spect(model):
    qubit = model.get_qubit(0)
    measure_channel = qubit.get_measure_channel()
    acquire_channel = qubit.get_acquire_channel()
    assert acquire_channel == measure_channel

    num_points = 10
    freq_range = 50e6
    center_freq = qubit.get_acquire_channel().frequency
    freqs = center_freq + np.linspace(-freq_range, freq_range, num_points)
    var_name = f"freq{qubit.index}"
    output_variable = f"Q{qubit.index}"

    builder = (
        get_builder(model)
        .sweep(SweepValue(var_name, freqs))
        .device_assign(measure_channel, "frequency", Variable(var_name))
        .device_assign(acquire_channel, "frequency", Variable(var_name))
        .measure_mean_signal(qubit, output_variable)
        .repeat(1000, 500e-6)
    )
    return builder
