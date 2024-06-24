import numpy as np

from qat.purr.backends.live import LiveHardwareModel
from qat.purr.compiler.instructions import Variable, SweepValue
from qat.purr.compiler.runtime import execute_instructions, get_builder
from qat.purr.utils.logger import get_default_logger
from qat.purr.backends.qblox.rnd.live import QbloxLiveEngine
from src.tests.qblox.utils import setup_qblox_hardware_model

log = get_default_logger()

cluster_kit = None


def test_resonator_spect():
    model = LiveHardwareModel()
    setup_qblox_hardware_model(model, cluster_kit)
    engine = QbloxLiveEngine(model)
    qubit = model.get_qubit(0)
    measure_channel = qubit.get_measure_channel()
    acquire_channel = qubit.get_acquire_channel()
    assert acquire_channel == measure_channel

    num_points = 10
    freq_range = 20e6
    center_freq = qubit.get_acquire_channel().frequency
    freqs = center_freq + np.linspace(-freq_range, freq_range, num_points)
    builder = (
        get_builder(model)
        .sweep(SweepValue(f'freq{0}', freqs))
        .device_assign(measure_channel, "frequency", Variable(f'freq{0}'))
        .device_assign(acquire_channel, "frequency", Variable(f'freq{0}'))
        .measure_mean_signal(qubit, output_variable=f'Q{0}')
        .repeat(400, 500e-6)
    )
    results, _ = execute_instructions(engine, builder.instructions)
    assert results is not None
