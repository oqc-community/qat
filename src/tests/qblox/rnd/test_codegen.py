import numpy as np
import pytest

from qat.purr.backends.live import LiveHardwareModel
from qat.purr.backends.qblox.rnd.codegen import QbloxEmitter, QatEmitter
from qat.purr.backends.qblox.rnd.instructions import ReducedSweep, ReducedSweepValue, HybridInstructionBuilder
from qat.purr.backends.qblox.rnd.live import QbloxLiveEngine
from src.tests.qblox.utils import setup_qblox_hardware_model
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.instructions import Sweep, SweepValue, Variable
from qat.purr.compiler.runtime import get_builder, execute_instructions

cluster_kit = None

def test_linear_sweep():
    model = LiveHardwareModel()
    setup_qblox_hardware_model(model)
    drive_channel = model.get_qubit(0).get_drive_channel()
    var_value = np.linspace(0, 11, 9)
    sweep = Sweep(SweepValue(f'var_name', var_value))
    linear_sweep = ReducedSweep(SweepValue(f'var_name', var_value))
    assert linear_sweep.name == "var_name"
    assert linear_sweep.start == 0
    assert linear_sweep.step == 1
    assert linear_sweep.count == 10


def test_fast_resonator_spect():
    model = LiveHardwareModel()
    setup_qblox_hardware_model(model, cluster_kit)
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
    qat_file = InstructionEmitter().emit(builder.instructions, model)
    qblox_packages = QbloxEmitter(qat_file).emit(qat_file)
    assert qblox_packages is not None


@pytest.mark.parametrize("model", [None], indirect=True)
def test_qat_graph(model):
    qubit = model.get_qubit(0)
    measure_channel = qubit.get_measure_channel()
    acquire_channel = qubit.get_acquire_channel()
    assert acquire_channel == measure_channel

    num_points = 10
    freq_range = 20e6
    center_freq = qubit.get_acquire_channel().frequency
    freqs = center_freq + np.linspace(-freq_range, freq_range, num_points)
    builder = (
        HybridInstructionBuilder(model)
        #.sweep(SweepValue(f'freq{0}', freqs))
        .repeat(400, 500e-6)
        .device_assign(measure_channel, "frequency", Variable(f'freq{0}'))
        .device_assign(acquire_channel, "frequency", Variable(f'freq{0}'))
        .measure_mean_signal(qubit, output_variable=f'Q{0}')
        .end_repeat()
    )
    qat_graph = QatEmitter().emit(builder.instructions)
    qblox_packages = QbloxEmitter().emit(qat_graph)
    assert qblox_packages is not None
