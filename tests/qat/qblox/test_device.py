import numpy as np
import pytest

from qat.purr.backends.qblox.codegen import QbloxEmitter
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.instructions import SweepValue, Variable
from qat.purr.compiler.runtime import execute_instructions, get_builder
from qat.purr.utils.logger import get_default_logger
from tests.qat.qblox.utils import ClusterInfo

log = get_default_logger()


@pytest.mark.parametrize("model", [ClusterInfo()], indirect=True)
class TestDummyQbloxControlHardware:
    def test_instruction_execution(self, model):
        amp = 1
        rise = 1.0 / 3.0
        phase = 0.72
        frequency = 500

        drive_channel = model.get_qubit(0).get_drive_channel()
        builder = (
            get_builder(model)
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .pulse(drive_channel, PulseShapeType.GAUSSIAN, width=150e-9, rise=rise)
            .phase_shift(drive_channel, phase)
            .frequency_shift(drive_channel, frequency)
        )

        engine = model.create_engine()
        results, _ = execute_instructions(engine, builder)
        assert results is not None

    def test_resource_allocation(self, model):
        qubit = model.get_qubit(0)
        drive_channel = qubit.get_drive_channel()

        num_points = 10
        freq_range = 10e6
        freqs = drive_channel.frequency + np.linspace(-freq_range, freq_range, num_points)
        builder = (
            get_builder(model)
            .synchronize(qubit)
            .device_assign(drive_channel, "scale", 1)
            .sweep(SweepValue(f"freq{0}", freqs))
            .device_assign(drive_channel, "frequency", Variable(f"freq{0}"))
            .pulse(
                drive_channel,
                PulseShapeType.SQUARE,
                width=5e-6,
                amp=1,
                phase=0.0,
                drag=0.0,
                rise=1.0 / 3.0,
            )
            .synchronize(qubit)
            .measure_mean_signal(qubit, output_variable=f"Q{0}")
        )

        qat_file = InstructionEmitter().emit(builder.instructions, model)
        qblox_packages = QbloxEmitter().emit(qat_file)
        model.control_hardware.set_data(qblox_packages)
        assert any(model.control_hardware._resources)
