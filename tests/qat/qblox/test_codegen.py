import pytest

from qat.purr.backends.qblox.codegen import (
    QbloxEmitter,
    get_nco_phase_arguments,
    get_nco_set_frequency_arguments,
)
from qat.purr.backends.qblox.device import QbloxPhysicalBaseband, QbloxPhysicalChannel
from qat.purr.backends.qblox.ir import Sequence
from qat.purr.compiler.devices import PulseChannel, PulseShapeType
from qat.purr.compiler.emitter import InstructionEmitter
from qat.purr.compiler.instructions import Acquire, MeasurePulse
from qat.purr.compiler.runtime import get_builder
from qat.purr.utils.logger import get_default_logger
from tests.qat.qblox.utils import ClusterInfo

log = get_default_logger()


@pytest.mark.parametrize("model", [ClusterInfo()], indirect=True)
class TestQbloxEmitter:
    def test_play_guassian(self, model):
        width = 100e-9
        rise = 1.0 / 5.0
        drive_channel = model.get_qubit(0).get_drive_channel()
        builder = get_builder(model).pulse(
            drive_channel, PulseShapeType.GAUSSIAN, width=width, rise=rise
        )
        qat_file = InstructionEmitter().emit(builder.instructions, model)
        qblox_packages = QbloxEmitter().emit(qat_file)
        for package in qblox_packages:
            assert package is not None
            assert isinstance(package.target, PulseChannel)
            assert isinstance(package.sequence, Sequence)
            assert "GAUSSIAN_0_I" in package.sequence.waveforms
            assert "GAUSSIAN_0_Q" in package.sequence.waveforms
            assert "play 0,1,4" in package.sequence.program

    def test_play_square(self, model):
        width = 100e-9
        amp = 1

        drive_channel = model.get_qubit(0).get_drive_channel()
        builder = get_builder(model).pulse(
            drive_channel, PulseShapeType.SQUARE, width=width, amp=amp
        )
        qat_file = InstructionEmitter().emit(builder.instructions, model)
        qblox_packages = QbloxEmitter().emit(qat_file)
        for package in qblox_packages:
            assert package is not None
            assert isinstance(package.target, PulseChannel)
            assert isinstance(package.sequence, Sequence)
            assert "SQUARE_0_I" in package.sequence.waveforms
            assert "SQUARE_0_Q" in package.sequence.waveforms
            assert "play 0,1,4" in package.sequence.program

    def test_phase_and_frequency_shift(self, model):
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
        qat_file = InstructionEmitter().emit(builder.instructions, model)
        qblox_packages = QbloxEmitter().emit(qat_file)
        for package in qblox_packages:
            expected_phase = get_nco_phase_arguments(phase)
            assert f"set_ph_delta {expected_phase}" in package.sequence.program
            expected_freq = get_nco_set_frequency_arguments(
                drive_channel.baseband_if_frequency + frequency
            )
            assert f"set_freq {expected_freq}" in package.sequence.program

    def test_measure_acquire(self, model):
        time = 7.5e-6
        delay = 150e-9
        qubit = model.get_qubit(0)
        acquire_channel = qubit.get_acquire_channel()
        measure_channel = qubit.get_measure_channel()
        assert measure_channel == acquire_channel

        builder = get_builder(model).acquire(acquire_channel, time, delay=delay)
        qat_file = InstructionEmitter().emit(builder.instructions, model)
        qblox_packages = QbloxEmitter().emit(qat_file)
        assert len(qblox_packages) == 0

        builder = get_builder(model)
        builder.add(
            [
                MeasurePulse(measure_channel, **qubit.pulse_measure),
                Acquire(acquire_channel, time=time, delay=delay),
            ]
        )
        qat_file = InstructionEmitter().emit(builder.instructions, model)
        qblox_packages = QbloxEmitter().emit(qat_file)
        assert len(qblox_packages) == 1
        package = qblox_packages[0]
        assert f"play {0},{1},{4}" in package.sequence.program
        assert f"wait {int(delay*1e9)}" in package.sequence.program
        assert f"acquire {0},R0,{16384}" in package.sequence.program

        channel = acquire_channel.physical_channel
        assert isinstance(channel, QbloxPhysicalChannel)
        assert isinstance(channel.baseband, QbloxPhysicalBaseband)
        assert len(channel.config.sequencers) > 0
        assert package.sequencer_config.square_weight_acq.integration_length == int(
            time * 1e9
        )
