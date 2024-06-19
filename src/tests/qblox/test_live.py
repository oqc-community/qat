import numpy as np
import pytest

from qat.purr.backends.qblox.live import QbloxLiveEngine
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.instructions import SweepValue, Variable
from qat.purr.compiler.runtime import execute_instructions, get_builder
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


@pytest.mark.parametrize("model", [None], indirect=True)
class TestQbloxLiveEngine:
    def test_measure_amp_sweep(self, model):
        engine = QbloxLiveEngine(model)
        q0 = model.get_qubit(0)

        for amp in [0.5, 1.0]:
            q0.pulse_measure["amp"] = amp
            builder = get_builder(model).measure(q0).repeat(10000)
            results, _ = execute_instructions(engine, builder.instructions)
            assert results is not None

    def test_measure_freq_sweep(self, model):
        engine = QbloxLiveEngine(model)
        q0 = model.get_qubit(0)

        q0.pulse_measure["amp"] = 0.3
        center = 9.7772e9
        size = 10
        freqs = center + np.linspace(-100e6, 100e6, size)
        builder = (
            get_builder(model)
            .sweep(SweepValue("frequency", freqs))
            .measure(q0)
            .device_assign(q0.get_measure_channel(), "frequency", Variable("frequency"))
            .device_assign(q0.get_acquire_channel(), "frequency", Variable("frequency"))
            .repeat(1000)
        )
        results, _ = execute_instructions(engine, builder.instructions)
        assert results is not None

    def test_instruction_execution(self, model):
        engine = QbloxLiveEngine(model)
        q0 = model.get_qubit(0)

        amp = 1
        rise = 1.0 / 3.0
        phase = 0.72
        frequency = 500

        drive_channel = q0.get_drive_channel()
        builder = (
            get_builder(model)
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .pulse(drive_channel, PulseShapeType.GAUSSIAN, width=150e-9, rise=rise)
            .phase_shift(drive_channel, phase)
            .frequency_shift(drive_channel, frequency)
        )

        results, _ = execute_instructions(engine, builder.instructions)
        assert results is not None

    def test_one_channel(self, model):
        engine = QbloxLiveEngine(model)
        q0 = model.get_qubit(0)

        amp = 1
        rise = 1.0 / 3.0

        drive_channel = q0.get_drive_channel()
        builder = (
            get_builder(model)
            .pulse(
                drive_channel,
                PulseShapeType.GAUSSIAN,
                width=100e-9,
                rise=rise,
                amp=amp / 2,
            )
            .delay(drive_channel, 100e-9)
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .delay(drive_channel, 100e-9)
        )

        results, _ = execute_instructions(engine, builder.instructions)
        assert results is not None

    def test_two_channels(self, model):
        engine = QbloxLiveEngine(model)
        q0 = model.get_qubit(0)
        q1 = model.get_qubit(1)

        amp = 1
        rise = 1.0 / 3.0

        drive_channel2 = q0.get_drive_channel()
        drive_channel3 = q1.get_drive_channel()
        builder = (
            get_builder(model)
            .pulse(
                drive_channel2,
                PulseShapeType.GAUSSIAN,
                width=100e-9,
                rise=rise,
                amp=amp / 2,
            )
            .pulse(drive_channel3, PulseShapeType.SQUARE, width=50e-9, amp=amp)
            .delay(drive_channel2, 100e-9)
            .delay(drive_channel3, 100e-9)
            .pulse(drive_channel2, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .pulse(
                drive_channel3,
                PulseShapeType.GAUSSIAN,
                width=50e-9,
                rise=rise,
                amp=amp / 2,
            )
        )

        results, _ = execute_instructions(engine, builder.instructions)
        assert results is not None

    def test_sync_two_channel(self, model):
        engine = QbloxLiveEngine(model)
        q0 = model.get_qubit(0)
        q1 = model.get_qubit(1)

        amp = 1
        rise = 1.0 / 3.0

        drive_channel0 = q0.get_drive_channel()
        drive_channel1 = q1.get_drive_channel()
        builder = (
            get_builder(model)
            .pulse(
                drive_channel0,
                PulseShapeType.GAUSSIAN,
                width=100e-9,
                rise=rise,
                amp=amp / 2,
            )
            .delay(drive_channel0, 100e-9)
            .pulse(drive_channel1, PulseShapeType.SQUARE, width=50e-9, amp=amp)
            .synchronize([q0, q1])
            .pulse(drive_channel0, PulseShapeType.SQUARE, width=100e-9, amp=1j * amp)
            .pulse(
                drive_channel1,
                PulseShapeType.GAUSSIAN,
                width=50e-9,
                rise=rise,
                amp=amp / 2j,
            )
        )

        results, _ = execute_instructions(engine, builder.instructions)
        assert results is not None

    def test_play_very_long_pulse(self, model):
        engine = QbloxLiveEngine(model)
        q0 = model.get_qubit(0)

        drive_channel = q0.get_drive_channel()
        builder = get_builder(model).pulse(
            drive_channel, PulseShapeType.SOFT_SQUARE, amp=0.1, width=1e-5, rise=1e-8
        )

        with pytest.raises(ValueError):
            results, _ = execute_instructions(engine, builder.instructions)

    def test_bare_measure(self, model):
        engine = QbloxLiveEngine(model)
        q0 = model.get_qubit(0)

        amp = 1
        qubit = q0
        qubit.pulse_measure["amp"] = amp
        drive_channel2 = qubit.get_drive_channel()
        builder = (
            get_builder(model)
            .pulse(drive_channel2, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .measure(qubit)
        )

        results, _ = execute_instructions(engine, builder.instructions)
        assert results is not None

    def test_measure_scope_mode(self, model):
        engine = QbloxLiveEngine(model)
        q0 = model.get_qubit(0)

        amp = 1
        qubit = q0
        qubit.pulse_measure["amp"] = amp
        drive_channel2 = qubit.get_drive_channel()
        builder = (
            get_builder(model)
            .pulse(drive_channel2, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .measure_scope_mode(qubit)
        )

        results, _ = execute_instructions(engine, builder.instructions)
        assert results is not None


@pytest.mark.parametrize("model", [None], indirect=True)
class Test1QMeasurements:
    def test_resonator_spect(self, model):
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
            .sweep(SweepValue(f"freq{0}", freqs))
            .device_assign(measure_channel, "frequency", Variable(f"freq{0}"))
            .device_assign(acquire_channel, "frequency", Variable(f"freq{0}"))
            .measure_mean_signal(qubit, output_variable=f"Q{0}")
            .repeat(400, 500e-6)
        )
        results, _ = execute_instructions(engine, builder.instructions)
        assert results is not None

    def test_qubit_spect(self, model):
        engine = QbloxLiveEngine(model)
        qubit = model.get_qubit(0)
        drive_channel = qubit.get_drive_channel()

        num_points = 10
        freq_range = 10e6
        center_freq = drive_channel.frequency
        freqs = center_freq + np.linspace(-freq_range, freq_range, num_points)
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
        results, _ = execute_instructions(engine, builder.instructions)
        assert results is not None

    def test_zmap(self, model):
        engine = QbloxLiveEngine(model)

        do_X = True  # excited state
        x12 = False
        measure_width = None
        delta_r = 0.0

        builder = get_builder(model)
        builder.synchronize([model.get_qubit(q) for q in [0]])

        if x12:
            for q in [0]:
                builder.X(model.get_qubit(q))
        builder.synchronize([model.get_qubit(q) for q in [0]])
        if do_X:
            for q in [0]:
                qubit = model.get_qubit(q)
                qubit_drive_channel = (
                    qubit.get_second_state_channel()
                    if x12
                    else qubit.get_drive_channel()
                )
                builder.add(model.get_gate_X(qubit, np.pi, qubit_drive_channel))
        builder.synchronize([model.get_qubit(q) for q in [0]])
        for q in [0]:
            qubit = model.get_qubit(q)
            measure_width = (
                qubit.pulse_measure["width"]
                if measure_width is None
                else measure_width[f"Q{q}"]
            )
            builder.device_assign(
                qubit.get_measure_channel(),
                "frequency",
                qubit.get_measure_channel().frequency + delta_r,
            )
            builder.measure_single_shot_signal(qubit, output_variable=f"Q{q}")

            results, _ = execute_instructions(engine, builder.instructions)
            assert results is not None
