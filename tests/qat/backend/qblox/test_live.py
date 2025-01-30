# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import numpy as np
import pytest

from qat.purr.backends.qblox.live import (
    NewQbloxLiveEngine,
    QbloxLiveEngine,
    QbloxLiveEngineAdapter,
)
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.instructions import SweepValue, Variable
from qat.purr.compiler.runtime import execute_instructions, get_builder
from qat.purr.utils.logger import get_default_logger

from tests.qat.utils.builder_nuggets import qubit_spect, resonator_spect, t1

log = get_default_logger()


@pytest.mark.parametrize("model", [None], indirect=True)
class TestQbloxLiveEngine:
    def test_measure_amp_sweep(self, model):
        engine = model.create_engine()
        q0 = model.get_qubit(0)

        for amp in [0.1, 0.2, 0.3]:
            q0.pulse_measure["amp"] = amp
            builder = get_builder(model).measure(q0).repeat(10000)
            results, _ = execute_instructions(engine, builder)
            assert results is not None

    def test_measure_freq_sweep(self, model):
        engine = model.create_engine()
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
        results, _ = execute_instructions(engine, builder)
        assert results is not None

    def test_instruction_execution(self, model):
        engine = model.create_engine()
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

        results, _ = execute_instructions(engine, builder)
        assert results is not None

    def test_one_channel(self, model):
        engine = model.create_engine()
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

        results, _ = execute_instructions(engine, builder)
        assert results is not None

    def test_two_channels(self, model):
        engine = model.create_engine()
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

        results, _ = execute_instructions(engine, builder)
        assert results is not None

    def test_sync_two_channel(self, model):
        engine = model.create_engine()
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

        results, _ = execute_instructions(engine, builder)
        assert results is not None

    def test_play_very_long_pulse(self, model):
        engine = model.create_engine()
        q0 = model.get_qubit(0)

        drive_channel = q0.get_drive_channel()
        builder = get_builder(model).pulse(
            drive_channel, PulseShapeType.SOFT_SQUARE, amp=0.1, width=1e-5, rise=1e-8
        )

        with pytest.raises(ValueError):
            results, _ = execute_instructions(engine, builder)

    def test_bare_measure(self, model):
        engine = model.create_engine()
        q0 = model.get_qubit(0)

        amp = 1
        qubit = q0
        qubit.pulse_measure["amp"] = amp
        drive_channel2 = qubit.get_drive_channel()

        # FluidBuilderWrapper nightmares
        builder, _ = (
            get_builder(model)
            .pulse(drive_channel2, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .measure(qubit)
        )

        results, _ = execute_instructions(engine, builder)
        assert results is not None

    def test_measure_scope_mode(self, model):
        engine = model.create_engine()
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

        results, _ = execute_instructions(engine, builder)
        assert results is not None

    def test_resonator_spect(self, model):
        engine = model.create_engine()
        builder = resonator_spect(model, [0, 1])

        results, _ = execute_instructions(engine, builder)
        assert results is not None

    def test_qubit_spect(self, model):
        engine = model.create_engine()
        builder = qubit_spect(model, [0, 1])

        results, _ = execute_instructions(engine, builder)
        assert results is not None


@pytest.mark.parametrize("model", [None], indirect=True)
class TestQbloxLiveEngineAdapter:
    def test_engine_adapter(self, model):
        runtime = model.create_runtime()
        assert isinstance(runtime.engine, QbloxLiveEngineAdapter)
        assert isinstance(runtime.engine._legacy_engine, QbloxLiveEngine)
        assert isinstance(runtime.engine._new_engine, NewQbloxLiveEngine)

    def test_resonator_spect(self, model):
        runtime = model.create_runtime()
        runtime.engine.enable_hax = True
        builder = resonator_spect(model)
        results = runtime.execute(builder)
        assert results is not None

    def test_qubit_spect(self, model):
        runtime = model.create_runtime()
        runtime.engine.enable_hax = True
        builder = qubit_spect(model)
        results = runtime.execute(builder)
        assert results is not None

    def test_t1(self, model):
        runtime = model.create_runtime()
        runtime.engine.enable_hax = True
        builder = t1(model)
        results = runtime.execute(builder)
        assert results is not None
