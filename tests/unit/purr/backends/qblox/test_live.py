# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd
import numpy as np
import pytest

from qat import qatconfig
from qat.purr.backends.qblox.constants import Constants
from qat.purr.backends.qblox.live import (
    QbloxLiveEngine1,
    QbloxLiveEngine2,
    QbloxLiveEngineAdapter,
)
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.instructions import SweepValue, Variable
from qat.purr.compiler.runtime import execute_instructions, get_builder
from qat.purr.utils.logger import get_default_logger

from tests.unit.utils.builder_nuggets import (
    delay_iteration,
    multi_readout,
    qubit_spect,
    readout_freq,
    resonator_spect,
    scope_acq,
    xpi2amp,
    zmap,
)

log = get_default_logger()


@pytest.mark.parametrize("model", [None], indirect=True)
@pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
@pytest.mark.parametrize("enable_hax", [False, True])
class Test1QMeasurements:
    def test_resonator_spect(self, enable_hax, model, qubit_indices):
        runtime = model.create_runtime()
        engine = runtime.engine
        engine.enable_hax = enable_hax

        builder = resonator_spect(model, qubit_indices)
        results, _ = execute_instructions(engine, builder)
        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == (10,)

    def test_qubit_spect(self, enable_hax, model, qubit_indices):
        runtime = model.create_runtime()
        engine = runtime.engine
        engine.enable_hax = enable_hax

        builder = qubit_spect(model, qubit_indices)
        results, _ = execute_instructions(engine, builder)
        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == (10,)

    def test_xpi2amp(self, enable_hax, model, qubit_indices):
        if enable_hax:
            pytest.skip("Needs more alignment as JIT execution work advances")

        runtime = model.create_runtime()
        engine = runtime.engine
        engine.enable_hax = enable_hax

        builder = xpi2amp(model, qubit_indices)
        results, _ = execute_instructions(engine, builder)
        assert len(results) == len(qubit_indices)
        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == (2, 10)

    def test_readout_freq(self, enable_hax, model, qubit_indices):
        if enable_hax:
            pytest.skip("Needs more alignment as JIT execution work advances")

        runtime = model.create_runtime()
        engine = runtime.engine
        engine.enable_hax = enable_hax

        builder = readout_freq(model, qubit_indices)
        results, _ = execute_instructions(engine, builder)
        assert len(results) == len(qubit_indices)
        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == (10, 2, 1000)

    def test_zmap(self, enable_hax, model, qubit_indices):
        if enable_hax:
            pytest.skip("Needs more alignment as JIT execution work advances")

        runtime = model.create_runtime()
        engine = runtime.engine
        engine.enable_hax = enable_hax

        builder = zmap(model, qubit_indices, do_X=True)
        results, _ = execute_instructions(engine, builder)
        assert len(results) == len(qubit_indices)
        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == (1, 1000)

    def test_t1(self, enable_hax, model, qubit_indices):
        runtime = model.create_runtime()
        engine = runtime.engine
        engine.enable_hax = enable_hax

        builder = delay_iteration(model, qubit_indices)
        results, _ = execute_instructions(engine, builder)
        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == (100,)


@pytest.mark.parametrize("model", [None], indirect=True)
class TestBuildingBlocks:
    @pytest.mark.parametrize("amp", [0.1, 0.2, 0.3])
    def test_measure_amp_sweep(self, model, amp):
        engine = model.create_engine()
        q0 = model.get_qubit(0)

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

    @pytest.mark.parametrize(
        "acq_width",
        (
            np.random.choice(
                np.arange(Constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS) * 1e-9,
                3,
            )
        ).tolist(),
    )
    @pytest.mark.parametrize("sync", [True, False])
    def test_scope_acq(self, model, acq_width, sync):
        qubit_indices = [0]
        engine = model.create_engine()

        for index in qubit_indices:
            qubit = model.get_qubit(index)
            qubit.measure_acquire["width"] = acq_width
            qubit.measure_acquire["sync"] = sync

        builder = scope_acq(model, qubit_indices)
        results, _ = execute_instructions(engine, builder)
        assert results is not None

        for index in qubit_indices:
            qubit = model.get_qubit(index)

            assert f"Q{index}" in results
            if sync:
                assert results[f"Q{index}"].shape == (
                    1,
                    int(qubit.pulse_measure["width"] * 1e9),
                )
            else:
                assert results[f"Q{index}"].shape == (1, int(acq_width * 1e9))

    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    def test_multi_readout(self, model, qubit_indices):
        old_value = qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT
        try:
            qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT = False
            engine = model.create_engine()
            builder = multi_readout(model, qubit_indices, do_X=True)

            results, _ = execute_instructions(engine, builder)
            assert results
            assert len(results) == 2 * len(qubit_indices)
            for index in qubit_indices:
                assert f"0_Q{index}" in results
                assert f"1_Q{index}" in results
        finally:
            qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT = old_value


@pytest.mark.parametrize("model", [None], indirect=True)
class TestQbloxLiveEngineAdapter:
    def test_engine_adapter(self, model):
        runtime = model.create_runtime()
        engine = runtime.engine

        assert isinstance(engine, QbloxLiveEngineAdapter)
        assert isinstance(engine._legacy_engine, QbloxLiveEngine1)
        assert isinstance(engine._new_engine, QbloxLiveEngine2)
