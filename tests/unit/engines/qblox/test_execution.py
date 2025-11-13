# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

import numpy as np
import pytest

from qat import qatconfig
from qat.backend.base import BaseBackend
from qat.backend.passes.purr.analysis import (
    TriageResult,
)
from qat.backend.qblox.codegen import QbloxBackend1, QbloxBackend2
from qat.backend.qblox.config.constants import Constants, QbloxTargetData
from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.engines.qblox.execution import QbloxEngine
from qat.pipelines.purr.qblox.compile import (
    backend_pipeline1,
    backend_pipeline2,
    middleend_pipeline1,
    middleend_pipeline2,
)
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.instructions import SweepValue, Variable
from qat.purr.compiler.runtime import get_builder
from qat.purr.utils.logger import get_default_logger

from tests.unit.utils.builder_nuggets import (
    delay_iteration,
    hidden_mode,
    measure_acquire,
    multi_readout,
    qubit_spect,
    readout_freq,
    resonator_spect,
    sweep_and_measure,
    xpi2amp,
    zmap,
)

log = get_default_logger()


def _do_emit(model, backend: BaseBackend, builder):
    res_mgr = ResultManager()
    met_mgr = MetricsManager()
    if isinstance(backend, QbloxBackend1):
        middleend_pipeline1(model, QbloxTargetData.default()).run(builder, res_mgr, met_mgr)
        backend_pipeline1().run(builder, res_mgr, met_mgr)
    elif isinstance(backend, QbloxBackend2):
        middleend_pipeline2(model, QbloxTargetData.default()).run(builder, res_mgr, met_mgr)
        backend_pipeline2().run(builder, res_mgr, met_mgr)
    triage_result = res_mgr.lookup_by_type(TriageResult)
    executable = backend.emit(builder, res_mgr, met_mgr)
    return executable.programs, triage_result


@pytest.mark.parametrize("qblox_model", [None], indirect=True)
@pytest.mark.parametrize("qblox_instrument", [None], indirect=True)
@pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
@pytest.mark.parametrize("backend_type", [QbloxBackend1, QbloxBackend2])
class Test1QMeasurements:
    """
    Tests execution of a standard 1Q measurements.
    """

    def test_resonator_spect(
        self, qblox_model, qblox_instrument, backend_type, qubit_indices
    ):
        backend = backend_type(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        builder = resonator_spect(qblox_model, qubit_indices)
        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == (10,)
            assert not np.any(np.isnan(results[f"Q{index}"]))

    def test_qubit_spect(self, qblox_model, qblox_instrument, backend_type, qubit_indices):
        backend = backend_type(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        builder = qubit_spect(qblox_model, qubit_indices)
        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == (10,)
            assert not np.any(np.isnan(results[f"Q{index}"]))

    def test_xpi2amp(self, qblox_model, qblox_instrument, backend_type, qubit_indices):
        if backend_type == QbloxBackend2:
            pytest.skip("Needs more alignment as JIT execution work advances")

        backend = backend_type(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        builder = xpi2amp(qblox_model, qubit_indices)
        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert len(results) == len(qubit_indices)
        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == (2, 10)
            assert not np.any(np.isnan(results[f"Q{index}"]))

    def test_readout_freq(self, qblox_model, qblox_instrument, backend_type, qubit_indices):
        if backend_type == QbloxBackend2:
            pytest.skip("Needs more alignment as JIT execution work advances")

        backend = backend_type(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        builder = readout_freq(qblox_model, qubit_indices)
        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert len(results) == len(qubit_indices)
        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == (10, 2, 1000)
            assert not np.any(np.isnan(results[f"Q{index}"]))

    def test_zmap(self, qblox_model, qblox_instrument, backend_type, qubit_indices):
        if backend_type == QbloxBackend2:
            pytest.skip("Needs more alignment as JIT execution work advances")

        backend = backend_type(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        builder = zmap(qblox_model, qubit_indices, do_X=True)
        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert len(results) == len(qubit_indices)
        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == (1, 1000)
            assert not np.any(np.isnan(results[f"Q{index}"]))

    def test_t1(self, qblox_model, qblox_instrument, backend_type, qubit_indices):
        backend = backend_type(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        builder = delay_iteration(qblox_model, qubit_indices)
        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == (100,)
            assert not np.any(np.isnan(results[f"Q{index}"]))


@pytest.mark.parametrize("qblox_model", [None], indirect=True)
@pytest.mark.parametrize("qblox_instrument", [None], indirect=True)
@pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
@pytest.mark.parametrize("backend_type", [QbloxBackend1, QbloxBackend2])
class TestExecutionSuite:
    """
    Tests execution of a plethora of IRs and combinations. These are specific programs
    to trigger certain aspects that have acquired attention over the
    """

    @pytest.mark.parametrize("amp", [0.1, 0.2, 0.3])
    def test_measure_amp_sweep(
        self, qblox_model, qblox_instrument, qubit_indices, backend_type, amp
    ):
        backend = backend_type(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        builder = get_builder(qblox_model)
        builder.repeat(10000)
        for index in qubit_indices:
            qubit = qblox_model.get_qubit(index)
            qubit.pulse_measure["amp"] = amp
            builder.measure(qubit)

        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert results is not None

    def test_measure_freq_sweep(
        self, qblox_model, qblox_instrument, qubit_indices, backend_type
    ):
        backend = backend_type(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        center = 9.7772e9
        size = 10
        freqs = center + np.linspace(-100e6, 100e6, size)

        builder = get_builder(qblox_model)
        builder.repeat(1000)
        for index in qubit_indices:
            qubit = qblox_model.get_qubit(index)
            qubit.pulse_measure["amp"] = 0.3
            builder.sweep(SweepValue("frequency", freqs))
            builder.measure(qubit)
            builder.device_assign(
                qubit.get_measure_channel(), "frequency", Variable("frequency")
            )
            builder.device_assign(
                qubit.get_acquire_channel(), "frequency", Variable("frequency")
            )

        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert results is not None

    def test_instruction_execution(
        self, qblox_model, qblox_instrument, qubit_indices, backend_type
    ):
        backend = backend_type(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        amp = 1
        rise = 1.0 / 3.0
        phase = 0.72
        frequency = 500

        builder = get_builder(qblox_model)
        for index in qubit_indices:
            qubit = qblox_model.get_qubit(index)
            drive_channel = qubit.get_drive_channel()
            builder.pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            builder.pulse(drive_channel, PulseShapeType.GAUSSIAN, width=150e-9, rise=rise)
            builder.phase_shift(drive_channel, phase)
            builder.frequency_shift(drive_channel, frequency)

        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert results is not None

    def test_one_channel(self, qblox_model, qblox_instrument, qubit_indices, backend_type):
        backend = backend_type(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        amp = 1
        rise = 1.0 / 3.0

        builder = get_builder(qblox_model)
        for index in qubit_indices:
            qubit = qblox_model.get_qubit(index)
            drive_channel = qubit.get_drive_channel()
            builder.pulse(
                drive_channel,
                PulseShapeType.GAUSSIAN,
                width=100e-9,
                rise=rise,
                amp=amp / 2,
            )
            builder.delay(drive_channel, 100e-9)
            builder.pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            builder.delay(drive_channel, 100e-9)

        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert results is not None

    def test_two_channels(self, qblox_model, qblox_instrument, qubit_indices, backend_type):
        backend = backend_type(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        amp = 1
        rise = 1.0 / 3.0

        builder = get_builder(qblox_model)
        for index in qubit_indices:
            qubit = qblox_model.get_qubit(index)
            drive_channel = qubit.get_drive_channel()
            second_state_channel = qubit.get_second_state_channel()

            builder.pulse(
                drive_channel,
                PulseShapeType.GAUSSIAN,
                width=100e-9,
                rise=rise,
                amp=amp / 2,
            )
            builder.pulse(second_state_channel, PulseShapeType.SQUARE, width=50e-9, amp=amp)
            builder.delay(drive_channel, 100e-9)
            builder.pulse(
                second_state_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp
            )
            builder.pulse(
                drive_channel,
                PulseShapeType.GAUSSIAN,
                width=50e-9,
                rise=rise,
                amp=amp / 2,
            )

        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert results is not None

    def test_sync_two_channel(
        self, qblox_model, qblox_instrument, qubit_indices, backend_type
    ):
        backend = backend_type(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        amp = 1
        rise = 1.0 / 3.0

        builder = get_builder(qblox_model)
        for index in qubit_indices:
            qubit = qblox_model.get_qubit(index)
            drive_channel = qubit.get_drive_channel()
            second_state_channel = qubit.get_second_state_channel()

            builder.pulse(
                drive_channel,
                PulseShapeType.GAUSSIAN,
                width=100e-9,
                rise=rise,
                amp=amp / 2,
            )
            builder.delay(drive_channel, 100e-9)
            builder.pulse(second_state_channel, PulseShapeType.SQUARE, width=50e-9, amp=amp)
            builder.synchronize(qubit)
            builder.pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=1j * amp)
            builder.pulse(
                second_state_channel,
                PulseShapeType.GAUSSIAN,
                width=50e-9,
                rise=rise,
                amp=amp / 2j,
            )

        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert results is not None

    def test_play_very_long_pulse(
        self, qblox_model, qblox_instrument, qubit_indices, backend_type
    ):
        backend = backend_type(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        builder = get_builder(qblox_model)
        for index in qubit_indices:
            qubit = qblox_model.get_qubit(index)
            drive_channel = qubit.get_drive_channel()
            builder.pulse(
                drive_channel, PulseShapeType.SOFT_SQUARE, amp=0.1, width=1e-5, rise=1e-8
            )

        with pytest.raises(ValueError):
            ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
            results = engine.execute(ordered_executables, triage_result)
            assert results is not None

    def test_bare_measure(self, qblox_model, qblox_instrument, qubit_indices, backend_type):
        backend = backend_type(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        amp = 1

        builder = get_builder(qblox_model)
        for index in qubit_indices:
            qubit = qblox_model.get_qubit(index)
            qubit.pulse_measure["amp"] = amp
            drive_channel = qubit.get_drive_channel()

            builder.pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            builder.measure(qubit)

        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
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
    def test_scope_acquisition(
        self, qblox_model, qblox_instrument, qubit_indices, backend_type, acq_width, sync
    ):
        backend = backend_type(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        for index in qubit_indices:
            qubit = qblox_model.get_qubit(index)
            qubit.measure_acquire["width"] = acq_width
            qubit.measure_acquire["sync"] = sync

        builder = measure_acquire(qblox_model, qubit_indices)
        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert results is not None

        for index in qubit_indices:
            qubit = qblox_model.get_qubit(index)

            assert f"Q{index}" in results
            if sync:
                assert results[f"Q{index}"].shape == (
                    1,
                    int(qubit.pulse_measure["width"] * 1e9),
                )
            else:
                assert results[f"Q{index}"].shape == (1, int(acq_width * 1e9))

    def test_multi_readout(
        self, qblox_model, qblox_instrument, qubit_indices, backend_type
    ):
        old_value = qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT
        try:
            qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT = False
            backend = backend_type(qblox_model)
            engine = QbloxEngine(qblox_instrument, qblox_model)

            builder = multi_readout(qblox_model, qubit_indices, do_X=True)

            ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
            results = engine.execute(ordered_executables, triage_result)
            assert results
            assert len(results) == 2 * len(qubit_indices)
            for index in qubit_indices:
                assert f"0_Q{index}" in results
                assert results[f"0_Q{index}"].shape == (1, 1000)
                assert results[f"0_Q{index}"].dtype == np.float64
                assert f"1_Q{index}" in results
                assert results[f"1_Q{index}"].shape == (1, 1000)
                assert results[f"1_Q{index}"].dtype == np.complex128
        finally:
            qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT = old_value

    def test_hidden_mode(self, qblox_model, qblox_instrument, qubit_indices, backend_type):
        if backend_type == QbloxBackend2:
            pytest.skip("Needs more alignment as JIT execution work advances")

        qubit_indices = [0]
        backend = backend_type(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        builder = hidden_mode(qblox_model, qubit_indices, num_points=3)

        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert results

    @pytest.mark.xfail(
        raises=ValueError, reason="Needs unification of Backend1 and Backend2's pipelines"
    )
    @pytest.mark.parametrize("num_points", [10])
    @pytest.mark.parametrize(
        "signal, single_shot, expected_shape, expected_dtype",
        [
            (True, True, (10, 1000), np.complex128),
            (True, False, (10,), np.complex128),
            (False, True, (10, 1000), np.float64),
            (False, False, (10,), np.float64),
        ],
    )
    def test_sweep_and_measure(
        self,
        qblox_model,
        qblox_instrument,
        qubit_indices,
        backend_type,
        num_points,
        signal,
        single_shot,
        expected_shape,
        expected_dtype,
    ):
        backend = backend_type(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        builder = sweep_and_measure(
            qblox_model, qubit_indices, num_points, signal, single_shot
        )

        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert results

        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == expected_shape
            assert results[f"Q{index}"].dtype == expected_dtype
