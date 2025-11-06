# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

import numpy as np
import pytest

from qat import qatconfig
from qat.backend.base import BaseBackend
from qat.backend.passes.purr.analysis import (
    TriagePass,
    TriageResult,
)
from qat.backend.qblox.codegen import QbloxBackend1, QbloxBackend2
from qat.backend.qblox.config.constants import Constants, QbloxTargetData
from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.engines.qblox.execution import QbloxEngine
from qat.middleend.passes.purr.transform import (
    DeviceUpdateSanitisation,
    PhaseOptimisation,
    PostProcessingSanitisation,
    RepeatSanitisation,
    ReturnSanitisation,
)
from qat.middleend.passes.purr.validation import InstructionValidation, ReadoutValidation
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
    xpi2amp,
    zmap,
)

log = get_default_logger()


def middleend_pipeline(model):
    target_data = QbloxTargetData.default()
    return (
        PassManager()
        | PhaseOptimisation()
        | PostProcessingSanitisation()
        | DeviceUpdateSanitisation()
        | InstructionValidation(target_data)
        | ReadoutValidation(model)
        | RepeatSanitisation(model, target_data)
        # | ScopeSanitisation()
        | ReturnSanitisation()
        # | DesugaringPass()
        | TriagePass()
        # | BindingPass()
        # | TILegalisationPass()
        # | QbloxLegalisationPass()
    )


def _do_emit(model, backend: BaseBackend, builder):
    res_mgr = ResultManager()
    met_mgr = MetricsManager()
    middleend_pipeline(model).run(builder, res_mgr, met_mgr)
    triage_result: TriageResult = res_mgr.lookup_by_type(TriageResult)
    executable = backend.emit(builder, res_mgr, met_mgr)
    return executable.programs, triage_result


@pytest.mark.parametrize("qblox_model", [None], indirect=True)
@pytest.mark.parametrize("qblox_instrument", [None], indirect=True)
@pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
class Test1QMeasurements:
    """
    Tests execution of a standard 1Q measurements.
    """

    def test_resonator_spect(self, qblox_model, qblox_instrument, qubit_indices):
        backend = QbloxBackend1(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        builder = resonator_spect(qblox_model, qubit_indices)
        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == (10,)
            assert not np.any(np.isnan(results[f"Q{index}"]))

    def test_qubit_spect(self, qblox_model, qblox_instrument, qubit_indices):
        backend = QbloxBackend1(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        builder = qubit_spect(qblox_model, qubit_indices)
        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == (10,)
            assert not np.any(np.isnan(results[f"Q{index}"]))

    def test_xpi2amp(self, qblox_model, qblox_instrument, qubit_indices):
        backend = QbloxBackend1(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        if isinstance(backend, QbloxBackend2):
            pytest.skip("Needs more alignment as JIT execution work advances")

        builder = xpi2amp(qblox_model, qubit_indices)
        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert len(results) == len(qubit_indices)
        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == (2, 10)
            assert not np.any(np.isnan(results[f"Q{index}"]))

    def test_readout_freq(self, qblox_model, qblox_instrument, qubit_indices):
        backend = QbloxBackend1(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        if isinstance(backend, QbloxBackend2):
            pytest.skip("Needs more alignment as JIT execution work advances")

        builder = readout_freq(qblox_model, qubit_indices)
        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert len(results) == len(qubit_indices)
        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == (10, 2, 1000)
            assert not np.any(np.isnan(results[f"Q{index}"]))

    def test_zmap(self, qblox_model, qblox_instrument, qubit_indices):
        backend = QbloxBackend1(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        if isinstance(backend, QbloxBackend2):
            pytest.skip("Needs more alignment as JIT execution work advances")

        builder = zmap(qblox_model, qubit_indices, do_X=True)
        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert len(results) == len(qubit_indices)
        for index in qubit_indices:
            assert f"Q{index}" in results
            assert results[f"Q{index}"].shape == (1, 1000)
            assert not np.any(np.isnan(results[f"Q{index}"]))

    def test_t1(self, qblox_model, qblox_instrument, qubit_indices):
        backend = QbloxBackend1(qblox_model)
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
class TestExecutionSuite:
    """
    Tests execution of a plethora of IRs and combinations. These are specific programs
    to trigger certain aspects that have acquired attention over the
    """

    @pytest.mark.parametrize("amp", [0.1, 0.2, 0.3])
    def test_measure_amp_sweep(self, qblox_model, qblox_instrument, amp):
        backend = QbloxBackend1(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        q0 = qblox_model.get_qubit(0)
        q0.pulse_measure["amp"] = amp
        builder = get_builder(qblox_model).measure(q0).repeat(10000)

        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert results is not None

    def test_measure_freq_sweep(self, qblox_model, qblox_instrument):
        backend = QbloxBackend1(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        q0 = qblox_model.get_qubit(0)
        q0.pulse_measure["amp"] = 0.3
        center = 9.7772e9
        size = 10
        freqs = center + np.linspace(-100e6, 100e6, size)
        builder = (
            get_builder(qblox_model)
            .sweep(SweepValue("frequency", freqs))
            .measure(q0)
            .device_assign(q0.get_measure_channel(), "frequency", Variable("frequency"))
            .device_assign(q0.get_acquire_channel(), "frequency", Variable("frequency"))
            .repeat(1000)
        )

        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert results is not None

    def test_instruction_execution(self, qblox_model, qblox_instrument):
        backend = QbloxBackend1(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        amp = 1
        rise = 1.0 / 3.0
        phase = 0.72
        frequency = 500
        q0 = qblox_model.get_qubit(0)
        drive_channel = q0.get_drive_channel()
        builder = (
            get_builder(qblox_model)
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .pulse(drive_channel, PulseShapeType.GAUSSIAN, width=150e-9, rise=rise)
            .phase_shift(drive_channel, phase)
            .frequency_shift(drive_channel, frequency)
        )

        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert results is not None

    def test_one_channel(self, qblox_model, qblox_instrument):
        backend = QbloxBackend1(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        q0 = qblox_model.get_qubit(0)

        amp = 1
        rise = 1.0 / 3.0

        drive_channel = q0.get_drive_channel()
        builder = (
            get_builder(qblox_model)
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

        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert results is not None

    def test_two_channels(self, qblox_model, qblox_instrument):
        backend = QbloxBackend1(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        q0 = qblox_model.get_qubit(0)
        q1 = qblox_model.get_qubit(1)

        amp = 1
        rise = 1.0 / 3.0

        drive_channel2 = q0.get_drive_channel()
        drive_channel3 = q1.get_drive_channel()
        builder = (
            get_builder(qblox_model)
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

        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert results is not None

    def test_sync_two_channel(self, qblox_model, qblox_instrument):
        backend = QbloxBackend1(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        q0 = qblox_model.get_qubit(0)
        q1 = qblox_model.get_qubit(1)

        amp = 1
        rise = 1.0 / 3.0

        drive_channel0 = q0.get_drive_channel()
        drive_channel1 = q1.get_drive_channel()
        builder = (
            get_builder(qblox_model)
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

        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert results is not None

    def test_bare_measure(self, qblox_model, qblox_instrument):
        backend = QbloxBackend1(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        q0 = qblox_model.get_qubit(0)

        amp = 1
        qubit = q0
        qubit.pulse_measure["amp"] = amp
        drive_channel2 = qubit.get_drive_channel()

        # FluidBuilderWrapper nightmares
        builder, _ = (
            get_builder(qblox_model)
            .pulse(drive_channel2, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .measure(qubit)
        )

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
    def test_scope_acq(self, qblox_model, qblox_instrument, acq_width, sync):
        backend = QbloxBackend1(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        qubit_indices = [0]

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

    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    def test_multi_readout(self, qblox_model, qblox_instrument, qubit_indices):
        backend = QbloxBackend1(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        old_value = qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT
        try:
            qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT = False
            builder = multi_readout(qblox_model, qubit_indices, do_X=True)

            ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
            results = engine.execute(ordered_executables, triage_result)
            assert results
            assert len(results) == 2 * len(qubit_indices)
            for index in qubit_indices:
                assert f"0_Q{index}" in results
                assert f"1_Q{index}" in results
        finally:
            qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT = old_value

    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_hidden_mode(self, qblox_model, qblox_instrument, qubit_indices):
        backend = QbloxBackend1(qblox_model)
        engine = QbloxEngine(qblox_instrument, qblox_model)

        builder = hidden_mode(qblox_model, qubit_indices, num_points=3)

        ordered_executables, triage_result = _do_emit(qblox_model, backend, builder)
        results = engine.execute(ordered_executables, triage_result)
        assert results
