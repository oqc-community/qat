# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd

import copy
import re
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from matplotlib import pyplot as plt
from more_itertools import partition

from qat import qatconfig
from qat.backend.passes.purr.analysis import ReadWriteResult, TILegalisationPass
from qat.backend.qblox.codegen import (
    QbloxBackend1,
    QbloxBackend2,
    QbloxContext,
    _check_shared_lo_conflicts,
    _debug_info_enabled,
    _get_lo_lanes,
)
from qat.backend.qblox.config.specification import (
    ConnectionConfig,
    LoConfig,
    ModuleConfig,
    SequencerConfig,
)
from qat.backend.qblox.execution import QbloxPackage
from qat.backend.qblox.ir import Opcode
from qat.backend.qblox.passes.analysis import AllocationManager, QbloxLegalisationPass
from qat.backend.qblox.target_data import CONTROL_SEQUENCER_DATA, Q1ASM_DATA, TARGET_DATA
from qat.backend.qblox.visualisation import plot_program
from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.pipelines.purr.qblox.compile import (
    backend_pipeline1,
    backend_pipeline2,
    middleend_pipeline1,
    middleend_pipeline2,
)
from qat.purr.backends.qblox.device import QbloxPhysicalBaseband, QbloxPhysicalChannel
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.instructions import (
    Acquire,
    DeviceUpdate,
    MeasurePulse,
    PhaseShift,
    Pulse,
    Repeat,
    Sweep,
    SweepValue,
    Synchronize,
    Variable,
    calculate_duration,
)
from qat.purr.compiler.runtime import get_builder
from qat.purr.utils.logger import get_default_logger

from tests.unit.backend.qblox.utils import do_emit
from tests.unit.utils.builder_nuggets import (
    delay_iteration,
    discrimination,
    empty,
    measure_acquire,
    multi_readout,
    pulse_amplitude_iteration,
    pulse_width_iteration,
    qubit_spect,
    resonator_spect,
    time_and_phase_iteration,
)

log = get_default_logger()


@pytest.fixture()
def copied_qblox_model(qblox_model):
    return copy.deepcopy(qblox_model)


class TestQbloxContext:
    """Tests properties of the QBloxContext, used by the two backends."""

    def test_duration_tracking_is_static_until_a_register_is_seen(self):
        context = QbloxContext(ReadWriteResult(), {})

        assert context.has_static_duration
        assert context.duration == 0

        context.ledger(40)
        context.ledger(60)

        assert context.has_static_duration
        assert context.duration == 100

        context.ledger("iter")

        assert not context.has_static_duration
        with pytest.raises(
            ValueError,
            match="Cannot determine duration statically in dynamic settings",
        ):
            _ = context.duration

    @pytest.mark.parametrize("qblox_model", [None], indirect=True)
    def test_synchronize_uses_static_wait_when_all_contexts_are_static(self, qblox_model):
        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        measure_channel = qblox_model.get_qubit(1).get_measure_channel()
        contexts = {
            drive_channel: QbloxContext(ReadWriteResult(), {}),
            measure_channel: QbloxContext(ReadWriteResult(), {}),
        }
        contexts[drive_channel].ledger(20)
        contexts[measure_channel].ledger(40)

        assert contexts[drive_channel].has_static_duration
        assert contexts[measure_channel].has_static_duration
        assert contexts[drive_channel].duration == 20
        assert contexts[measure_channel].duration == 40

        QbloxContext.synchronize(Synchronize([drive_channel, measure_channel]), contexts)

        assert contexts[drive_channel].has_static_duration
        assert contexts[measure_channel].has_static_duration
        assert contexts[drive_channel].duration == 40
        assert contexts[measure_channel].duration == 40
        assert (
            contexts[drive_channel].sequence_builder.q1asm_instructions[-1].opcode
            == Opcode.WAIT
        )
        assert contexts[drive_channel].sequence_builder.build().program == "wait 20"

    @pytest.mark.parametrize("qblox_model", [None], indirect=True)
    def test_synchronize_uses_wait_sync_when_any_context_is_dynamic(self, qblox_model):
        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        measure_channel = qblox_model.get_qubit(1).get_measure_channel()
        contexts = {
            drive_channel: QbloxContext(ReadWriteResult(), {}),
            measure_channel: QbloxContext(ReadWriteResult(), {}),
        }
        contexts[drive_channel].ledger(40)
        contexts[measure_channel].ledger("iter")

        assert contexts[drive_channel].has_static_duration
        assert not contexts[measure_channel].has_static_duration
        assert contexts[drive_channel].duration == 40
        with pytest.raises(ValueError, match="Cannot determine duration statically"):
            _ = contexts[measure_channel].duration

        QbloxContext.synchronize(Synchronize([drive_channel, measure_channel]), contexts)

        assert not contexts[drive_channel].has_static_duration
        assert not contexts[measure_channel].has_static_duration
        with pytest.raises(ValueError, match="Cannot determine duration statically"):
            _ = contexts[drive_channel].duration
        with pytest.raises(ValueError, match="Cannot determine duration statically"):
            _ = contexts[measure_channel].duration
        assert (
            contexts[drive_channel].sequence_builder.q1asm_instructions[-1].opcode
            == Opcode.WAIT_SYNC
        )
        assert (
            contexts[measure_channel].sequence_builder.q1asm_instructions[-1].opcode
            == Opcode.WAIT_SYNC
        )

    @pytest.mark.parametrize("qblox_model", [None], indirect=True)
    def test_synchronize_with_no_targets_does_nothing(self, qblox_model):
        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        contexts = {drive_channel: QbloxContext(ReadWriteResult(), {})}
        contexts[drive_channel].ledger(20)

        QbloxContext.synchronize(Synchronize([]), contexts)

        assert contexts[drive_channel].has_static_duration
        assert contexts[drive_channel].duration == 20
        assert contexts[drive_channel].sequence_builder.q1asm_instructions == []

    @pytest.mark.parametrize("qblox_model", [None], indirect=True)
    def test_timeline_concatenates_ledger_entries_in_order(self, qblox_model):
        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        context = QbloxContext(ReadWriteResult(), {}, enable_timeline_debug_info=True)

        context.ledger(8)
        context.ledger(4, np.ones(4, dtype=complex))

        pkg = context.create_package(drive_channel, 0, 0)
        np.testing.assert_array_equal(
            pkg.timeline, np.concatenate([np.zeros(8), np.ones(4)]).astype(complex)
        )

    @pytest.mark.parametrize("qblox_model", [None], indirect=True)
    @pytest.mark.parametrize(
        "phase, rotation",
        [
            pytest.param(np.pi / 2, 1j, id="static-phase-rotates"),
            pytest.param(Variable("phase"), 1, id="dynamic-phase-not-applied"),
        ],
    )
    def test_timeline_applies_only_static_phase(self, qblox_model, phase, rotation):
        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        alloc_mgr = AllocationManager()
        alloc_mgr.reg_alloc("phase")
        context = QbloxContext(
            ReadWriteResult(), {}, alloc_mgr=alloc_mgr, enable_timeline_debug_info=True
        )

        context.shift_phase(PhaseShift(drive_channel, phase))
        context.ledger(4, np.ones(4, dtype=complex))

        timeline = context.create_package(drive_channel, 0, 0).timeline
        np.testing.assert_allclose(timeline[-4:], np.full(4, rotation, dtype=complex))

    @staticmethod
    def _phase_context(enable_timeline_debug_info=False):
        alloc_mgr = AllocationManager()
        alloc_mgr.reg_alloc("phase")
        return QbloxContext(
            ReadWriteResult(),
            {},
            alloc_mgr=alloc_mgr,
            enable_timeline_debug_info=enable_timeline_debug_info,
        )

    @pytest.mark.parametrize("qblox_model", [None], indirect=True)
    def test_phase_tracking_accumulates_static_shifts(self, qblox_model):
        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        context = self._phase_context()

        assert context.phase == 0

        context.shift_phase(PhaseShift(drive_channel, 0.25))
        context.shift_phase(PhaseShift(drive_channel, 0.5))

        assert context.phase == pytest.approx(0.75)

    @pytest.mark.parametrize("qblox_model", [None], indirect=True)
    def test_phase_tracking_stays_dynamic_once_a_variable_is_seen(self, qblox_model):
        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        context = self._phase_context()

        context.shift_phase(PhaseShift(drive_channel, 0.25))
        context.shift_phase(PhaseShift(drive_channel, Variable("phase")))

        with pytest.raises(
            ValueError, match="Cannot determine phase statically in dynamic settings"
        ):
            _ = context.phase

        # A later static shift cannot make the phase known again.
        context.shift_phase(PhaseShift(drive_channel, 0.5))

        with pytest.raises(
            ValueError, match="Cannot determine phase statically in dynamic settings"
        ):
            _ = context.phase

    @pytest.mark.parametrize("qblox_model", [None], indirect=True)
    @pytest.mark.parametrize(
        "phase",
        [pytest.param(0.5, id="static"), pytest.param(Variable("phase"), id="dynamic")],
    )
    def test_reset_phase_restores_static_tracking(self, qblox_model, phase):
        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        context = self._phase_context()

        context.shift_phase(PhaseShift(drive_channel, phase))
        context.reset_phase()

        assert context.phase == 0

        context.shift_phase(PhaseShift(drive_channel, 0.25))

        assert context.phase == pytest.approx(0.25)

    @pytest.mark.parametrize("qblox_model", [None], indirect=True)
    def test_timeline_applies_accumulated_phase(self, qblox_model):
        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        context = self._phase_context(enable_timeline_debug_info=True)

        context.shift_phase(PhaseShift(drive_channel, np.pi / 4))
        context.shift_phase(PhaseShift(drive_channel, np.pi / 4))
        context.ledger(4, np.ones(4, dtype=complex))

        timeline = context.create_package(drive_channel, 0, 0).timeline
        np.testing.assert_allclose(timeline[-4:], np.full(4, 1j, dtype=complex))

    @pytest.mark.parametrize("qblox_model", [None], indirect=True)
    @pytest.mark.parametrize(
        "shifts, resets",
        [
            pytest.param(0, False, id="no-shifts"),
            pytest.param(2, True, id="reset-after-shifts"),
        ],
    )
    def test_timeline_is_not_rotated_when_accumulated_phase_is_zero(
        self, qblox_model, shifts, resets
    ):
        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        context = self._phase_context(enable_timeline_debug_info=True)
        pulse = np.full(4, 0.5 + 0.25j)

        for _ in range(shifts):
            context.shift_phase(PhaseShift(drive_channel, np.pi / 3))
        if resets:
            context.reset_phase()
        context.ledger(4, pulse)

        timeline = context.create_package(drive_channel, 0, 0).timeline
        np.testing.assert_array_equal(timeline[-4:], pulse)

    @pytest.mark.parametrize("qblox_model", [None], indirect=True)
    @pytest.mark.parametrize("phase", [0.0, np.pi / 3], ids=["no-phase", "static-phase"])
    def test_timeline_does_not_alias_the_pulse_and_is_complex(self, qblox_model, phase):
        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        context = self._phase_context(enable_timeline_debug_info=True)
        pulse = np.full(4, 0.5)  # Real-valued input

        if phase:
            context.shift_phase(PhaseShift(drive_channel, phase))
        context.ledger(4, pulse)
        pulse[:] = 0.0  # Mutating the input must not change the recorded timeline

        timeline = context.create_package(drive_channel, 0, 0).timeline
        assert timeline.dtype == complex
        np.testing.assert_allclose(timeline[-4:], np.full(4, 0.5 * np.exp(1.0j * phase)))

    @pytest.mark.parametrize("qblox_model", [None], indirect=True)
    def test_phase_is_tracked_when_timeline_is_not_recorded(self, qblox_model):
        """The phase is tracked even when the timeline is not being recorded."""
        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        context = self._phase_context(enable_timeline_debug_info=False)

        context.shift_phase(PhaseShift(drive_channel, 0.5))
        context.ledger(4, np.ones(4, dtype=complex))

        assert context.phase == pytest.approx(0.5)
        assert context.create_package(drive_channel, 0, 0).timeline is None

    @pytest.mark.parametrize("qblox_model", [None], indirect=True)
    def test_timeline_is_empty_array_when_nothing_is_recorded(self, qblox_model):
        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        context = QbloxContext(ReadWriteResult(), {}, enable_timeline_debug_info=True)

        pkg = context.create_package(drive_channel, 0, 0)

        assert isinstance(pkg.timeline, np.ndarray)
        assert pkg.timeline.size == 0

    @pytest.mark.parametrize("qblox_model", [None], indirect=True)
    @pytest.mark.parametrize(
        "enable_timeline_debug_info, durations",
        [
            pytest.param(False, [8], id="not-recorded"),
            pytest.param(True, [8, "iter"], id="dynamic-duration"),
        ],
    )
    def test_timeline_is_none(self, qblox_model, enable_timeline_debug_info, durations):
        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        context = QbloxContext(
            ReadWriteResult(), {}, enable_timeline_debug_info=enable_timeline_debug_info
        )

        for duration in durations:
            context.ledger(duration)

        assert context.create_package(drive_channel, 0, 0).timeline is None

    @pytest.mark.parametrize("enable_timeline_debug_info", [True, False])
    def test_ledger_rejects_pulse_with_mismatched_size(self, enable_timeline_debug_info):
        context = QbloxContext(
            ReadWriteResult(), {}, enable_timeline_debug_info=enable_timeline_debug_info
        )

        with pytest.raises(AssertionError):
            context.ledger(4, np.ones(3, dtype=complex))


@pytest.mark.parametrize("qblox_model", [None], indirect=True)
@pytest.mark.parametrize("backend_type", [QbloxBackend1, QbloxBackend2])
def test_sequencer_allocation(qblox_model, backend_type):
    if backend_type == QbloxBackend1:
        middleend_pipeline = middleend_pipeline1(qblox_model, TARGET_DATA)
        backend_pipeline = backend_pipeline1()
    elif backend_type == QbloxBackend2:
        middleend_pipeline = middleend_pipeline2(qblox_model, TARGET_DATA)
        backend_pipeline = backend_pipeline2()
    else:
        raise ValueError(f"Expected QbloxBackend1 or QbloxBackend2, got {backend_type}")

    drive_channel = qblox_model.get_qubit(0).get_drive_channel()
    gaussian = Pulse(drive_channel, PulseShapeType.GAUSSIAN, width=100e-9, rise=1.0 / 5.0)
    builder = qblox_model.create_builder().add(gaussian)

    res_mgr = ResultManager()
    met_mgr = MetricsManager()
    middleend_pipeline.run(builder, res_mgr, met_mgr, enable_hw_averaging=True)
    backend = backend_type(model=qblox_model, pipeline=backend_pipeline)

    assert not backend.allocations


@pytest.mark.parametrize(
    "compiler_config, expected",
    [
        pytest.param(None, False, id="no-compiler-config"),
        pytest.param(SimpleNamespace(), False, id="flag-not-supported"),
        pytest.param(SimpleNamespace(enable_debug_info=False), False, id="debug-off"),
        pytest.param(SimpleNamespace(enable_debug_info=True), True, id="debug-on"),
    ],
)
def test_debug_info_enabled(compiler_config, expected):
    assert _debug_info_enabled(compiler_config) is expected


@pytest.mark.parametrize("qblox_model", [None], indirect=True)
@pytest.mark.parametrize("backend_type", [QbloxBackend1, QbloxBackend2])
@pytest.mark.parametrize(
    "debug",
    [pytest.param(False, id="debug-off"), pytest.param(None, id="no-compiler-config")],
)
def test_timeline_omitted_unless_debug_enabled(qblox_model, backend_type, debug):
    builder = qblox_model.create_builder()
    qubit = qblox_model.get_qubit(0)
    drive_channel = qubit.get_drive_channel()
    builder.add(
        Pulse(
            drive_channel,
            PulseShapeType.GAUSSIAN,
            width=100e-9,
            rise=1.0 / 5.0,
        )
    )

    executable = do_emit(
        qblox_model,
        backend_type,
        builder,
        debug=debug,
    )

    pkg = executable.programs[0].packages[drive_channel.full_id()]
    assert pkg.timeline is None


@pytest.mark.parametrize("qblox_model", [None], indirect=True)
@pytest.mark.parametrize("backend_type", [QbloxBackend1, QbloxBackend2])
def test_timeline_records_square_pulse_when_debug_enabled(qblox_model, backend_type):
    drive_channel = qblox_model.get_qubit(0).get_drive_channel()
    builder = qblox_model.create_builder()
    builder.add(PhaseShift(drive_channel, np.pi / 2))
    builder.add(
        Pulse(
            drive_channel,
            PulseShapeType.SQUARE,
            width=100e-9,
            amp=0.5,
            ignore_channel_scale=True,
        )
    )

    executable = do_emit(qblox_model, backend_type, builder, debug=True)

    pkg = executable.programs[0].packages[drive_channel.full_id()]
    assert isinstance(pkg.timeline, np.ndarray)
    samples = pkg.timeline[np.nonzero(pkg.timeline)]
    expected = (0.5 + drive_channel.bias) * np.exp(1j * np.pi / 2)
    assert samples.size == 100
    np.testing.assert_allclose(samples, expected)


def _square_measure_builder(qblox_model):
    qubit = qblox_model.get_qubit(0)
    measure_channel = qubit.get_measure_channel()
    builder = qblox_model.create_builder()
    builder.add(
        [
            MeasurePulse(
                measure_channel,
                PulseShapeType.SQUARE,
                width=1e-6,
                amp=0.25,
                ignore_channel_scale=True,
            ),
            Acquire(
                qubit.get_acquire_channel(),
                time=7.5e-7,
                delay=qubit.measure_acquire["delay"],
            ),
        ]
    )
    return builder, measure_channel


@pytest.mark.parametrize("qblox_model", [None], indirect=True)
@pytest.mark.parametrize("backend_type", [QbloxBackend1, QbloxBackend2])
@pytest.mark.parametrize(
    "debug",
    [pytest.param(False, id="debug-off"), pytest.param(None, id="no-compiler-config")],
)
def test_square_measure_pulse_samples_not_built_without_debug(
    qblox_model, backend_type, debug, mocker
):
    builder, measure_channel = _square_measure_builder(qblox_model)
    spy = mocker.spy(QbloxContext, "_evaluate_waveform")

    executable = do_emit(qblox_model, backend_type, builder, debug=debug)

    pkg = executable.programs[0].packages[measure_channel.full_id()]
    assert pkg.timeline is None
    assert f"set_awg_offs {int(0.25 * Q1ASM_DATA.max_offset)},0" in pkg.sequence.program
    measure_calls = [c for c in spy.call_args_list if isinstance(c.args[1], MeasurePulse)]
    assert measure_calls
    assert all(c.kwargs.get("num_samples") == 1 for c in measure_calls)


@pytest.mark.parametrize("qblox_model", [None], indirect=True)
@pytest.mark.parametrize("backend_type", [QbloxBackend1, QbloxBackend2])
def test_timeline_records_square_measure_pulse_when_debug_enabled(
    qblox_model, backend_type
):
    builder, measure_channel = _square_measure_builder(qblox_model)

    executable = do_emit(qblox_model, backend_type, builder, debug=True)

    pkg = executable.programs[0].packages[measure_channel.full_id()]
    assert isinstance(pkg.timeline, np.ndarray)
    samples = pkg.timeline[np.nonzero(pkg.timeline)]
    assert samples.size == 1000
    np.testing.assert_allclose(samples, 0.25 + measure_channel.bias)


@pytest.mark.parametrize("qblox_model", [None], indirect=True)
class TestQbloxBackend1:
    def test_very_long_pulse(self, qblox_model):
        q0 = qblox_model.get_qubit(0)
        drive_channel = q0.get_drive_channel()
        builder = get_builder(qblox_model).pulse(
            drive_channel, PulseShapeType.SOFT_SQUARE, amp=0.1, width=1e-5, rise=1e-8
        )

        with pytest.raises(ValueError, match="No more waveform memory left"):
            do_emit(qblox_model, QbloxBackend1, builder)

    def test_play_gaussian(self, qblox_model):
        width = 100e-9
        rise = 1.0 / 5.0
        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        gaussian = Pulse(drive_channel, PulseShapeType.GAUSSIAN, width=width, rise=rise)
        builder = qblox_model.create_builder().add(gaussian)

        executable = do_emit(qblox_model, QbloxBackend1, builder)
        repeat = next(
            (inst for inst in builder.instructions if isinstance(inst, Repeat)), None
        )
        assert len(executable.acquires) == 0
        assert len(executable.assigns) == 0
        assert len(executable.returns) == 0
        assert executable.shots == (repeat.repeat_count if repeat else None)
        assert executable.calibration_id == qblox_model.calibration_id
        assert len(executable.programs) > 0
        for program in executable.programs:
            assert len(program.packages) == 1
            pulse_channel_id, pkg = next(iter(program.packages.items()))
            assert pulse_channel_id == drive_channel.full_id()
            assert f"GAUSSIAN_{hash(gaussian)}_I" in pkg.sequence.waveforms
            assert f"GAUSSIAN_{hash(gaussian)}_Q" in pkg.sequence.waveforms
            assert "play 0,1,100" in pkg.sequence.program

    @pytest.mark.parametrize("start_width, end_width", [(0, 100e-9), (50e-9, 100e-9)])
    def test_play_square(self, qblox_model, start_width, end_width):
        amp = 1
        num_points = 50

        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        time, _ = np.linspace(start_width, end_width, num_points, retstep=True)
        builder = get_builder(qblox_model)
        builder.sweep(SweepValue("t", time))
        builder.pulse(drive_channel, PulseShapeType.SQUARE, width=Variable("t"), amp=amp)
        executable = do_emit(qblox_model, QbloxBackend1, builder)
        assert len(executable.programs) == num_points

        ignored_indices = np.squeeze(
            np.where(time < CONTROL_SEQUENCER_DATA.grid_time * 1e-9)
        )
        non_ignored_indices = np.squeeze(
            np.where(time >= CONTROL_SEQUENCER_DATA.grid_time * 1e-9)
        )
        assert len(ignored_indices) + len(non_ignored_indices) == num_points

        for i in ignored_indices:
            assert not executable.programs[i].packages

        for i in non_ignored_indices:
            assert len(executable.programs[i].packages) == 1
            pkg = executable.programs[i].packages[drive_channel.full_id()]
            assert not pkg.sequence.waveforms
            assert f"set_awg_offs {Q1ASM_DATA.max_offset},0" in pkg.sequence.program
            assert "set_awg_offs 0,0" in pkg.sequence.program

    def test_phase_and_frequency_shift(self, qblox_model):
        amp = 1
        rise = 1.0 / 3.0
        phase = 0.72
        frequency = 500

        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        builder = (
            qblox_model.create_builder()
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .pulse(drive_channel, PulseShapeType.GAUSSIAN, width=150e-9, rise=rise)
            .phase_shift(drive_channel, phase)
            .frequency_shift(drive_channel, frequency)
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .pulse(drive_channel, PulseShapeType.GAUSSIAN, width=150e-9, rise=rise)
        )
        executable = do_emit(qblox_model, QbloxBackend1, builder)
        for program in executable.programs:
            for pkg in program.packages.values():
                expected_phase = QbloxLegalisationPass.phase_as_steps(phase)
                assert f"set_ph_delta {expected_phase}" in pkg.sequence.program
                expected_freq = QbloxLegalisationPass.freq_as_steps(
                    drive_channel.baseband_if_frequency + frequency
                )
                assert f"set_freq {expected_freq}" in pkg.sequence.program

    def test_measure_acquire(self, qblox_model):
        qubit = qblox_model.get_qubit(0)
        acquire_channel = qubit.get_acquire_channel()
        measure_channel = qubit.get_measure_channel()
        assert measure_channel == acquire_channel

        time = 7.5e-6
        i_offs_steps = int(qubit.pulse_measure["amp"] * Q1ASM_DATA.max_offset)
        delay = qubit.measure_acquire["delay"]

        builder = qblox_model.create_builder()
        builder.acquire(acquire_channel, time, delay=delay)
        executable = do_emit(qblox_model, QbloxBackend1, builder)
        for program in executable.programs:
            assert len(program.packages) == 0

        builder = qblox_model.create_builder()
        builder.add(
            [
                MeasurePulse(measure_channel, **qubit.pulse_measure),
                Acquire(acquire_channel, time=time, delay=delay),
            ]
        )
        executable = do_emit(qblox_model, QbloxBackend1, builder)
        remaining_width = int(qubit.pulse_measure["width"] * 1e9) - int(delay * 1e9)

        pattern = (
            rf"set_awg_offs {i_offs_steps},0\nupd_param {int(delay * 1e9)}\n"
            rf"acquire 0,R\d{{1,2}},{remaining_width}\nset_awg_offs 0,0\nupd_param 4"
        )
        for program in executable.programs:
            assert len(program.packages) == 1
            pkg = next(iter(program.packages.values()))
            assert not pkg.sequence.waveforms
            assert qubit.pulse_measure["shape"] == PulseShapeType.SQUARE
            assert re.search(pattern, pkg.sequence.program, re.MULTILINE)

            channel = acquire_channel.physical_channel
            assert isinstance(channel, QbloxPhysicalChannel)
            assert isinstance(channel.baseband, QbloxPhysicalBaseband)
            assert len(channel.config.sequencers) > 0
            assert pkg.seq_config.square_weight_acq.integration_length == int(time * 1e9)

    @pytest.mark.parametrize(
        ("width_seconds", "context"),
        [
            (np.full(10, 5e-6), nullcontext()),
            (
                np.full(
                    5,
                    (CONTROL_SEQUENCER_DATA.max_sample_size_waveforms / 2) * 1e-9,
                ),
                nullcontext(),
            ),  # 8.192e-6
            (
                np.full(1, CONTROL_SEQUENCER_DATA.max_sample_size_waveforms * 1e-9),
                pytest.raises(ValueError, match="No more waveform memory left"),
            ),  # 16.384e-6
        ],
    )
    def test_waveform_caching(self, qblox_model, width_seconds, context):
        width_samples = np.astype(width_seconds * 1e9, int)
        assert 2 * np.sum(width_samples) > CONTROL_SEQUENCER_DATA.max_sample_size_waveforms

        drive_channel = qblox_model.get_qubit(0).get_drive_channel()
        builder = qblox_model.create_builder()
        for val in width_seconds:
            builder.pulse(drive_channel, PulseShapeType.GAUSSIAN, width=val, rise=1.0 / 5.0)

        with context:
            executable = do_emit(qblox_model, QbloxBackend1, builder)
            for program in executable.programs:
                assert len(program.packages) == 1

    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    def test_resonator_spect(self, qblox_model, qubit_indices):
        qubit_indices = [0, 1]
        builder = resonator_spect(qblox_model, qubit_indices)
        executable = do_emit(qblox_model, QbloxBackend1, builder)
        for program in executable.programs:
            assert len(program.packages) == len(qubit_indices)

            for index in qubit_indices:
                qubit = qblox_model.get_qubit(index)
                acquire_channel = qubit.get_acquire_channel()
                acquire_pkg = next(
                    (
                        pkg
                        for pulse_channel_id, pkg in program.packages.items()
                        if pulse_channel_id == acquire_channel.full_id()
                    )
                )
                assert acquire_pkg.sequence.acquisitions

                measure_pulse = next(
                    inst
                    for inst in builder.instructions
                    if isinstance(inst, MeasurePulse)
                    and acquire_channel in inst.quantum_targets
                )

                if measure_pulse.shape == PulseShapeType.SQUARE:
                    assert not acquire_pkg.sequence.waveforms
                    assert "play" not in acquire_pkg.sequence.program
                    assert "set_awg_offs" in acquire_pkg.sequence.program
                    assert "upd_param" in acquire_pkg.sequence.program
                else:
                    assert acquire_pkg.sequence.waveforms
                    assert "play" in acquire_pkg.sequence.program
                    assert "set_awg_offs" not in acquire_pkg.sequence.program
                    assert "upd_param" not in acquire_pkg.sequence.program

    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    def test_qubit_spect(self, qblox_model, qubit_indices):
        builder = qubit_spect(qblox_model, qubit_indices)
        executable = do_emit(qblox_model, QbloxBackend1, builder)
        for program in executable.programs:
            assert len(program.packages) == 2 * len(qubit_indices)

            for index in qubit_indices:
                qubit = qblox_model.get_qubit(index)

                # Drive
                drive_channel = qubit.get_drive_channel()
                drive_pkg = next(
                    (
                        pkg
                        for pulse_channel_id, pkg in program.packages.items()
                        if pulse_channel_id == drive_channel.full_id()
                    )
                )
                drive_pulse = next(
                    inst
                    for inst in builder.instructions
                    if isinstance(inst, Pulse) and drive_channel in inst.quantum_targets
                )
                if drive_pulse.shape == PulseShapeType.SQUARE:
                    assert not drive_pkg.sequence.waveforms
                    assert "play" not in drive_pkg.sequence.program
                    assert "set_awg_offs" in drive_pkg.sequence.program
                    assert "upd_param" in drive_pkg.sequence.program
                else:
                    assert drive_pkg.sequence.waveforms
                    assert "play" in drive_pkg.sequence.program
                    assert "set_awg_offs" not in drive_pkg.sequence.program
                    assert "upd_param" not in drive_pkg.sequence.program

                # Readout
                acquire_channel = qubit.get_acquire_channel()
                acquire_pkg = next(
                    (
                        pkg
                        for pulse_channel_id, pkg in program.packages.items()
                        if pulse_channel_id == acquire_channel.full_id()
                    )
                )
                assert acquire_pkg.sequence.acquisitions
                measure_pulse = next(
                    inst
                    for inst in builder.instructions
                    if isinstance(inst, MeasurePulse)
                    and acquire_channel in inst.quantum_targets
                )

                if measure_pulse.shape == PulseShapeType.SQUARE:
                    assert not acquire_pkg.sequence.waveforms
                    assert "play" not in acquire_pkg.sequence.program
                    assert "set_awg_offs" in acquire_pkg.sequence.program
                    assert "upd_param" in acquire_pkg.sequence.program
                else:
                    assert acquire_pkg.sequence.waveforms
                    assert "play" in acquire_pkg.sequence.program
                    assert "set_awg_offs" not in acquire_pkg.sequence.program
                    assert "upd_param" not in acquire_pkg.sequence.program

    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    @pytest.mark.parametrize("enable_weights", [True, False])
    @pytest.mark.parametrize("do_X", [True, False])
    def test_scope_acquisition(self, qblox_model, qubit_indices, enable_weights, do_X):
        builder = measure_acquire(qblox_model, qubit_indices, do_X=do_X)

        executable = do_emit(qblox_model, QbloxBackend1, builder)
        for program in executable.programs:
            if do_X:
                assert len(program.packages) == 2 * len(qubit_indices)
            else:
                assert len(program.packages) == len(qubit_indices)

            acquire_pkg = next(pkg for pkg in program.packages.values())
            assert "weighed_acquire" not in acquire_pkg.sequence.program

        builder = measure_acquire(qblox_model, qubit_indices, enable_weights=enable_weights)
        executable = do_emit(qblox_model, QbloxBackend1, builder)
        for program in executable.programs:
            for acquire_pkg in program.packages.values():
                if enable_weights:
                    assert "acquire_weighed" in acquire_pkg.sequence.program
                else:
                    assert "acquire" in acquire_pkg.sequence.program

    @pytest.mark.parametrize("qubit_indices", [[0]])
    @pytest.mark.parametrize("num_acquires", [1, 2, 3])
    def test_multi_readout(self, qblox_model, qubit_indices, num_acquires):
        builder = multi_readout(
            qblox_model, qubit_indices, do_X=False, num_acquires=num_acquires
        )
        executable = do_emit(qblox_model, QbloxBackend1, builder)
        for program in executable.programs:
            assert len(program.packages) == len(qubit_indices)

            for index in qubit_indices:
                qubit = qblox_model.get_qubit(index)
                measure_channel = qubit.get_measure_channel()

                # Readout
                measure_pkg = next(
                    (
                        pkg
                        for pulse_channel_id, pkg in program.packages.items()
                        if pulse_channel_id == measure_channel.full_id()
                    )
                )
                assert measure_pkg.sequence.acquisitions
                assert measure_pkg.sequence.program.count("acquire") == 2 * num_acquires

        builder = multi_readout(
            qblox_model, qubit_indices, do_X=True, num_acquires=num_acquires
        )
        with pytest.raises(
            ValueError, match="Mid-circuit measurements currently unable to be used"
        ):
            do_emit(qblox_model, QbloxBackend1, builder)

        old_value = qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT
        try:
            qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT = False
            executable = do_emit(qblox_model, QbloxBackend1, builder)
            for program in executable.programs:
                assert len(program.packages) == 2 * len(qubit_indices)

                for index in qubit_indices:
                    qubit = qblox_model.get_qubit(index)
                    measure_channel = qubit.get_measure_channel()

                    # Readout
                    measure_pkg = next(
                        (
                            pkg
                            for pulse_channel_id, pkg in program.packages.items()
                            if pulse_channel_id == measure_channel.full_id()
                        )
                    )
                    assert measure_pkg.sequence.acquisitions
                    assert measure_pkg.sequence.program.count("acquire") == 2 * num_acquires
        finally:
            qatconfig.INSTRUCTION_VALIDATION.NO_MID_CIRCUIT_MEASUREMENT = old_value

    @pytest.mark.parametrize("pulse_width", [0, CONTROL_SEQUENCER_DATA.grid_time, 1e3, 1e6])
    @pytest.mark.parametrize(
        "delay_width",
        [
            0,
            CONTROL_SEQUENCER_DATA.grid_time,
            100,
            1e3 - CONTROL_SEQUENCER_DATA.grid_time,
            1e3,
        ],
    )
    def test_measure_acquire_operands(self, copied_qblox_model, pulse_width, delay_width):
        qubit_indices = [0]
        for index in qubit_indices:
            qubit = copied_qblox_model.get_qubit(index)
            qubit.pulse_measure["width"] = pulse_width * 1e-9
            qubit.measure_acquire["delay"] = delay_width * 1e-9

        builder = measure_acquire(copied_qblox_model, qubit_indices)

        effective_width = max(
            min(pulse_width, delay_width), CONTROL_SEQUENCER_DATA.grid_time
        )
        if 0 < pulse_width < effective_width + CONTROL_SEQUENCER_DATA.grid_time:
            if pulse_width < delay_width + CONTROL_SEQUENCER_DATA.grid_time:
                with pytest.raises(
                    ValueError, match=r"Expected pulse width >= delay width \+ 4 ns"
                ):
                    do_emit(copied_qblox_model, QbloxBackend1, builder)
            else:
                with pytest.raises(
                    ValueError, match=r"Remaining width after delay must be at least 4 ns"
                ):
                    do_emit(copied_qblox_model, QbloxBackend1, builder)

        else:
            executable = do_emit(copied_qblox_model, QbloxBackend1, builder)
            for program in executable.programs:
                packages = program.packages
                if pulse_width < CONTROL_SEQUENCER_DATA.grid_time:
                    assert len(packages) == 0
                else:
                    assert len(packages) == len(qubit_indices)
                    for pkg in packages.values():
                        program = pkg.sequence.program
                        quotient = effective_width // Q1ASM_DATA.max_wait_time
                        remainder = effective_width % Q1ASM_DATA.max_wait_time
                        if quotient > 1:
                            assert f"wait {remainder}" in program

    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    def test_discrimination(self, qblox_model, qubit_indices):
        qubits = [qblox_model.get_qubit(index) for index in qubit_indices]

        builder = discrimination(qblox_model, qubit_indices)
        acquire = next(inst for inst in builder.instructions if isinstance(inst, Acquire))
        acq_width = int(calculate_duration(acquire))

        executable = do_emit(qblox_model, QbloxBackend1, builder)
        for program in executable.programs:
            assert len(program.packages) == len(qubits)
            qub_pkg_zip = [
                (qub, pkg)
                for (qub, (pulse_channel_id, pkg)) in zip(
                    qubits, program.packages.items(), strict=True
                )
                if qub.get_measure_channel()
                == qblox_model.get_pulse_channel_from_id(pulse_channel_id)
            ]

            for qubit, measure_pkg in qub_pkg_zip:
                A, B = qubit.mean_z_map_args[0], qubit.mean_z_map_args[1]
                mean_g = (1 - B) / A
                mean_e = (-1 - B) / A

                rotation = np.mod(-np.angle(mean_e - mean_g), 2 * np.pi)
                threshold = (np.exp(1j * rotation) * (mean_e + mean_g)).real / 2

                assert measure_pkg.seq_config.thresholded_acq.rotation == np.rad2deg(
                    rotation
                )
                assert (
                    measure_pkg.seq_config.thresholded_acq.threshold
                    == acq_width * threshold
                )

    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    def test_timeline_visualisation(self, qblox_model, qubit_indices, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda: None)

        width = 100e-9
        rise = 1.0 / 5.0

        builder = qblox_model.create_builder()
        for index in qubit_indices:
            qubit = qblox_model.get_qubit(index)
            drive_channel = qubit.get_drive_channel()
            gaussian = Pulse(drive_channel, PulseShapeType.GAUSSIAN, width=width, rise=rise)
            builder.add(gaussian)

        executable = do_emit(qblox_model, QbloxBackend1, builder, debug=True)
        for program in executable.programs:
            assert len(program.packages) == len(qubit_indices)
            for pkg in program.packages.values():
                assert pkg.timeline is not None
                assert isinstance(pkg.timeline, np.ndarray)
                assert pkg.timeline.size > 0
                assert not np.any(np.equal(pkg.timeline, None))
                assert not np.any(np.isnan(pkg.timeline))
            plot_program(program)

    def test_driver_fw_version(self, qblox_model):
        qubit_indices = [0]
        builder = resonator_spect(qblox_model, qubit_indices)
        executable = do_emit(qblox_model, QbloxBackend1, builder)

        for program in executable.programs:
            assert program.driver_version == TARGET_DATA.driver_version
            assert program.fw_version == TARGET_DATA.fw_version


@pytest.mark.parametrize("qblox_model", [None], indirect=True)
class TestQbloxBackend2:
    def test_prologue_epilogue(self, qblox_model):
        builder = empty(qblox_model)
        assert len(builder.instructions) == 3

        executable = do_emit(
            qblox_model,
            QbloxBackend2,
            builder,
            ignore_empty=False,
            debug=True,
        )
        assert len(executable.programs) == 1
        for program in executable.programs:
            assert len(program.packages) == 2

            for pkg in program.packages.values():
                assert pkg.timeline.size == 4
                assert not pkg.sequence.waveforms
                assert not pkg.sequence.acquisitions
                assert not pkg.sequence.weights

                assert "set_mrk 3\nset_latch_en 1,4\nupd_param 4" in pkg.sequence.program
                assert "stop" in pkg.sequence.program

    @pytest.mark.parametrize("num_points", [1, 10, 100])
    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    def test_resonator_spect(self, qblox_model, num_points, qubit_indices):
        builder = resonator_spect(qblox_model, qubit_indices, num_points)
        sweeps = [inst for inst in builder.instructions if isinstance(inst, Sweep)]
        executable = do_emit(qblox_model, QbloxBackend2, builder)
        assert len(executable.programs) == 1
        for program in executable.programs:
            assert len(program.packages) == len(qubit_indices)

            for index in qubit_indices:
                qubit = qblox_model.get_qubit(index)
                acquire_channel = qubit.get_acquire_channel()
                acquire_pkg = next(
                    (
                        pkg
                        for pulse_channel_id, pkg in program.packages.items()
                        if pulse_channel_id == acquire_channel.full_id()
                    )
                )
                measure_pulse = next(
                    inst
                    for inst in builder.instructions
                    if isinstance(inst, MeasurePulse)
                    and acquire_channel in inst.quantum_targets
                )

                assert acquire_pkg.sequence.acquisitions
                for sweep in sweeps:
                    assert f"{hash(sweep)}_0" in acquire_pkg.sequence.program

                assert "wait_sync" in acquire_pkg.sequence.program

                if measure_pulse.shape == PulseShapeType.SQUARE:
                    assert not acquire_pkg.sequence.waveforms
                    assert "play" not in acquire_pkg.sequence.program
                    assert "set_awg_offs" in acquire_pkg.sequence.program
                    assert "upd_param" in acquire_pkg.sequence.program
                else:
                    assert acquire_pkg.sequence.waveforms
                    assert "play" in acquire_pkg.sequence.program
                    assert "set_awg_offs" not in acquire_pkg.sequence.program
                    assert "upd_param" not in acquire_pkg.sequence.program

    @pytest.mark.parametrize("num_points", [1, 10, 100])
    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    def test_qubit_spect(self, qblox_model, num_points, qubit_indices):
        builder = qubit_spect(qblox_model, qubit_indices, num_points)

        remaining, static_dus = partition(
            lambda inst: (
                isinstance(inst, DeviceUpdate) and not isinstance(inst.value, Variable)
            ),
            builder.instructions,
        )
        remaining, static_dus = list(remaining), list(static_dus)

        assert len(static_dus) == len(qubit_indices)
        executable = do_emit(qblox_model, QbloxBackend2, builder)
        assert len(executable.programs) == 1

        for inst in static_dus:
            assert inst not in builder.instructions

        for program in executable.programs:
            assert len(program.packages) == 2 * len(qubit_indices)

            for index in qubit_indices:
                qubit = qblox_model.get_qubit(index)
                drive_channel = qubit.get_drive_channel()
                acquire_channel = qubit.get_acquire_channel()

                # Drive
                drive_pkg = next(
                    (
                        pkg
                        for pulse_channel_id, pkg in program.packages.items()
                        if pulse_channel_id == drive_channel.full_id()
                    )
                )
                drive_pulse = next(
                    inst
                    for inst in builder.instructions
                    if isinstance(inst, Pulse) and drive_channel in inst.quantum_targets
                )

                assert not drive_pkg.sequence.acquisitions

                assert "wait_sync" in drive_pkg.sequence.program

                if drive_pulse.shape == PulseShapeType.SQUARE:
                    assert not drive_pkg.sequence.waveforms
                    assert "play" not in drive_pkg.sequence.program
                    assert "set_awg_offs" in drive_pkg.sequence.program
                    assert "upd_param" in drive_pkg.sequence.program
                else:
                    assert drive_pkg.sequence.waveforms
                    assert "play" in drive_pkg.sequence.program
                    assert "set_awg_offs" not in drive_pkg.sequence.program
                    assert "upd_param" not in drive_pkg.sequence.program

                # Readout
                acquire_pkg = next(
                    (
                        pkg
                        for pulse_channel_id, pkg in program.packages.items()
                        if pulse_channel_id == acquire_channel.full_id()
                    )
                )
                measure_pulse = next(
                    inst
                    for inst in builder.instructions
                    if isinstance(inst, MeasurePulse)
                    and acquire_channel in inst.quantum_targets
                )

                assert acquire_pkg.sequence.acquisitions

                assert "wait_sync" in acquire_pkg.sequence.program

                if measure_pulse.shape == PulseShapeType.SQUARE:
                    assert not acquire_pkg.sequence.waveforms
                    assert "play" not in acquire_pkg.sequence.program
                    assert "set_awg_offs" in acquire_pkg.sequence.program
                    assert "upd_param" in acquire_pkg.sequence.program
                else:
                    assert acquire_pkg.sequence.waveforms
                    assert "play" in acquire_pkg.sequence.program
                    assert "set_awg_offs" not in acquire_pkg.sequence.program
                    assert "upd_param" not in acquire_pkg.sequence.program

    @pytest.mark.parametrize("num_points", [100])
    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_delay_iteration(self, qblox_model, num_points, qubit_indices):
        builder = delay_iteration(qblox_model, qubit_indices, num_points)
        executable = do_emit(qblox_model, QbloxBackend2, builder)
        assert len(executable.programs) == 1
        for program in executable.programs:
            assert len(program.packages) == 2 * len(qubit_indices)

    @pytest.mark.parametrize("num_points", [10])
    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_pulse_width_iteration(self, qblox_model, num_points, qubit_indices):
        builder = pulse_width_iteration(qblox_model, qubit_indices, num_points)
        executable = do_emit(qblox_model, QbloxBackend2, builder)
        assert len(executable.programs) == 1
        for program in executable.programs:
            assert len(program.packages) == 2 * len(qubit_indices)

    @pytest.mark.parametrize("num_points", [10])
    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_pulse_amplitude_iteration(self, qblox_model, num_points, qubit_indices):
        builder = pulse_amplitude_iteration(qblox_model, qubit_indices, num_points)
        executable = do_emit(qblox_model, QbloxBackend2, builder)
        assert len(executable.programs) == 1
        for program in executable.programs:
            assert len(program.packages) == 2 * len(qubit_indices)

    @pytest.mark.parametrize("num_points", [10])
    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_pulse_amplitude_iteration_records_silent_timeline_when_debug_enabled(
        self, qblox_model, num_points, qubit_indices
    ):
        builder = pulse_amplitude_iteration(qblox_model, qubit_indices, num_points)
        executable = do_emit(qblox_model, QbloxBackend2, builder, debug=True)

        drive_channel = qblox_model.get_qubit(qubit_indices[0]).get_drive_channel()
        pkg = executable.programs[0].packages[drive_channel.full_id()]
        # The swept amplitude lives in a register, so the pulse is recorded as zeros.
        assert isinstance(pkg.timeline, np.ndarray)
        assert pkg.timeline.size > 0
        assert np.count_nonzero(pkg.timeline) == 0

    @pytest.mark.parametrize("num_points", [10])
    @pytest.mark.parametrize("qubit_indices", [[0]])
    def test_time_and_phase_iteration(self, qblox_model, num_points, qubit_indices):
        builder = time_and_phase_iteration(qblox_model, qubit_indices, num_points)
        executable = do_emit(qblox_model, QbloxBackend2, builder)
        assert len(executable.programs) == 1
        for program in executable.programs:
            assert len(program.packages) == 2 * len(qubit_indices)

    @pytest.mark.parametrize("num_points", [10])
    @pytest.mark.parametrize("qubit_indices", [[0], [0, 1]])
    @pytest.mark.parametrize("do_X", [True, False])
    def test_timeline_visualisation(
        self, qblox_model, num_points, qubit_indices, do_X, monkeypatch
    ):
        monkeypatch.setattr(plt, "show", lambda: None)

        # Static settings
        builder = measure_acquire(qblox_model, qubit_indices, do_X)
        executable = do_emit(qblox_model, QbloxBackend2, builder, debug=True)
        for program in executable.programs:
            for pkg in program.packages.values():
                assert pkg.timeline is not None
                assert isinstance(pkg.timeline, np.ndarray)
                assert pkg.timeline.size > 0
                assert not np.any(np.equal(pkg.timeline, None))
                assert not np.any(np.isnan(pkg.timeline))
            plot_program(program)

        # Dynamic settings
        builder = time_and_phase_iteration(qblox_model, qubit_indices, num_points)
        executable = do_emit(qblox_model, QbloxBackend2, builder, debug=True)

        default_pulse_channel_ids = [
            qblox_model.get_qubit(index).get_default_pulse_channel().full_id()
            for index in qubit_indices
        ]

        for program in executable.programs:
            for pkg in program.packages.values():
                if pkg.pulse_channel_id in default_pulse_channel_ids:
                    assert pkg.timeline is None
                else:
                    assert pkg.timeline is not None
                    assert isinstance(pkg.timeline, np.ndarray)
                    assert pkg.timeline.size > 0
                    assert not np.any(np.equal(pkg.timeline, None))
                    assert not np.any(np.isnan(pkg.timeline))

            plot_program(program)


# Legacy model compares by identity, so compare copies of its pydantic configs.
def _config_snapshot(model):
    return {
        pc.id: pc.config.model_copy(deep=True)
        for pc in model.physical_channels.values()
        if isinstance(pc, QbloxPhysicalChannel)
    }


@pytest.mark.parametrize("qblox_model", [None], indirect=True)
@pytest.mark.parametrize("backend_type", [QbloxBackend1, QbloxBackend2])
@pytest.mark.parametrize("builder_func", [qubit_spect, resonator_spect, measure_acquire])
def test_compilation_leaves_model_config_unchanged(
    copied_qblox_model, backend_type, builder_func
):
    model = copied_qblox_model
    # Sentinels for values compilation overwrites, so any write to the model shows.
    for channel in model.physical_channels.values():
        if isinstance(channel, QbloxPhysicalChannel):
            channel.config.module.scope_acq.sequencer_select = 7
            for seq_config in channel.config.sequencers.values():
                seq_config.nco.freq = 1.0
    snapshot = _config_snapshot(model)
    assert snapshot

    do_emit(model, backend_type, builder_func(model, [0, 1]))

    assert _config_snapshot(model) == snapshot


@pytest.mark.parametrize(
    "lo, connections, lanes",
    [
        (LoConfig(out0_in0_en=True), {"out0"}, {"out0_in0"}),
        (LoConfig(out0_in0_en=True), {"in0"}, {"out0_in0"}),
        (LoConfig(out0_in0_en=True), {"out1"}, {"out1_in1"}),
        (LoConfig(out0_in0_en=True), {"in1"}, {"out1_in1"}),
        (LoConfig(out0_in0_en=True), {"out0", "in0"}, {"out0_in0"}),
        (LoConfig(out0_in0_en=True), {"out2", "out5"}, {"out2", "out5"}),
        (
            LoConfig(out0_en=True, out1_en=True),
            {"out0", "out1"},
            {"out0", "out1"},
        ),
        (LoConfig(), {"out0"}, set()),
        (LoConfig(), {"out1"}, {"out1_in1"}),
    ],
)
def test_get_lo_lanes(lo, connections, lanes):
    assert _get_lo_lanes(lo, connections) == lanes


@pytest.mark.parametrize(
    "instrument, slot, connections, lo",
    [
        ("qrc", 1, ["in0"], LoConfig(out0_in0_en=True, out0_in0_freq=1e9)),
        ("qrc", 2, ["out0"], LoConfig(out0_in0_en=True, out0_in0_freq=2e9)),
        ("other_qrc", 1, ["out0"], LoConfig(out0_in0_en=True, out0_in0_freq=2e9)),
        ("qrc", 1, ["out2"], LoConfig(out0_in0_en=True, out2_freq=2e9)),
    ],
    ids=["same_freq", "different_slot", "different_instrument", "different_lane"],
)
def test_check_shared_lo_conflicts_allows(instrument, slot, connections, lo):
    first = QbloxPackage(
        pulse_channel_id="a",
        instrument_id="qrc",
        slot_idx=1,
        mod_config=ModuleConfig(lo=LoConfig(out0_in0_en=True, out0_in0_freq=1e9)),
        seq_config=SequencerConfig(connection=ConnectionConfig(bulk_value=["out0"])),
    )
    second = QbloxPackage(
        pulse_channel_id="b",
        instrument_id=instrument,
        slot_idx=slot,
        mod_config=ModuleConfig(lo=lo),
        seq_config=SequencerConfig(connection=ConnectionConfig(bulk_value=connections)),
    )

    _check_shared_lo_conflicts([first, second])


def test_check_shared_lo_conflicts_raises_for_different_freqs_on_one_lane():
    first = QbloxPackage(
        pulse_channel_id="a",
        instrument_id="qrc",
        slot_idx=1,
        mod_config=ModuleConfig(lo=LoConfig(out0_in0_en=True, out0_in0_freq=1e9)),
        seq_config=SequencerConfig(connection=ConnectionConfig(bulk_value=["out0"])),
    )
    second = QbloxPackage(
        pulse_channel_id="b",
        instrument_id="qrc",
        slot_idx=1,
        mod_config=ModuleConfig(lo=LoConfig(out0_in0_en=True, out0_in0_freq=2e9)),
        seq_config=SequencerConfig(connection=ConnectionConfig(bulk_value=["in0"])),
    )

    with pytest.raises(ValueError, match="can only have one frequency"):
        _check_shared_lo_conflicts([first, second])


@pytest.mark.parametrize("qblox_model", [None], indirect=True)
@pytest.mark.parametrize("backend_type", [QbloxBackend1, QbloxBackend2])
@pytest.mark.parametrize(
    "connections, field", [(["out0"], "out0_in0_freq"), (["out1"], "out1_in1_freq")]
)
def test_qrc_drive_on_out0_or_out1_gets_shared_lo(
    copied_qblox_model, backend_type, connections, field
):
    model = copied_qblox_model
    drive = model.get_qubit(0).get_drive_channel()
    for seq_config in drive.physical_channel.config.sequencers.values():
        seq_config.connection.bulk_value = connections
    builder = model.create_builder().add(
        Pulse(drive, PulseShapeType.GAUSSIAN, width=100e-9, rise=1.0 / 5.0)
    )

    executable = do_emit(model, backend_type, builder)

    package = next(
        pkg
        for program in executable.programs
        for pkg in program.packages.values()
        if pkg.pulse_channel_id == drive.full_id()
    )
    expected = TILegalisationPass.decompose_freq(drive.frequency, drive)[0]
    assert getattr(package.mod_config.lo, field) == expected


@pytest.mark.parametrize("qblox_model", [None], indirect=True)
@pytest.mark.parametrize("backend_type", [QbloxBackend1, QbloxBackend2])
def test_shared_lo_conflict_fails_compilation(copied_qblox_model, backend_type):
    model = copied_qblox_model
    drive = model.get_qubit(0).get_drive_channel()
    for seq_config in drive.physical_channel.config.sequencers.values():
        seq_config.connection.bulk_value = ["out0"]
    builder = model.create_builder().add(
        Pulse(drive, PulseShapeType.GAUSSIAN, width=100e-9, rise=1.0 / 5.0)
    )
    builder.add(measure_acquire(model, [0]))

    with pytest.raises(ValueError, match="can only have one frequency"):
        do_emit(model, backend_type, builder)
