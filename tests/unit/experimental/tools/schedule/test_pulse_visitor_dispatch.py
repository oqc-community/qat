# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from types import SimpleNamespace

import numpy as np
import pytest
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import IndexType, ModuleOp, StringAttr
from xdsl.ir import Block

from qat.experimental.dialect.pulse.ir.attributes import (
    FrequencyAttr,
    PhaseAttr,
    SampledWaveformAttr,
    TimeAttr,
)
from qat.experimental.dialect.pulse.ir.interfaces import IsAnalyticalWaveformInterface
from qat.experimental.dialect.pulse.ir.ops import (
    AcquireOp,
    AddOp,
    ConstantOp,
    CreateFrameOp,
    MixOp,
    PhaseOp,
    PhaseSetOp,
    PhaseShiftOp,
    PulseOp,
    StartContinuousWaveformOp,
    StopContinuousWaveformOp,
    SubOp,
    SynchronizeOp,
    WaitOp,
)
from qat.experimental.dialect.pulse.ir.schedule import (
    PulseScheduleVisitor,
    build_pulse_schedule,
)
from qat.experimental.dialect.pulse.ir.types import WaveformType
from qat.experimental.tools.schedule import ScheduleTracker


def test_pulse_schedule_visitor_dispatches_scheduling_operations(mocker):
    schedule_tracker = ScheduleTracker()
    schedule_visitor = PulseScheduleVisitor(schedule_tracker)
    schedule_visitor._scalar = lambda value: 1.0
    registry = PulseScheduleVisitor.__dict__["visit"].dispatcher.registry
    frame = object()
    phased_frame = object()
    waited_frame = object()
    acquired_frame = object()
    other_frame = object()
    synchronised_frame = object()

    registry[CreateFrameOp](
        schedule_visitor,
        SimpleNamespace(
            frequency=SimpleNamespace(),
            port=SimpleNamespace(data="q0"),
            result=frame,
        ),
    )
    registry[PhaseOp](
        schedule_visitor,
        SimpleNamespace(
            frame=frame,
            phase=SimpleNamespace(),
            result=phased_frame,
            name=PhaseSetOp.name,
        ),
    )
    registry[WaitOp](
        schedule_visitor,
        SimpleNamespace(
            frame=phased_frame,
            duration=SimpleNamespace(),
            result=waited_frame,
        ),
    )
    registry[AcquireOp](
        schedule_visitor,
        SimpleNamespace(
            frame=waited_frame,
            duration=SimpleNamespace(),
            label=None,
            frame_result=acquired_frame,
        ),
    )
    registry[CreateFrameOp](
        schedule_visitor,
        SimpleNamespace(
            frequency=SimpleNamespace(),
            port=SimpleNamespace(data="q1"),
            result=other_frame,
        ),
    )
    registry[SynchronizeOp](
        schedule_visitor,
        SimpleNamespace(
            frames=[acquired_frame, other_frame],
            results=[synchronised_frame, object()],
        ),
    )

    schedule_visitor._waveforms["waveform"] = (2.0, np.array([1.0 + 0.0j]))
    registry[PulseOp](
        schedule_visitor,
        SimpleNamespace(
            frame=synchronised_frame,
            waveform="waveform",
            result=object(),
        ),
    )

    assert schedule_tracker.timelines["q0"][-1].label == "pulse"
    assert schedule_tracker.timelines["q0"][-1].end == 4.0
    assert schedule_tracker.timelines["q1"] == ()
    np.testing.assert_array_equal(
        schedule_tracker.timelines["q0"][0].signal,
        np.zeros(2),
    )


def test_pulse_schedule_visitor_preserves_acquisition_label():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = PulseScheduleVisitor(schedule_tracker)
    schedule_visitor._scalar = lambda value: 2.0
    frame = object()
    registry = PulseScheduleVisitor.__dict__["visit"].dispatcher.registry
    registry[CreateFrameOp](
        schedule_visitor,
        SimpleNamespace(
            frequency=SimpleNamespace(),
            port=SimpleNamespace(data="q0"),
            result=frame,
        ),
    )

    registry[AcquireOp](
        schedule_visitor,
        SimpleNamespace(
            frame=frame,
            duration=SimpleNamespace(),
            label=SimpleNamespace(data="readout"),
            frame_result=object(),
        ),
    )

    assert schedule_tracker.records[0].label == "readout"


def test_pulse_schedule_visitor_configures_signal_metadata():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = PulseScheduleVisitor(
        schedule_tracker,
        signal_unit="normalized amplitude",
        signal_limits=None,
    )
    schedule_visitor._scalar = lambda value: 1.0

    PulseScheduleVisitor.__dict__["visit"].dispatcher.registry[CreateFrameOp](
        schedule_visitor,
        SimpleNamespace(
            frequency=SimpleNamespace(),
            port=SimpleNamespace(data="q0"),
            result=object(),
        ),
    )

    resource = schedule_tracker.resource_info("q0")
    assert resource.signal_unit == "normalized amplitude"
    assert resource.signal_limits is None


def test_pulse_schedule_visitor_samples_analytical_waveforms(mocker):
    schedule_tracker = ScheduleTracker()
    schedule_visitor = PulseScheduleVisitor(schedule_tracker)
    schedule_visitor._scalar = lambda value: 2.0
    mocker.patch(
        "qat.experimental.dialect.pulse.ir.schedule.evaluate_waveform",
        return_value=np.array([1.0 + 0.0j]),
    )
    operation = SimpleNamespace(
        width=SimpleNamespace(),
        amplitude=SimpleNamespace(),
        result="waveform",
        name="test_waveform",
        build_shape=lambda: object(),
    )

    PulseScheduleVisitor.__dict__["visit"].dispatcher.registry[
        IsAnalyticalWaveformInterface
    ](schedule_visitor, operation)

    assert schedule_visitor._waveforms["waveform"][0] == 2.0


@pytest.mark.parametrize(
    ("width", "amplitude", "shape"),
    [
        (None, 1.0, object()),
        (2.0, None, object()),
        (2.0, 1.0, None),
    ],
)
def test_pulse_schedule_visitor_rejects_dynamic_analytical_waveforms(
    width, amplitude, shape
):
    schedule_visitor = PulseScheduleVisitor(ScheduleTracker())
    width_operand = object()
    amplitude_operand = object()
    values = {width_operand: width, amplitude_operand: amplitude}
    schedule_visitor._scalar = values.get
    operation = SimpleNamespace(
        width=width_operand,
        amplitude=amplitude_operand,
        result="waveform",
        name="test_waveform",
        build_shape=lambda: shape,
    )

    with pytest.raises(ValueError, match="requires static operands"):
        PulseScheduleVisitor.__dict__["visit"].dispatcher.registry[
            IsAnalyticalWaveformInterface
        ](schedule_visitor, operation)


def test_pulse_schedule_visitor_records_sampled_constants(mocker):
    schedule_tracker = ScheduleTracker()
    schedule_visitor = PulseScheduleVisitor(schedule_tracker)
    waveform = SampledWaveformAttr([1.0 + 0.0j], TimeAttr(1.0), TimeAttr(1.0))
    mocker.patch(
        "qat.experimental.dialect.pulse.ir.schedule.ConstantLike.get_constant_value",
        return_value=waveform,
    )

    PulseScheduleVisitor.__dict__["visit"].dispatcher.registry[ConstantOp](
        schedule_visitor, SimpleNamespace(result=object())
    )

    assert schedule_visitor._waveforms


@pytest.mark.parametrize(
    ("operation_type", "expected"),
    [
        (AddOp, np.array([4.0, 6.0])),
        (SubOp, np.array([-2.0, -2.0])),
        (MixOp, np.array([3.0, 8.0])),
    ],
)
def test_build_pulse_schedule_evaluates_static_waveform_expressions(
    operation_type, expected
):
    width = TimeAttr(2e-9)
    sample_time = TimeAttr(1e-9)
    lhs = ConstantOp(SampledWaveformAttr([1.0, 2.0], width, sample_time))
    rhs = ConstantOp(SampledWaveformAttr([3.0, 4.0], width, sample_time))
    if operation_type is MixOp:
        expression = operation_type(lhs, rhs)
    else:
        expression = operation_type(lhs, rhs, WaveformType())
    frequency = ConstantOp(FrequencyAttr(0.0))
    frame = CreateFrameOp(frequency, StringAttr("drive"))
    module = ModuleOp([lhs, rhs, expression, frequency, frame, PulseOp(frame, expression)])

    schedule = build_pulse_schedule(module)

    np.testing.assert_array_equal(schedule.records[0].signal, expected.astype(complex))


def test_build_pulse_schedule_rejects_nested_control_flow():
    index_type = IndexType()
    lower = arith.ConstantOp.from_int_and_width(0, index_type)
    upper = arith.ConstantOp.from_int_and_width(2, index_type)
    step = arith.ConstantOp.from_int_and_width(1, index_type)
    loop = scf.ForOp(
        lower,
        upper,
        step,
        [],
        Block(ops=[scf.YieldOp()], arg_types=[index_type]),
    )

    with pytest.raises(ValueError, match="requires flat control flow"):
        build_pulse_schedule(ModuleOp([lower, upper, step, loop]))


def test_pulse_schedule_visitor_rejects_incomplete_waveform_expressions():
    visitor = PulseScheduleVisitor(ScheduleTracker())
    operation = SimpleNamespace(
        lhs=object(),
        rhs=object(),
        result=object(),
        name=MixOp.name,
        py_operation=np.multiply,
    )

    with pytest.raises(ValueError, match="requires static operands"):
        PulseScheduleVisitor.__dict__["visit"].dispatcher.registry[MixOp](
            visitor, operation
        )


def test_pulse_schedule_visitor_rejects_mismatched_waveform_expressions():
    visitor = PulseScheduleVisitor(ScheduleTracker())
    lhs, rhs = object(), object()
    visitor._waveforms[lhs] = (1e-9, np.array([1.0]))
    visitor._waveforms[rhs] = (2e-9, np.array([1.0, 2.0]))
    operation = SimpleNamespace(
        lhs=lhs,
        rhs=rhs,
        result=object(),
        name=AddOp.name,
        py_operation=np.add,
    )

    with pytest.raises(ValueError, match="requires matching operands"):
        PulseScheduleVisitor.__dict__["visit"].dispatcher.registry[AddOp](
            visitor, operation
        )


def test_pulse_schedule_visitor_ignores_non_waveform_constants(mocker):
    schedule_visitor = PulseScheduleVisitor(ScheduleTracker())
    mocker.patch(
        "qat.experimental.dialect.pulse.ir.schedule.ConstantLike.get_constant_value",
        return_value=object(),
    )

    PulseScheduleVisitor.__dict__["visit"].dispatcher.registry[ConstantOp](
        schedule_visitor, SimpleNamespace(result=object())
    )

    assert schedule_visitor._waveforms == {}


@pytest.mark.parametrize(
    ("operation_type", "operation", "message"),
    [
        (
            CreateFrameOp,
            SimpleNamespace(frequency=object(), port=SimpleNamespace(data="q0")),
            "frequency must be statically known",
        ),
        (
            PhaseOp,
            SimpleNamespace(
                frame=SimpleNamespace(
                    type=SimpleNamespace(port=SimpleNamespace(data="q0"))
                ),
                phase=object(),
                name="pulse.phase",
            ),
            "requires a static phase",
        ),
        (
            WaitOp,
            SimpleNamespace(
                frame=SimpleNamespace(
                    type=SimpleNamespace(port=SimpleNamespace(data="q0"))
                ),
                duration=object(),
            ),
            "duration must be statically known",
        ),
        (
            AcquireOp,
            SimpleNamespace(
                frame=SimpleNamespace(
                    type=SimpleNamespace(port=SimpleNamespace(data="q0"))
                ),
                duration=object(),
                label=None,
            ),
            "duration must be statically known",
        ),
    ],
)
def test_pulse_schedule_visitor_rejects_dynamic_operations(
    operation_type, operation, message
):
    schedule_tracker = ScheduleTracker()
    schedule_visitor = PulseScheduleVisitor(schedule_tracker)
    schedule_visitor._scalar = lambda value: None
    registry = PulseScheduleVisitor.__dict__["visit"].dispatcher.registry

    with pytest.raises(ValueError, match=message):
        registry[operation_type](schedule_visitor, operation)


def test_pulse_schedule_visitor_rejects_unknown_waveform():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = PulseScheduleVisitor(schedule_tracker)
    frame = object()
    registry = PulseScheduleVisitor.__dict__["visit"].dispatcher.registry
    schedule_visitor._scalar = lambda value: 1.0
    registry[CreateFrameOp](
        schedule_visitor,
        SimpleNamespace(
            frequency=SimpleNamespace(),
            port=SimpleNamespace(data="q0"),
            result=frame,
        ),
    )

    with pytest.raises(ValueError, match="must be visited before"):
        registry[PulseOp](
            schedule_visitor,
            SimpleNamespace(frame=frame, waveform="missing", result=object()),
        )


def test_pulse_schedule_visitor_tracks_frame_ssa_lineage_and_phase_semantics():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = PulseScheduleVisitor(schedule_tracker)
    registry = PulseScheduleVisitor.__dict__["visit"].dispatcher.registry
    frequency, set_phase, shift_phase, duration = (object() for _ in range(4))
    values = {
        frequency: 5e9,
        set_phase: np.pi / 2,
        shift_phase: np.pi / 4,
        duration: 1e-9,
    }
    schedule_visitor._scalar = values.get
    initial_frame, set_frame, shifted_frame = (object() for _ in range(3))

    registry[CreateFrameOp](
        schedule_visitor,
        SimpleNamespace(
            frequency=frequency,
            port=SimpleNamespace(data="drive"),
            result=initial_frame,
        ),
    )
    registry[PhaseOp](
        schedule_visitor,
        SimpleNamespace(
            frame=initial_frame,
            phase=set_phase,
            result=set_frame,
            name=PhaseSetOp.name,
        ),
    )
    registry[PhaseOp](
        schedule_visitor,
        SimpleNamespace(
            frame=set_frame,
            phase=shift_phase,
            result=shifted_frame,
            name=PhaseShiftOp.name,
        ),
    )
    registry[WaitOp](
        schedule_visitor,
        SimpleNamespace(frame=shifted_frame, duration=duration, result=object()),
    )

    event = schedule_tracker.timelines["drive"][0]
    assert event.frequency == 5e9
    assert event.phase == pytest.approx(3 * np.pi / 4)


def test_pulse_operations_propagate_real_frame_ssa_values():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = PulseScheduleVisitor(schedule_tracker)
    frequency = ConstantOp(FrequencyAttr(5e9))
    initial_frame = CreateFrameOp(frequency, StringAttr("drive"))
    set_phase = PhaseSetOp(initial_frame, ConstantOp(PhaseAttr(np.pi / 2)))
    shift_phase = PhaseShiftOp(set_phase, ConstantOp(PhaseAttr(np.pi / 4)))
    wait = WaitOp(shift_phase, ConstantOp(TimeAttr(1e-9)))

    for operation in (initial_frame, set_phase, shift_phase, wait):
        operation.accept(schedule_visitor)

    event = schedule_tracker.timelines["drive"][0]
    assert event.frequency == 5e9
    assert event.phase == pytest.approx(3 * np.pi / 4)


def test_pulse_schedule_visitor_tracks_frames_independently_on_the_same_port():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = PulseScheduleVisitor(schedule_tracker)
    schedule_visitor._scalar = lambda value: 1.0
    registry = PulseScheduleVisitor.__dict__["visit"].dispatcher.registry
    first_frame, second_frame = object(), object()

    for frame in (first_frame, second_frame):
        registry[CreateFrameOp](
            schedule_visitor,
            SimpleNamespace(
                frequency=object(),
                port=SimpleNamespace(data="drive"),
                result=frame,
            ),
        )
        registry[WaitOp](
            schedule_visitor,
            SimpleNamespace(frame=frame, duration=object(), result=object()),
        )

    assert schedule_tracker.resources == ("drive", "drive [2]")
    assert [len(events) for events in schedule_tracker.timelines.values()] == [1, 1]


def test_pulse_schedule_visitor_treats_synchronisation_as_lineage_only():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = PulseScheduleVisitor(schedule_tracker)
    schedule_visitor._scalar = lambda value: 1.0
    registry = PulseScheduleVisitor.__dict__["visit"].dispatcher.registry
    first_frame, second_frame = object(), object()
    first_result, second_result = object(), object()

    for port, frame in (("first", first_frame), ("second", second_frame)):
        registry[CreateFrameOp](
            schedule_visitor,
            SimpleNamespace(
                frequency=object(),
                port=SimpleNamespace(data=port),
                result=frame,
            ),
        )
    registry[SynchronizeOp](
        schedule_visitor,
        SimpleNamespace(
            frames=[first_frame, second_frame],
            results=[first_result, second_result],
        ),
    )
    registry[WaitOp](
        schedule_visitor,
        SimpleNamespace(frame=second_result, duration=object(), result=object()),
    )

    assert schedule_tracker.timelines["first"] == ()
    assert schedule_tracker.timelines["second"][0].label == "wait"


def test_pulse_schedule_visitor_rejects_unknown_frame_ssa_value():
    schedule_visitor = PulseScheduleVisitor(ScheduleTracker())
    schedule_visitor._scalar = lambda value: 1.0

    with pytest.raises(ValueError, match="frame must be visited"):
        PulseScheduleVisitor.__dict__["visit"].dispatcher.registry[WaitOp](
            schedule_visitor,
            SimpleNamespace(frame=object(), duration=object(), result=object()),
        )


def test_pulse_schedule_visitor_plots_continuous_waveforms_during_timed_ops():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = PulseScheduleVisitor(schedule_tracker, sample_time=1.0)
    registry = PulseScheduleVisitor.__dict__["visit"].dispatcher.registry
    schedule_visitor._scalar = lambda value: 2.0
    initial_frame, started_frame, waited_frame = (object() for _ in range(3))

    registry[CreateFrameOp](
        schedule_visitor,
        SimpleNamespace(
            frequency=object(),
            port=SimpleNamespace(data="drive"),
            result=initial_frame,
        ),
    )
    registry[StartContinuousWaveformOp](
        schedule_visitor,
        SimpleNamespace(
            frame=initial_frame,
            amplitude=object(),
            result=started_frame,
        ),
    )
    registry[WaitOp](
        schedule_visitor,
        SimpleNamespace(frame=started_frame, duration=object(), result=waited_frame),
    )
    registry[StopContinuousWaveformOp](
        schedule_visitor,
        SimpleNamespace(frame=waited_frame, result=object()),
    )

    start, wait, stop = schedule_tracker.timelines["drive"]
    assert start.label == "start continuous waveform"
    np.testing.assert_allclose(wait.signal, [2.0, 2.0])
    assert stop.label == "stop continuous waveform"


def test_pulse_schedule_visitor_rejects_dynamic_continuous_waveform_amplitude():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = PulseScheduleVisitor(schedule_tracker)
    registry = PulseScheduleVisitor.__dict__["visit"].dispatcher.registry
    frame = object()
    schedule_visitor._scalar = lambda value: 1.0
    registry[CreateFrameOp](
        schedule_visitor,
        SimpleNamespace(
            frequency=object(),
            port=SimpleNamespace(data="drive"),
            result=frame,
        ),
    )
    schedule_visitor._scalar = lambda value: None

    with pytest.raises(ValueError, match="amplitude must be statically known"):
        registry[StartContinuousWaveformOp](
            schedule_visitor,
            SimpleNamespace(frame=frame, amplitude=object(), result=object()),
        )


def test_pulse_schedule_visitor_rejects_unsupported_operation():
    schedule_visitor = PulseScheduleVisitor(ScheduleTracker())

    with pytest.raises(NotImplementedError, match="does not support"):
        schedule_visitor.visit(object())  # type: ignore[arg-type]


def test_pulse_schedule_visitor_extracts_static_scalar(mocker):
    mocker.patch(
        "qat.experimental.dialect.pulse.ir.schedule.ConstantLike.get_constant_value",
        return_value=SimpleNamespace(literal_value=2.5),
    )

    assert PulseScheduleVisitor._scalar(object()) == 2.5


def test_pulse_schedule_visitor_returns_none_for_dynamic_scalar(mocker):
    mocker.patch(
        "qat.experimental.dialect.pulse.ir.schedule.ConstantLike.get_constant_value",
        return_value=object(),
    )

    assert PulseScheduleVisitor._scalar(object()) is None
