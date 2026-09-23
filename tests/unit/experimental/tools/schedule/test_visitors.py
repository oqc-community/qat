# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import numpy as np
import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import ArrayAttr, ModuleOp
from xdsl.ir import Block, Region

from qat.experimental.dialect.q1.ir.imm_desc import (
    DurationImm,
    SI16Imm,
    SI32Imm,
    SU32Imm,
    UI5Imm,
    UI10Imm,
)
from qat.experimental.dialect.q1.ir.ops import (
    AcquireImmRsImmOp,
    MoveImmRdOp,
    NopOp,
    PlayImmImmImmOp,
    PlayRsRsImmOp,
    SetAwgGainImmImmOp,
    SetFreqImmOp,
    StopOp,
    UpdParamImmOp,
    WaitImmOp,
    WaitSyncImmOp,
)
from qat.experimental.dialect.q1.ir.reg_desc import IntRegisterType
from qat.experimental.dialect.q1.ir.schedule import Q1ScheduleVisitor, build_q1_schedule
from qat.experimental.dialect.q1_sequence.ir.attrs import make_waveform
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.tools.schedule import ScheduleTracker


def test_q1_operations_accept_q1_schedule_visitor():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = Q1ScheduleVisitor(schedule_tracker, "q0")

    NopOp().accept(schedule_visitor)
    SetFreqImmOp(SI32Imm(125)).accept(schedule_visitor)
    WaitImmOp(DurationImm(8)).accept(schedule_visitor)

    schedule_events = schedule_tracker.timelines["q0"]
    assert [event.label for event in schedule_events] == ["nop", "wait"]
    assert schedule_events[-1].frequency == 0.0
    assert schedule_events[-1].end == 12

    UpdParamImmOp(DurationImm(4)).accept(schedule_visitor)

    assert schedule_tracker.timelines["q0"][-1].frequency == 31.25


def test_q1_schedule_visitor_rejects_unsupported_operations():
    schedule_tracker = ScheduleTracker()
    schedule_visitor = Q1ScheduleVisitor(schedule_tracker)

    with pytest.raises(NotImplementedError, match="does not support"):
        schedule_visitor.visit(object())  # type: ignore[arg-type]


def test_q1_schedule_visitor_registers_sequence_resource():
    schedule_tracker = ScheduleTracker()
    Q1ScheduleVisitor(schedule_tracker)

    assert schedule_tracker.resources == ("sequence",)
    assert schedule_tracker.timelines["sequence"] == ()


def test_build_q1_schedule_walks_module():
    module = ModuleOp([NopOp(), WaitImmOp(DurationImm(8))])

    schedule_tracker = build_q1_schedule(module)

    assert [event.label for event in schedule_tracker.timelines["sequence"]] == [
        "nop",
        "wait",
    ]
    assert schedule_tracker.timelines["sequence"][-1].end == 12


def test_q1_schedule_builder_returns_a_fresh_tracker():
    module = ModuleOp([NopOp()])

    first_schedule = build_q1_schedule(module)
    second_schedule = build_q1_schedule(module)

    assert first_schedule is not second_schedule
    assert len(first_schedule.records) == len(second_schedule.records) == 1
    assert first_schedule.records[0].label == second_schedule.records[0].label
    np.testing.assert_array_equal(
        first_schedule.records[0].signal,
        second_schedule.records[0].signal,
    )


def test_build_q1_schedule_reads_sequence_waveform_table():
    sequence = SequenceOp(
        "drive",
        [
            SetAwgGainImmImmOp(SI16Imm(16_384), SI16Imm(16_384)),
            PlayImmImmImmOp(UI10Imm(0), UI10Imm(1), DurationImm(4)),
            WaitImmOp(DurationImm(4)),
            StopOp(),
        ],
        waveforms=ArrayAttr(
            [
                make_waveform("i", 0, [1.0, 0.5, 0.0, -0.5, -1.0, 0.0]),
                make_waveform("q", 1, [0.0, 0.5, 1.0, 0.5, 0.0, -0.5]),
            ]
        ),
    )

    schedule_tracker = build_q1_schedule(ModuleOp([sequence]))

    play, wait = schedule_tracker.timelines["drive"]
    scale = 0.5 / np.sqrt(2)
    np.testing.assert_allclose(
        play.signal,
        scale * np.array([1.0, 0.5 + 0.5j, 1.0j, -0.5 + 0.5j]),
    )
    np.testing.assert_allclose(
        wait.signal,
        np.zeros(4),
    )


def test_build_q1_schedule_accepts_lowered_register_acquisition():
    bin_index = MoveImmRdOp(SU32Imm(0), IntRegisterType.unallocated())
    sequence = SequenceOp(
        "readout",
        [
            bin_index,
            AcquireImmRsImmOp(UI5Imm(0), bin_index.rd, DurationImm(8)),
            StopOp(),
        ],
    )

    schedule_tracker = build_q1_schedule(ModuleOp([sequence]))

    assert [event.label for event in schedule_tracker.records] == ["acquire"]
    assert schedule_tracker.records[0].duration == 8


def test_build_q1_schedule_accepts_static_register_playback():
    wave0 = MoveImmRdOp(SU32Imm(0), IntRegisterType.unallocated())
    wave1 = MoveImmRdOp(SU32Imm(1), IntRegisterType.unallocated())
    sequence = SequenceOp(
        "drive",
        [
            wave0,
            wave1,
            PlayRsRsImmOp(wave0.rd, wave1.rd, DurationImm(4)),
            StopOp(),
        ],
        waveforms=ArrayAttr(
            [
                make_waveform("i", 0, [1.0, 0.5]),
                make_waveform("q", 1, [0.0, 0.5]),
            ]
        ),
    )

    schedule_tracker = build_q1_schedule(ModuleOp([sequence]))

    np.testing.assert_allclose(
        schedule_tracker.records[0].signal,
        np.array([1.0, 0.5 + 0.5j, 0.0, 0.0]) / np.sqrt(2),
    )


def test_build_q1_schedule_rejects_multi_sequence_wait_sync():
    first = SequenceOp("first", [WaitSyncImmOp(DurationImm(4)), StopOp()])
    second = SequenceOp("second", [WaitImmOp(DurationImm(8)), StopOp()])

    with pytest.raises(ValueError, match="wait_sync across multiple sequences"):
        build_q1_schedule(ModuleOp([first, second]))


def test_build_q1_schedule_rejects_multi_block_control_flow():
    sequence = SequenceOp(
        "drive",
        Region([Block([StopOp()]), Block([StopOp()])]),
    )

    with pytest.raises(ValueError, match="linearised single-block"):
        build_q1_schedule(ModuleOp([sequence]))


def test_build_q1_schedule_rejects_structured_operations_in_single_block():
    residual = arith.ConstantOp.from_int_and_width(1, 32)
    sequence = SequenceOp("drive", [residual, StopOp()])

    with pytest.raises(ValueError, match="requires flat Q1 operations"):
        build_q1_schedule(ModuleOp([sequence]))


def test_build_q1_schedule_rejects_structured_operations_in_flat_module():
    residual = arith.ConstantOp.from_int_and_width(1, 32)

    with pytest.raises(ValueError, match="requires flat Q1 operations"):
        build_q1_schedule(ModuleOp([residual]))
