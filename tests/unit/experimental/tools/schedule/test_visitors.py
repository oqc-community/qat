# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import numpy as np
import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import ArrayAttr, ModuleOp
from xdsl.ir import Block, Region

from qat.experimental.dialect.q1.ir.imm_desc import (
    BoolImm,
    DurationImm,
    NcoPhaseImm,
    SI16Imm,
    SI32Imm,
    SU32Imm,
    UI4Imm,
    UI5Imm,
    UI10Imm,
)
from qat.experimental.dialect.q1.ir.ops import (
    AcquireImmRsImmOp,
    AddRsImmRdOp,
    JmpImmOp,
    LabelOp,
    LoopRdImmOp,
    MoveImmRdOp,
    NopOp,
    NotRsRdOp,
    PlayImmImmImmOp,
    PlayRsRsImmOp,
    ResetPhOp,
    SetAwgGainImmImmOp,
    SetFreqImmOp,
    SetLatchEnImmImmOp,
    SetMrkImmOp,
    SetPhDeltaImmOp,
    SetPhImmOp,
    StopOp,
    UpdParamImmOp,
    WaitImmOp,
    WaitSyncImmOp,
)
from qat.experimental.dialect.q1.ir.reg_desc import IntRegisterType, Registers
from qat.experimental.dialect.q1.ir.schedule import Q1ScheduleVisitor, build_q1_schedule
from qat.experimental.dialect.q1_sequence.ir.attrs import (
    AwgConfigAttr,
    NcoConfigAttr,
    SequencerConfigAttr,
    make_waveform,
)
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


def test_build_q1_schedule_uses_configured_nco_until_runtime_override():
    sequence = SequenceOp(
        "drive",
        [
            PlayImmImmImmOp(UI10Imm(0), UI10Imm(1), DurationImm(4)),
            SetFreqImmOp(SI32Imm(500_000_000)),
            PlayImmImmImmOp(UI10Imm(0), UI10Imm(1), DurationImm(4)),
            StopOp(),
        ],
        waveforms=ArrayAttr(
            [
                make_waveform("i", 0, [1.0, 1.0, 1.0, 1.0]),
                make_waveform("q", 1, [0.0, 0.0, 0.0, 0.0]),
            ]
        ),
        sequencer_config=SequencerConfigAttr(nco=NcoConfigAttr(frequency=250_000_000)),
    )

    schedule = build_q1_schedule(ModuleOp([sequence]))

    first, second = schedule.timelines["drive"]
    assert first.frequency == 250_000_000
    assert second.frequency == 125_000_000
    np.testing.assert_allclose(
        first.signal,
        np.exp(1j * np.arange(4) * np.pi / 2) / np.sqrt(2),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        second.signal,
        np.exp(1j * np.arange(4) * np.pi / 4) / np.sqrt(2),
        atol=1e-12,
    )


def test_build_q1_schedule_uses_configured_phase_until_runtime_override():
    sequence = SequenceOp(
        "drive",
        [
            PlayImmImmImmOp(UI10Imm(0), UI10Imm(1), DurationImm(4)),
            SetPhImmOp(NcoPhaseImm(0)),
            PlayImmImmImmOp(UI10Imm(0), UI10Imm(1), DurationImm(4)),
            StopOp(),
        ],
        waveforms=ArrayAttr(
            [
                make_waveform("i", 0, [1.0, 1.0, 1.0, 1.0]),
                make_waveform("q", 1, [0.0, 0.0, 0.0, 0.0]),
            ]
        ),
        sequencer_config=SequencerConfigAttr(nco=NcoConfigAttr(phase_offs=90.0)),
    )

    schedule = build_q1_schedule(ModuleOp([sequence]))

    first, second = schedule.timelines["drive"]
    assert first.phase == pytest.approx(np.pi / 2)
    assert second.phase == pytest.approx(0.0)
    assert first.phase * first.phase_scale == pytest.approx(250_000_000)
    assert second.phase * second.phase_scale == pytest.approx(0)
    np.testing.assert_allclose(first.signal, 1.0j / np.sqrt(2), atol=1e-12)
    np.testing.assert_allclose(second.signal, 1.0 / np.sqrt(2), atol=1e-12)


def test_build_q1_schedule_does_not_modulate_signal_when_awg_modulation_is_disabled():
    sequence = SequenceOp(
        "drive",
        [
            PlayImmImmImmOp(UI10Imm(0), UI10Imm(1), DurationImm(4)),
            StopOp(),
        ],
        waveforms=ArrayAttr(
            [
                make_waveform("i", 0, [1.0, 1.0, 1.0, 1.0]),
                make_waveform("q", 1, [0.0, 0.0, 0.0, 0.0]),
            ]
        ),
        sequencer_config=SequencerConfigAttr(
            nco=NcoConfigAttr(frequency=250_000_000, phase_offs=90.0),
            awg=AwgConfigAttr(mod_en=False),
        ),
    )

    schedule = build_q1_schedule(ModuleOp([sequence]))

    event = schedule.timelines["drive"][0]
    assert event.frequency == 250_000_000
    assert event.phase == pytest.approx(np.pi / 2)
    assert not event.phase_modulates_signal
    np.testing.assert_allclose(
        event.signal,
        np.array([1.0, 1.0, 1.0, 1.0]) / np.sqrt(2),
        atol=1e-12,
    )


def test_build_q1_schedule_preserves_configured_phase_near_full_turn():
    phase_steps = 999_999_999
    sequence = SequenceOp(
        "drive",
        [
            PlayImmImmImmOp(UI10Imm(0), UI10Imm(1), DurationImm(4)),
            StopOp(),
        ],
        waveforms=ArrayAttr(
            [
                make_waveform("i", 0, [1.0]),
                make_waveform("q", 1, [0.0]),
            ]
        ),
        sequencer_config=SequencerConfigAttr(
            nco=NcoConfigAttr(phase_offs=phase_steps * 360 / 1_000_000_000)
        ),
    )

    schedule = build_q1_schedule(ModuleOp([sequence]))

    event = schedule.timelines["drive"][0]
    assert event.phase * event.phase_scale == pytest.approx(phase_steps)
    expected_signal = np.zeros(4, dtype=complex)
    expected_signal[0] = np.exp(2j * np.pi * phase_steps / 1_000_000_000) / np.sqrt(2)
    np.testing.assert_allclose(
        event.signal,
        expected_signal,
        atol=1e-12,
    )


def test_build_q1_schedule_reset_preserves_configured_phase_offset():
    sequence = SequenceOp(
        "drive",
        [
            SetPhDeltaImmOp(NcoPhaseImm(250_000_000)),
            PlayImmImmImmOp(UI10Imm(0), UI10Imm(1), DurationImm(4)),
            ResetPhOp(),
            PlayImmImmImmOp(UI10Imm(0), UI10Imm(1), DurationImm(4)),
            StopOp(),
        ],
        waveforms=ArrayAttr(
            [
                make_waveform("i", 0, [1.0, 1.0, 1.0, 1.0]),
                make_waveform("q", 1, [0.0, 0.0, 0.0, 0.0]),
            ]
        ),
        sequencer_config=SequencerConfigAttr(nco=NcoConfigAttr(phase_offs=90.0)),
    )

    schedule = build_q1_schedule(ModuleOp([sequence]))

    before_reset, after_reset = schedule.timelines["drive"]
    assert before_reset.phase == pytest.approx(np.pi)
    assert after_reset.phase == pytest.approx(np.pi / 2)
    assert before_reset.phase * before_reset.phase_scale == pytest.approx(500_000_000)
    assert after_reset.phase * after_reset.phase_scale == pytest.approx(250_000_000)
    np.testing.assert_allclose(before_reset.signal, -1.0 / np.sqrt(2), atol=1e-12)
    np.testing.assert_allclose(after_reset.signal, 1.0j / np.sqrt(2), atol=1e-12)


def test_build_q1_schedule_visualises_one_lowered_loop_iteration():
    counter = MoveImmRdOp(SU32Imm(2), Registers.R0)
    inverted_index = NotRsRdOp(counter.rd, Registers.R1)
    acquisition_index = AddRsImmRdOp(inverted_index.rd, SU32Imm(3), Registers.R2)
    sequence = SequenceOp(
        "drive",
        [
            counter,
            LabelOp("shot"),
            inverted_index,
            acquisition_index,
            PlayImmImmImmOp(UI10Imm(0), UI10Imm(1), DurationImm(4)),
            AcquireImmRsImmOp(UI5Imm(0), acquisition_index.rd, DurationImm(8)),
            LoopRdImmOp(Registers.R0, "shot"),
            StopOp(),
        ],
        waveforms=ArrayAttr(
            [
                make_waveform("i", 0, [1.0]),
                make_waveform("q", 1, [0.0]),
            ]
        ),
    )
    module = ModuleOp([sequence])
    original_operations = tuple(sequence.body.block.ops)

    schedule = build_q1_schedule(module)

    assert [event.label for event in schedule.records] == ["play", "acquire"]
    assert tuple(sequence.body.block.ops) == original_operations


def test_build_q1_schedule_rejects_dynamic_waveform_index_inside_lowered_loop():
    counter = MoveImmRdOp(SU32Imm(2), Registers.R0)
    waveform_index = AddRsImmRdOp(counter.rd, SU32Imm(1), Registers.R1)
    sequence = SequenceOp(
        "drive",
        [
            counter,
            LabelOp("shot"),
            waveform_index,
            PlayRsRsImmOp(
                waveform_index.rd,
                waveform_index.rd,
                DurationImm(4),
            ),
            LoopRdImmOp(Registers.R0, "shot"),
            StopOp(),
        ],
    )

    with pytest.raises(ValueError, match="requires a static register value"):
        build_q1_schedule(ModuleOp([sequence]))


def test_build_q1_schedule_rejects_general_jump_control_flow():
    sequence = SequenceOp(
        "drive",
        [
            LabelOp("body"),
            JmpImmOp("body"),
            StopOp(),
        ],
    )

    with pytest.raises(ValueError, match="does not support general jump control flow"):
        build_q1_schedule(ModuleOp([sequence]))


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


def test_build_q1_schedule_renders_mission_sequences_from_one_sync_origin():
    sync_config = SequencerConfigAttr(enable_sync=True)
    sequences = []
    for name, duration in (
        ("acquire", 8),
        ("readout", 8),
        ("control_0", 12),
        ("control_1", 16),
    ):
        counter = MoveImmRdOp(SU32Imm(2), Registers.R1)
        operations = [
            SetMrkImmOp(UI4Imm(3)),
            SetLatchEnImmImmOp(BoolImm(1), DurationImm(4)),
            UpdParamImmOp(DurationImm(4)),
            counter,
            NopOp(),
            LabelOp("shot"),
            WaitSyncImmOp(DurationImm(4)),
            ResetPhOp(),
            UpdParamImmOp(DurationImm(4)),
        ]
        waveforms = ArrayAttr([])
        if name == "acquire":
            inverted_index = NotRsRdOp(counter.rd, Registers.R2)
            acquisition_index = AddRsImmRdOp(inverted_index.rd, SU32Imm(3), Registers.R2)
            operations.extend(
                [
                    inverted_index,
                    NopOp(),
                    acquisition_index,
                    NopOp(),
                    AcquireImmRsImmOp(
                        UI5Imm(0), acquisition_index.rd, DurationImm(duration)
                    ),
                ]
            )
        else:
            waveforms = ArrayAttr(
                [
                    make_waveform("i", 0, [1.0] * duration),
                    make_waveform("q", 1, [0.0] * duration),
                ]
            )
            operations.append(
                PlayImmImmImmOp(UI10Imm(0), UI10Imm(1), DurationImm(duration))
            )
        operations.extend([LoopRdImmOp(Registers.R1, "shot"), StopOp()])
        sequences.append(
            SequenceOp(
                name,
                operations,
                waveforms=waveforms,
                sequencer_config=sync_config,
            )
        )

    schedule = build_q1_schedule(ModuleOp(sequences))

    assert schedule.resources == ("acquire", "readout", "control_0", "control_1")
    assert {
        resource: [
            (event.start, event.end)
            for event in schedule.timelines[resource]
            if event.label == "wait_sync"
        ]
        for resource in schedule.resources
    } == {
        "acquire": [(12, 16)],
        "readout": [(12, 16)],
        "control_0": [(12, 16)],
        "control_1": [(12, 16)],
    }
    assert {
        resource: [
            (event.label, event.start, event.end)
            for event in schedule.timelines[resource]
            if event.label in {"play", "acquire"}
        ]
        for resource in schedule.resources
    } == {
        "acquire": [("acquire", 28, 36)],
        "readout": [("play", 20, 28)],
        "control_0": [("play", 20, 32)],
        "control_1": [("play", 20, 36)],
    }


def test_build_q1_schedule_accepts_unmatched_multi_sequence_wait_sync():
    first = SequenceOp(
        "first",
        [WaitSyncImmOp(DurationImm(4)), StopOp()],
    )
    second = SequenceOp("second", [WaitImmOp(DurationImm(8)), StopOp()])

    schedule = build_q1_schedule(ModuleOp([first, second]))

    assert [
        (event.label, event.start, event.end) for event in schedule.timelines["first"]
    ] == [("wait_sync", 0, 4)]
    assert [
        (event.label, event.start, event.end) for event in schedule.timelines["second"]
    ] == [("wait", 0, 8)]


def test_build_q1_schedule_aligns_each_enabled_multi_sequence_wait_sync():
    sync_config = SequencerConfigAttr(enable_sync=True)
    first = SequenceOp(
        "first",
        [
            WaitSyncImmOp(DurationImm(4)),
            WaitSyncImmOp(DurationImm(8)),
            StopOp(),
        ],
        sequencer_config=sync_config,
    )
    second = SequenceOp(
        "second",
        [
            WaitSyncImmOp(DurationImm(8)),
            WaitSyncImmOp(DurationImm(4)),
            StopOp(),
        ],
        sequencer_config=sync_config,
    )

    schedule = build_q1_schedule(ModuleOp([first, second]))

    assert [
        (event.label, event.start, event.end) for event in schedule.timelines["first"]
    ] == [("wait_sync", 0, 4), ("synchronise", 4, 8), ("wait_sync", 8, 16)]
    assert [
        (event.label, event.start, event.end) for event in schedule.timelines["second"]
    ] == [("wait_sync", 0, 8), ("wait_sync", 8, 12)]


def test_build_q1_schedule_rejects_mismatched_enabled_wait_sync_counts():
    sync_config = SequencerConfigAttr(enable_sync=True)
    first = SequenceOp(
        "first",
        [
            WaitSyncImmOp(DurationImm(4)),
            WaitSyncImmOp(DurationImm(4)),
            StopOp(),
        ],
        sequencer_config=sync_config,
    )
    second = SequenceOp(
        "second",
        [WaitSyncImmOp(DurationImm(4)), StopOp()],
        sequencer_config=sync_config,
    )

    with pytest.raises(ValueError, match="matching wait_sync counts"):
        build_q1_schedule(ModuleOp([first, second]))


def test_build_q1_schedule_blocks_early_participant_at_wait_sync():
    sync_config = SequencerConfigAttr(enable_sync=True)
    first = SequenceOp(
        "first",
        [
            WaitImmOp(DurationImm(8)),
            WaitSyncImmOp(DurationImm(4)),
            PlayImmImmImmOp(UI10Imm(0), UI10Imm(1), DurationImm(4)),
            StopOp(),
        ],
        waveforms=ArrayAttr(
            [
                make_waveform("i", 0, [1.0]),
                make_waveform("q", 1, [0.0]),
            ]
        ),
        sequencer_config=sync_config,
    )
    second = SequenceOp(
        "second",
        [
            WaitSyncImmOp(DurationImm(4)),
            PlayImmImmImmOp(UI10Imm(0), UI10Imm(1), DurationImm(4)),
            StopOp(),
        ],
        waveforms=ArrayAttr(
            [
                make_waveform("i", 0, [1.0]),
                make_waveform("q", 1, [0.0]),
            ]
        ),
        sequencer_config=sync_config,
    )

    schedule = build_q1_schedule(ModuleOp([first, second]))

    assert [
        (event.label, event.start, event.end) for event in schedule.timelines["first"]
    ] == [("wait", 0, 8), ("wait_sync", 8, 12), ("play", 12, 16)]
    assert [
        (event.label, event.start, event.end) for event in schedule.timelines["second"]
    ] == [
        ("synchronise", 0, 8),
        ("wait_sync", 8, 12),
        ("play", 12, 16),
    ]


def test_build_q1_schedule_falls_back_when_sequence_sync_is_disabled():
    first = SequenceOp(
        "first",
        [WaitImmOp(DurationImm(8)), WaitSyncImmOp(DurationImm(4)), StopOp()],
        sequencer_config=SequencerConfigAttr(enable_sync=True),
    )
    second = SequenceOp(
        "second",
        [WaitSyncImmOp(DurationImm(4)), StopOp()],
        sequencer_config=SequencerConfigAttr(enable_sync=False),
    )

    schedule = build_q1_schedule(ModuleOp([first, second]))

    assert [
        (event.label, event.start, event.end) for event in schedule.timelines["first"]
    ] == [("wait", 0, 8), ("wait_sync", 8, 12)]
    assert [
        (event.label, event.start, event.end) for event in schedule.timelines["second"]
    ] == [("wait_sync", 0, 4)]


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
