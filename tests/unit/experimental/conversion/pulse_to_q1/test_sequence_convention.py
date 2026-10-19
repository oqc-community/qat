# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from dataclasses import replace

import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.irdl import IRDLOperation, irdl_op_definition, traits_def
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.conversion.pulse_to_q1.hardware_binding import (
    QbloxHardwareBindingPass,
)
from qat.experimental.conversion.pulse_to_q1.sequence_convention import (
    QbloxSequenceInitialisationPass,
)
from qat.experimental.dialect.pulse.ir import ConstantOp, CreateFrameOp, FrequencyAttr
from qat.experimental.dialect.q1 import (
    BoolImm,
    DurationImm,
    SetLatchEnImmImmOp,
    SetMrkImmOp,
    StopOp,
    UpdParamImmOp,
)
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.system_data.hardware.qblox import DEFAULT_QBLOX_TARGET

from tests.unit.experimental.conversion.pulse_to_q1.qblox_configuration.helpers import (
    canonical_data,
    sequencer,
    supplied,
)


@irdl_op_definition
class _UnsupportedTerminatorOp(IRDLOperation):
    """Terminator that is invalid for an executable Q1 sequence."""

    name = "test.unsupported_sequence_terminator"
    traits = traits_def(IsTerminator())


def _bound_module(*prefix) -> tuple[ModuleOp, SequenceOp]:
    frequency = ConstantOp(FrequencyAttr(4_200_000_000))
    frame = CreateFrameOp(frequency, StringAttr("port-0"))
    sequence = SequenceOp(
        "port-0-channel-0",
        [*prefix, frequency, frame, StopOp()],
        "port-0",
    )
    module = ModuleOp([sequence])
    data = canonical_data(configurations=[supplied([sequencer(0)])])
    QbloxHardwareBindingPass(data).apply(Context(), module)
    return module, sequence


def test_sequence_lowering_convention_requires_hardware_binding():
    frequency = ConstantOp(FrequencyAttr(4_200_000_000))
    frame = CreateFrameOp(frequency, StringAttr("port-0"))
    sequence = SequenceOp(
        "port-0-channel-0",
        [frequency, frame, StopOp()],
        "port-0",
    )
    module = ModuleOp([sequence])

    with pytest.raises(PassFailedException, match="must be hardware-bound"):
        QbloxSequenceInitialisationPass().apply(Context(), module)


def test_sequence_lowering_convention_rejects_empty_body_before_binding():
    sequence = SequenceOp("port-0-channel-0", [], "port-0")
    module = ModuleOp([sequence])

    with pytest.raises(PassFailedException, match="has an empty body"):
        QbloxSequenceInitialisationPass().apply(Context(), module)


def test_sequence_lowering_convention_rejects_non_stop_terminator_without_mutating():
    module, sequence = _bound_module()
    sequence.body.block.last_op.detach()
    sequence.body.block.add_op(_UnsupportedTerminatorOp())
    original_operations = tuple(sequence.body.block.ops)

    with pytest.raises(PassFailedException, match="must end with q1 stop"):
        QbloxSequenceInitialisationPass().apply(Context(), module)

    assert tuple(sequence.body.block.ops) == original_operations


def test_sequence_lowering_convention_orders_prologue_payload_and_final_stop():
    """The prologue precedes payload and the structural stop remains final."""

    module, sequence = _bound_module()

    QbloxSequenceInitialisationPass().apply(Context(), module)

    marker, latch, update, frequency, frame, stop = sequence.body.block.ops
    duration = DEFAULT_QBLOX_TARGET.q1asm.min_rt_duration_ns
    assert isinstance(marker, SetMrkImmOp)
    assert marker.mrk.data == 3
    assert isinstance(latch, SetLatchEnImmImmOp)
    assert latch.latch_en == BoolImm(1)
    assert latch.duration == DurationImm(duration)
    assert isinstance(update, UpdParamImmOp)
    assert update.duration == DurationImm(duration)
    assert isinstance(frequency, ConstantOp)
    assert isinstance(frame, CreateFrameOp)
    assert isinstance(stop, StopOp)


def test_sequence_lowering_convention_uses_target_minimum_rt_duration():
    """The latch and applying update use the configured minimum RT duration."""
    module, sequence = _bound_module()
    target_data = replace(
        DEFAULT_QBLOX_TARGET,
        q1asm=replace(DEFAULT_QBLOX_TARGET.q1asm, min_rt_duration_ns=8),
    )

    QbloxSequenceInitialisationPass(target_data).apply(Context(), module)

    _, latch, update, *_ = sequence.body.block.ops
    assert isinstance(latch, SetLatchEnImmImmOp)
    assert latch.duration == DurationImm(8)
    assert isinstance(update, UpdParamImmOp)
    assert update.duration == DurationImm(8)


def test_sequence_lowering_convention_ignores_non_sequence_operations():
    """Top-level operations outside sequence envelopes remain unchanged."""
    frequency = ConstantOp(FrequencyAttr(4_200_000_000))
    module = ModuleOp([frequency])

    QbloxSequenceInitialisationPass().apply(Context(), module)

    assert tuple(module.body.block.ops) == (frequency,)
