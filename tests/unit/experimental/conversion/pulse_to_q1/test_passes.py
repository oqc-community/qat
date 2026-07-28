# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from xdsl.context import Context
from xdsl.dialects import func
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Block, Region

from qat.experimental.conversion.pulse_to_q1.passes import (
    PulseToQ1LoweringPass,
    create_default_pulse_to_q1_pipeline,
)
from qat.experimental.conversion.pulse_to_q1.sequence_outlining import Q1OutliningPass
from qat.experimental.dialect.pulse.ir import ConstantOp, CreateFrameOp, FrequencyAttr
from qat.experimental.dialect.q1 import StopOp
from qat.experimental.dialect.q1_sequence import SequenceOp


def _module_with_main(ops) -> ModuleOp:
    return ModuleOp([func.FuncOp("main", ((), ()), Region(Block(ops)))])


def test_default_pulse_to_q1_pipeline_runs_outlining_pass():
    """Verify that the default pipeline outlines one sequence per frame."""
    freq = ConstantOp(FrequencyAttr(4.8e9))
    frame = CreateFrameOp(freq, StringAttr("q0.drive"))
    module = _module_with_main([freq, frame, func.ReturnOp()])

    pipeline = create_default_pulse_to_q1_pipeline()
    pipeline.apply(Context(), module)

    [seq] = list(module.body.block.ops)
    assert isinstance(seq, SequenceOp)
    assert seq.channel_id.data == "q0.drive"
    assert isinstance(seq.body.block.last_op, StopOp)


def test_default_pulse_to_q1_pipeline_includes_lowering_pass():
    """Verify that the default pipeline contains both the outlining and lowering stages."""
    pipeline = create_default_pulse_to_q1_pipeline()

    assert len(pipeline.passes) == 2
    assert isinstance(pipeline.passes[0], Q1OutliningPass)
    assert isinstance(pipeline.passes[1], PulseToQ1LoweringPass)
