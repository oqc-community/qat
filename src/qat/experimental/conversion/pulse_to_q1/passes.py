# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Pass and pipeline definitions for the Pulse-to-Q1 conversion."""

from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass, PassPipeline
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriteWalker

from qat.backend.qblox.target_data import TARGET_DATA, QbloxTargetData
from qat.experimental.conversion.pulse_to_q1.rewrite_patterns import (
    create_pulse_to_q1_lowering_patterns,
)
from qat.experimental.conversion.pulse_to_q1.sequence_outlining import Q1OutliningPass


@dataclass(frozen=True)
class PulseToQ1LoweringPass(ModulePass):
    """Apply the Pulse-to-Q1 rewrite stage inside outlined sequences.

    `Q1OutliningPass` first isolates one logical sequence envelope for each
    frame partition. This pass then traverses those envelopes and applies the
    per-operation rewrite set that converts Pulse-level instructions into the
    flat Q1 instruction dialect.
    """

    name = "pulse-to-q1-lowering"
    target_data: QbloxTargetData = field(default=TARGET_DATA)

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                create_pulse_to_q1_lowering_patterns(self.target_data)
            ),
            apply_recursively=False,
        ).rewrite_module(op)


def create_default_pulse_to_q1_pipeline(
    target_data: QbloxTargetData = TARGET_DATA,
) -> PassPipeline:
    """Create the default pass pipeline for Pulse-to-Q1 conversion.

    The default pipeline proceeds in two stages. It first outlines the Pulse
    program into per-frame `q1_sequence.sequence` envelopes. It then applies
    the lowering rewrite set within those emitted sequences.

    :param target_data: Optional QBlox target description used by both stages.
    :returns: Pass pipeline for the default Pulse-to-Q1 conversion flow.
    """

    return PassPipeline(
        (
            Q1OutliningPass(target_data=target_data),
            PulseToQ1LoweringPass(target_data=target_data),
        )
    )
