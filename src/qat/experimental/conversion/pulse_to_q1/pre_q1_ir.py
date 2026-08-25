# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Intermediate ops bridging Pulse IR and Q1 for the Pulse-to-Q1 conversion.

These ops sit between the Pulse dialect and the flat Q1 instruction dialect. They capture
QBlox-specific acquisition context (such as the result store index and repetition count)
that is derived from the surrounding Pulse control flow, so that the final lowering to Q1
instructions can be a straightforward, context-free rewrite.
"""

from xdsl.dialects.builtin import IndexType, IntAttr, StringAttr
from xdsl.ir import Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    prop_def,
    result_def,
    traits_def,
)

from qat.experimental.dialect.pulse.ir import (
    AcquisitionType,
    AdvancesTimeTrait,
    FrameType,
    TimeType,
    WeightsAttr,
)


@irdl_op_definition
class PreQ1AcquireOp(IRDLOperation):
    """A pre-lowering form of ``pulse.acquire`` enriched with QBlox context.

    ``pulse.acquire`` describes only *what* to acquire; QBlox hardware additionally needs to
    know *where* the result is stored (the bin/store index) and *how many* times the
    acquisition repeats (derived from the enclosing loop nest).
    :class:`Q1PreAcquireTransformationPass` computes that context and replaces each
    ``pulse.acquire`` with this op, which :class:`RewritePreQ1AcquireOp` then lowers to a
    concrete Q1 acquire instruction.

    :ivar frame: The frame on which the acquisition is performed.
    :ivar duration: The acquisition duration, of type ``pulse.time``.
    :ivar store_idx: The ``index``-typed bin into which results are written.
    :ivar frame_result: The frame threaded to downstream time-ordered ops.
    :ivar acquisition_result: The acquisition result value.
    :ivar number_runs: The number of times the acquisition executes across the loop nest.
    :ivar weights: Optional integration weights.
    :ivar label: Optional label used for observability and debugging.
    """

    name = "pre_q1_pulse.acquire"
    traits = traits_def(AdvancesTimeTrait())

    frame = operand_def(FrameType)
    duration = operand_def(TimeType)
    store_idx = operand_def(IndexType)
    frame_result = result_def(FrameType)
    acquisition_result = result_def(AcquisitionType)
    number_runs = prop_def(IntAttr)
    weights = opt_attr_def(WeightsAttr)
    label = opt_attr_def(StringAttr)

    def __init__(
        self,
        frame: SSAValue | Operation,
        duration: SSAValue | Operation,
        store_idx: SSAValue | Operation,
        number_runs: int | IntAttr,
        weights: WeightsAttr | None = None,
        label: str | StringAttr | None = None,
    ) -> None:
        """Build a ``pre_q1_pulse.acquire`` op.

        :param frame: The SSA value representing the frame on which to perform the
            acquisition.
        :param duration: The SSA value representing the duration of the acquisition, of type
            ``pulse.time``.
        :param store_idx: The ``index``-typed SSA value giving the bin into which the
            acquisition result is stored.
        :param number_runs: The total number of acquisition repetitions across the enclosing
            loop nest, used to size the acquisition's bin allocation.
        :param weights: Optional weights attribute for the acquisition.
        :param label: Optional string attribute used to label the acquisition for
            observability and debugging.
        """
        frame_ssa = SSAValue.get(frame, type=FrameType)
        duration_ssa = SSAValue.get(duration, type=TimeType)
        store_idx_ssa = SSAValue.get(store_idx, type=IndexType)

        attributes = {} if weights is None else {"weights": weights}
        if label is not None:
            attributes["label"] = StringAttr(label) if isinstance(label, str) else label

        super().__init__(
            operands=[frame_ssa, duration_ssa, store_idx_ssa],
            result_types=[frame_ssa.type, AcquisitionType()],
            attributes=attributes,
            properties={
                "number_runs": IntAttr(number_runs)
                if isinstance(number_runs, int)
                else number_runs
            },
        )
