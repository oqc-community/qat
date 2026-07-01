# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import json
from dataclasses import dataclass
from io import StringIO
from typing import IO, Any

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.utils.target import Target

from qat.experimental.dialect.q1 import emit_program
from qat.experimental.dialect.q1_sequence.ir.attrs import (
    AcquisitionAttr,
    WaveformAttr,
    WeightAttr,
)
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp


def _waveform_to_dict(attr: WaveformAttr) -> dict[str, Any]:
    data = list(attr.data.iter_values())
    return {"data": data, "index": attr.index.data}


def _weight_to_dict(attr: WeightAttr) -> dict[str, Any]:
    data = list(attr.data.iter_values())
    return {"data": data, "index": attr.index.data}


def _acquisition_to_dict(
    attr: AcquisitionAttr,
) -> dict[str, Any]:
    return {
        "num_bins": attr.num_bins.data,
        "index": attr.index.data,
    }


def emit_sequence(seq_op: SequenceOp) -> dict[str, Any]:
    """Emits a Qblox Sequence dict for a single SequenceOp.

    :param seq_op: SequenceOp to emit.
    :returns: Dict matching ``qblox_instruments.types.Sequence``.
    :raises VerifyException: If the SequenceOp fails verification.
    """

    seq_op.verify()

    waveforms = {wf.waveform_name.data: _waveform_to_dict(wf) for wf in seq_op.waveforms}
    weights = {w.weight_name.data: _weight_to_dict(w) for w in seq_op.weights}
    acquisitions = {
        a.acquisition_name.data: _acquisition_to_dict(a) for a in seq_op.acquisitions
    }

    stream = StringIO()
    emit_program(seq_op.body, stream)
    program = stream.getvalue()

    return {
        "program": program,
        "waveforms": waveforms,
        "weights": weights,
        "acquisitions": acquisitions,
    }


def emit_module(
    module: ModuleOp,
) -> dict[str, dict[str, Any]]:
    """Emits all sequences in a module as a keyed dict.

    :param module: ModuleOp containing SequenceOps.
    :returns: ``{channel_id: Sequence_dict, ...}`` — one entry per
        sequencer/channel.
    """

    result: dict[str, dict[str, Any]] = {}
    for op in module.body.block.ops:
        if isinstance(op, SequenceOp):
            cid = op.channel_id.data
            if cid in result:
                raise ValueError(f"Duplicate channel_id '{cid}' in module")
            result[cid] = emit_sequence(op)
    return result


@dataclass(frozen=True)
class Q1SequenceTarget(Target):
    name = "q1_sequence"

    def emit(self, ctx: Context, module: ModuleOp, output: IO[str]) -> None:
        """Emits a Q1 sequence module as JSON.

        :param ctx: xDSL context for the emission target.
        :param module: Module containing SequenceOps.
        :param output: Text stream receiving the JSON output.
        """

        result = emit_module(module)
        json.dump(result, output)
