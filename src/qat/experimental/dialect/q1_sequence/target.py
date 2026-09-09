# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd

from dataclasses import dataclass
from io import StringIO
from json import dump
from math import isfinite
from typing import IO, Any

from xdsl.context import Context
from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IntAttr,
    IntegerAttr,
    ModuleOp,
    NoneAttr,
    StringAttr,
    i1,
)
from xdsl.ir import Attribute, EnumAttribute, ParametrizedAttribute
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.target import Target

from qat.experimental.dialect.q1 import emit_program
from qat.experimental.dialect.q1.ir.imm_desc import Q1Imm
from qat.experimental.dialect.q1_sequence.ir.attrs import (
    AcquisitionAttr,
    ConfigAttr,
    ModuleConfigAttr,
    SequencerConfigAttr,
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


def _emit_attribute(attr: Attribute) -> Any:
    attr.verify()
    if isinstance(attr, NoneAttr):
        return None
    if isinstance(attr, StringAttr):
        return attr.data
    if isinstance(attr, FloatAttr):
        value = float(attr.value.data)
        if not isfinite(value):
            raise ValueError("Q1 sequence configuration floats must be finite")
        return value
    if isinstance(attr, IntegerAttr):
        value = attr.value.data
        return bool(value) if attr.type == i1 else value
    if isinstance(attr, IntAttr):
        return attr.data
    if isinstance(attr, Q1Imm):
        return attr.data
    if isinstance(attr, EnumAttribute):
        return attr.data.value
    if isinstance(attr, ArrayAttr):
        return [_emit_attribute(item) for item in attr]
    if isinstance(attr, ParametrizedAttribute):
        # TODO(COMPILER-1446): Replace implicit IRDL reflection with an explicit
        # target-side configuration emission contract.
        return {
            name: _emit_attribute(parameter)
            for (name, _), parameter in zip(
                type(attr).get_irdl_definition().parameters,
                attr.parameters,
                strict=True,
            )
        }
    raise TypeError(f"Cannot emit Q1 sequence attribute {type(attr).__name__}")


def emit_config(attr: ConfigAttr) -> dict[str, Any]:
    """Emit a Q1 sequence configuration attribute as runtime data.

    IRDL parameter declarations are the single configuration schema. This target translation
    recursively emits their names and values instead of copying each attribute into a
    parallel runtime model.

    :param attr: Module or sequencer configuration attribute to emit.
    :returns: JSON-compatible configuration mapping.
    :raises VerifyException: If required resolved routing state is absent.
    """

    attr.verify()
    if isinstance(attr, SequencerConfigAttr):
        for name in (
            "connections",
            "output_path_connections",
            "acquisition_path_connections",
            "disabled_outputs",
            "disabled_acquisition_paths",
        ):
            if isinstance(getattr(attr, name), NoneAttr):
                raise VerifyException(
                    f"SequencerConfigAttr.{name} must be resolved before emission"
                )

    emitted = _emit_attribute(attr)
    if not isinstance(emitted, dict):
        raise TypeError(f"Expected configuration mapping, got {type(emitted).__name__}")
    if isinstance(attr, ModuleConfigAttr):
        emitted["outputs"].sort(key=lambda item: item["output_id"])
        emitted["inputs"].sort(key=lambda item: item["input_id"])
        emitted["local_oscillators"].sort(key=lambda item: item["oscillator_id"])
    return emitted


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
        if not isinstance(op, SequenceOp):
            raise TypeError(
                "Q1 sequence emission requires top-level SequenceOps; "
                f"found {type(op).__name__}"
            )
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
        dump(result, output)
