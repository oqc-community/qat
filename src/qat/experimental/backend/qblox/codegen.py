# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Direct typed Qblox emission from configured experimental Q1 sequence IR."""

from __future__ import annotations

from collections import defaultdict

from pydantic import JsonValue
from xdsl.dialects.builtin import ModuleOp

from qat.backend.qblox.execution import DEFAULT_TIMEOUT_SECONDS, QbloxPackage, QbloxProgram
from qat.backend.qblox.target_data import TARGET_DATA, QbloxTargetData
from qat.experimental.backend.qblox.pre_emission_verification import (
    verify_qblox_pre_emission,
)
from qat.experimental.backend.qblox.translation import (
    translate_module_config,
    translate_package,
)
from qat.experimental.dialect.q1 import (
    AcquireWeightedImmRsRsRsImmOp,
    MoveImmRdOp,
    PlayRsRsImmOp,
)
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp


def _validate_table_references(sequence_op: SequenceOp) -> None:
    waveform_indices = {item.index.data for item in sequence_op.waveforms}
    weight_indices = {item.index.data for item in sequence_op.weights}

    for op in sequence_op.body.block.ops:
        if isinstance(op, PlayRsRsImmOp):
            for reference in (op.wave0, op.wave1):
                index = _resolve_static_table_index(sequence_op, reference)
                if not 0 <= index <= 1023:
                    raise ValueError(
                        f"Sequence {sequence_op.channel_id.data!r} has out-of-range "
                        f"waveform index {index}. Expected [0, 1023]"
                    )
                if index not in waveform_indices:
                    raise ValueError(
                        f"Sequence {sequence_op.channel_id.data!r} references missing "
                        f"waveform index {index}"
                    )
        if isinstance(op, AcquireWeightedImmRsRsRsImmOp):
            for reference in (op.weight_idx0, op.weight_idx1):
                index = _resolve_static_table_index(sequence_op, reference)
                if not 0 <= index <= 31:
                    raise ValueError(
                        f"Sequence {sequence_op.channel_id.data!r} has out-of-range weight "
                        f"index {index}. Expected [0, 31]"
                    )
                if index not in weight_indices:
                    raise ValueError(
                        f"Sequence {sequence_op.channel_id.data!r} references missing "
                        f"weight index {index}"
                    )


# TODO(COMPILER-1462): Replace this with reusable Q1 static value/bounds analysis.
def _resolve_static_table_index(sequence: SequenceOp, value) -> int:
    """Resolve table indices defined directly by a static immediate move.

    General SSA/register value bounds require a dedicated analysis outside code generation.
    """

    owner = value.owner
    if isinstance(owner, MoveImmRdOp):
        return owner.imm.data
    owner_name = getattr(owner, "name", "block argument")
    raise ValueError(
        f"Sequence {sequence.channel_id.data!r} has a non-static register table reference "
        f"from {owner_name}."
    )


def emit_qblox_program(
    module: ModuleOp,
    target_data: QbloxTargetData = TARGET_DATA,
    metadata: dict[str, JsonValue] | None = None,
) -> QbloxProgram:
    """Emit the shared Qblox runtime program from configured Q1 sequence IR.

    The module must satisfy Qblox pre-emission verification, which runs before any payload
    is constructed.

    :param module: Configured, allocated and flat Qblox Q1 sequence module.
    :param target_data: Qblox numeric limits used for final emission validation.
    :param metadata: Optional linker/runtime metadata reserved for later consumers.
    :returns: Compiler payload snapshotted from the verified sequence IR.
    """

    sequences = verify_qblox_pre_emission(module, target_data)

    module_sequences: dict[tuple[str, int], list[SequenceOp]] = defaultdict(list)
    for sequence_op in sequences:
        module_sequences[
            (sequence_op.instrument_id.data, sequence_op.slot_idx.data)
        ].append(sequence_op)

    translated_module_configs = {}
    for module_location, allocated_sequences in module_sequences.items():
        module_config = allocated_sequences[0].module_config
        translated_module_configs[module_location] = translate_module_config(
            module_config,
            (
                sequence_op.sequencer_config
                for sequence_op in allocated_sequences
                if sequence_op.sequencer_config is not None
            ),
        )

    packages: dict[str, QbloxPackage] = {}
    for sequence_op in sequences:
        _validate_table_references(sequence_op)
        module_location = (
            sequence_op.instrument_id.data,
            sequence_op.slot_idx.data,
        )
        package = translate_package(sequence_op, translated_module_configs[module_location])
        package_key = sequence_op.channel_id.data
        if package_key in packages:
            raise ValueError(f"Duplicate Qblox package key {package_key!r}")
        packages[package_key] = package

    return QbloxProgram(
        packages=packages,
        driver_version=target_data.driver_version,
        fw_version=target_data.fw_version,
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        metadata=metadata,
    )
