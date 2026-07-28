# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Q1 sequence outlining pass: partitions a Pulse entry block into per-frame sequence envelopes."""

import re
from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass

from qat.backend.qblox.target_data import TARGET_DATA, QbloxTargetData
from qat.experimental.dialect.pulse.ir import CreateFrameOp
from qat.experimental.dialect.pulse.transforms.partition_by_frame import (
    FrameLineageAnalysis,
    build_frame_lineage_analysis,
)
from qat.experimental.dialect.pulse.utils import pulse_entry_block
from qat.experimental.dialect.q1 import StopOp
from qat.experimental.dialect.q1_sequence import SequenceOp

_NON_SYMBOL_CHARS = re.compile(r"[^0-9A-Za-z_$.]")
_MULTI_UNDERSCORE = re.compile(r"_+")


def _normalize_sequence_symbol(channel_token: str) -> str:
    """Normalize a channel token to a valid xDSL symbol.

    xDSL symbols are restricted to alphanumeric, underscore, dollar, and dot.
    QBlox channels use `/` separators (e.g., `q0/drive`), which are replaced with
    underscores. Leading digits are prefixed with `_`.

    :param channel_token: Channel identifier to normalize.
    :returns: Valid symbol name, or fallback to `sequence` if normalization
              yields empty string.
    """
    symbol = _NON_SYMBOL_CHARS.sub("_", channel_token)
    symbol = _MULTI_UNDERSCORE.sub("_", symbol).strip("_")
    if not symbol:
        return "sequence"
    if symbol[0].isdigit():
        return f"_{symbol}"
    return symbol


@dataclass
class OutliningState:
    """State carried by Pulse-to-Q1 outlining.

    This state records the sequence scaffolding introduced during the first stage of Pulse-
    to-Q1 conversion. Its organisation remains structurally parallel to the legacy
    QbloxProgram model, with one emitted sequence for each logical frame partition.

    :ivar frame_to_port: Partition metadata keyed by frame id.
    :ivar frame_to_sequence: Emitted sequence symbol keyed by frame id.
    """

    frame_to_port: dict[str, str] = field(default_factory=dict)
    frame_to_sequence: dict[str, str] = field(default_factory=dict)


@dataclass
class _SymbolAllocator:
    """Allocate stable outlined-sequence symbols for a frame partition.

    When a physical channel token uniquely identifies one logical frame, the
    emitted symbol is derived from that token after normalisation. The
    normalised form is used only if it has not already been claimed by a prior
    partition; otherwise the allocator falls back to deterministic ``frame_i``
    naming. When the token is shared across multiple frames, the fallback is
    applied unconditionally.

    All ``frame_i`` fallback names are pre-reserved on construction so that
    normalised tokens can never claim them. This guarantees that the fallback
    path always produces a unique symbol.

    :ivar symbol_counts: Number of logical frames mapped to each channel token.
    :ivar used_sequence_symbols: Symbols already emitted or reserved during the
        current outline run, used to detect normalisation collisions.
    """

    symbol_counts: dict[str, int]
    used_sequence_symbols: set[str] = field(default_factory=set)

    def allocate(
        self,
        frame_id: str,
        frame: SSAValue,
        analysis: FrameLineageAnalysis,
    ) -> tuple[str, str]:
        """Return the physical channel token and emitted sequence symbol."""

        channel_token = analysis.frame_to_port[frame]
        sequence_symbol = frame_id
        if self.symbol_counts[channel_token] == 1:
            normalized_symbol = _normalize_sequence_symbol(channel_token)
            if normalized_symbol not in self.used_sequence_symbols:
                sequence_symbol = normalized_symbol

        self.used_sequence_symbols.add(sequence_symbol)
        return channel_token, sequence_symbol


def _partition_dependency_closure(
    lineage_ops: tuple[Operation, ...],
    op_by_result: dict[SSAValue, Operation],
) -> set[Operation]:
    """Return the transitive dependency closure for one frame partition.

    ``FrameLineageAnalysis`` identifies the operations that belong to a frame
    lineage. This helper expands that membership to include any entry-block
    definitions required to clone the lineage into a self-contained sequence
    body.
    """

    needed_ops: set[Operation] = set()
    pending_ops = list(lineage_ops)
    while pending_ops:
        op = pending_ops.pop()
        if op in needed_ops:
            continue
        needed_ops.add(op)
        for operand in op.operands:
            defining_op = op_by_result.get(operand)
            if defining_op is not None and defining_op not in needed_ops:
                pending_ops.append(defining_op)
    return needed_ops


def _build_partition_sequence_body(
    entry_block_ops: list[Operation],
    lineage_ops: tuple[Operation, ...],
    op_by_result: dict[SSAValue, Operation],
) -> list[Operation]:
    """Build the cloned body for one outlined sequence.

    The analysis result describes lineage membership. The outlining pass still needs to
    clone the supporting entry-block operations that feed that lineage so the emitted
    sequence body remains valid on its own.
    """

    needed_ops = _partition_dependency_closure(lineage_ops, op_by_result)
    value_mapper: dict[SSAValue, SSAValue] = {}
    return [op.clone(value_mapper) for op in entry_block_ops if op in needed_ops]


@dataclass(frozen=True)
class Q1OutliningPass(ModulePass):
    """Outline one q1_sequence per logical Pulse frame.

    This pass partitions the Pulse instruction stream by logical frame lineage
    and emits one `q1_sequence.sequence` operation for each partition. The
    resulting structure mirrors the legacy QbloxProgram organisation at the
    scaffolding level and establishes the unit on which subsequent lowering
    rewrites operate. The emitted symbol acts as a stable handle in the xDSL
    symbol table, which keeps the outlined sequence observable and makes later
    references explicit.

    Example::

        pulse.create_frame %freq, "q0/drive"
        pulse.create_frame %freq, "q1/drive"

    becomes two independent sequence envelopes::

        q1_sequence.sequence @q0_drive { q1.stop }
        q1_sequence.sequence @q1_drive { q1.stop }
    """

    name = "pulse-to-q1-outlining"
    target_data: QbloxTargetData = field(default=TARGET_DATA)
    state: OutliningState = field(default_factory=OutliningState, init=False)

    def _sequence_op_for_partition(
        self,
        frame_id: str,
        frame: SSAValue,
        analysis: FrameLineageAnalysis,
        sequence_body: list[Operation],
        symbol_allocator: _SymbolAllocator,
    ) -> tuple[SequenceOp, str, str]:
        """Construct one outlined sequence together with its recorded metadata.

        :param frame_id: Synthetic frame label used for fallback naming.
        :param frame: Root SSA value for the logical frame partition.
        :param analysis: Frame-lineage analysis for the enclosing module. This provides the
            lineage membership and physical-port metadata.
        :param sequence_body: Cloned operations to place in the sequence body. The body
            already contains the lineage ops plus any cloned definitions required to make
            the envelope self-contained.
        :param symbol_allocator: Symbol allocator for the outline run.
        :returns: The emitted sequence op, the physical channel token, and the final
            sequence symbol.
        """

        if not any(isinstance(op, CreateFrameOp) for op in sequence_body):
            raise ValueError(f"Partition {frame_id} does not contain pulse.create_frame.")

        channel_token, sequence_symbol = symbol_allocator.allocate(
            frame_id, frame, analysis
        )
        sequence_ops = [*sequence_body, StopOp()]
        return SequenceOp(sequence_symbol, sequence_ops), channel_token, sequence_symbol

    def _emit_sequence_ops(
        self, module: ModuleOp, analysis: FrameLineageAnalysis
    ) -> tuple[list[SequenceOp], dict[str, str], dict[str, str]]:
        """Emit sequence operations from frame-lineage analysis.

        Each logical frame partition yields one emitted sequence. When a
        physical channel token identifies a unique partition, the emitted
        symbol is derived from that token after symbol normalisation. When the
        token is shared or the derived symbol would collide, the pass falls
        back to deterministic ``frame_i`` naming.

        :param analysis: Frame-lineage analysis computed by FrameLineagePass.
        :returns: Triple of emitted SequenceOp list, frame→port mapping, and
                  frame→sequence symbol mapping.
        """
        symbol_counts: dict[str, int] = {}
        for symbol in analysis.frame_to_port.values():
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        n_frames = len(analysis.frame_to_operations)
        reserved = {f"frame_{i}" for i in range(n_frames)}
        symbol_allocator = _SymbolAllocator(symbol_counts, reserved)

        entry_block = pulse_entry_block(module)
        entry_block_ops = list(entry_block.ops)
        op_by_result = {result: op for op in entry_block_ops for result in op.results}

        sequences: list[SequenceOp] = []
        frame_to_port: dict[str, str] = {}
        frame_to_sequence: dict[str, str] = {}
        for frame_index, (frame, lineage_ops) in enumerate(
            analysis.frame_to_operations.items()
        ):
            # TODO - Reuse a frame's optional identifier once: COMPILER-1379
            frame_id = f"frame_{frame_index}"
            sequence_body = _build_partition_sequence_body(
                entry_block_ops,
                lineage_ops,
                op_by_result,
            )
            sequence_op, channel_token, sequence_symbol = self._sequence_op_for_partition(
                frame_id,
                frame,
                analysis,
                sequence_body,
                symbol_allocator,
            )

            frame_to_port[frame_id] = channel_token
            sequences.append(sequence_op)
            frame_to_sequence[frame_id] = sequence_symbol
        return sequences, frame_to_port, frame_to_sequence

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        """Replace the Pulse entry stream with one outlined sequence per frame.

        This stage detaches the entry block operations from the module and
        replaces them with ``q1_sequence.sequence`` envelopes that carry the
        frame-local operations for each lineage. Each emitted sequence body is
        self-contained and ends with ``q1.stop``. The per-operation rewrite
        patterns introduced by COMPILER-1343–1346 will later lower the Pulse
        payload inside those envelopes.
        """
        analysis = build_frame_lineage_analysis(op)
        sequence_ops, frame_to_port, frame_to_sequence = self._emit_sequence_ops(
            op, analysis
        )
        self.state.frame_to_port = frame_to_port
        self.state.frame_to_sequence = frame_to_sequence

        module_block = op.body.block
        for old_op in list(module_block.ops):
            old_op.detach()
        module_block.add_ops(sequence_ops)
