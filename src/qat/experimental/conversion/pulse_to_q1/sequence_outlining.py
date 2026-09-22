# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Q1 sequence outlining pass: partitions a Pulse entry block into per-frame sequence envelopes."""

from dataclasses import dataclass, field
from re import compile

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.scf import ForOp
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.conversion.pulse_to_q1.loop_fission import (
    fission_for_lineage,
    is_results_op,
)
from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    CreateFrameOp,
    FrameType,
    IntegrateOp,
)
from qat.experimental.dialect.pulse.transforms.partition_by_frame import (
    FrameLineage,
    FrameLineageAnalysis,
    build_frame_lineage_analysis,
)
from qat.experimental.dialect.pulse.utils import pulse_entry_block
from qat.experimental.dialect.q1 import SetMrkImmOp, StopOp, UI4Imm
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.passes.pass_ordering import OrderedPass

_NON_SYMBOL_CHARS = compile(r"[^0-9A-Za-z_$.]")
_MULTI_UNDERSCORE = compile(r"_+")

# TODO(COMPILER-1450): Move marker initialisation after physical binding and derive it
# from the module kind. This legacy RF-enable mask is not validated for QRC.
_MARKER_BITMASK = 0b0011


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


class OutliningState:
    """State carried by Pulse-to-Q1 outlining.

    This state records the sequence scaffolding introduced during the first stage of Pulse-
    to-Q1 conversion. Its organisation remains structurally parallel to the legacy
    QbloxProgram model, with one emitted sequence for each logical frame partition.

    :ivar frame_to_port: Partition metadata keyed by frame id.
    :ivar frame_to_sequence: Emitted sequence symbol keyed by frame id.
    """

    def __init__(self) -> None:
        self.frame_to_port: dict[str, str] = {}
        self.frame_to_sequence: dict[str, str] = {}


class _SymbolAllocator:
    """Allocate stable outlined-sequence symbols for a frame partition.

    When a physical channel token uniquely identifies one logical frame, the
    emitted symbol is derived from that token after normalisation. The
    normalised form is used only if it has not already been claimed by a prior
    partition. Otherwise the allocator falls back to deterministic ``frame_i``
    naming. When the token is shared across multiple frames, the fallback is
    applied unconditionally.

    All ``frame_i`` fallback names are pre-reserved on construction so that
    normalised tokens can never claim them. This guarantees that the fallback
    path always produces a unique symbol.

    :ivar symbol_counts: Number of logical frames mapped to each channel token.
    :ivar used_sequence_symbols: Symbols already emitted or reserved during the
        current outline run, used to detect normalisation collisions.
    """

    def __init__(
        self,
        symbol_counts: dict[str, int],
        used_sequence_symbols: set[str] | None = None,
    ) -> None:
        self.symbol_counts = symbol_counts
        self.used_sequence_symbols = (
            used_sequence_symbols if used_sequence_symbols is not None else set()
        )

    def allocate(
        self,
        frame_id: str,
        lineage: FrameLineage,
    ) -> tuple[str, str]:
        """Return the physical channel token and emitted sequence symbol."""

        channel_token = lineage.port
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
    body, including constants and other ops captured as free variables inside
    nested regions.
    """

    needed_ops: set[Operation] = set()
    pending_ops = list(lineage_ops)
    while pending_ops:
        op = pending_ops.pop()
        if op in needed_ops:
            continue
        needed_ops.add(op)
        if isinstance(op, AcquireOp):
            # Integration is a downstream semantic marker for the acquisition rather than
            # an operand dependency, so preserve it explicitly in the hardware partition.
            pending_ops.extend(
                use.operation
                for use in op.acquisition_result.uses
                if isinstance(use.operation, IntegrateOp)
            )
        # op.walk() includes op itself, so this also covers op's own operands as well as
        # free variables captured by any nested regions.
        for nested_op in op.walk():
            for operand in nested_op.operands:
                defining_op = op_by_result.get(operand)
                if defining_op is not None and defining_op not in needed_ops:
                    pending_ops.append(defining_op)
    return needed_ops


def _build_partition_sequence_body(
    entry_block_ops: list[Operation],
    lineage: FrameLineage,
    op_by_result: dict[SSAValue, Operation],
    entry_lineages: dict[Operation, list[FrameLineage]],
    analysis: FrameLineageAnalysis,
) -> list[Operation]:
    """Build the cloned body for one outlined sequence.

    The analysis result describes lineage membership. The outlining pass still needs to
    clone the supporting entry-block operations that feed that lineage so the emitted
    sequence body remains valid on its own.

    A region-bearing entry-block operation enclosing this lineage's work, such as the
    shot loop, is rebuilt from the operations belonging to this lineage alone rather than
    copied wholesale. That applies whether or not other lineages are enclosed too: a loop
    enclosing one frame still carries a results array the sequence must not keep.

    Entry-block results bookkeeping is never copied: a hardware sequence records
    acquisitions into bins and carries no results arrays, and nothing in the Q1 pipeline
    lowers them.
    """

    needed_ops = _partition_dependency_closure(lineage.entry_ops, op_by_result)
    for op in needed_ops:
        owners = entry_lineages.get(op, ())
        if len(owners) == 1 and owners[0] is not lineage:
            raise PassFailedException(
                f"{op.name} is owned by another frame lineage and cannot be cloned as "
                "a dependency"
            )

    value_mapper: dict[SSAValue, SSAValue] = {}
    sequence_body: list[Operation] = []
    for op in entry_block_ops:
        if op not in needed_ops or is_results_op(op):
            continue
        if _needs_rebuilding(op, lineage, entry_lineages, analysis):
            _reject_consumed_loop_results(op, needed_ops)
            sequence_body.append(fission_for_lineage(op, lineage, analysis, value_mapper))
        else:
            sequence_body.append(op.clone(value_mapper))
    return sequence_body


def _reject_consumed_loop_results(op: Operation, needed_ops: set[Operation]) -> None:
    """Reject a rebuilt loop whose results this partition still needs.

    A rebuilt loop carries nothing between iterations and so produces no results. Any
    operation in the partition consuming one would be left holding a value owned by the
    original loop, outside the emitted sequence.

    :raises PassFailedException: If a needed operation consumes a result of ``op``.
    """
    for result in op.results:
        for use in result.uses:
            if use.operation in needed_ops and not is_results_op(use.operation):
                raise PassFailedException(
                    f"{use.operation.name} consumes a result of {op.name}, which is "
                    "rebuilt without loop-carried values and so produces none"
                )


def _reject_frames_crossing_regions(module: ModuleOp) -> None:
    """Reject entry-block regions that take frames as block arguments.

    The lineage analysis cannot resolve a frame arriving as a block argument, so it
    attributes nothing inside such a region. Outlining would then emit sequences missing
    that region's work entirely, which is valid IR describing a different program.

    :raises PassFailedException: If a region-bearing entry-block operation has a block
        taking a frame argument.
    """
    for op in pulse_entry_block(module).ops:
        for region in op.regions:
            for block in region.blocks:
                if any(isinstance(arg.type, FrameType) for arg in block.args):
                    raise PassFailedException(
                        f"{op.name} takes a frame as a block argument; frame lineage "
                        "cannot be resolved across that boundary, so its body cannot be "
                        "outlined"
                    )


def _needs_rebuilding(
    op: Operation,
    lineage: FrameLineage,
    entry_lineages: dict[Operation, list[FrameLineage]],
    analysis: FrameLineageAnalysis,
) -> bool:
    """Return whether ``op`` must be rebuilt per partition rather than copied wholesale.

    Two cases need it. Several lineages cannot share one copy of a region-bearing operation,
    since each emitted sequence is independent. And a shot loop enclosing even a single
    lineage still carries results bookkeeping a hardware sequence must not keep.
    """
    if not op.regions:
        return False
    if len(entry_lineages.get(op, ())) > 1:
        return True
    return isinstance(op, ForOp) and any(
        lineage in analysis.lineages_for_op(inner) for inner in op.walk() if inner is not op
    )


@dataclass(frozen=True)
class Q1OutliningPass(OrderedPass, ModulePass):
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

    state: OutliningState = field(default_factory=OutliningState, init=False)

    def _sequence_op_for_partition(
        self,
        frame_id: str,
        lineage: FrameLineage,
        sequence_body: list[Operation],
        symbol_allocator: _SymbolAllocator,
    ) -> tuple[SequenceOp, str, str]:
        """Construct one outlined sequence together with its recorded metadata.

        :param frame_id: Synthetic frame label used for fallback naming.
        :param lineage: Frame lineage for this partition and its port metadata.
        :param sequence_body: Cloned operations to place in the sequence body. The body
            already contains the lineage ops plus any cloned definitions required to make
            the envelope self-contained.
        :param symbol_allocator: Symbol allocator for the outline run.
        :returns: The emitted sequence op, the physical channel token, and the final
            sequence symbol.
        """

        if not any(
            isinstance(nested, CreateFrameOp)
            for op in sequence_body
            for nested in op.walk()
        ):
            raise ValueError(f"Partition {frame_id} does not contain pulse.create_frame.")

        channel_token, sequence_symbol = symbol_allocator.allocate(frame_id, lineage)
        sequence_ops = [
            SetMrkImmOp(UI4Imm(_MARKER_BITMASK)),
            *sequence_body,
            StopOp(),
        ]
        return (
            SequenceOp(sequence_symbol, sequence_ops, port_id=channel_token),
            channel_token,
            sequence_symbol,
        )

    def _emit_sequence_ops(
        self, module: ModuleOp, analysis: FrameLineageAnalysis
    ) -> tuple[list[SequenceOp], dict[str, str], dict[str, str]]:
        """Emit sequence operations from frame-lineage analysis.

        Each logical frame partition yields one emitted sequence. When a
        physical channel token identifies a unique partition, the emitted
        symbol is derived from that token after symbol normalisation. When the
        token is shared or the derived symbol would collide, the pass falls
        back to deterministic ``frame_i`` naming.

        :param module: Pulse module containing the entry block to partition.
        :param analysis: Frame-lineage analysis computed for ``module``.
        :returns: Triple of emitted SequenceOp list, frame→port mapping, and
                  frame→sequence symbol mapping.
        """
        _reject_frames_crossing_regions(module)

        # Region-bearing control flow shared by several lineages is fissioned below.
        # A shared *pulse* operation is synchronisation that should already have been
        # lowered to per-sequencer waits; fission cannot preserve its semantics.
        if shared_ops := analysis.shared_ops:
            shared_op = shared_ops[0]
            raise PassFailedException(
                f"{shared_op.name} spans multiple frame lineages and cannot be outlined "
                "into independent Q1 sequences"
            )
        entry_lineages: dict[Operation, list[FrameLineage]] = {}
        for frame_lineage in analysis.lineages:
            for entry_op in frame_lineage.entry_ops:
                entry_lineages.setdefault(entry_op, []).append(frame_lineage)
        symbol_counts = analysis.port_counts
        n_frames = len(analysis.lineages)
        reserved = {f"frame_{i}" for i in range(n_frames)}
        symbol_allocator = _SymbolAllocator(symbol_counts, reserved)

        entry_block = pulse_entry_block(module)
        entry_block_ops = list(entry_block.ops)
        op_by_result = {result: op for op in entry_block_ops for result in op.results}

        sequences: list[SequenceOp] = []
        frame_to_port: dict[str, str] = {}
        frame_to_sequence: dict[str, str] = {}
        for frame_index, lineage in enumerate(analysis.lineages):
            # TODO(COMPILER-1379): Prefer the optional frame label once it is available.
            frame_id = f"frame_{frame_index}"
            sequence_body = _build_partition_sequence_body(
                entry_block_ops,
                lineage,
                op_by_result,
                entry_lineages,
                analysis,
            )
            sequence_op, channel_token, sequence_symbol = self._sequence_op_for_partition(
                frame_id,
                lineage,
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
        patterns later in the configured Q1 pipeline lower the Pulse payload
        inside those envelopes.
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
