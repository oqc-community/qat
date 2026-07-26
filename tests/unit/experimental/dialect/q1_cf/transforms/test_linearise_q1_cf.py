# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Unit tests for the q1_cf to q1 linearisation pass.

The pass collapses a multi-block ``q1_cf`` CFG inside a ``SequenceOp`` into a
single flat ``q1`` block whose control transfer is by label.

Coverage:
* Result shape: one block, only q1 ops, ends in ``Stop*``, no q1_cf residue.
* Terminator lowering: each conditional predicate selects the correct
  conditional jump after ``cmp``/``test``; the loop back-edge becomes a counted
  ``loop``.
* Constant folding plus unreachable pruning collapses a decided branch.
* Fall-through elision drops a jump to the layout-next block.
* Block-argument erasure coalesces an incoming register without emitting a move,
  and rejects a body whose incoming register cannot be coalesced by a rename.
* Halts of one terminal state converge to a single terminal op: clean ``stop``
  variants fold together regardless of code, and ``illegal`` traps fold together;
  a body exiting through both a ``stop`` and an ``illegal`` is rejected.
* The pass rewrites every sequence in a module and rejects a body with no
  halting path.
"""

from __future__ import annotations

from io import StringIO

import pytest
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, Operation, Region
from xdsl.utils.exceptions import PassFailedException
from xdsl.utils.test_value import create_ssa_value

from qat.experimental.dialect.q1 import (
    IllegalOp,
    JmpImmOp,
    LabelOp,
    LoopRdImmOp,
    MoveImmRdOp,
    NotRsRdOp,
    Registers,
    StopImmOp,
    StopOp,
    StopRsOp,
)
from qat.experimental.dialect.q1.ir.imm_desc import SI32Imm, SU32Imm
from qat.experimental.dialect.q1.target import emit_program
from qat.experimental.dialect.q1_cf import (
    BinaryPredicate,
    BinaryPredicateBranchOp,
    JmpBranchOp,
    LoopBranchOp,
    UnaryPredicate,
    UnaryPredicateBranchOp,
)
from qat.experimental.dialect.q1_cf.transforms.linearise_q1_cf import (
    LineariseQ1CfToQ1Pass,
    _assemble_as_single_block,
    _coalescing_rename_block_arguments,
    _fold_and_prune,
    _fold_branch,
    _fresh_aux_label,
    _fresh_labels,
    _Layout,
    _linearise_sequence,
    _lower_terminator,
    _primary_incoming,
    _referenced_label,
)
from qat.experimental.dialect.q1_sequence import SequenceOp

R0, R1, R2 = Registers.R0, Registers.R1, Registers.R2


def _linearise(seq: SequenceOp) -> list[str]:
    """Run the pass on a lone sequence and return its emitted assembly lines.

    The result is verified to hold the pass's structural contract: a single
    block of q1 ops ending in a halt (``Stop*`` or ``illegal``) with no q1_cf
    residue.
    """
    _linearise_sequence(seq)
    seq.verify()
    assert len(seq.body.blocks) == 1
    body = seq.body.blocks[0]
    assert isinstance(body.last_op, StopOp | StopImmOp | StopRsOp | IllegalOp)
    for op in body.ops:
        assert op.dialect_name() == "q1"
    output = StringIO()
    emit_program(seq.body, output)
    return [line.strip() for line in output.getvalue().splitlines() if line.strip()]


def _opaque(seed_value: int, reg):
    """Return an op whose result is opaque to constant folding, in ``reg``.

    A ``move`` feeds an ALU op so the result is not a direct immediate move and
    therefore never folds, forcing the branch to lower rather than resolve.
    """
    seed = MoveImmRdOp(SU32Imm(seed_value), reg)
    return seed, NotRsRdOp(seed.rd, reg)


def _binary_predicate_diamond(
    predicate: BinaryPredicate, channel: str = "Q0"
) -> SequenceOp:
    """A diamond whose entry branches on an opaque comparison of two registers."""
    merge = Block([StopOp()])
    then_b = Block([JmpBranchOp([], merge)])
    else_b = Block([JmpBranchOp([], merge)])
    lhs_seed, lhs = _opaque(1, R0)
    rhs_seed, rhs = _opaque(2, R1)
    entry = Block(
        [
            lhs_seed,
            lhs,
            rhs_seed,
            rhs,
            BinaryPredicateBranchOp(predicate, lhs.rd, rhs.rd, [], [], then_b, else_b),
        ]
    )
    return SequenceOp(channel, Region([entry, else_b, then_b, merge]))


def _unary_predicate_diamond(predicate: UnaryPredicate, channel: str = "Q0") -> SequenceOp:
    """A diamond whose entry branches on an opaque unary predicate test."""
    merge = Block([StopOp()])
    then_b = Block([JmpBranchOp([], merge)])
    else_b = Block([JmpBranchOp([], merge)])
    seed, rs = _opaque(1, R0)
    entry = Block(
        [seed, rs, UnaryPredicateBranchOp(predicate, rs.rd, [], [], then_b, else_b)]
    )
    return SequenceOp(channel, Region([entry, else_b, then_b, merge]))


def test_single_block_sequence_is_left_unchanged():
    body = Block([MoveImmRdOp(SU32Imm(3), R0), StopOp()])
    seq = SequenceOp("Q0", Region([body]))

    _linearise_sequence(seq)

    assert seq.body.blocks[0] is body
    output = StringIO()
    emit_program(seq.body, output)
    lines = [line.strip() for line in output.getvalue().splitlines() if line.strip()]
    assert lines == ["move 3, R0", "stop"]


def test_diamond_lowers_to_single_flat_block():
    lines = _linearise(_binary_predicate_diamond(BinaryPredicate.slt))

    assert lines == [
        "move 1, R0",
        "not R0, R0",
        "move 2, R1",
        "not R1, R1",
        "cmp R0, R1",
        "jl @bb2",
        "jmp @bb3",
        "bb2:",
        "bb3:",
        "stop",
    ]


_BINARY_PREDICATE_JUMPS = [
    (BinaryPredicate.eq, "jz"),
    (BinaryPredicate.ne, "jnz"),
    (BinaryPredicate.slt, "jl"),
    (BinaryPredicate.sle, "jle"),
    (BinaryPredicate.sgt, "jg"),
    (BinaryPredicate.sge, "jge"),
    (BinaryPredicate.ult, "jb"),
    (BinaryPredicate.ule, "jbe"),
    (BinaryPredicate.ugt, "ja"),
    (BinaryPredicate.uge, "jae"),
]


@pytest.mark.parametrize("predicate,mnemonic", _BINARY_PREDICATE_JUMPS)
def test_binary_predicate_selects_conditional_jump(predicate, mnemonic):
    lines = _linearise(_binary_predicate_diamond(predicate))

    assert "cmp R0, R1" in lines
    assert f"{mnemonic} @bb2" in lines


_UNARY_PREDICATE_JUMPS = [
    (UnaryPredicate.eqz, "jz"),
    (UnaryPredicate.nez, "jnz"),
    (UnaryPredicate.ltz, "js"),
    (UnaryPredicate.gez, "jns"),
]


@pytest.mark.parametrize("predicate,mnemonic", _UNARY_PREDICATE_JUMPS)
def test_unary_predicate_selects_conditional_jump(predicate, mnemonic):
    lines = _linearise(_unary_predicate_diamond(predicate))

    assert "test R0, R0" in lines
    assert f"{mnemonic} @bb2" in lines


def test_constant_comparison_folds_and_prunes():
    merge = Block([StopImmOp(SI32Imm(0))])
    then_b = Block([JmpBranchOp([], merge)])
    else_b = Block([JmpBranchOp([], merge)])
    lhs = MoveImmRdOp(SU32Imm(3), R0)
    rhs = MoveImmRdOp(SU32Imm(5), R1)
    entry = Block(
        [
            lhs,
            rhs,
            BinaryPredicateBranchOp(
                BinaryPredicate.slt, lhs.rd, rhs.rd, [], [], then_b, else_b
            ),
        ]
    )
    seq = SequenceOp("Q0", Region([entry, else_b, then_b, merge]))

    lines = _linearise(seq)

    assert not any(line.startswith("cmp") for line in lines)
    assert not any(line.startswith("jl") for line in lines)
    assert lines == ["move 3, R0", "move 5, R1", "stop 0"]


def test_constant_unary_predicate_branch_folds():
    merge = Block([StopOp()])
    then_b = Block([JmpBranchOp([], merge)])
    else_b = Block([JmpBranchOp([], merge)])
    rs = MoveImmRdOp(SU32Imm(0), R0)
    entry = Block(
        [rs, UnaryPredicateBranchOp(UnaryPredicate.eqz, rs.rd, [], [], then_b, else_b)]
    )
    seq = SequenceOp("Q0", Region([entry, else_b, then_b, merge]))

    lines = _linearise(seq)

    assert not any(line.startswith("test") for line in lines)
    assert not any(line.startswith("jz") for line in lines)


def test_fold_branch_unary_false_selects_else_edge():
    merge = Block([StopOp()])
    then_b = Block([JmpBranchOp([], merge)])
    else_b = Block([JmpBranchOp([], merge)])
    rs = MoveImmRdOp(SU32Imm(1), R0)
    term = UnaryPredicateBranchOp(UnaryPredicate.eqz, rs.rd, [], [], then_b, else_b)

    folded = _fold_branch(term)

    assert isinstance(folded, JmpBranchOp)
    assert folded.successor is else_b


def test_fold_branch_binary_false_selects_else_edge():
    merge = Block([StopOp()])
    then_b = Block([JmpBranchOp([], merge)])
    else_b = Block([JmpBranchOp([], merge)])
    lhs = MoveImmRdOp(SU32Imm(5), R0)
    rhs = MoveImmRdOp(SU32Imm(3), R1)
    term = BinaryPredicateBranchOp(
        BinaryPredicate.slt, lhs.rd, rhs.rd, [], [], then_b, else_b
    )

    folded = _fold_branch(term)

    assert isinstance(folded, JmpBranchOp)
    assert folded.successor is else_b


def test_fold_and_prune_rejects_detached_entry():
    with pytest.raises(AssertionError, match="detached from its region"):
        _fold_and_prune(Block())


def test_fold_and_prune_handles_block_without_terminator():
    entry = Block()
    tail = Block([StopOp()])
    region = Region([entry, tail])

    _fold_and_prune(region.blocks[0])

    assert list(region.blocks) == [entry]


def test_loop_lowers_to_counted_loop():
    exit_b = Block([StopOp()])
    body = Block(arg_types=[R2])
    body.add_op(LoopBranchOp(body.args[0], [body.args[0]], [], body, exit_b))
    init = MoveImmRdOp(SU32Imm(10), R2)
    entry = Block([init, JmpBranchOp([init.rd], body)])
    seq = SequenceOp("Q0", Region([entry, body, exit_b]))

    lines = _linearise(seq)

    assert lines == [
        "move 10, R2",
        "bb1:",
        "loop R2, @bb1",
        "stop",
    ]


def test_fall_through_edge_emits_no_jump():
    """The else edge of a diamond is the layout neighbour, so it needs no jump."""
    seq = _binary_predicate_diamond(BinaryPredicate.slt)

    lines = _linearise(seq)

    # A jump follows the conditional jump only for the then and merge edges,
    # never for the elided fall-through into the else block.
    assert lines.count("jl @bb2") == 1
    assert "bb1:" not in lines


def test_block_argument_coalesced_without_move():
    """A block argument fed from its own register needs no move on erasure."""
    exit_b = Block([StopImmOp(SI32Imm(7))])
    body = Block(arg_types=[R2])
    body.add_op(JmpBranchOp([], exit_b))
    init = MoveImmRdOp(SU32Imm(7), R2)
    entry = Block([init, JmpBranchOp([init.rd], body)])
    seq = SequenceOp("Q0", Region([entry, body, exit_b]))

    lines = _linearise(seq)

    assert not any("move R" in line for line in lines)
    assert lines == ["move 7, R2", "stop 7"]


def test_register_mismatch_is_rejected():
    """A forwarded value in the wrong register cannot be coalesced by a rename.

    q1_cf requires per-edge register equality, so this state cannot arise from a verified
    body. The pass rejects it rather than emit a move that a later single-block form would
    have to sequence, since chained or cyclic moves on an edge are not expressible without a
    scratch register.
    """
    exit_b = Block([StopOp()])
    body = Block(arg_types=[R2])
    body.add_op(JmpBranchOp([], exit_b))
    init = MoveImmRdOp(SU32Imm(7), R1)
    entry = Block([init, JmpBranchOp([init.rd], body)])
    seq = SequenceOp("Q0", Region([entry, body, exit_b]))

    with pytest.raises(PassFailedException, match="register-coalesced"):
        _linearise_sequence(seq)


def test_entry_block_argument_is_rejected():
    """An entry block argument has no incoming edge to coalesce against.

    Only the entry lacks predecessors, so an argument on it cannot be sourced by a rename. A
    well-formed sequence entry takes no arguments, and the pass rejects the malformed case
    with a controlled error rather than an index fault.
    """
    exit_b = Block([StopOp()])
    entry = Block(arg_types=[R2])
    entry.add_op(JmpBranchOp([], exit_b))
    seq = SequenceOp("Q0", Region([entry, exit_b]))

    with pytest.raises(PassFailedException, match="no incoming edge"):
        _linearise_sequence(seq)


def test_pass_lowers_every_sequence_in_module():
    module = ModuleOp(
        [
            _binary_predicate_diamond(BinaryPredicate.eq, channel="Q0"),
            _unary_predicate_diamond(UnaryPredicate.nez, channel="Q1"),
        ]
    )

    LineariseQ1CfToQ1Pass().apply(Context(), module)
    module.verify()

    for seq in module.body.walk():
        if isinstance(seq, SequenceOp):
            assert len(seq.body.blocks) == 1


def test_body_without_halting_path_is_rejected():
    body = Block()
    body.add_op(JmpBranchOp([], body))
    seq = SequenceOp("Q0", Region([body, Block([JmpBranchOp([], body)])]))

    with pytest.raises(PassFailedException, match="no halt terminator"):
        _linearise_sequence(seq)


def test_primary_incoming_can_take_backward_only_edge():
    block = Block(arg_types=[R2])
    pred = Block()
    incoming = create_ssa_value(R2)
    layout = _Layout(
        ordered=[block, pred],
        index_of={block: 0, pred: 1},
        label_of={block: "bb0", pred: "bb1"},
        predecessors={block: [(pred, [incoming])], pred: []},
        sink=pred,
    )

    assert _primary_incoming(layout, block, 0) is incoming


def test_coalescing_rejects_self_fed_block_argument():
    block = Block(arg_types=[R2])
    pred = Block()
    seq = SequenceOp("Q0", Region([block, pred]))
    layout = _Layout(
        ordered=[block, pred],
        index_of={block: 0, pred: 1},
        label_of={block: "bb0", pred: "bb1"},
        predecessors={block: [(pred, [block.args[0]])], pred: []},
        sink=pred,
    )

    with pytest.raises(PassFailedException, match="only fed from itself"):
        _coalescing_rename_block_arguments(seq, layout)


def test_lower_terminator_rejects_block_without_terminator():
    block = Block()
    sink = Block([StopOp()])
    layout = _Layout(
        ordered=[block, sink],
        index_of={block: 0, sink: 1},
        label_of={block: "bb0", sink: "bb1"},
        predecessors={block: [], sink: []},
        sink=sink,
    )

    with pytest.raises(AssertionError, match="no terminator to lower"):
        _lower_terminator(layout, block, None, [], layout.label_of[sink])


def test_fresh_aux_label_deconflicts_defined_and_referenced_labels():
    block = Block([LabelOp("halt"), JmpImmOp("halt_1"), StopOp()])
    sink = Block([StopOp()])
    layout = _Layout(
        ordered=[block, sink],
        index_of={block: 0, sink: 1},
        label_of={block: "bb0", sink: "bb1"},
        predecessors={block: [], sink: []},
        sink=sink,
    )

    fresh = _fresh_aux_label(layout, "halt")
    assert fresh not in {"halt", "halt_1", "bb0", "bb1"}


def test_assemble_as_single_block_rejects_sink_without_halt():
    sink = Block()
    layout = _Layout(
        ordered=[sink],
        index_of={sink: 0},
        label_of={sink: "bb0"},
        predecessors={sink: []},
        sink=sink,
    )

    with pytest.raises(AssertionError, match="sink block has no halt"):
        _assemble_as_single_block(layout)


def _early_halt_diamond(early: Operation, merge: Operation | None = None) -> SequenceOp:
    """A flag diamond whose taken block halts early and whose merge halts.

    ``early`` is the terminator of the taken block and ``merge`` that of the merge
    block, defaulting to a clean ``stop``. The two halting paths exercise
    convergence to a single terminal halt and rejection of mixed terminal states.
    """
    merge_b = Block([merge if merge is not None else StopOp()])
    else_b = Block([JmpBranchOp([], merge_b)])
    then_b = Block([early])
    seed, rs = _opaque(1, R0)
    entry = Block(
        [
            seed,
            rs,
            UnaryPredicateBranchOp(UnaryPredicate.nez, rs.rd, [], [], then_b, else_b),
        ]
    )
    return SequenceOp("Q0", Region([entry, else_b, then_b, merge_b]))


def test_early_halt_redirects_to_single_stop():
    """A conditional early ``stop`` folds into the single terminal ``stop``."""
    lines = _linearise(_early_halt_diamond(StopOp()))

    assert lines.count("stop") == 1
    assert lines[-1] == "stop"


def test_distinct_early_halt_converges_to_terminal_stop():
    """An early ``stop`` with any code redirects to the terminal halt.

    A stop halts the sequencer identically on every path, so the early halt is
    not a distinct exit to preserve. It lowers to a jump onto the single terminal
    ``stop``, and its own code does not survive.
    """
    lines = _linearise(_early_halt_diamond(StopImmOp(SI32Imm(1))))

    assert sum(line.startswith("stop") for line in lines) == 1
    assert "stop 1" not in lines
    assert lines[-1] == "stop"


def test_early_halt_redirect_avoids_sink_prelude_ops():
    """An early halt must not be redirected through sink pre-halt instructions."""
    merge = Block([MoveImmRdOp(SU32Imm(9), R1), StopOp()])
    else_b = Block([JmpBranchOp([], merge)])
    then_b = Block([StopImmOp(SI32Imm(1))])
    seed, rs = _opaque(1, R0)
    entry = Block(
        [
            seed,
            rs,
            UnaryPredicateBranchOp(UnaryPredicate.nez, rs.rd, [], [], then_b, else_b),
        ]
    )
    seq = SequenceOp("Q0", Region([entry, else_b, then_b, merge]))

    lines = _linearise(seq)

    assert "move 9, R1" in lines
    assert any(line.endswith("_halt:") for line in lines)
    assert any(line.startswith("jmp @") and "_halt" in line for line in lines)
    assert sum(line.startswith("stop") for line in lines) == 1
    assert lines[-1] == "stop"


def test_illegal_traps_converge_to_single_illegal():
    """Two ``illegal`` traps share one terminal state and converge to one op."""
    lines = _linearise(_early_halt_diamond(IllegalOp(), IllegalOp()))

    assert lines.count("illegal") == 1
    assert "stop" not in lines
    assert lines[-1] == "illegal"


def test_mixed_stop_and_illegal_is_rejected():
    """A body exiting through both a clean ``stop`` and an ``illegal`` is rejected.

    An ``illegal`` raises an error flag that drives the sequencer status to error,
    a terminal state distinct from a clean ``stop``. A single flat block admits one
    terminal halt and neither state redirects to the other without changing the
    observed status, so the pass rejects the body rather than mask a trap.
    """
    seq = _early_halt_diamond(IllegalOp())

    with pytest.raises(PassFailedException, match="illegal trap"):
        _linearise_sequence(seq)


def test_fresh_labels_preserve_scheme_when_no_labels_exist():
    """With no label already present, blocks take the plain ``bb{i}`` scheme."""
    blocks = [
        Block([StopOp()]),
        Block([StopImmOp(SI32Imm(42))]),
        Block([StopRsOp(create_ssa_value(R0))]),
    ]

    labels = _fresh_labels(blocks)

    assert [labels[block] for block in blocks] == ["bb0", "bb1", "bb2"]


def test_fresh_labels_deconflict_against_existing_definition_and_reference():
    """Generated names skip labels a body already defines or a jump references.

    A ``q1.x.label`` definition and a jump's label operand both reserve a name.
    The counter steps over the reserved names, so no generated label duplicates a
    definition or silently retargets an existing jump.
    """
    reserving = Block([LabelOp("bb0"), JmpImmOp("bb2"), StopOp()])
    plain = Block([StopOp()])
    blocks = [reserving, plain]

    labels = _fresh_labels(blocks)

    assert set(labels.values()).isdisjoint({"bb0", "bb2"})
    assert len(set(labels.values())) == len(blocks)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (JmpImmOp("foo"), "foo"),
        (LoopRdImmOp(R0, "loop_top"), "loop_top"),
        (JmpImmOp(16), None),
        (LoopRdImmOp(R0, 8), None),
        (MoveImmRdOp(SU32Imm(3), R0), None),
    ],
)
def test_referenced_label_reads_only_label_jump_targets(op, expected):
    """A label is read from a jump or loop target, and only when it is symbolic.

    An address-immediate target and a non-jump op reference no label, so the typed
    accessor returns ``None`` rather than duck-typing an ``imm`` property.
    """
    assert _referenced_label(op) == expected


def test_fresh_labels_deconflict_against_loop_reference():
    """A counted loop's label operand reserves a name the counter steps over."""
    reserving = Block([LoopRdImmOp(R0, "bb0"), StopOp()])
    plain = Block([StopOp()])
    blocks = [reserving, plain]

    labels = _fresh_labels(blocks)

    assert "bb0" not in set(labels.values())
    assert len(set(labels.values())) == len(blocks)
