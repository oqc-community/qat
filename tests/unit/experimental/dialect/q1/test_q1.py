# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd


from io import StringIO

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp

from qat.experimental.dialect.q1 import Q1, Q1asmTarget, print_assembly, q1_code

_expected_q1_op_names = {
    "q1.x.label",
    "q1.xx.DEF",
    "q1..illegal",
    "q1..nop",
    "q1..reset_ph",
    "q1..stop",
    "q1.i.ja",
    "q1.i.jae",
    "q1.i.jb",
    "q1.i.jbe",
    "q1.i.jg",
    "q1.i.jl",
    "q1.i.jle",
    "q1.i.jmp",
    "q1.i.jno",
    "q1.i.jns",
    "q1.i.jnz",
    "q1.i.jo",
    "q1.i.js",
    "q1.i.jz",
    "q1.i.set_freq",
    "q1.i.set_mrk",
    "q1.i.set_ph",
    "q1.i.set_ph_delta",
    "q1.i.stop",
    "q1.ii.set_awg_gain",
    "q1.ii.set_awg_offs",
    "q1.iiii.set_cond",
    "q1.ir.move",
    "q1.ir.not",
    "q1.r.ja",
    "q1.r.jae",
    "q1.r.jb",
    "q1.r.jbe",
    "q1.r.jg",
    "q1.r.jl",
    "q1.r.jle",
    "q1.r.jmp",
    "q1.r.jno",
    "q1.r.jns",
    "q1.r.jnz",
    "q1.r.jo",
    "q1.r.js",
    "q1.r.jz",
    "q1.r.set_freq",
    "q1.r.set_mrk",
    "q1.r.set_ph",
    "q1.r.set_ph_delta",
    "q1.r.stop",
    "q1.ri.loop",
    "q1.rii.jge",
    "q1.rii.jlt",
    "q1.rir.add",
    "q1.rir.and",
    "q1.rir.asl",
    "q1.rir.asr",
    "q1.rir.jge",
    "q1.rir.jlt",
    "q1.rir.or",
    "q1.rir.sub",
    "q1.rir.xor",
    "q1.rr.loop",
    "q1.rr.move",
    "q1.rr.not",
    "q1.rr.set_awg_gain",
    "q1.rr.set_awg_offs",
    "q1.rrr.add",
    "q1.rrr.and",
    "q1.rrr.asl",
    "q1.rrr.asr",
    "q1.rrr.or",
    "q1.rrr.sub",
    "q1.rrr.xor",
    "q1.rrri.set_cond",
}


def test_q1_module_helpers_emit_and_register_ops():
    ctx = Context()
    module = ModuleOp([])

    explicit_stream = StringIO()
    print_assembly(module, explicit_stream)

    generated = q1_code(module)
    assert generated == explicit_stream.getvalue()

    target_stream = StringIO()
    Q1asmTarget().emit(ctx, module, target_stream)
    assert target_stream.getvalue() == generated

    q1_op_names = {op.name for op in Q1.operations}

    assert q1_op_names == _expected_q1_op_names
