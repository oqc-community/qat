# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from xdsl.dialects.builtin import ModuleOp

from qat.experimental.dialect.q1.ir.ops import StopOp
from qat.experimental.dialect.q1_sequence import Q1Sequence, SequenceOp


class TestDialectRegistration:
    def test_dialect_ops(self):
        op_names = {op.name for op in Q1Sequence.operations}
        assert op_names == {"q1_sequence.sequence"}

    def test_dialect_attrs(self):
        attr_names = {attr.name for attr in Q1Sequence.attributes}
        assert attr_names == {
            "q1_sequence.waveform",
            "q1_sequence.weight",
            "q1_sequence.acquisition",
        }

    def test_sequence_in_module(self):
        seq = SequenceOp("ch0", [StopOp()])
        module = ModuleOp([seq])
        found = [op for op in module.body.block.ops if isinstance(op, SequenceOp)]
        assert len(found) == 1
        assert found[0].channel_id.data == "ch0"

    def test_multiple_sequences_in_module(self):
        s0 = SequenceOp("drive", [StopOp()])
        s1 = SequenceOp("readout", [StopOp()])
        module = ModuleOp([s0, s1])
        names = [
            op.channel_id.data for op in module.body.block.ops if isinstance(op, SequenceOp)
        ]
        assert names == ["drive", "readout"]
