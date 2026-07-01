# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from xdsl.ir import Dialect

from qat.experimental.dialect.q1_sequence.ir.attrs import (
    AcquisitionAttr,
    WaveformAttr,
    WeightAttr,
)
from qat.experimental.dialect.q1_sequence.ir.imm_desc import (
    AcqTableIndex,
    BinCountImm,
    WaveformTableIndex,
    WeightTableIndex,
)
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.dialect.q1_sequence.target import Q1SequenceTarget

Q1Sequence = Dialect(
    "q1_sequence",
    [SequenceOp],
    [
        WaveformAttr,
        WeightAttr,
        AcquisitionAttr,
        WaveformTableIndex,
        WeightTableIndex,
        AcqTableIndex,
        BinCountImm,
    ],
)

__all__ = [
    "AcqTableIndex",
    "AcquisitionAttr",
    "BinCountImm",
    "Q1Sequence",
    "Q1SequenceTarget",
    "SequenceOp",
    "WaveformAttr",
    "WaveformTableIndex",
    "WeightAttr",
    "WeightTableIndex",
]
