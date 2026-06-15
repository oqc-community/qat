# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from xdsl.ir import Dialect

from qat.experimental.dialect.q1_sequence.attrs import (
    AcquisitionAttr,
    WaveformAttr,
    WeightAttr,
)
from qat.experimental.dialect.q1_sequence.ops import SequenceOp
from qat.experimental.dialect.q1_sequence.target import Q1SequenceTarget

Q1Sequence = Dialect(
    "q1_sequence",
    [SequenceOp],
    [WaveformAttr, WeightAttr, AcquisitionAttr],
)

__all__ = [
    "AcquisitionAttr",
    "Q1Sequence",
    "Q1SequenceTarget",
    "SequenceOp",
    "WaveformAttr",
    "WeightAttr",
]
