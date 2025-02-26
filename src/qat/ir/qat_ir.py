# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from pydantic import BaseModel

from qat.ir.instructions import Instruction


class QatIR(BaseModel):
    """Stores in the intermediate representation of compiled quantum programs using QAT's
    instruction set.

    Currently only contains the list as a basic block.
    """

    instructions: list[Instruction] = []
