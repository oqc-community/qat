# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd


from qat.core.pass_base import PassManager
from qat.middleend.base import CustomMiddleend


class FallthroughMiddleend(CustomMiddleend):
    """
    A middle end that passes through an input :class:`InstructionBuilder` and does not alter it.
    """

    def __init__(self, model: None = None):
        super().__init__(model=None, pipeline=PassManager())
