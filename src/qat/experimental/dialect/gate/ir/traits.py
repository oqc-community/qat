# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from xdsl.traits import OpTrait


class CliffordTrait(OpTrait):
    """A trait that signifies every instance of the operation is a Clifford gate.

    Passes use it to ask whether a gate is Clifford without listing gates.
    """

    ...
