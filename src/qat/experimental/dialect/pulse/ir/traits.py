# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from xdsl.traits import OpTrait


class AdvancesTimeTrait(OpTrait):
    """A trait that signifies an operation advances time on the frame(s) it acts on.

    The time does not need to be known at compile time, and in that sense, can be runtime
    dynamic.
    """

    ...
