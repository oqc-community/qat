# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Traits describing how Q1 instructions interact with the sequencer's parameter latch.

On Qblox, parameter instructions such as ``set_ph`` and ``set_awg_offs`` take no time and
do not take effect when issued. The next ``upd_param``, ``play`` or ``acquire`` applies
everything pending.
"""

from xdsl.traits import OpTrait


class UpdatesParameterTrait(OpTrait):
    """The instruction sets a parameter that waits for the next latching instruction."""


class LatchesParametersTrait(OpTrait):
    """The instruction applies every pending parameter update when it starts."""
