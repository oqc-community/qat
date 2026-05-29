# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""
imm_desc.py
=================

This module defines the set of immediate types for the QBlox ISA. QBlox used
to support only *unsigned* 32-bit integers (see :class:`UI32`) and the programmer had
to legalize negative values by 2's complement. QBlox nowadays are increasingly
supporting more types to improve the programmer's experience. While not fully
specified/documented, we're anticipating such improvements to the Q1 ISA here by
specifying the following:

* UI32: Unsigned 32-bit IntegerType alias
* SI32: Signed 32-bit IntegerType alias

* UI16: Unsigned 16-bit IntegerType alias
* SI16: Signed 16-bit IntegerType alias
"""

from typing import Literal, TypeAlias

from xdsl.dialects.builtin import IntegerType, Signedness

UI32: TypeAlias = IntegerType[Literal[32], Literal[Signedness.UNSIGNED]]
UI16: TypeAlias = IntegerType[Literal[16], Literal[Signedness.UNSIGNED]]

SI32: TypeAlias = IntegerType[Literal[32], Literal[Signedness.SIGNED]]
SI16: TypeAlias = IntegerType[Literal[16], Literal[Signedness.SIGNED]]

ui32: UI32 = UI32(32, Signedness.UNSIGNED)
ui16: UI16 = UI16(16, Signedness.UNSIGNED)

si32: SI32 = SI32(32, Signedness.SIGNED)
si16: SI16 = SI16(16, Signedness.SIGNED)
