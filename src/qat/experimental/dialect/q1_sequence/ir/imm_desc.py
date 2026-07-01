# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
"""Semantic immediate attribute types for the Qblox q1_sequence dialect.

These types refine :class:`~qat.experimental.dialect.q1.ir.imm_desc.Q1Imm` with
the per-sequencer table sizes and per-module bin budget published by Qblox.
Out-of-range values fail at construction time, before they can land in the IR.
"""

from typing import ClassVar

from xdsl.irdl import irdl_attr_definition

from qat.experimental.dialect.q1.ir.imm_desc import Q1Imm


@irdl_attr_definition
class WaveformTableIndex(Q1Imm):
    """Index into a sequencer's waveform table: ``[0, 1023]``.

    A Q1 sequencer can hold up to 1024 waveform entries shared across both
    output paths; ``play.wave0`` / ``play.wave1`` reference them by index.
    """

    name = "q1_sequence.waveform_table_index"
    _MIN: ClassVar[int] = 0
    _MAX: ClassVar[int] = 1023


@irdl_attr_definition
class WeightTableIndex(Q1Imm):
    """Index into a sequencer's weight table: ``[0, 31]``.

    Up to 32 integration-weight arrays are stored per sequencer; the
    ``acquire_weighted`` instruction selects one via its 6-bit
    ``weight_idx{0,1}`` fields (encoded as :class:`UI6Imm`), of which only
    the low 5 bits address an entry on current hardware.
    """

    name = "q1_sequence.weight_table_index"
    _MIN: ClassVar[int] = 0
    _MAX: ClassVar[int] = 31


@irdl_attr_definition
class AcqTableIndex(Q1Imm):
    """Index into a sequencer's acquisition table: ``[0, 31]``.

    Up to 32 acquisition entries per sequencer; ``acquire*`` instructions
    reference them via ``acq_idx``.
    """

    name = "q1_sequence.acq_table_index"
    _MIN: ClassVar[int] = 0
    _MAX: ClassVar[int] = 31


@irdl_attr_definition
class BinCountImm(Q1Imm):
    """Number of acquisition bins: ``[0, 7_000_000]``.

    The hardware budget is per-module, not per-acquisition: 3 M bins total
    for QRM / QRM-RF and 7 M total for QRC. We allow the worst-case max
    here; cross-acquisition aggregation is checked elsewhere.
    """

    name = "q1_sequence.bin_count_imm"
    _MIN: ClassVar[int] = 0
    _MAX: ClassVar[int] = 7_000_000
