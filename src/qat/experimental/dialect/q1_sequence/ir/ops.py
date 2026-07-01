# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from collections.abc import Sequence

from xdsl.dialects.builtin import ArrayAttr, StringAttr, SymbolNameConstraint
from xdsl.ir import Block, Operation, Region
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    prop_def,
    region_def,
    traits_def,
)
from xdsl.traits import IsolatedFromAbove, IsTerminator, SymbolOpInterface
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.q1_sequence.ir.attrs import (
    AcquisitionAttr,
    WaveformAttr,
    WeightAttr,
)


@irdl_op_definition
class SequenceOp(IRDLOperation):
    """A sequence op represents the payload for a single Sequencer/PPU in a Qblox
    instrument. The body region holds Q1 assembly ops and must be terminated by a Stop* op.
    Data table attributes (waveforms, weights, acquisitions) are static lookup tables
    referenced by instruction indices.

    :param sym_name: Channel/sequencer identifier (e.g. ``"Q0_drive"``).
    :param body: Single-block region of Q1 instruction ops.
    :param waveforms: Waveform data table entries.
    :param weights: Weight data table entries.
    :param acquisitions: Acquisition data table entries.
    """

    name = "q1_sequence.sequence"

    sym_name = attr_def(SymbolNameConstraint())
    body = region_def("single_block")

    waveforms = prop_def(ArrayAttr[WaveformAttr])
    weights = prop_def(ArrayAttr[WeightAttr])
    acquisitions = prop_def(ArrayAttr[AcquisitionAttr])

    traits = traits_def(
        SymbolOpInterface(),
        IsolatedFromAbove(),
    )

    @property
    def channel_id(self) -> StringAttr:
        """Alias for `sym_name`, the channel identifier."""

        return self.sym_name

    def __init__(
        self,
        channel_id: str | StringAttr,
        program: Sequence[Operation] | Region,
        waveforms: ArrayAttr[WaveformAttr] | None = None,
        weights: ArrayAttr[WeightAttr] | None = None,
        acquisitions: ArrayAttr[AcquisitionAttr] | None = None,
    ):
        if isinstance(channel_id, str):
            channel_id = StringAttr(channel_id)
        if waveforms is None:
            waveforms = ArrayAttr([])
        if weights is None:
            weights = ArrayAttr([])
        if acquisitions is None:
            acquisitions = ArrayAttr([])

        region = program if isinstance(program, Region) else Region(Block(list(program)))

        super().__init__(
            attributes={"sym_name": channel_id},
            properties={
                "waveforms": waveforms,
                "weights": weights,
                "acquisitions": acquisitions,
            },
            regions=[region],
        )

    def verify_(self) -> None:
        """Verifies SequenceOp invariants.

        - channel_id must be non-empty.
        - Program body must be terminated by an IsTerminator op.
        - Indices must be unique within each data table.
        - Names must be unique within each data table.
        """

        if not self.channel_id.data:
            raise VerifyException("SequenceOp channel_id must be non-empty")

        last_op = self.body.block.last_op
        if last_op is None or not last_op.has_trait(IsTerminator):
            raise VerifyException(
                f"Sequence '{self.channel_id.data}' body must end"
                f" with a terminator op (e.g. stop)"
            )

        for table_name, table, name_key in (
            ("waveforms", self.waveforms, "waveform_name"),
            ("weights", self.weights, "weight_name"),
            ("acquisitions", self.acquisitions, "acquisition_name"),
        ):
            indices: set[int] = set()
            names: set[str] = set()
            for entry in table:
                idx = entry.index.data
                if idx in indices:
                    raise VerifyException(
                        f"Duplicate index {idx} in {table_name}"
                        f" of sequence"
                        f" '{self.channel_id.data}'"
                    )
                indices.add(idx)

                entry_name = getattr(entry, name_key).data
                if entry_name in names:
                    raise VerifyException(
                        f"Duplicate name '{entry_name}'"
                        f" in {table_name}"
                        f" of sequence"
                        f" '{self.channel_id.data}'"
                    )
                names.add(entry_name)
