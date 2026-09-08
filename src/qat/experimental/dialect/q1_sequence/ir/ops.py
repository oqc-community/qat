# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025-2026 Oxford Quantum Circuits Ltd

from collections.abc import Sequence

from xdsl.dialects.builtin import ArrayAttr, ModuleOp, NoneAttr, StringAttr
from xdsl.ir import Attribute, Block, Operation, Region
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    region_def,
    traits_def,
)
from xdsl.traits import IsolatedFromAbove, IsTerminator
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.q1 import ACQUISITION_OP_TYPES
from qat.experimental.dialect.q1_sequence.ir.attrs import (
    AcquisitionAttr,
    ModuleConfigAttr,
    SequencerConfigAttr,
    WaveformAttr,
    WeightAttr,
)
from qat.experimental.dialect.q1_sequence.ir.imm_desc import (
    SequencerIndexAttr,
    SlotIndexAttr,
)
from qat.experimental.system_data.hardware.qblox.target import (
    DEFAULT_QBLOX_TARGET,
    Q1SequencerFeature,
)


@irdl_op_definition
class SequenceOp(IRDLOperation):
    """A sequence op represents the payload for a single Sequencer/PPU in a Qblox
    instrument. The body region holds Q1 assembly ops across one or more blocks, each
    terminated by an ``IsTerminator`` op: ``q1_cf`` branches wire the control-flow graph and
    ``Stop*`` ops end terminating paths. In multi-block (CFG) form the region must be
    linearised to a single block before emission, since ``emit_sequence`` lowers via
    ``q1.emit_program`` which requires all ops to be ``AssemblyPrintable`` (``q1_cf``
    terminators are not). Data table attributes (waveforms, weights, acquisitions) are
    static lookup tables referenced by instruction indices. ``instrument_id``, ``slot_idx``
    and ``seq_idx`` identify the physical allocation. ``sequencer_config`` owns the digital
    configuration and connections, while ``module_config`` owns the authoritative analogue
    configuration of the physical module.

    :param channel_id: Stable identifier for the emitted sequencer program.
    :param program: Q1 instruction operations or a region containing one or more blocks.
    :param port_id: Canonical port identifier (e.g. ``"Q0/drive"``). Defaults to
        ``channel_id`` for an unbound sequence.
    :param waveforms: Waveform data table entries.
    :param weights: Weight data table entries.
    :param acquisitions: Acquisition data table entries.
    :param instrument_id: Physical Cluster instrument, absent until hardware binding.
    :param slot_idx: Physical Cluster slot, absent until hardware binding.
    :param seq_idx: Physical sequencer index, absent until hardware binding.
    :param sequencer_config: Per-sequencer configuration and connections, absent until
        resolved.
    :param module_config: Analogue module configuration, absent until hardware binding.
    """

    name = "q1_sequence.sequence"

    channel_id = attr_def(StringAttr)
    port_id = attr_def(StringAttr)
    body = region_def()

    waveforms = prop_def(ArrayAttr[WaveformAttr])
    weights = prop_def(ArrayAttr[WeightAttr])
    acquisitions = prop_def(ArrayAttr[AcquisitionAttr])
    instrument_id = opt_prop_def(StringAttr)
    slot_idx = opt_prop_def(SlotIndexAttr)
    seq_idx = opt_prop_def(SequencerIndexAttr)
    sequencer_config = opt_prop_def(SequencerConfigAttr)
    module_config = opt_prop_def(ModuleConfigAttr)

    traits = traits_def(IsolatedFromAbove())

    def __init__(
        self,
        channel_id: str | StringAttr,
        program: Sequence[Operation] | Region,
        port_id: str | StringAttr | None = None,
        waveforms: ArrayAttr[WaveformAttr] | None = None,
        weights: ArrayAttr[WeightAttr] | None = None,
        acquisitions: ArrayAttr[AcquisitionAttr] | None = None,
        instrument_id: str | StringAttr | None = None,
        slot_idx: int | SlotIndexAttr | None = None,
        seq_idx: int | SequencerIndexAttr | None = None,
        sequencer_config: SequencerConfigAttr | None = None,
        module_config: ModuleConfigAttr | None = None,
    ):
        if isinstance(channel_id, str):
            channel_id = StringAttr(channel_id)
        if port_id is None:
            port_id = channel_id
        elif isinstance(port_id, str):
            port_id = StringAttr(port_id)
        if waveforms is None:
            waveforms = ArrayAttr([])
        if weights is None:
            weights = ArrayAttr([])
        if acquisitions is None:
            acquisitions = ArrayAttr([])
        if isinstance(instrument_id, str):
            instrument_id = StringAttr(instrument_id)
        if isinstance(slot_idx, int):
            slot_idx = SlotIndexAttr(slot_idx)
        if isinstance(seq_idx, int):
            seq_idx = SequencerIndexAttr(seq_idx)

        region = program if isinstance(program, Region) else Region(Block(list(program)))

        properties: dict[str, Attribute] = {
            "waveforms": waveforms,
            "weights": weights,
            "acquisitions": acquisitions,
        }
        optional: dict[str, Attribute | None] = {
            "instrument_id": instrument_id,
            "slot_idx": slot_idx,
            "seq_idx": seq_idx,
            "sequencer_config": sequencer_config,
            "module_config": module_config,
        }
        properties.update(
            {name: value for name, value in optional.items() if value is not None}
        )

        super().__init__(
            attributes={"channel_id": channel_id, "port_id": port_id},
            properties=properties,
            regions=[region],
        )

    def verify_(self) -> None:
        """Verifies SequenceOp invariants.

        - channel_id must be non-empty.
        - Body must contain at least one block.
        - Each block in the body must end with an IsTerminator op.
        - Indices must be unique within each data table.
        - Names must be unique within each data table.
        """

        if not self.channel_id.data:
            raise VerifyException("SequenceOp channel_id must be non-empty")
        if not self.port_id.data:
            raise VerifyException("SequenceOp port_id must be non-empty")

        allocation = (self.instrument_id, self.slot_idx, self.seq_idx)
        if any(part is None for part in allocation) and any(
            part is not None for part in allocation
        ):
            raise VerifyException(
                "SequenceOp instrument_id, slot_idx and seq_idx must be set together"
            )
        if self.module_config is not None and any(part is None for part in allocation):
            raise VerifyException(
                "SequenceOp module_config requires instrument_id, slot_idx and seq_idx"
            )
        if self.instrument_id is not None and not self.instrument_id.data:
            raise VerifyException("SequenceOp instrument_id must be non-empty")
        if not self.body.blocks:
            raise VerifyException(
                f"Sequence '{self.channel_id.data}' body must contain at least one block"
            )

        for block in self.body.blocks:
            last_op = block.last_op
            if last_op is None or not last_op.has_trait(IsTerminator):
                raise VerifyException(
                    f"Sequence '{self.channel_id.data}': each block must end"
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

        self._verify_physical_allocation()

    def _verify_physical_allocation(self) -> None:
        """Verify allocation against the attached authoritative module configuration."""

        if (
            self.sequencer_config is not None
            and isinstance(self.sequencer_config.port_id, StringAttr)
            and self.sequencer_config.port_id != self.port_id
        ):
            raise VerifyException(
                "SequenceOp port_id conflicts with its sequencer configuration"
            )

        if self.instrument_id is None or self.slot_idx is None or self.seq_idx is None:
            if self.module_config is not None or self.sequencer_config is not None:
                raise VerifyException(
                    "SequenceOp configuration requires a complete physical allocation"
                )
            return

        module_config = self.module_config
        if module_config is not None and (
            module_config.instrument_id != self.instrument_id
            or module_config.slot_idx != self.slot_idx
        ):
            raise VerifyException(
                "SequenceOp module_config does not match its instrument_id and slot_idx"
            )

        parent = self.parent_op()
        while parent is not None and not isinstance(parent, ModuleOp):
            parent = parent.parent_op()

        if parent is not None:
            siblings = [
                op
                for op in parent.body.block.ops
                if isinstance(op, SequenceOp) and op is not self
            ]
            if any(
                sibling.instrument_id == self.instrument_id
                and sibling.slot_idx == self.slot_idx
                and sibling.seq_idx == self.seq_idx
                for sibling in siblings
            ):
                raise VerifyException(
                    f"Duplicate physical allocation ({self.instrument_id.data!r}, "
                    f"{self.slot_idx.data}, {self.seq_idx.data})"
                )
            if module_config is not None and any(
                sibling.instrument_id == self.instrument_id
                and sibling.slot_idx == self.slot_idx
                and sibling.module_config is not None
                and sibling.module_config != module_config
                for sibling in siblings
            ):
                raise VerifyException(
                    "Sequences allocated to the same physical module have conflicting "
                    "module configurations"
                )

        if module_config is None:
            return

        module_spec = DEFAULT_QBLOX_TARGET.module_spec(module_config.kind.data)
        if self.seq_idx.data >= module_spec.sequencer_count:
            raise VerifyException(
                f"Sequencer index {self.seq_idx.data} is invalid for "
                f"{module_config.kind.data.value}"
            )

        supports_acquisition = DEFAULT_QBLOX_TARGET.supports(
            module_config.kind.data,
            self.seq_idx.data,
            Q1SequencerFeature.acquisition,
        )
        uses_acquisition = bool(self.acquisitions) or any(
            isinstance(op, ACQUISITION_OP_TYPES) for op in self.walk()
        )
        if self.sequencer_config is not None:
            uses_acquisition = (
                uses_acquisition or self.sequencer_config.has_acquisition_config
            )
            connections = (
                self.sequencer_config.connections
                if isinstance(self.sequencer_config.connections, ArrayAttr)
                else ()
            )
            uses_acquisition = uses_acquisition or any(
                connection.input_ids for connection in connections
            )
            uses_acquisition = uses_acquisition or (
                isinstance(self.sequencer_config.acquisition_path_connections, ArrayAttr)
                and bool(self.sequencer_config.acquisition_path_connections)
            )
        if uses_acquisition and not supports_acquisition:
            raise VerifyException(
                "Acquisition connections, configuration, table data, or instructions require "
                "an acquisition-capable sequencer"
            )

        if parent is not None and module_spec.acquisition_memory_bins is not None:
            module_sequences = (
                op
                for op in parent.body.block.ops
                if isinstance(op, SequenceOp)
                and op.instrument_id == self.instrument_id
                and op.slot_idx == self.slot_idx
            )
            used_bins = sum(
                acquisition.num_bins.data
                for sequence in module_sequences
                for acquisition in sequence.acquisitions
            )
            if used_bins > module_spec.acquisition_memory_bins:
                raise VerifyException(
                    f"{module_config.kind.data.value} module acquisition tables require "
                    f"{used_bins} bins, exceeding the "
                    f"{module_spec.acquisition_memory_bins}-bin module limit"
                )
        if self.sequencer_config is None:
            return
        if not module_spec.supports_mixer_correction and not isinstance(
            self.sequencer_config.mixer, NoneAttr
        ):
            raise VerifyException(
                f"{module_config.kind.data.value} does not support mixer correction"
            )
        self._verify_connections(module_config)

    def _verify_connections(self, module_config: ModuleConfigAttr) -> None:
        """Verify the sequencer connections against the module's configured lanes.

        :param module_config: Authoritative configuration of the allocated module.
        """

        config = self.sequencer_config
        seq_idx = self.seq_idx.data
        configured_outputs = {item.output_id.data for item in module_config.outputs}
        configured_inputs = {item.input_id.data for item in module_config.inputs}
        connections = (
            config.connections if isinstance(config.connections, ArrayAttr) else ()
        )
        for connection in connections:
            missing_outputs = {
                output_id
                for output_id in connection.output_ids
                if output_id not in configured_outputs
            }
            if missing_outputs:
                raise VerifyException(
                    f"SequenceOp connection references unconfigured outputs "
                    f"{sorted(missing_outputs)}"
                )
            unreachable_outputs = {
                output_id
                for output_id in connection.output_ids
                if seq_idx
                not in DEFAULT_QBLOX_TARGET.output_sequencers(
                    module_config.kind.data, output_id
                )
            }
            if unreachable_outputs:
                raise VerifyException(
                    f"{module_config.kind.data.value} sequencer {seq_idx} cannot "
                    f"drive connection outputs {sorted(unreachable_outputs)}"
                )
            missing_inputs = {
                input_id
                for input_id in connection.input_ids
                if input_id not in configured_inputs
            }
            if missing_inputs:
                raise VerifyException(
                    f"SequenceOp connection references unconfigured inputs "
                    f"{sorted(missing_inputs)}"
                )
        output_path_connections = (
            config.output_path_connections
            if isinstance(config.output_path_connections, ArrayAttr)
            else ()
        )
        for connection in output_path_connections:
            output_id = connection.output_id.data
            if output_id not in configured_outputs:
                raise VerifyException(
                    f"SequenceOp output {output_id} is absent from the "
                    f"configuration of module ({module_config.instrument_id.data!r}, "
                    f"{module_config.slot_idx.data})"
                )
            if seq_idx not in DEFAULT_QBLOX_TARGET.output_sequencers(
                module_config.kind.data, output_id
            ):
                raise VerifyException(
                    f"{module_config.kind.data.value} sequencer {seq_idx} cannot drive "
                    f"output {output_id}"
                )

        acquisition_path_connections = (
            config.acquisition_path_connections
            if isinstance(config.acquisition_path_connections, ArrayAttr)
            else ()
        )
        for connection in acquisition_path_connections:
            if connection.input_id.data not in configured_inputs:
                raise VerifyException(
                    f"SequenceOp input {connection.input_id.data} is absent from the configuration "
                    f"of module ({module_config.instrument_id.data!r}, "
                    f"{module_config.slot_idx.data})"
                )

        oscillator_id = config.local_oscillator_id
        if isinstance(oscillator_id, StringAttr) and all(
            item.oscillator_id.data != oscillator_id.data
            for item in module_config.local_oscillators
        ):
            raise VerifyException(
                f"SequenceOp references unknown local oscillator '{oscillator_id.data}'"
            )


def find_enclosing_sequence(op: Operation) -> SequenceOp:
    """Walk the parent chain to find the enclosing ``SequenceOp``.

    :param op: The operation from which to start the upward search.
    :returns: The nearest ancestor ``SequenceOp`` of ``op``.
    :raises ValueError: If no ``SequenceOp`` is found in the parent chain.
    """
    tracked_op = op
    while not isinstance(tracked_op, SequenceOp):
        tracked_op = tracked_op.parent_op()
        if tracked_op is None:
            raise ValueError("No SequenceOp found in the parent chain.")
    return tracked_op
