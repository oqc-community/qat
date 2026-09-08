# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Bind outlined Pulse sequences to their physical Qblox configuration."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose

from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import PassFailedException, VerifyException

from qat.experimental.conversion.pulse_to_q1.qblox_configuration.models import (
    SequencerBinding,
)
from qat.experimental.conversion.pulse_to_q1.qblox_configuration.resolution import (
    resolve_sequencer_bindings,
)
from qat.experimental.dialect.pulse.ir import CreateFrameOp
from qat.experimental.dialect.pulse.utils import extract_frequency_hz
from qat.experimental.dialect.q1_sequence.ir.imm_desc import (
    SequencerIndexAttr,
    SlotIndexAttr,
)
from qat.experimental.dialect.q1_sequence.ir.ops import SequenceOp
from qat.experimental.passes.pass_ordering import OrderedPass
from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.system_data.hardware.qblox.configuration import (
    supplied_configurations,
)
from qat.experimental.system_data.hardware.qblox.models import QbloxChannelBinding
from qat.experimental.system_data.hardware.qblox.view import QbloxHardwareView


@dataclass(frozen=True)
class QbloxHardwareBindingPass(OrderedPass, ModulePass):
    """Bind each outlined sequence to the physical Qblox sequencer it runs on.

    Each ``SequenceOp`` must contain one constant ``pulse.create_frame`` whose port and
    carrier frequency identify a canonical channel. The pass records that channel's
    allocated instrument, slot, and sequencer index on the sequence, and attaches the
    resolved sequencer and module configuration. Configuration a sequence already carries
    is merged rather than replaced, so program-owned acquisition data survives binding.
    """

    name = "qblox-hardware-binding"
    canonical_data: CanonicalSystemData

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        try:
            hardware_view = QbloxHardwareView.derive(self.canonical_data)
            bindings = resolve_sequencer_bindings(
                hardware_view, supplied_configurations(self.canonical_data)
            )
        except (ValueError, VerifyException) as error:
            raise PassFailedException(str(error)) from error

        sequences = [
            sequence for sequence in op.body.block.ops if isinstance(sequence, SequenceOp)
        ]
        resolved_bindings = [
            self._resolve_sequence(sequence, hardware_view.channel_bindings, bindings)
            for sequence in sequences
        ]
        physical_allocations = [
            (binding.module_location, binding.sequencer_index)
            for binding in resolved_bindings
        ]
        if len(physical_allocations) != len(set(physical_allocations)):
            raise PassFailedException(
                "Outlined sequences resolve to a duplicate Qblox physical allocation."
            )

        for sequence, binding in zip(sequences, resolved_bindings, strict=True):
            self._verify_existing(sequence, binding)
        for sequence, binding in zip(sequences, resolved_bindings, strict=True):
            self._bind(sequence, binding)

    def _resolve_sequence(
        self,
        sequence: SequenceOp,
        channels: Mapping[str, QbloxChannelBinding],
        bindings: Mapping[str, SequencerBinding],
    ) -> SequencerBinding:
        """Resolve the sequencer a sequence binds to from its Pulse frame.

        :param sequence: The outlined sequence to resolve.
        :param channels: Canonical channels keyed by identifier.
        :param bindings: Resolved sequencer bindings keyed by canonical channel.
        :returns: The binding for the sequence.
        :raises PassFailedException: If the frame does not map to exactly one canonical
            channel with a resolved sequencer.
        """

        frames = [
            nested
            for nested in sequence.walk()
            if isinstance(nested, CreateFrameOp) and nested.port == sequence.port_id
        ]
        if len(frames) != 1:
            raise PassFailedException(
                f"Sequence {sequence.channel_id.data!r} must contain exactly one "
                f"pulse.create_frame matching port {sequence.port_id.data!r} before "
                "Qblox hardware binding."
            )
        carrier = extract_frequency_hz(frames[0])
        candidates = [
            channel
            for channel in channels.values()
            if channel.port_id == sequence.port_id.data
            and isclose(channel.carrier_frequency, carrier, rel_tol=0.0, abs_tol=1e-6)
        ]
        if len(candidates) != 1:
            raise PassFailedException(
                f"Sequence {sequence.channel_id.data!r} maps to {len(candidates)} canonical "
                f"channels for port {sequence.port_id.data!r} at {carrier} Hz; expected one."
            )
        binding = bindings.get(candidates[0].channel_id)
        if binding is None:
            raise PassFailedException(
                f"Canonical channel {candidates[0].channel_id!r} has no Qblox sequencer."
            )
        return binding

    def _verify_existing(self, sequence: SequenceOp, binding: SequencerBinding) -> None:
        """Verify configuration a sequence already carries against the resolved binding.

        :param sequence: The outlined sequence being bound.
        :param binding: The resolved binding of the sequence.
        :raises PassFailedException: If existing allocation or module configuration
            contradicts the canonical system data.
        """

        expected = (
            StringAttr(binding.module_location.instrument_id),
            SlotIndexAttr(binding.module_location.slot),
            SequencerIndexAttr(binding.sequencer_index),
        )
        existing = (sequence.instrument_id, sequence.slot_idx, sequence.seq_idx)
        if any(value is not None for value in existing) and existing != expected:
            raise PassFailedException(
                f"Sequence {sequence.channel_id.data!r} has physical allocation "
                "conflicting with the canonical system data."
            )
        if (
            sequence.module_config is not None
            and sequence.module_config != binding.module_config
        ):
            raise PassFailedException(
                f"Sequence {sequence.channel_id.data!r} has module configuration "
                "conflicting with the canonical system data."
            )

    def _bind(self, sequence: SequenceOp, binding: SequencerBinding) -> None:
        """Record the resolved allocation and configuration on a sequence.

        :param sequence: The outlined sequence being bound.
        :param binding: The resolved binding of the sequence.
        :raises PassFailedException: If existing sequencer configuration conflicts with the
            resolved configuration.
        """

        try:
            sequencer_config = (
                sequence.sequencer_config.merge_bound(binding.sequencer_config)
                if sequence.sequencer_config is not None
                else binding.sequencer_config
            )
        except VerifyException as error:
            raise PassFailedException(
                f"Sequence {sequence.channel_id.data!r} has sequencer configuration "
                f"conflicting with the canonical system data: {error}"
            ) from error
        sequence.properties["instrument_id"] = StringAttr(
            binding.module_location.instrument_id
        )
        sequence.properties["slot_idx"] = SlotIndexAttr(binding.module_location.slot)
        sequence.properties["seq_idx"] = SequencerIndexAttr(binding.sequencer_index)
        sequence.properties["sequencer_config"] = sequencer_config
        sequence.properties["module_config"] = binding.module_config
