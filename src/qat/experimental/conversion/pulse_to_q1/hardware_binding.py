# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Bind outlined Pulse sequences to their physical Qblox sequencer hardware.

This pass assigns each outlined sequence to a Qblox sequencer based on its frame's
port and carrier frequency. A sequence's frame nominally identifies a canonical channel;
when a port exposes multiple channels at the same frequency (ambiguous case), the pass
selects one deterministically: preferring an explicit sequence channel_id match, then
defaulting to the first available candidate.

Once selected, each sequence is resolved to a sequencer binding via the allocation
strategy. When multiple sequencers can satisfy a candidate, a tie-breaker applies:
the sequencer with minimum (instrument_id, slot, sequencer_index) is chosen to ensure
deterministic and reproducible allocation across runs.

Frequency matching tolerates ±0.5 Hz to account for Q1asm's 1 Hz frequency resolution.
Only channels actively selected by sequences consume allocation; calibrated but unplayed
channels do not reserve sequencers, enabling efficient hardware utilization.

The pass merges existing program-owned sequencer configuration with the resolved
configuration during binding, preserving acquisition settings while enforcing canonical
module constraints.
"""

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
from qat.experimental.conversion.pulse_to_q1.sequence_outlining import Q1OutliningPass
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
    """Bind each outlined sequence op to the physical Qblox sequencer it runs on.

    Each ``SequenceOp`` must contain one constant ``pulse.create_frame`` whose port and
    carrier frequency identify a canonical channel. The pass records that channel's
    allocated instrument, slot, and sequencer index on the sequence, and attaches the
    resolved sequencer and module configuration. Existing program-owned acquisition
    configuration is preserved by merging it with the resolved configuration during binding.
    """

    name = "qblox-hardware-binding"
    canonical_data: CanonicalSystemData

    def required_predecessors(self) -> frozenset[type[ModulePass]]:
        return frozenset({Q1OutliningPass})

    def runs_before(self) -> frozenset[type[ModulePass]]:
        from qat.experimental.conversion.pulse_to_q1.passes import Q1PulseValidationPass

        return frozenset({Q1PulseValidationPass})

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        sequence_ops = [
            sequence_op
            for sequence_op in op.body.block.ops
            if isinstance(sequence_op, SequenceOp)
        ]
        try:
            hardware_view = QbloxHardwareView.derive(self.canonical_data)
            sequence_channel_ids = self._sequence_channel_ids(
                sequence_ops, hardware_view.channel_bindings
            )
            sequence_bindings = resolve_sequencer_bindings(
                hardware_view,
                supplied_configurations(self.canonical_data),
                sequence_channel_ids,
            )
        except (ValueError, VerifyException) as error:
            raise PassFailedException(str(error)) from error

        # Validate that all requested channels have resolved bindings
        failed_channels = [
            channel_id
            for channel_id in sequence_channel_ids
            if channel_id not in sequence_bindings
        ]
        if failed_channels:
            raise PassFailedException(
                f"Sequencer allocation failed for channels {failed_channels!r}. "
                "No Qblox sequencer available for these channels."
            )

        # Resolve each sequence with tie-breaking logic
        consumed_channel_ids: set[str] = set()
        resolved_bindings = []
        for sequence_op in sequence_ops:
            binding, selected_channel_id = self._resolve_sequence(
                sequence_op,
                hardware_view.channel_bindings,
                sequence_bindings,
                consumed_channel_ids,
            )
            consumed_channel_ids.add(selected_channel_id)
            resolved_bindings.append(binding)
        physical_allocations = [
            (binding.module_location, binding.sequencer_index)
            for binding in resolved_bindings
        ]
        if len(physical_allocations) != len(set(physical_allocations)):
            raise PassFailedException(
                "Outlined sequences resolve to a duplicate Qblox physical allocation."
            )

        for sequence_op, binding in zip(sequence_ops, resolved_bindings, strict=True):
            self._verify_existing(sequence_op, binding)
        for sequence_op, binding in zip(sequence_ops, resolved_bindings, strict=True):
            self._bind(sequence_op, binding)

    def _sequence_channel_ids(
        self,
        sequence_ops: list[SequenceOp],
        channels: Mapping[str, QbloxChannelBinding],
    ) -> list[str]:
        """Select one canonical channel for each outlined sequence.

        A sequence's frame identifies a canonical channel by port and carrier frequency; an
        ambiguous frame maps to several channels. Selection happens in sequence order,
        preferring an explicit sequence channel identifier when it matches one of the
        available candidates. Only the selected channels are later allocated and resolved,
        so ambiguous siblings that this program never chooses do not consume sequencers.

        :param sequence_ops: The outlined sequences to bind.
        :param channels: Canonical channels keyed by identifier.
        :returns: The selected canonical channel identifier for each sequence, in order.
        :raises PassFailedException: If a sequence has no single matching frame.
        """

        consumed_channel_ids: set[str] = set()
        sequence_channel_ids: list[str] = []
        for sequence_op in sequence_ops:
            available_candidates = self._available_candidates(
                sequence_op, channels, consumed_channel_ids
            )
            selected_candidate = self._select_preferred_or_first_candidate(
                sequence_op, available_candidates
            )
            consumed_channel_ids.add(selected_candidate.channel_id)
            sequence_channel_ids.append(selected_candidate.channel_id)
        return sequence_channel_ids

    def _select_preferred_or_first_candidate(
        self,
        sequence_op: SequenceOp,
        available_candidates: list[QbloxChannelBinding],
    ) -> QbloxChannelBinding:
        """Select a candidate, preferring explicit sequence channel ID."""

        preferred_candidate = next(
            (
                candidate
                for candidate in available_candidates
                if candidate.channel_id == sequence_op.channel_id.data
            ),
            None,
        )
        return preferred_candidate or available_candidates[0]

    def _available_candidates(
        self,
        sequence_op: SequenceOp,
        channels: Mapping[str, QbloxChannelBinding],
        consumed_channel_ids: set[str],
    ) -> list[QbloxChannelBinding]:
        """Return the unconsumed canonical candidates for a sequence.

        Finds all canonical channels matching the sequence's port and frame carrier
        frequency (within ±0.5 Hz tolerance). Filters out channels already consumed by
        earlier sequences in the pass.
        """

        carrier, candidates = self._frame_candidates(sequence_op, channels)
        available_candidates = [
            candidate
            for candidate in candidates
            if candidate.channel_id not in consumed_channel_ids
        ]
        if not available_candidates:
            raise PassFailedException(
                f"Sequence {sequence_op.channel_id.data!r} maps to {len(candidates)} "
                f"canonical channels for port {sequence_op.port_id.data!r} at "
                f"{carrier} Hz, but none are available."
            )
        return available_candidates

    def _frame_candidates(
        self,
        sequence_op: SequenceOp,
        channels: Mapping[str, QbloxChannelBinding],
    ) -> tuple[float, list[QbloxChannelBinding]]:
        """Return the carrier and canonical channels a sequence's frame maps to.

        :param sequence_op: The outlined sequence to inspect.
        :param channels: Canonical channels keyed by identifier.
        :returns: The frame carrier frequency and the channels matching the sequence's port
            at that frequency.
        :raises PassFailedException: If the sequence does not contain exactly one
            ``pulse.create_frame`` matching its port.
        """

        frames = [
            nested
            for nested in sequence_op.walk()
            if isinstance(nested, CreateFrameOp) and nested.port == sequence_op.port_id
        ]
        if len(frames) != 1:
            raise PassFailedException(
                f"Sequence {sequence_op.channel_id.data!r} must contain exactly one "
                f"pulse.create_frame matching port {sequence_op.port_id.data!r} before "
                "Qblox hardware binding."
            )
        carrier = extract_frequency_hz(frames[0])

        # Q1asm allows a frequency resolution of 1Hz, so we allow a tolerance of ±0.5 Hz
        _FREQUENCY_RESOLUTION = 0.5
        candidates = [
            channel
            for channel in channels.values()
            if channel.port_id == sequence_op.port_id.data
            and isclose(
                channel.carrier_frequency, carrier, rel_tol=0, abs_tol=_FREQUENCY_RESOLUTION
            )
        ]
        return carrier, candidates

    def _resolve_sequence(
        self,
        sequence_op: SequenceOp,
        channels: Mapping[str, QbloxChannelBinding],
        bindings: Mapping[str, SequencerBinding],
        consumed_channel_ids: set[str],
    ) -> tuple[SequencerBinding, str]:
        """Resolve the sequencer binding for a sequence.

        Selects a canonical channel from available candidates (preferring an explicit
        channel_id match), retrieves its sequencer binding, and applies a tie-breaker
        if multiple candidates have bindings.

        The tie-breaker ensures deterministic selection: when multiple candidates have
        resolved sequencer bindings, the one with the minimum (instrument_id, slot,
        sequencer_index) tuple is returned.

        :param sequence_op: The outlined sequence to resolve.
        :param channels: Canonical channels keyed by identifier.
        :param bindings: Resolved sequencer bindings keyed by canonical channel.
        :param consumed_channel_ids: Canonical channels already assigned to a previous
            sequence during this pass application.
        :returns: Tuple of (sequencer binding, selected channel_id).
        :raises PassFailedException: If the frame does not map to any available canonical
            channel with a resolved sequencer.
        """

        available_candidates = self._available_candidates(
            sequence_op, channels, consumed_channel_ids
        )
        selected_candidate = self._select_preferred_or_first_candidate(
            sequence_op, available_candidates
        )
        selected_binding = bindings.get(selected_candidate.channel_id)
        if selected_binding is not None:
            return selected_binding, selected_candidate.channel_id

        resolved_candidates = [
            (candidate.channel_id, bindings[candidate.channel_id])
            for candidate in available_candidates
            if candidate.channel_id in bindings
        ]
        if len(resolved_candidates) == 0:
            # Distinguish between preferred channel unavailable vs no binding found
            is_preferred = selected_candidate.channel_id == sequence_op.channel_id.data
            if is_preferred:
                ch_id = selected_candidate.channel_id
                raise PassFailedException(
                    f"Can't find binding for preferred channel {ch_id!r}; "
                    "Pulse channel not assigned to any QBlox sequencer."
                )
            # Include at least one candidate channel in the error for debugging
            first_candidate = available_candidates[0].channel_id
            seq_id = sequence_op.channel_id.data
            raise PassFailedException(
                f"No available sequencer binding found for any candidate of "
                f"sequence {seq_id!r}; candidate {first_candidate!r} "
                "has no allocated sequencer."
            )
        # Tie-breaker if multiple candidates are available; gives a deterministic choice
        selected_channel_id, best_binding = min(
            resolved_candidates,
            key=lambda item: (
                item[1].module_location.instrument_id,
                item[1].module_location.slot,
                item[1].sequencer_index,
            ),
        )
        return best_binding, selected_channel_id

    def _verify_existing(self, sequence_op: SequenceOp, binding: SequencerBinding) -> None:
        """Verify a sequence's existing allocation against the resolved binding.

        Runs after resolution but before binding to catch conflicts early. Allows sequences
        to carry partial or unspecified allocation (None values), but rejects any mismatch
        between existing and canonical values. This protects against sequences that were
        pre-allocated to an invalid or conflicting sequencer.

        :param sequence_op: The outlined sequence being bound.
        :param binding: The resolved binding from canonical system data.
        :raises PassFailedException: If existing allocation (instrument, slot, sequencer)
            contradicts the canonical binding, or if module configuration conflicts.
        """

        expected = (
            StringAttr(binding.module_location.instrument_id),
            SlotIndexAttr(binding.module_location.slot),
            SequencerIndexAttr(binding.sequencer_index),
        )
        existing = (
            sequence_op.instrument_id,
            sequence_op.slot_idx,
            sequence_op.seq_idx,
        )
        if any(value is not None for value in existing) and existing != expected:
            raise PassFailedException(
                f"Sequence {sequence_op.channel_id.data!r} has physical allocation "
                "conflicting with the canonical system data."
            )
        if (
            sequence_op.module_config is not None
            and sequence_op.module_config != binding.module_config
        ):
            raise PassFailedException(
                f"Sequence {sequence_op.channel_id.data!r} has module configuration "
                "conflicting with the canonical system data."
            )

    def _bind(self, sequence_op: SequenceOp, binding: SequencerBinding) -> None:
        """Record the resolved allocation and configuration on a sequence.

        :param sequence_op: The outlined sequence being bound (modified in-place).
        :param binding: The resolved binding from canonical allocation.
        :raises PassFailedException: If sequencer configuration merge fails, indicating an
            irreconcilable conflict between existing and canonical settings.
        """

        try:
            sequencer_config = (
                sequence_op.sequencer_config.merge_bound(binding.sequencer_config)
                if sequence_op.sequencer_config is not None
                else binding.sequencer_config
            )
        except VerifyException as error:
            raise PassFailedException(
                f"Sequence {sequence_op.channel_id.data!r} has sequencer configuration "
                f"conflicting with the canonical system data: {error}"
            ) from error
        sequence_op.properties["instrument_id"] = StringAttr(
            binding.module_location.instrument_id
        )
        sequence_op.properties["slot_idx"] = SlotIndexAttr(binding.module_location.slot)
        sequence_op.properties["seq_idx"] = SequencerIndexAttr(binding.sequencer_index)
        sequence_op.properties["sequencer_config"] = sequencer_config
        sequence_op.properties["module_config"] = binding.module_config
