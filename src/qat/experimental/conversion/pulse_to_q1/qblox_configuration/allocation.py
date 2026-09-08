# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Allocate physical Qblox sequencers from the banks canonical ports supply.

The allocation reproduces the legacy ``AllocatingBackend`` semantics: a canonical channel
is allocated out of the bank its own port supplies, minus the sequencers already taken on
the physical module, and every channel of one port keeps its assignment for the whole
compilation. Nothing is synthesised - a sequencer the source never described is never
allocated, so the supplied bank alone decides what a module may use.
"""

from __future__ import annotations

from collections.abc import Mapping

from frozendict import frozendict

from qat.experimental.conversion.pulse_to_q1.qblox_configuration.models import (
    ReconciledModuleConfiguration,
    SequencerPlacement,
)
from qat.experimental.system_data.hardware.qblox.models import QbloxModuleLocation


def allocate_sequencers(
    module_configurations: Mapping[QbloxModuleLocation, ReconciledModuleConfiguration],
) -> frozendict[str, SequencerPlacement]:
    """Allocate a physical sequencer to every canonical channel of every module.

    :param module_configurations: Configurations keyed by physical module location.
    :returns: The placement of each canonical channel, keyed by channel identifier.
    :raises ValueError: If a port supplies no bank, or its bank cannot satisfy the channels
        routed through it.
    """

    placements: dict[str, SequencerPlacement] = {}
    for module_configuration in module_configurations.values():
        placements.update(_allocate_on_module(module_configuration))
    return frozendict(placements)


def _allocate_on_module(
    module_configuration: ReconciledModuleConfiguration,
) -> dict[str, SequencerPlacement]:
    """Allocate the sequencers of one physical module.

    Allocated indices are tracked module-wide because two canonical ports exposing the same
    module still contend for its physical sequencers.

    :param module_configuration: Configuration assembled from every canonical port exposing
        the physical module.
    :returns: The placement of each channel routed through the module.
    :raises ValueError: If a port supplies no bank, or its bank is exhausted.
    """

    module_location = module_configuration.module_view.location
    allocated_indices: set[int] = set()
    placements: dict[str, SequencerPlacement] = {}
    for channel_binding in module_configuration.module_view.channel_bindings:
        sequencer_bank = module_configuration.sequencer_banks.get(
            channel_binding.port_id, frozendict()
        )
        if not sequencer_bank:
            raise ValueError(
                f"Qblox port {channel_binding.port_id!r} on module "
                f"{module_location!r} supplies no sequencer bank"
            )
        available_indices = sorted(set(sequencer_bank) - allocated_indices)
        if not available_indices:
            raise ValueError(
                f"Qblox port {channel_binding.port_id!r} on module "
                f"{module_location!r} has no sequencer left for channel "
                f"{channel_binding.channel_id!r}; its supplied bank "
                f"{sorted(sequencer_bank)} is fully allocated"
            )
        sequencer_index = available_indices[0]
        allocated_indices.add(sequencer_index)
        placements[channel_binding.channel_id] = SequencerPlacement(
            channel_id=channel_binding.channel_id,
            port_id=channel_binding.port_id,
            module_location=module_location,
            sequencer_index=sequencer_index,
            sequencer_configuration=sequencer_bank[sequencer_index],
        )
    return placements
