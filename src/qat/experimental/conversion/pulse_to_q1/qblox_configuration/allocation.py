# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Allocate physical Qblox sequencers from the banks canonical ports supply.

The allocation matches each used canonical channel to a distinct sequencer of the bank its
own port supplies, using a maximum bipartite matching over the whole physical module. It
follows ``AllocatingBackend.allocate_bipartite`` in the legacy backend
(``qat.backend.base``), with the same sorted, hash-seed-independent ordering and
unplaced-channel error. Any valid assignment is found whatever order the channels arrive
in, and ports sharing a module never share a sequencer index. Nothing is synthesised - a
sequencer the source never described is never allocated, so the supplied bank alone decides
what a module may use.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from math import degrees, isclose

import networkx as nx
from frozendict import frozendict

from qat.experimental.conversion.pulse_to_q1.qblox_configuration.models import (
    ReconciledModuleConfiguration,
    SequencerPlacement,
)
from qat.experimental.system_data.hardware.qblox.configuration import (
    QbloxSequencerConfiguration,
)
from qat.experimental.system_data.hardware.qblox.models import (
    QbloxChannelBinding,
    QbloxModuleLocation,
)


def allocate_sequencers(
    module_configurations: Mapping[QbloxModuleLocation, ReconciledModuleConfiguration],
    used_channel_ids: Collection[str],
) -> frozendict[str, SequencerPlacement]:
    """Allocate a physical sequencer to the canonical channels a program uses.

    :param module_configurations: Configurations keyed by physical module location.
    :param used_channel_ids: A collection of canonical channels used within a program.
    :returns: The placement of each allocated canonical channel, keyed by channel
        identifier.
    :raises ValueError: If a port supplies no bank, or its bank cannot satisfy the channels
        routed through it.
    """

    placements: dict[str, SequencerPlacement] = {}
    for module_configuration in module_configurations.values():
        placements.update(_allocate_on_module(module_configuration, used_channel_ids))
    return frozendict(placements)


def _allocate_on_module(
    module_configuration: ReconciledModuleConfiguration,
    used_channel_ids: Collection[str],
) -> dict[str, SequencerPlacement]:
    """Allocate the sequencers of one physical module.

    Allocated indices are tracked module-wide because two canonical ports exposing the same
    module still contend for its physical sequencers.

    :param module_configuration: Configuration assembled from every canonical port exposing
        the physical module.
    :param used_channel_ids: Canonical channels a program actually drives. A channel absent
        from the set is skipped, so a calibrated-but-unplayed channel is never allocated.
    :returns: The placement of each allocated channel routed through the module.
    :raises ValueError: If a port supplies no bank, or its bank is exhausted.
    """

    module_location = module_configuration.module_view.location
    banks = module_configuration.sequencer_banks

    # Sorting on ``channel_id`` gives a deterministic order, so the node indices below do
    # not depend on the order the module view happens to list its bindings in.
    used_channel_bindings = sorted(
        (
            binding
            for binding in module_configuration.module_view.channel_bindings
            if binding.channel_id in used_channel_ids
        ),
        key=lambda binding: binding.channel_id,
    )

    # NetworkX turns ``top_nodes`` into a set, and string hashes vary with PYTHONHASHSEED.
    # Tuples of integers hash the same in every process, so with sorted bindings and
    # candidates the matching is reproducible. The leading 0 or 1 keeps channel indices
    # and sequencer ids apart, as both are small integers.
    graph = nx.Graph()
    incompatible_edges: list[tuple[int, int]] = []
    for index, binding in enumerate(used_channel_bindings):
        sequencer_bank = banks.get(binding.port_id, frozendict())
        if not sequencer_bank:
            raise ValueError(
                f"Qblox port {binding.port_id!r} on module "
                f"{module_location!r} supplies no sequencer bank"
            )
        graph.add_node((0, index), bipartite=0)
        for sequencer_id, sequencer_configuration in sorted(sequencer_bank.items()):
            if _is_compatible(binding, sequencer_configuration):
                graph.add_edge((0, index), (1, sequencer_id))
            else:
                incompatible_edges.append((index, sequencer_id))
    top_nodes = [(0, index) for index in range(len(used_channel_bindings))]
    matching = nx.bipartite.maximum_matching(graph, top_nodes=top_nodes)

    if incompatible_edges and not _all_placed(matching, len(used_channel_bindings)):
        # No assignment honours every pinned value, so the supplied configuration
        # contradicts the calibration. Match on the banks alone and let resolution report
        # the specific conflict.
        graph.add_edges_from(
            ((0, index), (1, sequencer_id)) for index, sequencer_id in incompatible_edges
        )
        matching = nx.bipartite.maximum_matching(graph, top_nodes=top_nodes)

    unplaced = [
        binding
        for index, binding in enumerate(used_channel_bindings)
        if (0, index) not in matching
    ]
    if unplaced:
        details = "\n".join(
            f"  {binding.channel_id!r} (port {binding.port_id!r}, "
            f"candidate sequencers {sorted(banks[binding.port_id])})"
            for binding in unplaced
        )
        raise ValueError(
            f"Could not allocate a sequencer to {len(unplaced)} of "
            f"{len(used_channel_bindings)} channel(s) on module {module_location!r}; "
            f"no assignment gives every channel a distinct sequencer from its port "
            f"bank:\n{details}"
        )

    placements: dict[str, SequencerPlacement] = {}
    for index, binding in enumerate(used_channel_bindings):
        _, sequencer_id = matching[(0, index)]
        placements[binding.channel_id] = SequencerPlacement(
            channel_id=binding.channel_id,
            port_id=binding.port_id,
            module_location=module_location,
            sequencer_index=sequencer_id,
            sequencer_configuration=banks[binding.port_id][sequencer_id],
        )
    return placements


def _is_compatible(
    binding: QbloxChannelBinding, sequencer_configuration: QbloxSequencerConfiguration
) -> bool:
    """Return whether a supplied sequencer can drive the channel as configured.

    A sequencer whose supplied NCO frequency or mixer correction is pinned belongs to the
    channel whose calibration it names, so it must not be matched to another. This mirrors
    the checks resolution makes on the placement.

    TODO(COMPILER-1631): This assumes a calibrated canonical channel. Revisit once the spike
    decides how requests are bound by port and capability.
    """

    values = sequencer_configuration.values
    nco = values.get("nco")
    mixer = values.get("mixer")
    nco = nco if isinstance(nco, Mapping) else {}
    mixer = mixer if isinstance(mixer, Mapping) else {}

    required_frequency = float(
        binding.carrier_frequency - (binding.oscillator_frequency or 0)
    )
    checks = (
        (nco.get("freq"), required_frequency, {"rel_tol": 1e-12, "abs_tol": 1.0}),
        (mixer.get("phase_offset"), degrees(binding.phase_offset), {"abs_tol": 1e-9}),
        (mixer.get("gain_ratio"), binding.imbalance, {"abs_tol": 1e-9}),
    )
    # A value that is not a real number is left for resolution to reject with context.
    return all(
        not _is_real_number(supplied) or isclose(supplied, required, **tolerance)
        for supplied, required, tolerance in checks
    )


def _is_real_number(value: object) -> bool:
    """Return whether a supplied value is a real number, as resolution defines one."""

    return isinstance(value, int | float) and not isinstance(value, bool)


def _all_placed(matching: Mapping, channel_count: int) -> bool:
    """Return whether a matching gives every channel node a sequencer."""

    return all((0, index) in matching for index in range(channel_count))
