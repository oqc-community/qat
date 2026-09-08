# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Records describing how canonical channels occupy physical Qblox sequencers.

The records here carry the intermediate state of Qblox configuration resolution: the
per-module join of supplied configuration fragments, the sequencer each canonical channel
is allocated, and the Q1 attributes that allocation resolves to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from frozendict import frozendict

from qat.experimental.dialect.q1_sequence.ir.attrs import (
    ModuleConfigAttr,
    SequencerConfigAttr,
)
from qat.experimental.system_data.hardware.qblox.configuration import (
    ConfigValue,
    QbloxSequencerConfiguration,
)
from qat.experimental.system_data.hardware.qblox.models import (
    QbloxModuleLocation,
    QbloxModuleView,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReconciledModuleConfiguration:
    """Supplied configuration of one physical module, joined across its ports.

    A source may describe one module through several canonical ports. Reconciliation
    resolves shared-configuration references and verifies that repeated module-wide values
    agree before storing one copy. Sequencer banks stay port-specific because each port
    supplies the bank its channels may be allocated from, although ports sharing a module
    still contend for the same physical sequencers.

    :ivar module_view: The hardware view of the physical module.
    :ivar module_values: Agreed module-wide values keyed by source field name.
    :ivar sequencer_banks: Supplied sequencer bank of each canonical port, keyed by physical
        sequencer index.
    """

    module_view: QbloxModuleView
    module_values: frozendict[str, ConfigValue] = field(default_factory=frozendict)
    sequencer_banks: frozendict[str, frozendict[int, QbloxSequencerConfiguration]] = field(
        default_factory=frozendict
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class SequencerPlacement:
    """The physical sequencer one canonical channel is allocated.

    :ivar channel_id: Canonical channel identifier.
    :ivar port_id: Canonical port whose supplied bank the sequencer came from.
    :ivar module_location: Physical instrument and slot of the allocated module.
    :ivar sequencer_index: Allocated physical sequencer index.
    :ivar sequencer_configuration: Configuration supplied for the selected sequencer.
    """

    channel_id: str
    port_id: str
    module_location: QbloxModuleLocation
    sequencer_index: int
    sequencer_configuration: QbloxSequencerConfiguration


@dataclass(frozen=True, slots=True, kw_only=True)
class SequencerBinding:
    """Resolved Q1 configuration of one canonical channel's physical sequencer.

    :ivar channel_id: Canonical channel identifier.
    :ivar port_id: Canonical port the sequencer drives.
    :ivar module_location: Physical instrument and slot of the allocated module.
    :ivar sequencer_index: Allocated physical sequencer index.
    :ivar sequencer_config: Resolved digital configuration and connections.
    :ivar module_config: Resolved analogue configuration of the physical module.
    """

    channel_id: str
    port_id: str
    module_location: QbloxModuleLocation
    sequencer_index: int
    sequencer_config: SequencerConfigAttr
    module_config: ModuleConfigAttr
