# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from qat.experimental.conversion.pulse_to_q1.qblox_configuration.allocation import (
    allocate_sequencers,
)
from qat.experimental.conversion.pulse_to_q1.qblox_configuration.reconciliation import (
    reconcile_configurations,
)
from qat.experimental.system_data.hardware.qblox.configuration import (
    QbloxSuppliedConfiguration,
    supplied_configurations,
)
from qat.experimental.system_data.hardware.qblox.view import QbloxHardwareView

from tests.unit.experimental.conversion.pulse_to_q1.qblox_configuration.helpers import (
    canonical_data,
    sequencer,
    supplied,
)


def _allocate(configurations, **kwargs):
    data = canonical_data(configurations=configurations, **kwargs)
    configurations = reconcile_configurations(
        QbloxHardwareView.derive(data), supplied_configurations(data)
    )
    return allocate_sequencers(configurations)


def test_channels_take_the_lowest_available_index_of_their_port_bank():
    placements = _allocate(
        [supplied([sequencer(0), sequencer(2), sequencer(4)])], channels_per_port=3
    )

    assert [
        placements[f"port-0-channel-{index}"].sequencer_index for index in range(3)
    ] == [0, 2, 4]


def test_ports_sharing_a_module_do_not_share_sequencers():
    bank = [sequencer(index) for index in range(4)]
    placements = _allocate(
        [supplied(bank), supplied(bank)],
        channels_per_port=2,
    )

    assert [placement.sequencer_index for placement in placements.values()] == [0, 1, 2, 3]
    assert {placement.port_id for placement in placements.values()} == {
        "port-0",
        "port-1",
    }


def test_placement_carries_the_supplied_sequencer_of_the_selected_port():
    described = supplied([sequencer(3, outputs=[1]), sequencer(5, outputs=[1])])
    placements = _allocate([QbloxSuppliedConfiguration(is_reference=True), described])

    placement = placements["port-0-channel-0"]
    assert placement.sequencer_index == 3
    assert placement.sequencer_configuration is described.sequencers[0]
    assert placement.module_location.slot == 2
    assert placements["port-1-channel-0"].sequencer_index == 5


def test_exhausted_port_bank_names_the_channel_and_its_bank():
    with pytest.raises(ValueError, match=r"channel 'port-0-channel-1'; its supplied bank"):
        _allocate([supplied([sequencer(0)])], channels_per_port=2)


def test_port_supplying_no_bank_is_rejected():
    with pytest.raises(ValueError, match="supplies no sequencer bank"):
        _allocate([supplied()])


def test_port_without_supplied_configuration_is_rejected():
    with pytest.raises(ValueError, match="supplies no sequencer bank"):
        _allocate([None])
