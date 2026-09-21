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
from qat.experimental.system_data.hardware.qblox.models import QbloxModuleKind
from qat.experimental.system_data.hardware.qblox.view import QbloxHardwareView

from tests.unit.experimental.conversion.pulse_to_q1.qblox_configuration.helpers import (
    canonical_data,
    sequencer,
    supplied,
)


def _allocate(configurations, **kwargs):
    data = canonical_data(configurations=configurations, **kwargs)
    reconciled = reconcile_configurations(
        QbloxHardwareView.derive(data), supplied_configurations(data)
    )
    used_channel_ids = {channel.id for channel in data.channels}
    return allocate_sequencers(reconciled, used_channel_ids)


def test_channels_take_the_lowest_available_index_of_their_port_bank():
    placements = _allocate(
        [supplied([sequencer(0), sequencer(2), sequencer(4)])], channels_per_port=3
    )

    assert [
        placements[f"port-0-channel-{index}"].sequencer_index for index in range(3)
    ] == [0, 2, 4]


def test_qrc_channels_take_the_highest_available_index_of_their_port_bank():
    # A QRC allocates in reverse so its low readout sequencers stay free for the channels
    # that need them; here three channels take the top three supplied control sequencers.
    placements = _allocate(
        [
            supplied(
                [
                    sequencer(8, outputs=[2]),
                    sequencer(9, outputs=[3]),
                    sequencer(10, outputs=[4]),
                ]
            )
        ],
        kind=QbloxModuleKind.qrc,
        channels_per_port=3,
    )

    assert [
        placements[f"port-0-channel-{index}"].sequencer_index for index in range(3)
    ] == [10, 9, 8]


def test_qrc_reverse_allocation_frees_readout_sequencers_for_a_readout_port():
    # A QRC exposes two ports on one module: a readout port that can only use the eight
    # readout sequencers (0-7), and a control port whose drive outputs are reachable from
    # any sequencer (0-11). The control port is calibrated first, so a forward, lowest-first
    # allocation would let its four channels consume readout sequencers 0-3 and starve the
    # readout port. Reverse allocation gives the control channels the four high control
    # sequencers (8-11), leaving every readout sequencer free for the readout channels.
    control_bank = [sequencer(index, outputs=[2]) for index in range(12)]
    readout_bank = [sequencer(index, inputs=[0]) for index in range(8)]

    placements = _allocate(
        [supplied(control_bank), supplied(readout_bank)],
        kind=QbloxModuleKind.qrc,
        channels_per_port=[4, 8],
    )

    control_indices = sorted(
        placements[f"port-0-channel-{index}"].sequencer_index for index in range(4)
    )
    readout_indices = sorted(
        placements[f"port-1-channel-{index}"].sequencer_index for index in range(8)
    )
    assert control_indices == [8, 9, 10, 11]
    assert readout_indices == [0, 1, 2, 3, 4, 5, 6, 7]


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
