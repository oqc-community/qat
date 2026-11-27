# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import itertools
import os
import subprocess
import sys
from dataclasses import replace

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


def _port_indices(placements, port):
    return [
        placement.sequencer_index
        for placement in placements.values()
        if placement.port_id == f"port-{port}"
    ]


def test_each_channel_gets_a_distinct_sequencer_from_its_port_bank():
    placements = _allocate(
        [supplied([sequencer(0), sequencer(2), sequencer(4)])], channels_per_port=3
    )

    assert sorted(_port_indices(placements, 0)) == [0, 2, 4]


def test_qrc_channels_are_placed_on_distinct_sequencers_of_their_bank():
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

    assert sorted(_port_indices(placements, 0)) == [8, 9, 10]


def test_qrc_control_and_readout_ports_both_fit_when_their_banks_overlap():
    # The control port can use sequencers 0-11 and the readout port only 0-7, so the one
    # valid assignment gives the control channels 8-11. A greedy lowest-first pass would
    # let them take 0-3 and starve the readout port.
    control_bank = [sequencer(index, outputs=[2]) for index in range(12)]
    readout_bank = [sequencer(index, inputs=[0]) for index in range(8)]

    placements = _allocate(
        [supplied(control_bank), supplied(readout_bank)],
        kind=QbloxModuleKind.qrc,
        channels_per_port=[4, 8],
    )

    assert sorted(_port_indices(placements, 0)) == [8, 9, 10, 11]
    assert sorted(_port_indices(placements, 1)) == list(range(8))


def test_ports_sharing_a_module_do_not_share_sequencers():
    bank = [sequencer(index) for index in range(4)]
    placements = _allocate(
        [supplied(bank), supplied(bank)],
        channels_per_port=2,
    )

    assert sorted(placement.sequencer_index for placement in placements.values()) == [
        0,
        1,
        2,
        3,
    ]
    assert {placement.port_id for placement in placements.values()} == {
        "port-0",
        "port-1",
    }


def test_placement_carries_the_supplied_sequencer_of_the_selected_port():
    described = supplied([sequencer(3, outputs=[1]), sequencer(5, outputs=[1])])
    placements = _allocate([QbloxSuppliedConfiguration(is_reference=True), described])

    placement = placements["port-1-channel-0"]
    assert placement.sequencer_index in {3, 5}
    assert placement.sequencer_configuration in described.sequencers
    assert placement.sequencer_configuration.index == placement.sequencer_index
    assert placement.module_location.slot == 2


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_a_valid_assignment_is_found_for_any_order_of_the_channels(order):
    # Only one assignment works: port 0 -> 2, port 1 -> 0, port 2 -> 1. Greedy allocation
    # finds it for some orderings of the module's channels only.
    data = canonical_data(
        configurations=[
            supplied([sequencer(0), sequencer(1), sequencer(2)]),
            supplied([sequencer(0)]),
            supplied([sequencer(0), sequencer(1)]),
        ]
    )
    [(location, configuration)] = reconcile_configurations(
        QbloxHardwareView.derive(data), supplied_configurations(data)
    ).items()
    bindings = configuration.module_view.channel_bindings
    shuffled = replace(
        configuration,
        module_view=replace(
            configuration.module_view,
            channel_bindings=tuple(bindings[index] for index in order),
        ),
    )

    placements = allocate_sequencers(
        {location: shuffled}, {channel.id for channel in data.channels}
    )

    assert {
        channel_id: placement.sequencer_index
        for channel_id, placement in placements.items()
    } == {"port-0-channel-0": 2, "port-1-channel-0": 0, "port-2-channel-0": 1}


@pytest.mark.parametrize("bank_sizes", list(itertools.permutations((4, 8, 12))))
def test_a_twelve_sequencer_slot_is_fully_allocated_for_any_port_order(bank_sizes):
    # Shaped like slot 1 of qft_5q: nested banks that together need all 12 sequencers.
    # Greedy allocation fails for every ordering except smallest bank first.
    banks = [[sequencer(index) for index in range(count)] for count in bank_sizes]
    counts = [4, 4, 4]

    placements = _allocate([supplied(bank) for bank in banks], channels_per_port=counts)

    assert sorted(p.sequencer_index for p in placements.values()) == list(range(12))
    for port, bank in enumerate(banks):
        assert set(_port_indices(placements, port)) <= {c.index for c in bank}


def test_unplaceable_channels_are_all_named_with_their_port_and_bank():
    with pytest.raises(ValueError, match="Could not allocate") as error:
        _allocate([supplied([sequencer(0)])], channels_per_port=3)

    message = str(error.value)
    assert "Could not allocate a sequencer to 2 of 3 channel(s)" in message
    assert message.count("(port 'port-0', candidate sequencers [0])") == 2


def test_port_supplying_no_bank_is_rejected():
    with pytest.raises(ValueError, match="supplies no sequencer bank"):
        _allocate([supplied()])


def test_port_without_supplied_configuration_is_rejected():
    with pytest.raises(ValueError, match="supplies no sequencer bank"):
        _allocate([None])


def _two_channels_sharing_a_bank():
    return _allocate([supplied([sequencer(0), sequencer(1)])], channels_per_port=2)


def _two_ports_sharing_a_bank():
    bank = [sequencer(index) for index in range(4)]
    return _allocate([supplied(bank), supplied(bank)], channels_per_port=2)


def _nested_banks():
    banks = [[sequencer(index) for index in range(count)] for count in (4, 8, 12)]
    return _allocate([supplied(bank) for bank in banks], channels_per_port=[4, 4, 4])


_TIE_BREAK_CASES = {
    "two_channels_sharing_a_bank": (
        _two_channels_sharing_a_bank,
        {"port-0-channel-0": 1, "port-0-channel-1": 0},
    ),
    "two_ports_sharing_a_bank": (
        _two_ports_sharing_a_bank,
        {
            "port-0-channel-0": 3,
            "port-0-channel-1": 0,
            "port-1-channel-0": 1,
            "port-1-channel-1": 2,
        },
    ),
    "nested_banks": (
        _nested_banks,
        {
            "port-0-channel-0": 3,
            "port-0-channel-1": 0,
            "port-0-channel-2": 2,
            "port-0-channel-3": 1,
            "port-1-channel-0": 7,
            "port-1-channel-1": 6,
            "port-1-channel-2": 5,
            "port-1-channel-3": 4,
            "port-2-channel-0": 11,
            "port-2-channel-1": 10,
            "port-2-channel-2": 9,
            "port-2-channel-3": 8,
        },
    ),
}


def _tie_break_placements(case: str) -> dict[str, int]:
    builder, _ = _TIE_BREAK_CASES[case]
    return {
        channel_id: placement.sequencer_index
        for channel_id, placement in sorted(builder().items())
    }


@pytest.mark.parametrize("case", _TIE_BREAK_CASES)
def test_tie_break_between_equivalent_matchings_is_unchanged(case):
    # Canary. Several maximum matchings are valid in these cases, and which one is returned
    # is left to networkx (Hopcroft-Karp), whose docs don't promise it. A failure after a
    # networkx upgrade means its tie-break changed. Check the new placements are acceptable
    # and update the expected values, as nothing requires these particular indices.
    _, expected = _TIE_BREAK_CASES[case]

    assert _tie_break_placements(case) == expected


def test_placements_do_not_depend_on_pythonhashseed(request):
    script = (
        "import json;"
        "from tests.unit.experimental.conversion.pulse_to_q1.qblox_configuration"
        ".test_allocation import _TIE_BREAK_CASES, _tie_break_placements;"
        "print(json.dumps({c: _tie_break_placements(c) for c in _TIE_BREAK_CASES}))"
    )

    outputs = {
        subprocess.run(  # noqa: S603 - interpreter and script are fixed test inputs
            [sys.executable, "-c", script],
            cwd=request.config.rootpath,
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("0", "1")
    }

    assert len(outputs) == 1
