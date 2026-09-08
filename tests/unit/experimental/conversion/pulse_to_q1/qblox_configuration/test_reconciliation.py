# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from frozendict import frozendict

from qat.experimental.conversion.pulse_to_q1.qblox_configuration.reconciliation import (
    reconcile_configurations,
)
from qat.experimental.system_data.hardware.qblox.configuration import (
    QbloxSuppliedConfiguration,
    supplied_configurations,
)
from qat.experimental.system_data.hardware.qblox.models import (
    QbloxModuleKind,
    QbloxModuleLocation,
)
from qat.experimental.system_data.hardware.qblox.view import QbloxHardwareView

from tests.unit.experimental.conversion.pulse_to_q1.qblox_configuration.helpers import (
    canonical_data,
    sequencer,
    supplied,
)

MODULE_LOCATION = QbloxModuleLocation("cluster", 2)


def _reconciled(configurations, **kwargs):
    data = canonical_data(configurations=configurations, **kwargs)
    return reconcile_configurations(
        QbloxHardwareView.derive(data), supplied_configurations(data)
    )


def test_sequencer_banks_stay_port_specific():
    configurations = _reconciled(
        [
            supplied([sequencer(0), sequencer(1)]),
            supplied([sequencer(2), sequencer(3)]),
        ]
    )

    banks = configurations[MODULE_LOCATION].sequencer_banks
    assert set(banks["port-0"]) == {0, 1}
    assert set(banks["port-1"]) == {2, 3}


def test_module_fragments_are_reconciled_across_ports():
    configurations = _reconciled(
        [
            supplied([sequencer(0)], module_values={"lo": {"out0_en": True}}),
            supplied(
                [sequencer(1, outputs=[1])],
                module_values={"lo": {"out1_en": True}, "attenuation": {"out1": 6}},
            ),
        ]
    )

    assert configurations[MODULE_LOCATION].module_values == frozendict(
        {
            "lo": frozendict({"out0_en": True, "out1_en": True}),
            "attenuation": frozendict({"out1": 6}),
        }
    )


def test_conflicting_module_fragments_name_the_field_and_ports():
    with pytest.raises(ValueError, match=r"'attenuation.out0' values: port 'port-0'"):
        _reconciled(
            [
                supplied([sequencer(0)], module_values={"attenuation": {"out0": 0}}),
                supplied([sequencer(1)], module_values={"attenuation": {"out0": 6}}),
            ]
        )


def test_module_fragment_may_not_change_shape_between_ports():
    with pytest.raises(ValueError, match="a group of fields and port 'port-1' does not"):
        _reconciled(
            [
                supplied([sequencer(0)], module_values={"lo": {"out0_en": True}}),
                supplied([sequencer(1)], module_values={"lo": True}),
            ]
        )


def test_reference_resolves_to_the_sibling_port_configuration():
    described = supplied([sequencer(0), sequencer(1)], module_values={"attenuation": {}})
    configurations = _reconciled([QbloxSuppliedConfiguration(is_reference=True), described])

    banks = configurations[MODULE_LOCATION].sequencer_banks
    assert set(banks["port-0"]) == set(banks["port-1"]) == {0, 1}
    assert banks["port-0"][0] is described.sequencers[0]


def test_reference_without_a_sibling_fragment_is_rejected():
    with pytest.raises(ValueError, match="0 distinct supplied configurations"):
        _reconciled([QbloxSuppliedConfiguration(is_reference=True)])


def test_reference_with_ambiguous_siblings_is_rejected():
    with pytest.raises(ValueError, match="2 distinct supplied configurations"):
        _reconciled(
            [
                QbloxSuppliedConfiguration(is_reference=True),
                supplied([sequencer(0)]),
                supplied([sequencer(1)]),
            ]
        )


def test_port_supplying_no_configuration_has_no_bank():
    configurations = _reconciled([supplied([sequencer(0)]), None])

    assert set(configurations[MODULE_LOCATION].sequencer_banks) == {"port-0"}


def test_reconciliation_covers_every_module_of_the_view():
    configurations = _reconciled(
        [supplied([sequencer(0)])], kind=QbloxModuleKind.qrm_rf, slot=14
    )

    location = QbloxModuleLocation("cluster", 14)
    assert set(configurations) == {location}
    assert configurations[location].module_view.kind is QbloxModuleKind.qrm_rf
