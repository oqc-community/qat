# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import json
from pathlib import Path

import pytest
from frozendict import frozendict

from qat.experimental.system_data.hardware.qblox.configuration import (
    QBLOX_CONFIGURATION_ATTRIBUTE,
    AcquisitionPathConnection,
    OutputPathConnection,
    PortConnection,
)
from qat.experimental.system_data.hardware.qblox.models import (
    DirectionKind,
    QbloxModuleKind,
    QbloxModuleLocation,
    SignalPath,
)
from qat.experimental.system_data.materialisers.boundary import materialise
from qat.experimental.system_data.materialisers.errors import MaterialisationIntegrityError
from qat.experimental.system_data.materialisers.purr.decoder import (
    decode_jsonpickle_payload,
)
from qat.experimental.system_data.materialisers.purr.extensions.qblox import (
    QBLOX_PORT_REFERENCE_ATTRIBUTE,
    decode_qblox_configuration,
    decode_qblox_port_attributes,
    decode_qblox_port_reference,
)


def _payload(config=None, port_id="A-CH-QCM-RF-2"):
    return {
        "id": port_id,
        "baseband": {
            "id": "A-LO-0-QCM-RF-2",
            "instrument_id": "cluster",
            "slot_idx": 2,
            "config": config,
        },
    }


def test_port_reference_carries_the_physical_identity():
    reference = decode_qblox_port_reference(_payload())

    assert reference.kind is QbloxModuleKind.qcm_rf
    assert reference.module_location == QbloxModuleLocation("cluster", 2)
    assert reference.oscillator_id == "A-LO-0-QCM-RF-2"


def test_channels_of_other_targets_are_left_untouched():
    assert decode_qblox_port_reference({"id": "CH1", "baseband": {}}) is None
    assert decode_qblox_port_attributes({"id": "CH1"}) is None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"id": "A-CH-QCM-RF-2"}, "no baseband payload"),
        (_payload() | {"baseband": {"instrument_id": "", "slot_idx": 2}}, "no instrument"),
        (
            _payload() | {"baseband": {"instrument_id": "cluster"}},
            "no module slot",
        ),
        (
            _payload() | {"baseband": {"instrument_id": "cluster", "slot_idx": True}},
            "no module slot",
        ),
        (
            _payload()
            | {"baseband": {"instrument_id": "cluster", "slot_idx": 2, "id": ""}},
            "invalid baseband id",
        ),
    ],
)
def test_malformed_physical_identity_is_rejected(payload, expected):
    with pytest.raises(ValueError, match=expected):
        decode_qblox_port_reference(payload)


def test_serialisation_markers_and_unset_fields_are_dropped():
    configuration = decode_qblox_configuration(
        {
            "py/object": "qat.purr.backends.qblox.config.QbloxConfig",
            "slot_idx": 2,
            "module": {
                "py/object": "qat.purr.backends.qblox.config.ModuleConfig",
                "lo": {"out0_en": True, "out0_freq": None},
                "offset": {"out0_path0": None, "out0_path1": None},
                "scope_acq": None,
            },
            "sequencers": {},
        }
    )

    assert configuration.module_configuration.values == frozendict(
        {"lo": frozendict({"out0_en": True})}
    )
    assert configuration.sequencers == ()


def test_pydantic_serialisation_state_decodes_as_qblox_configuration():
    decoded = decode_jsonpickle_payload(
        {
            "py/object": "qat.backend.qblox.config.specification.QbloxConfig",
            "py/state": {
                "__dict__": {
                    "slot_idx": None,
                    "module": {
                        "py/object": "qat.backend.qblox.config.specification.ModuleConfig",
                        "py/state": {
                            "__dict__": {"lo": {"out0_in0_en": True}},
                            "__pydantic_extra__": None,
                            "__pydantic_fields_set__": {"py/set": ["lo"]},
                            "__pydantic_private__": None,
                        },
                    },
                    "sequencers": {},
                },
                "__pydantic_extra__": None,
                "__pydantic_fields_set__": {"py/set": ["module", "sequencers"]},
                "__pydantic_private__": None,
            },
        }
    )

    configuration = decode_qblox_configuration(decoded)

    assert configuration.module_configuration.values == frozendict(
        {"lo": frozendict({"out0_in0_en": True})}
    )
    assert configuration.sequencers == ()


def test_reference_stubs_become_configuration_references():
    configuration = decode_qblox_configuration({"_adapter_reference": "mapping"})

    assert configuration.is_reference
    assert configuration.module_configuration is None
    assert configuration.sequencers == ()


def test_connection_fields_decode_into_typed_routing():
    configuration = decode_qblox_configuration(
        {
            "module": None,
            "sequencers": {
                "0": {
                    "sync_en": True,
                    "connection": {
                        "bulk_value": ["out0_1", "in0", "io0_1"],
                        "out0": "I",
                        "out2": "off",
                        "acq_I": "in0",
                        "acq_Q": "off",
                    },
                    "nco": {"freq": 1.0e8, "phase_offs": None},
                }
            },
        }
    )

    sequencer = configuration.sequencers[0]
    assert sequencer.index == 0
    assert sequencer.values == frozendict(
        {"sync_en": True, "nco": frozendict({"freq": 1.0e8})}
    )
    connection = sequencer.connection
    assert connection.connections == (
        PortConnection(direction=DirectionKind.output, port_ids=(0, 1)),
        PortConnection(direction=DirectionKind.input, port_ids=(0,)),
        PortConnection(direction=DirectionKind.io, port_ids=(0, 1)),
    )
    assert connection.output_path_connections == (
        OutputPathConnection(output_id=0, path=SignalPath.i),
    )
    assert connection.disabled_outputs == frozenset({2})
    assert connection.acquisition_path_connections == (
        AcquisitionPathConnection(input_id=0, path=SignalPath.i),
    )
    assert connection.disabled_acquisition_paths == frozenset({SignalPath.q})


def test_a_complex_input_connection_decodes_into_two_acquisition_lanes():
    """``in0_1`` is valid complex-mode acquisition syntax on baseband modules."""

    configuration = decode_qblox_configuration(
        {"sequencers": {0: {"connection": {"bulk_value": ["out0_1", "in0_1"]}}}}
    )

    assert configuration.sequencers[0].connection.connections == (
        PortConnection(direction=DirectionKind.output, port_ids=(0, 1)),
        PortConnection(direction=DirectionKind.input, port_ids=(0, 1)),
    )


@pytest.mark.parametrize(
    ("acq", "expected_paths", "expected_disabled"),
    [
        ("in1", (AcquisitionPathConnection(input_id=1, path=SignalPath.iq),), None),
        ("off", (), True),
    ],
)
def test_combined_acquisition_field_decodes(acq, expected_paths, expected_disabled):
    configuration = decode_qblox_configuration(
        {"sequencers": {0: {"connection": {"acq": acq}}}}
    )

    connection = configuration.sequencers[0].connection
    assert connection.acquisition_path_connections == expected_paths
    assert connection.acquisition_disabled is expected_disabled


def test_boolean_acquisition_field_is_an_explicit_enable():
    configuration = decode_qblox_configuration(
        {"sequencers": {0: {"connection": {"acq": True}}}}
    )

    assert configuration.sequencers[0].connection.acquisition_enabled is True


def test_empty_connection_is_reported_as_absent():
    configuration = decode_qblox_configuration(
        {"sequencers": {0: {"connection": {"bulk_value": None}}}}
    )

    assert configuration.sequencers[0].connection is None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"module": 1}, "must be a mapping"),
        ({"sequencers": []}, "sequencers must be a mapping"),
        ({"sequencers": {"a": {}}}, "keyed by sequencer index"),
        ({"sequencers": {0: 1}}, "must be a mapping"),
        ({"module": {"lo": object()}}, "unsupported object value"),
        (
            {"module": {"lo": {"_adapter_reference": "mapping"}}},
            "unresolved shared-object reference",
        ),
        ({"colour": "green"}, "unsupported fields"),
        (
            {"_adapter_reference": "mapping", "colour": "green"},
            "refers to another port but carries fields",
        ),
        ({"sequencers": {0: {"connection": {"gain": 1}}}}, "unsupported fields"),
        (
            {"sequencers": {0: {"connection": {"bulk_value": ["sideways"]}}}},
            "not a Qblox connection string",
        ),
        (
            {"sequencers": {0: {"connection": {"bulk_value": "out0"}}}},
            "must be a list of connection strings",
        ),
        ({"sequencers": {0: {"connection": {"out0": 1}}}}, "must be a sequencer path"),
        ({"sequencers": {0: {"connection": {"out0": "Z"}}}}, "must be a sequencer path"),
        ({"sequencers": {0: {"connection": {"acq_I": "out0"}}}}, "physical input"),
        ({"sequencers": {0: {"connection": {"acq_I": "in0_1"}}}}, "physical input"),
        ({"sequencers": {0: {"connection": {"acq": 1}}}}, "boolean or name a physical"),
        (
            {"sequencers": {0: {"connection": {"acq": "off", "bulk_value": ["in0"]}}}},
            "disables acquisition while binding physical inputs",
        ),
        (
            {"sequencers": {0: {"connection": {"out0": "off", "bulk_value": ["out0"]}}}},
            "both binds and disables outputs",
        ),
    ],
)
def test_malformed_configuration_payloads_are_rejected(payload, expected):
    with pytest.raises(ValueError, match=expected):
        decode_qblox_configuration(payload)


def test_port_attributes_carry_both_typed_extensions():
    attributes = decode_qblox_port_attributes(
        _payload({"module": {"lo": {"out0_en": True}}})
    )

    assert set(attributes) == {
        QBLOX_PORT_REFERENCE_ATTRIBUTE,
        QBLOX_CONFIGURATION_ATTRIBUTE,
    }
    assert attributes[
        QBLOX_CONFIGURATION_ATTRIBUTE
    ].module_configuration.values == frozendict({"lo": frozendict({"out0_en": True})})


def test_port_without_supplied_configuration_carries_only_its_identity():
    assert set(decode_qblox_port_attributes(_payload())) == {QBLOX_PORT_REFERENCE_ATTRIBUTE}


def test_the_source_boundary_replaces_raw_baseband_metadata_with_typed_extensions():
    payload = json.loads(
        Path("tests/files/calibrations/qblox_calibration.json").read_text()
    )

    data = materialise(source_payload=payload, source_additional_data={})

    ports = [
        resource for resource in data.external_resources if resource.object_type == "port"
    ]
    assert ports
    for resource in ports:
        keys = {entry.key for entry in resource.attributes}
        assert "baseband" not in keys
        assert QBLOX_PORT_REFERENCE_ATTRIBUTE in keys
        assert QBLOX_CONFIGURATION_ATTRIBUTE in keys


def test_malformed_source_metadata_fails_materialisation_at_the_boundary(mocker):
    mocker.patch(
        "qat.experimental.system_data.materialisers.purr.materialisers.signal_paths."
        "decode_qblox_port_attributes",
        side_effect=ValueError("baseband.config has unsupported fields ['colour']"),
    )
    payload = json.loads(
        Path("tests/files/calibrations/qblox_calibration.json").read_text()
    )

    with pytest.raises(MaterialisationIntegrityError) as failure:
        materialise(source_payload=payload, source_additional_data={})

    assert failure.value.details["reason"] == (
        "baseband.config has unsupported fields ['colour']"
    )
    assert failure.value.path.startswith("$.physical_channels.")
