# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from frozendict import frozendict

from qat.experimental.system_data.canonical.schema import (
    AttributeEntry,
    CanonicalSystemData,
    ExternalResourceData,
)
from qat.experimental.system_data.hardware.qblox.configuration import (
    QBLOX_CONFIGURATION_ATTRIBUTE,
    AcquisitionPathConnection,
    OutputPathConnection,
    PortConnection,
    QbloxModuleConfiguration,
    QbloxSequencerConfiguration,
    QbloxSuppliedConfiguration,
    SequencerConnection,
    immutable_config_value,
    supplied_configurations,
)
from qat.experimental.system_data.hardware.qblox.models import (
    DirectionKind,
    SignalPath,
    connection_input_ids,
    connection_output_ids,
)


def _canonical(*attributes: AttributeEntry) -> CanonicalSystemData:
    return CanonicalSystemData(
        acquire_limit=100,
        external_resources=(
            ExternalResourceData(
                id="port-resource", object_type="port", attributes=attributes
            ),
        ),
    )


def test_config_values_are_deeply_immutable():
    value = immutable_config_value({"lo": {"out0_en": True}, "paths": [1, 2]})

    assert value == frozendict({"lo": frozendict({"out0_en": True}), "paths": (1, 2)})
    with pytest.raises(TypeError):
        value["lo"] = None


def test_config_values_reject_unrepresentable_objects():
    with pytest.raises(ValueError, match=r"config.lo holds an unsupported object value"):
        immutable_config_value({"lo": object()})


def test_config_values_reject_non_string_keys():
    with pytest.raises(ValueError, match="non-string mapping key"):
        immutable_config_value({1: True})


@pytest.mark.parametrize(
    ("direction", "port_ids", "expected"),
    [
        (DirectionKind.output, (0,), "out0"),
        (DirectionKind.output, (0, 1), "out0_1"),
        (DirectionKind.input, (1,), "in1"),
        (DirectionKind.input, (0, 1), "in0_1"),
        (DirectionKind.io, (0, 1), "io0_1"),
    ],
)
def test_port_connection_renders_the_qblox_connection_string(direction, port_ids, expected):
    assert PortConnection(direction=direction, port_ids=port_ids).connection == expected


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (
            lambda: PortConnection(direction=DirectionKind.output, port_ids=()),
            "at least one I/O port",
        ),
        (
            lambda: PortConnection(direction=DirectionKind.output, port_ids=(0, 0)),
            "must be distinct",
        ),
        (
            lambda: PortConnection(direction=DirectionKind.input, port_ids=(0, 1, 2)),
            "at most two I/O ports",
        ),
        (
            lambda: PortConnection(direction=DirectionKind.io, port_ids=(0, 1, 2)),
            "at most two I/O ports",
        ),
        (
            lambda: PortConnection(direction=DirectionKind.output, port_ids=(-1,)),
            "non-negative integers",
        ),
        (
            lambda: OutputPathConnection(output_id=-1, path=SignalPath.i),
            "needs a physical output",
        ),
        (
            lambda: AcquisitionPathConnection(input_id=-1, path=SignalPath.i),
            "needs a physical input",
        ),
        (
            lambda: SequencerConnection(
                output_path_connections=(
                    OutputPathConnection(output_id=0, path=SignalPath.i),
                    OutputPathConnection(output_id=0, path=SignalPath.q),
                )
            ),
            "only one sequencer path",
        ),
        (
            lambda: SequencerConnection(
                acquisition_path_connections=(
                    AcquisitionPathConnection(input_id=0, path=SignalPath.i),
                    AcquisitionPathConnection(input_id=1, path=SignalPath.i),
                )
            ),
            "may select only one input",
        ),
        (
            lambda: QbloxSequencerConfiguration(index=-1),
            "non-negative integer",
        ),
        (
            lambda: QbloxSuppliedConfiguration(
                sequencers=(
                    QbloxSequencerConfiguration(index=1),
                    QbloxSequencerConfiguration(index=0),
                )
            ),
            "ordered by sequencer index",
        ),
        (
            lambda: QbloxSuppliedConfiguration(
                sequencers=(
                    QbloxSequencerConfiguration(index=0),
                    QbloxSequencerConfiguration(index=0),
                )
            ),
            "must not repeat",
        ),
        (
            lambda: QbloxSuppliedConfiguration(
                is_reference=True, module_configuration=QbloxModuleConfiguration()
            ),
            "reference carries no values",
        ),
    ],
)
def test_configuration_records_reject_invalid_values(factory, expected):
    with pytest.raises(ValueError, match=expected):
        factory()


def test_connection_lane_sets_cover_every_binding_form():
    """``io0_1`` drives and acquires on every lane it names."""

    connection = SequencerConnection(
        connections=(
            PortConnection(direction=DirectionKind.io, port_ids=(0, 1)),
            PortConnection(direction=DirectionKind.output, port_ids=(2,)),
            PortConnection(direction=DirectionKind.input, port_ids=(3,)),
        ),
        output_path_connections=(OutputPathConnection(output_id=4, path=SignalPath.i),),
        acquisition_path_connections=(
            AcquisitionPathConnection(input_id=5, path=SignalPath.q),
        ),
        disabled_outputs=frozenset({6}),
    )

    assert connection.output_ids == frozenset({0, 1, 2, 4, 6})
    assert connection.input_ids == frozenset({0, 1, 3, 5})


def test_supplied_configurations_are_keyed_by_external_resource():
    configuration = QbloxSuppliedConfiguration(is_reference=True)
    data = _canonical(
        AttributeEntry(key=QBLOX_CONFIGURATION_ATTRIBUTE, value=configuration)
    )

    assert supplied_configurations(data) == frozendict({"port-resource": configuration})


def test_resources_without_the_extension_are_skipped():
    assert supplied_configurations(_canonical()) == frozendict()


def test_duplicate_extensions_are_rejected():
    configuration = QbloxSuppliedConfiguration()
    data = _canonical(
        AttributeEntry(key=QBLOX_CONFIGURATION_ATTRIBUTE, value=configuration),
        AttributeEntry(key=QBLOX_CONFIGURATION_ATTRIBUTE, value=configuration),
    )

    with pytest.raises(ValueError, match="duplicate 'qblox_config' attributes"):
        supplied_configurations(data)


def test_untyped_extension_values_are_rejected():
    data = _canonical(AttributeEntry(key=QBLOX_CONFIGURATION_ATTRIBUTE, value={}))

    with pytest.raises(ValueError, match="invalid materialised Qblox configuration"):
        supplied_configurations(data)


@pytest.mark.parametrize(
    "connection",
    [
        SequencerConnection(
            connections=(PortConnection(direction=DirectionKind.input, port_ids=(0,)),)
        ),
        SequencerConnection(
            acquisition_path_connections=(
                AcquisitionPathConnection(input_id=0, path=SignalPath.i),
            )
        ),
        SequencerConnection(acquisition_enabled=True),
        SequencerConnection(acquisition_disabled=True),
        SequencerConnection(disabled_acquisition_paths=frozenset({SignalPath.q})),
    ],
)
def test_every_acquisition_routing_form_counts_as_acquisition_use(connection):
    assert connection.uses_acquisition


@pytest.mark.parametrize(
    "connection",
    [
        SequencerConnection(),
        SequencerConnection(
            connections=(PortConnection(direction=DirectionKind.output, port_ids=(0,)),),
            output_path_connections=(OutputPathConnection(output_id=1, path=SignalPath.i),),
            disabled_outputs=frozenset({2}),
        ),
    ],
)
def test_output_only_routing_is_not_acquisition_use(connection):
    assert not connection.uses_acquisition


@pytest.mark.parametrize(
    ("direction", "port_ids", "outputs", "inputs"),
    [
        (DirectionKind.output, (0, 1), (0, 1), ()),
        (DirectionKind.input, (1,), (), (1,)),
        (DirectionKind.io, (0, 1), (0, 1), (0, 1)),
        (DirectionKind.io, (2,), (2,), (2,)),
    ],
)
def test_connection_lane_helpers_are_direction_aware(direction, port_ids, outputs, inputs):
    assert connection_output_ids(direction, port_ids) == outputs
    assert connection_input_ids(direction, port_ids) == inputs
