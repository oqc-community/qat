# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest
from frozendict import frozendict
from pydantic_extra_types.semantic_version import SemanticVersion

from qat.experimental.system_data.canonical.schema import (
    AttributeEntry,
    CanonicalSystemData,
    ChannelData,
    ExternalResourceData,
    OscillatorData,
    PortData,
)
from qat.experimental.system_data.hardware.qblox.models import (
    PortReference,
    QbloxModuleKind,
    QbloxModuleLocation,
)
from qat.experimental.system_data.hardware.qblox.view import QbloxHardwareView


def _reference(
    instrument_id="cluster",
    slot_idx=2,
    oscillator_id="canonical-oscillator",
    kind=QbloxModuleKind.qcm,
) -> PortReference:
    return PortReference(
        kind=kind,
        module_location=QbloxModuleLocation(instrument_id, slot_idx),
        oscillator_id=oscillator_id,
    )


def _canonical(
    kind: str = "QCM",
    instrument_id="cluster",
    slot_idx=2,
) -> CanonicalSystemData:
    port_id = "canonical-port"
    port_resource_id = f"A-CH-{kind}-2"
    oscillator_id = "canonical-oscillator"
    oscillator_resource_id = f"A-LO-0-{kind}-2"
    return CanonicalSystemData(
        acquire_limit=100,
        external_resources=(
            ExternalResourceData(
                id=port_resource_id,
                object_type="port",
                attributes=(
                    AttributeEntry(
                        key="qblox",
                        value=_reference(
                            instrument_id=instrument_id,
                            slot_idx=slot_idx,
                            oscillator_id=oscillator_id,
                            kind=QbloxModuleKind.from_qblox_name(kind),
                        ),
                    ),
                ),
            ),
            ExternalResourceData(id=oscillator_resource_id, object_type="oscillator"),
        ),
        ports=(
            PortData(
                id=port_id,
                sample_time=1000,
                external_resource_id=port_resource_id,
                acquire_allowed="QRM" in kind or kind == "QRC",
            ),
        ),
        oscillators=(
            OscillatorData(
                id=oscillator_id,
                frequency=4_000_000_000,
                external_resource_id=oscillator_resource_id,
            ),
        ),
        channels=(
            ChannelData(
                id="channel",
                port_id=port_id,
                frequency=4_200_000_000,
                oscillator_reference=oscillator_id,
                scale=0.75 + 0.25j,
                imbalance=0.9,
                phase_offset=0.125,
            ),
        ),
        metadata=(
            AttributeEntry(key="driver_version", value="1.2.3"),
            AttributeEntry(key="fw_version", value="4.5.6"),
        ),
    )


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("QCM", QbloxModuleKind.qcm),
        ("QCM-RF", QbloxModuleKind.qcm_rf),
        ("QRM", QbloxModuleKind.qrm),
        ("QRM-RF", QbloxModuleKind.qrm_rf),
        ("QRC", QbloxModuleKind.qrc),
    ],
)
def test_derives_representative_module_kinds(name, kind):
    view = QbloxHardwareView.derive(_canonical(name))

    assert view.acquire_limit == 100
    assert len(view.modules) == 1
    module_location = QbloxModuleLocation("cluster", 2)
    module = view.modules[module_location]
    assert module.kind is kind
    assert module.location.instrument_id == "cluster"
    assert module.location.slot == 2
    assert module.ports[0].port_id == "canonical-port"
    assert module.oscillators[0].oscillator_id == "canonical-oscillator"
    binding = module.channel_bindings[0]
    assert view.modules[module_location] is module
    assert view.port_bindings["canonical-port"] is module.ports[0]
    assert view.oscillator_bindings["canonical-oscillator"] is module.oscillators[0]
    assert view.channel_bindings["channel"] is binding
    assert all(
        isinstance(mapping, frozendict)
        for mapping in (
            view.modules,
            view.port_bindings,
            view.oscillator_bindings,
            view.channel_bindings,
        )
    )
    with pytest.raises(TypeError):
        view.modules[module_location] = module
    assert binding.channel_id == "channel"
    assert binding.port_id == "canonical-port"
    assert binding.port_resource_id == f"A-CH-{name}-2"
    assert binding.carrier_frequency == 4_200_000_000
    assert binding.oscillator_id == "canonical-oscillator"
    assert binding.oscillator_frequency == 4_000_000_000
    assert binding.oscillator_resource_id == f"A-LO-0-{name}-2"
    assert binding.scale == 0.75 + 0.25j
    assert binding.imbalance == 0.9
    assert binding.phase_offset == 0.125
    assert binding.module_location is module.location
    assert binding.module_location.instrument_id == "cluster"
    assert binding.module_location.slot == 2
    assert module.ports[0].block_size == 1
    assert module.ports[0].min_blocks == 1
    assert module.ports[0].max_blocks == -1


def test_derivation_preserves_port_timing_constraints():
    canonical = _canonical()
    constrained_port = PortData(
        id=canonical.ports[0].id,
        sample_time=500,
        block_size=4,
        min_blocks=2,
        max_blocks=16,
        external_resource_id=canonical.ports[0].external_resource_id,
    )

    view = QbloxHardwareView.derive(
        CanonicalSystemData(
            external_resources=canonical.external_resources,
            ports=(constrained_port,),
            oscillators=canonical.oscillators,
            channels=canonical.channels,
            metadata=canonical.metadata,
        )
    )

    binding = view.port_bindings[constrained_port.id]
    assert binding.sample_time == 500
    assert binding.block_size == 4
    assert binding.min_blocks == 2
    assert binding.max_blocks == 16


def test_direct_construction_copies_module_mapping():
    derived = QbloxHardwareView.derive(_canonical())
    modules = dict(derived.modules)
    view = QbloxHardwareView(
        SemanticVersion.parse("0.0.0"), SemanticVersion.parse("0.0.0"), 100, modules
    )

    modules.clear()

    assert view.modules == derived.modules


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (
            lambda canonical: (
                canonical.external_resources[0].attributes
                + (AttributeEntry(key="qblox", value=_reference()),)
            ),
            "duplicate 'qblox' attributes",
        ),
        (
            lambda canonical: (AttributeEntry(key="qblox", value={}),),
            "materialised Qblox reference",
        ),
    ],
)
def test_rejects_invalid_target_attribute_shapes(mutator, expected):
    canonical = _canonical()
    resource = canonical.external_resources[0]
    replacement = ExternalResourceData(
        id=resource.id,
        object_type=resource.object_type,
        attributes=mutator(canonical),
    )
    invalid = CanonicalSystemData(
        external_resources=(replacement, *canonical.external_resources[1:]),
        ports=canonical.ports,
        oscillators=canonical.oscillators,
        channels=canonical.channels,
        metadata=canonical.metadata,
    )
    with pytest.raises(ValueError, match=expected):
        QbloxHardwareView.derive(invalid)


def test_ignores_duplicate_attributes_on_unrelated_resources():
    canonical = _canonical()
    unrelated = ExternalResourceData(
        id="camera",
        object_type="camera",
        attributes=(
            AttributeEntry(key="setting", value=1),
            AttributeEntry(key="setting", value=2),
        ),
    )

    view = QbloxHardwareView.derive(
        CanonicalSystemData(
            external_resources=(*canonical.external_resources, unrelated),
            ports=canonical.ports,
            oscillators=canonical.oscillators,
            channels=canonical.channels,
            metadata=canonical.metadata,
        )
    )

    assert len(view.modules) == 1


def test_does_not_infer_qblox_identity_from_resource_names():
    canonical = _canonical()
    resource = ExternalResourceData(id="A-CH-QCM-2", object_type="QCM")
    port = PortData(
        id="A-CH-QCM-2",
        sample_time=1000,
        external_resource_id=resource.id,
    )

    view = QbloxHardwareView.derive(
        CanonicalSystemData(
            external_resources=(resource,),
            ports=(port,),
            metadata=canonical.metadata,
        )
    )

    assert not view.modules


@pytest.mark.parametrize("external_resource_id", [None, "missing-resource"])
def test_ignores_ports_without_a_resolved_external_resource(external_resource_id):
    canonical = _canonical()
    port = PortData(
        id="unresolved-port",
        sample_time=1000,
        external_resource_id=external_resource_id,
    )

    view = QbloxHardwareView.derive(
        CanonicalSystemData(ports=(port,), metadata=canonical.metadata)
    )

    assert not view.modules


def test_ignores_channels_without_a_projected_port():
    canonical = _canonical()
    orphan_channel = ChannelData(
        id="orphan-channel",
        port_id="missing-port",
        frequency=4_200_000_000,
    )

    view = QbloxHardwareView.derive(
        CanonicalSystemData(
            external_resources=canonical.external_resources,
            ports=canonical.ports,
            oscillators=canonical.oscillators,
            channels=(orphan_channel,),
            metadata=canonical.metadata,
        )
    )

    assert "orphan-channel" not in view.channel_bindings


def test_typed_reference_is_authoritative_over_resource_object_type():
    canonical = _canonical()
    resources = tuple(
        ExternalResourceData(
            id=resource.id,
            object_type="QCM",
            attributes=resource.attributes,
        )
        for resource in canonical.external_resources
    )

    view = QbloxHardwareView.derive(
        CanonicalSystemData(
            external_resources=resources,
            ports=canonical.ports,
            oscillators=canonical.oscillators,
            channels=canonical.channels,
            metadata=canonical.metadata,
        )
    )

    assert view.modules[QbloxModuleLocation("cluster", 2)].kind is QbloxModuleKind.qcm


def test_typed_reference_is_authoritative_over_port_identifier():
    canonical = _canonical()
    contradictory_port = PortData(
        id="canonical-CH-QRM-2",
        sample_time=1000,
        external_resource_id=canonical.ports[0].external_resource_id,
    )

    view = QbloxHardwareView.derive(
        CanonicalSystemData(
            external_resources=canonical.external_resources,
            ports=(contradictory_port,),
            oscillators=canonical.oscillators,
            metadata=canonical.metadata,
        )
    )

    assert view.modules[QbloxModuleLocation("cluster", 2)].kind is QbloxModuleKind.qcm


@pytest.mark.parametrize(
    ("port_resource_id", "oscillator_resource_id"),
    [
        ("A-CH-QCM-3", "A-LO-0-QCM-2"),
        ("A-CH-QCM-2", "A-LO-0-QCM-3"),
    ],
)
def test_typed_module_location_is_authoritative_over_resource_identifiers(
    port_resource_id, oscillator_resource_id
):
    canonical = _canonical()
    port_resource = ExternalResourceData(
        id=port_resource_id,
        object_type="port",
        attributes=canonical.external_resources[0].attributes,
    )
    oscillator_resource = ExternalResourceData(
        id=oscillator_resource_id,
        object_type="oscillator",
    )
    port = PortData(
        id=canonical.ports[0].id,
        sample_time=canonical.ports[0].sample_time,
        external_resource_id=port_resource_id,
    )
    oscillator = OscillatorData(
        id=canonical.oscillators[0].id,
        frequency=canonical.oscillators[0].frequency,
        external_resource_id=oscillator_resource_id,
    )

    view = QbloxHardwareView.derive(
        CanonicalSystemData(
            external_resources=(port_resource, oscillator_resource),
            ports=(port,),
            oscillators=(oscillator,),
            channels=canonical.channels,
            metadata=canonical.metadata,
        )
    )

    assert QbloxModuleLocation("cluster", 2) in view.modules


def test_projects_custom_resource_identifiers():
    canonical = _canonical()
    port_resource = ExternalResourceData(
        id="tenant-CH-routing-key-17",
        object_type="QCM",
        attributes=(
            AttributeEntry(
                key="qblox",
                value=_reference(oscillator_id="tenant-LO-routing-key-18"),
            ),
        ),
    )
    oscillator_resource = ExternalResourceData(
        id="tenant-LO-resource-key-19",
        object_type="QCM",
    )
    port = PortData(
        id=canonical.ports[0].id,
        sample_time=canonical.ports[0].sample_time,
        external_resource_id=port_resource.id,
    )
    oscillator = OscillatorData(
        id="tenant-LO-routing-key-18",
        frequency=canonical.oscillators[0].frequency,
        external_resource_id=oscillator_resource.id,
    )
    channel = ChannelData(
        id=canonical.channels[0].id,
        port_id=port.id,
        frequency=canonical.channels[0].frequency,
        oscillator_reference=oscillator.id,
    )

    view = QbloxHardwareView.derive(
        CanonicalSystemData(
            external_resources=(port_resource, oscillator_resource),
            ports=(port,),
            oscillators=(oscillator,),
            channels=(channel,),
            metadata=canonical.metadata,
        )
    )

    assert QbloxModuleLocation("cluster", 2) in view.modules


def test_rejects_missing_oscillator_join_and_cross_module_use():
    canonical = _canonical()
    oscillator = canonical.oscillators[0]
    with pytest.raises(ValueError, match="[Oo]scillator.*missing external resource"):
        QbloxHardwareView.derive(
            CanonicalSystemData(
                external_resources=(canonical.external_resources[0],),
                ports=canonical.ports,
                oscillators=(oscillator,),
                channels=canonical.channels,
                metadata=canonical.metadata,
            )
        )


def test_rejects_port_reference_to_missing_oscillator():
    canonical = _canonical()
    port_resource = canonical.external_resources[0]
    replacement_resource = ExternalResourceData(
        id=port_resource.id,
        object_type=port_resource.object_type,
        attributes=(
            AttributeEntry(
                key="qblox",
                value=_reference(oscillator_id="missing-oscillator"),
            ),
        ),
    )

    with pytest.raises(ValueError, match="port references missing oscillator"):
        QbloxHardwareView.derive(
            CanonicalSystemData(
                external_resources=(
                    replacement_resource,
                    *canonical.external_resources[1:],
                ),
                ports=canonical.ports,
                oscillators=canonical.oscillators,
                channels=canonical.channels,
                metadata=canonical.metadata,
            )
        )


def test_rejects_channel_reference_to_missing_oscillator():
    canonical = _canonical()
    channel = ChannelData(
        id=canonical.channels[0].id,
        port_id=canonical.channels[0].port_id,
        frequency=canonical.channels[0].frequency,
        oscillator_reference="missing-oscillator",
    )

    with pytest.raises(ValueError, match="channel references missing oscillator"):
        QbloxHardwareView.derive(
            CanonicalSystemData(
                external_resources=canonical.external_resources,
                ports=canonical.ports,
                oscillators=canonical.oscillators,
                channels=(channel,),
                metadata=canonical.metadata,
            )
        )


def test_rejects_ports_with_different_module_kinds_at_same_location():
    canonical = _canonical()
    second_resource_id = "second-port-resource"
    second_resource = ExternalResourceData(
        id=second_resource_id,
        object_type="port",
        attributes=(
            AttributeEntry(
                key="qblox",
                value=_reference(oscillator_id=None, kind=QbloxModuleKind.qrm),
            ),
        ),
    )
    second_port = PortData(
        id="second-port",
        sample_time=1000,
        external_resource_id=second_resource_id,
    )

    with pytest.raises(ValueError, match="inconsistent module kinds"):
        QbloxHardwareView.derive(
            CanonicalSystemData(
                external_resources=(*canonical.external_resources, second_resource),
                ports=(*canonical.ports, second_port),
                oscillators=canonical.oscillators,
                channels=canonical.channels,
                metadata=canonical.metadata,
            )
        )


def test_rejects_oscillator_without_external_resource_join():
    canonical = _canonical()
    oscillator = OscillatorData(
        id=canonical.oscillators[0].id,
        frequency=canonical.oscillators[0].frequency,
        external_resource_id=None,
    )

    with pytest.raises(ValueError, match="has no external-resource join"):
        QbloxHardwareView.derive(
            CanonicalSystemData(
                external_resources=canonical.external_resources,
                ports=canonical.ports,
                oscillators=(oscillator,),
                channels=canonical.channels,
                metadata=canonical.metadata,
            )
        )

    second_id = "B-CH-QCM-3"
    second_resource = ExternalResourceData(
        id=second_id,
        object_type="port",
        attributes=(
            AttributeEntry(
                key="qblox",
                value=_reference(
                    instrument_id="other",
                    slot_idx=3,
                    oscillator_id=oscillator.id,
                ),
            ),
        ),
    )
    shared_oscillator_resource = ExternalResourceData(
        id="shared-oscillator-resource",
        object_type="QCM",
    )
    shared_oscillator = OscillatorData(
        id=oscillator.id,
        frequency=oscillator.frequency,
        external_resource_id=shared_oscillator_resource.id,
    )
    second_port = PortData(id=second_id, sample_time=1000, external_resource_id=second_id)
    with pytest.raises(ValueError, match="used across inconsistent modules"):
        QbloxHardwareView.derive(
            CanonicalSystemData(
                external_resources=(
                    canonical.external_resources[0],
                    shared_oscillator_resource,
                    second_resource,
                ),
                ports=(*canonical.ports, second_port),
                oscillators=(shared_oscillator,),
                channels=(
                    canonical.channels[0],
                    ChannelData(
                        id="other-channel",
                        port_id=second_port.id,
                        frequency=4_200_000_000,
                        oscillator_reference=shared_oscillator.id,
                    ),
                ),
                metadata=canonical.metadata,
            )
        )


def test_local_oscillator_cannot_span_modules_without_channel_references():
    canonical = _canonical()
    second_resource_id = "B-CH-QCM-3"
    second_port_id = "second-canonical-port"
    second_resource = ExternalResourceData(
        id=second_resource_id,
        object_type="QCM",
        attributes=(
            AttributeEntry(
                key="qblox",
                value=_reference(
                    instrument_id="other",
                    slot_idx=3,
                    oscillator_id=canonical.oscillators[0].id,
                ),
            ),
        ),
    )
    shared_oscillator_resource = ExternalResourceData(
        id="shared-oscillator-resource",
        object_type="QCM",
    )
    shared_oscillator = OscillatorData(
        id=canonical.oscillators[0].id,
        frequency=canonical.oscillators[0].frequency,
        external_resource_id=shared_oscillator_resource.id,
    )

    with pytest.raises(ValueError, match="used across inconsistent modules"):
        QbloxHardwareView.derive(
            CanonicalSystemData(
                external_resources=(
                    canonical.external_resources[0],
                    shared_oscillator_resource,
                    second_resource,
                ),
                ports=(
                    canonical.ports[0],
                    PortData(
                        id=second_port_id,
                        sample_time=1000,
                        external_resource_id=second_resource_id,
                    ),
                ),
                oscillators=(shared_oscillator,),
                metadata=canonical.metadata,
            )
        )


def test_port_channels_must_share_the_local_oscillator():
    canonical = _canonical()
    second_oscillator = OscillatorData(
        id="second-oscillator",
        frequency=4_100_000_000,
        external_resource_id="B-LO-0-QCM-2",
    )
    second_resource = ExternalResourceData(id="B-LO-0-QCM-2", object_type="oscillator")
    second_channel = ChannelData(
        id="second-channel",
        port_id=canonical.ports[0].id,
        frequency=4_300_000_000,
        oscillator_reference=second_oscillator.id,
    )

    with pytest.raises(ValueError, match="local oscillator conflicts"):
        QbloxHardwareView.derive(
            CanonicalSystemData(
                external_resources=(*canonical.external_resources, second_resource),
                ports=canonical.ports,
                oscillators=(*canonical.oscillators, second_oscillator),
                channels=(*canonical.channels, second_channel),
                metadata=canonical.metadata,
            )
        )


def test_driver_firmware_versions_extracted_from_metadata():
    canonical = _canonical()
    view = QbloxHardwareView.derive(canonical)

    assert view.driver_version == SemanticVersion.parse("1.2.3")
    assert view.firmware_version == SemanticVersion.parse("4.5.6")
    assert all(
        module_details.firmware_version == "4.5.6"
        for module_location, module_details in view.modules.items()
    )


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        (
            (
                AttributeEntry(key="driver_version", value=123),
                AttributeEntry(key="fw_version", value="4.5.6"),
            )
        ),
        (
            (
                AttributeEntry(key="driver_version", value="1.2.3"),
                AttributeEntry(key="fw_version", value=None),
            )
        ),
    ],
)
def test_invalid_driver_firmware_version_raises_error(invalid_metadata):
    canonical = _canonical()
    invalid_canonical = CanonicalSystemData(
        external_resources=canonical.external_resources,
        ports=canonical.ports,
        oscillators=canonical.oscillators,
        channels=canonical.channels,
        metadata=invalid_metadata,
    )

    with pytest.raises(ValueError, match="Invalid driver or firmware version"):
        QbloxHardwareView.derive(invalid_canonical)


def test_missing_driver_firmware_versions_raise_error():
    canonical = _canonical()
    missing_metadata = (AttributeEntry(key="some_other_key", value="value"),)
    missing_canonical = CanonicalSystemData(
        external_resources=canonical.external_resources,
        ports=canonical.ports,
        oscillators=canonical.oscillators,
        channels=canonical.channels,
        metadata=missing_metadata,
    )

    with pytest.raises(ValueError, match="Missing required Qblox metadata"):
        QbloxHardwareView.derive(missing_canonical)
