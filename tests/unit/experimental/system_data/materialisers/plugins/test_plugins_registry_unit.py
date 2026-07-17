# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from qat.experimental.system_data.materialisers import plugins as plugin_api
from qat.experimental.system_data.materialisers.types import SourceType


class _ValidAdditionalData(plugin_api.SourceAdditionalDataModel):
    pass


class _OtherAdditionalData(plugin_api.SourceAdditionalDataModel):
    flag: bool = False


class _ValidPlugin:
    source_type = SourceType.PURR
    source_version = "0.1.0"
    additional_data_model = _ValidAdditionalData

    def verify_integrity(self, source_payload):
        _ = source_payload

    def materialise(self, *, source_payload, source_version, additional_data):
        _ = source_payload
        _ = source_version
        _ = additional_data
        return "ok"


class _ConflictingPlugin:
    source_type = SourceType.PURR
    source_version = "0.1.0"
    additional_data_model = _OtherAdditionalData

    def verify_integrity(self, source_payload):
        _ = source_payload

    def materialise(self, *, source_payload, source_version, additional_data):
        _ = source_payload
        _ = source_version
        _ = additional_data
        return "ok"


class _ExtensionAdditionalData(plugin_api.SourceAdditionalDataModel):
    required_value: str


class _ExtensionPlugin:
    source_type = SourceType.PURR
    source_version = "9.9.9"
    additional_data_model = _ExtensionAdditionalData

    def verify_integrity(self, source_payload):
        _ = source_payload

    def materialise(self, *, source_payload, source_version, additional_data):
        _ = source_payload
        _ = source_version
        return additional_data.required_value


@pytest.fixture(autouse=True)
def _isolated_plugin_registry(monkeypatch):
    monkeypatch.setattr(plugin_api, "_PLUGIN_REGISTRY", {})


def test_register_materialiser_plugin_rejects_invalid_source_type():
    plugin = _ValidPlugin()
    plugin.source_type = "purr"

    with pytest.raises(ValueError, match="source_type"):
        plugin_api.register_materialiser_plugin(plugin=plugin)


def test_register_materialiser_plugin_rejects_empty_source_version():
    plugin = _ValidPlugin()
    plugin.source_version = ""

    with pytest.raises(ValueError, match="source_version"):
        plugin_api.register_materialiser_plugin(plugin=plugin)


def test_register_materialiser_plugin_rejects_invalid_additional_data_model():
    plugin = _ValidPlugin()
    plugin.additional_data_model = dict

    with pytest.raises(ValueError, match="additional_data_model"):
        plugin_api.register_materialiser_plugin(plugin=plugin)


def test_register_materialiser_plugin_rejects_conflicting_duplicate_registration():
    plugin_api.register_materialiser_plugin(plugin=_ValidPlugin())

    with pytest.raises(ValueError, match="already registered"):
        plugin_api.register_materialiser_plugin(plugin=_ConflictingPlugin())


def test_register_materialiser_plugin_replace_overwrites_existing_registration():
    plugin_one = _ValidPlugin()
    plugin_two = _ConflictingPlugin()

    plugin_api.register_materialiser_plugin(plugin=plugin_one)
    plugin_api.register_materialiser_plugin(plugin=plugin_two, replace=True)

    resolved = plugin_api.get_materialiser_plugin(
        source_type=SourceType.PURR,
        source_version="0.1.0",
    )
    assert resolved is plugin_two


def test_register_materialiser_plugin_supports_extension_registration():
    plugin = _ExtensionPlugin()

    plugin_api.register_materialiser_plugin(plugin=plugin)

    resolved = plugin_api.get_materialiser_plugin(
        source_type=SourceType.PURR,
        source_version="9.9.9",
    )
    assert resolved is plugin


def test_register_materialiser_plugin_is_idempotent_for_same_identity():
    plugin_one = _ExtensionPlugin()
    plugin_two = _ExtensionPlugin()

    plugin_api.register_materialiser_plugin(plugin=plugin_one)
    plugin_api.register_materialiser_plugin(plugin=plugin_two)

    resolved = plugin_api.get_materialiser_plugin(
        source_type=SourceType.PURR,
        source_version="9.9.9",
    )
    assert resolved is plugin_one


def test_get_materialiser_plugin_returns_none_for_invalid_source_type_string():
    assert (
        plugin_api.get_materialiser_plugin(
            source_type="not-a-source",
            source_version="0.1.0",
        )
        is None
    )


def test_get_registered_source_versions_returns_sorted_versions():
    plugin = _ValidPlugin()
    plugin.source_version = "9.9.9"
    plugin_api.register_materialiser_plugin(plugin=plugin)

    plugin_second = _ValidPlugin()
    plugin_second.source_version = "0.0.1"
    plugin_api.register_materialiser_plugin(plugin=plugin_second)

    versions = plugin_api.get_registered_source_versions(SourceType.PURR)
    assert versions == ("0.0.1", "9.9.9")
