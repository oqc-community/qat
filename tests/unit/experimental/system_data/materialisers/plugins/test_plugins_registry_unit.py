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


class _StringSourcePlugin:
    source_type = "ext-source"
    source_version = "1.0.0"
    additional_data_model = _ValidAdditionalData

    def verify_integrity(self, source_payload):
        _ = source_payload

    def materialise(self, *, source_payload, source_version, additional_data):
        return "ok-ext"


@pytest.fixture(autouse=True)
def _isolated_plugin_registry(monkeypatch):
    monkeypatch.setattr(plugin_api, "_PLUGIN_REGISTRY", {})


def test_register_materialiser_plugin_rejects_non_string_non_enum_source_type():
    plugin = _ValidPlugin()
    plugin.source_type = 42  # type: ignore[assignment]

    with pytest.raises(ValueError, match="source_type"):
        plugin_api.register_materialiser_plugin(plugin=plugin)


def test_register_materialiser_plugin_rejects_empty_string_source_type():
    plugin = _ValidPlugin()
    plugin.source_type = ""  # type: ignore[assignment]

    with pytest.raises(ValueError, match="source_type"):
        plugin_api.register_materialiser_plugin(plugin=plugin)


def test_register_materialiser_plugin_rejects_enum_reserved_string_source_type():
    plugin = _StringSourcePlugin()
    plugin.source_type = "purr"

    with pytest.raises(ValueError, match="conflicts"):
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


def test_get_registered_source_versions_returns_empty_for_unknown_source_type():
    plugin_api.register_materialiser_plugin(plugin=_ValidPlugin())

    assert plugin_api.get_registered_source_versions("unknown-source") == ()


def test_get_registered_source_versions_returns_sorted_versions():
    plugin = _ValidPlugin()
    plugin.source_version = "9.9.9"
    plugin_api.register_materialiser_plugin(plugin=plugin)

    plugin_second = _ValidPlugin()
    plugin_second.source_version = "0.0.1"
    plugin_api.register_materialiser_plugin(plugin=plugin_second)

    versions = plugin_api.get_registered_source_versions(SourceType.PURR)
    assert versions == ("0.0.1", "9.9.9")


# ── String source-type tests ──────────────────────────────────────────────────


def test_register_materialiser_plugin_accepts_string_source_type():
    plugin = _StringSourcePlugin()
    plugin_api.register_materialiser_plugin(plugin=plugin)

    resolved = plugin_api.get_materialiser_plugin(
        source_type="ext-source",
        source_version="1.0.0",
    )
    assert resolved is plugin


def test_get_materialiser_plugin_returns_none_for_unsupported_string_source_version():
    plugin_api.register_materialiser_plugin(plugin=_StringSourcePlugin())

    assert (
        plugin_api.get_materialiser_plugin(
            source_type="ext-source",
            source_version="9.9.9",
        )
        is None
    )


def test_get_materialiser_plugin_coerces_string_to_enum_for_enum_keyed_plugins():
    plugin_api.register_materialiser_plugin(plugin=_ValidPlugin())

    resolved = plugin_api.get_materialiser_plugin(
        source_type="purr",
        source_version="0.1.0",
    )
    assert resolved is not None


def test_get_materialiser_plugin_returns_exact_registry_match():
    plugin = _ValidPlugin()
    plugin_api.register_materialiser_plugin(plugin=plugin)

    assert (
        plugin_api.get_materialiser_plugin(
            source_type=SourceType.PURR,
            source_version="0.1.0",
        )
        is plugin
    )


def test_get_registered_source_versions_works_for_string_source_type():
    plugin_api.register_materialiser_plugin(plugin=_StringSourcePlugin())

    versions = plugin_api.get_registered_source_versions("ext-source")
    assert versions == ("1.0.0",)


def test_get_registered_source_versions_coerces_string_to_enum():
    plugin_api.register_materialiser_plugin(plugin=_ValidPlugin())

    versions = plugin_api.get_registered_source_versions("purr")
    assert versions == ("0.1.0",)


def test_get_registered_materialiser_plugins_includes_string_typed_plugins():
    plugin_api.register_materialiser_plugin(plugin=_StringSourcePlugin())

    all_plugins = plugin_api.get_registered_materialiser_plugins()
    assert any(isinstance(p, _StringSourcePlugin) for p in all_plugins)


def test_get_registered_materialiser_plugins_filters_by_string_source_type():
    plugin_api.register_materialiser_plugin(plugin=_ValidPlugin())
    plugin_api.register_materialiser_plugin(plugin=_StringSourcePlugin())

    ext_plugins = plugin_api.get_registered_materialiser_plugins("ext-source")
    assert len(ext_plugins) == 1
    assert isinstance(ext_plugins[0], _StringSourcePlugin)


def test_get_registered_materialiser_plugins_filters_by_enum_excludes_string_plugins():
    plugin_api.register_materialiser_plugin(plugin=_ValidPlugin())
    plugin_api.register_materialiser_plugin(plugin=_StringSourcePlugin())

    purr_plugins = plugin_api.get_registered_materialiser_plugins(SourceType.PURR)
    assert len(purr_plugins) == 1
    assert isinstance(purr_plugins[0], _ValidPlugin)


def test_get_registered_materialiser_plugins_returns_empty_for_unknown_source_type():
    plugin_api.register_materialiser_plugin(plugin=_ValidPlugin())

    assert plugin_api.get_registered_materialiser_plugins("unknown-source") == ()


# ── Supersedes tests ──────────────────────────────────────────────────────────


class _SupersedesPlugin:
    source_type = SourceType.PURR
    source_version = "0.2.0"
    additional_data_model = _ValidAdditionalData
    supersedes = ((SourceType.PURR, "0.1.0"),)

    def verify_integrity(self, source_payload):
        _ = source_payload

    def materialise(self, *, source_payload, source_version, additional_data):
        return "supersedes-ok"


class _BadSupersedesPlugin:
    source_type = SourceType.PURR
    source_version = "0.2.0"
    additional_data_model = _ValidAdditionalData
    supersedes = ("not-a-tuple",)  # invalid format

    def verify_integrity(self, source_payload):
        _ = source_payload

    def materialise(self, *, source_payload, source_version, additional_data):
        return "ok"


def test_register_materialiser_plugin_accepts_valid_supersedes():
    plugin = _SupersedesPlugin()
    plugin_api.register_materialiser_plugin(plugin=plugin)

    resolved = plugin_api.get_materialiser_plugin(
        source_type=SourceType.PURR,
        source_version="0.2.0",
    )
    assert resolved is plugin


def test_register_materialiser_plugin_rejects_invalid_supersedes_format():
    with pytest.raises(ValueError, match="supersedes"):
        plugin_api.register_materialiser_plugin(plugin=_BadSupersedesPlugin())


def test_register_materialiser_plugin_rejects_supersedes_entry_with_wrong_tuple_length():
    plugin = _ValidPlugin()
    plugin.supersedes = ((SourceType.PURR,),)  # 1-tuple, not 2-tuple

    with pytest.raises(ValueError, match="supersedes"):
        plugin_api.register_materialiser_plugin(plugin=plugin)


def test_register_materialiser_plugin_rejects_supersedes_entry_with_invalid_source_type():
    plugin = _ValidPlugin()
    plugin.supersedes = ((42, "0.1.0"),)  # invalid source type

    with pytest.raises(ValueError, match="supersedes"):
        plugin_api.register_materialiser_plugin(plugin=plugin)


def test_register_materialiser_plugin_rejects_supersedes_entry_with_empty_source_type():
    plugin = _ValidPlugin()
    plugin.supersedes = (("", "0.1.0"),)  # empty source type

    with pytest.raises(ValueError, match="supersedes"):
        plugin_api.register_materialiser_plugin(plugin=plugin)


@pytest.mark.parametrize("source_version", [None, ""])
def test_register_materialiser_plugin_rejects_invalid_supersedes_version(source_version):
    plugin = _ValidPlugin()
    plugin.supersedes = ((SourceType.PURR, source_version),)  # invalid version

    with pytest.raises(ValueError, match="supersedes"):
        plugin_api.register_materialiser_plugin(plugin=plugin)


@pytest.mark.parametrize("supersedes", [None, 42, "not-a-tuple"])
def test_register_materialiser_plugin_rejects_invalid_supersedes_container(supersedes):
    plugin = _ValidPlugin()
    plugin.supersedes = supersedes

    with pytest.raises(ValueError, match="supersedes"):
        plugin_api.register_materialiser_plugin(plugin=plugin)
