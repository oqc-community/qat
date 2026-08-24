# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from qat.experimental.system_data.materialisers import descriptor_resolution as dr
from qat.experimental.system_data.materialisers.errors import (
    SourceValidationError,
    UnsupportedSourceError,
    UnsupportedSourceVersionError,
)
from qat.experimental.system_data.materialisers.types import SourceType


class _FakePlugin:
    def __init__(self, *, source_type, source_version, detected):
        self.source_type = source_type
        self.source_version = source_version
        self._detected = detected

    def resolve_type_and_version(self, source_payload):
        _ = source_payload
        return self._detected


def _plugins_getter(plugins):
    def _get_registered_materialiser_plugins(source_type=None):
        if source_type is None:
            return tuple(plugins)
        return tuple(plugin for plugin in plugins if plugin.source_type == source_type)

    return _get_registered_materialiser_plugins


def test_resolve_descriptor_warns_and_falls_back_when_metadata_is_not_mapping(
    monkeypatch, caplog
):
    plugin = _FakePlugin(
        source_type=SourceType.PURR,
        source_version="0.1.0",
        detected=(SourceType.PURR, "0.1.0"),
    )
    monkeypatch.setattr(
        dr,
        "get_registered_materialiser_plugins",
        _plugins_getter([plugin]),
    )
    monkeypatch.setattr(
        dr, "get_registered_source_versions", lambda _source_type: ("0.1.0",)
    )

    resolved = dr.resolve_source_descriptor({"metadata": "bad"})

    assert resolved == (SourceType.PURR, "0.1.0")
    assert any(
        "Ignoring non-mapping payload metadata" in rec.message for rec in caplog.records
    )


def test_resolve_descriptor_raises_on_detector_descriptor_mismatch(monkeypatch):
    plugin = _FakePlugin(
        source_type=SourceType.PURR,
        source_version="0.1.0",
        detected=(SourceType.PURR, "9.9.9"),
    )
    monkeypatch.setattr(
        dr,
        "get_registered_materialiser_plugins",
        _plugins_getter([plugin]),
    )

    with pytest.raises(SourceValidationError, match="descriptor mismatch"):
        dr.resolve_source_descriptor({})


def test_resolve_descriptor_raises_on_ambiguous_plugin_detection(monkeypatch):
    plugins = [
        _FakePlugin(
            source_type=SourceType.PURR,
            source_version="0.1.0",
            detected=(SourceType.PURR, "0.1.0"),
        ),
        _FakePlugin(
            source_type=SourceType.PURR,
            source_version="0.2.0",
            detected=(SourceType.PURR, "0.2.0"),
        ),
    ]
    monkeypatch.setattr(
        dr,
        "get_registered_materialiser_plugins",
        _plugins_getter(plugins),
    )

    with pytest.raises(SourceValidationError, match="Ambiguous source descriptor"):
        dr.resolve_source_descriptor({})


def test_resolve_descriptor_invalid_source_type_hint_raises_unsupported(monkeypatch):
    with pytest.raises(UnsupportedSourceError, match="Unsupported source type"):
        dr.resolve_source_descriptor(
            {
                "metadata": {
                    "source_type": "not_a_source",
                    "source_version": "0.1.0",
                }
            }
        )


def test_resolve_descriptor_hint_without_version_uses_candidate_type_detector(monkeypatch):
    requested_candidate_types = []

    def _get_registered_materialiser_plugins(source_type=None):
        requested_candidate_types.append(source_type)
        return (purr_plugin,)

    purr_plugin = _FakePlugin(
        source_type=SourceType.PURR,
        source_version="0.2.0",
        detected=(SourceType.PURR, "0.2.0"),
    )
    monkeypatch.setattr(
        dr,
        "get_registered_materialiser_plugins",
        _get_registered_materialiser_plugins,
    )
    monkeypatch.setattr(
        dr,
        "get_registered_source_versions",
        lambda source_type: ("0.1.0", "0.2.0") if source_type == SourceType.PURR else (),
    )

    resolved = dr.resolve_source_descriptor({"metadata": {"source_type": "purr"}})

    assert resolved == (SourceType.PURR, "0.2.0")
    assert requested_candidate_types == [SourceType.PURR]


def test_resolve_descriptor_hint_without_version_raises_when_not_inferable(monkeypatch):
    monkeypatch.setattr(
        dr,
        "get_registered_materialiser_plugins",
        _plugins_getter([]),
    )
    monkeypatch.setattr(
        dr,
        "get_registered_source_versions",
        lambda source_type: ("0.1.0", "0.2.0") if source_type == SourceType.PURR else (),
    )

    with pytest.raises(SourceValidationError, match="Could not infer source_version"):
        dr.resolve_source_descriptor({"metadata": {"source_type": "purr"}})


def test_resolve_source_version_hint_unsupported_version_raises(monkeypatch):
    monkeypatch.setattr(
        dr,
        "get_registered_source_versions",
        lambda source_type: ("0.1.0",) if source_type == SourceType.PURR else (),
    )

    with pytest.raises(UnsupportedSourceVersionError, match="Unsupported source version"):
        dr.resolve_source_descriptor(
            {
                "metadata": {
                    "source_type": "purr",
                    "source_version": "9.9.9",
                }
            }
        )


def test_resolve_source_version_no_registered_versions_raises_unsupported(monkeypatch):
    monkeypatch.setattr(dr, "get_registered_source_versions", lambda _source_type: ())

    with pytest.raises(UnsupportedSourceError, match="Unsupported source type"):
        dr.resolve_source_descriptor(
            {
                "metadata": {
                    "source_type": "purr",
                    "source_version": "0.1.0",
                }
            }
        )


def test_resolve_descriptor_ignores_none_detector_matches(monkeypatch):
    plugins = [
        _FakePlugin(
            source_type=SourceType.PURR,
            source_version="0.1.0",
            detected=None,
        )
    ]
    monkeypatch.setattr(
        dr,
        "get_registered_materialiser_plugins",
        _plugins_getter(plugins),
    )

    with pytest.raises(UnsupportedSourceError, match="Unsupported source type"):
        dr.resolve_source_descriptor({})


def test_resolve_descriptor_hint_without_version_falls_back_to_single_registered_version(
    monkeypatch,
):
    monkeypatch.setattr(
        dr,
        "get_registered_materialiser_plugins",
        _plugins_getter([]),
    )
    monkeypatch.setattr(
        dr,
        "get_registered_source_versions",
        lambda source_type: ("0.1.0",) if source_type == SourceType.PURR else (),
    )

    resolved = dr.resolve_source_descriptor({"metadata": {"source_type": "purr"}})

    assert resolved == (SourceType.PURR, "0.1.0")


# ── Supersedes tests ──────────────────────────────────────────────────────────


class _FakePluginWithSupersedes(_FakePlugin):
    def __init__(self, *, source_type, source_version, detected, supersedes=()):
        super().__init__(
            source_type=source_type, source_version=source_version, detected=detected
        )
        self.supersedes = supersedes


def test_supersession_resolves_ambiguous_matches_to_single_winner(monkeypatch):
    base_plugin = _FakePlugin(
        source_type=SourceType.PURR,
        source_version="0.1.0",
        detected=(SourceType.PURR, "0.1.0"),
    )
    ddq_plugin = _FakePluginWithSupersedes(
        source_type=SourceType.PURR,
        source_version="0.2.0",
        detected=(SourceType.PURR, "0.2.0"),
        supersedes=((SourceType.PURR, "0.1.0"),),
    )
    monkeypatch.setattr(
        dr,
        "get_registered_materialiser_plugins",
        _plugins_getter([base_plugin, ddq_plugin]),
    )

    resolved = dr.resolve_source_descriptor({})

    assert resolved == (SourceType.PURR, "0.2.0")


def test_circular_supersession_raises_with_message(monkeypatch):
    plugin_a = _FakePluginWithSupersedes(
        source_type=SourceType.PURR,
        source_version="0.1.0",
        detected=(SourceType.PURR, "0.1.0"),
        supersedes=((SourceType.PURR, "0.2.0"),),
    )
    plugin_b = _FakePluginWithSupersedes(
        source_type=SourceType.PURR,
        source_version="0.2.0",
        detected=(SourceType.PURR, "0.2.0"),
        supersedes=((SourceType.PURR, "0.1.0"),),
    )
    monkeypatch.setattr(
        dr,
        "get_registered_materialiser_plugins",
        _plugins_getter([plugin_a, plugin_b]),
    )

    with pytest.raises(SourceValidationError, match="Circular supersession"):
        dr.resolve_source_descriptor({})


def test_partial_supersession_still_raises_ambiguity(monkeypatch):
    """Plugin A supersedes B but C also matches and no one supersedes C."""
    plugin_a = _FakePluginWithSupersedes(
        source_type=SourceType.PURR,
        source_version="0.1.0",
        detected=(SourceType.PURR, "0.1.0"),
        supersedes=((SourceType.PURR, "0.2.0"),),
    )
    plugin_b = _FakePlugin(
        source_type=SourceType.PURR,
        source_version="0.2.0",
        detected=(SourceType.PURR, "0.2.0"),
    )
    plugin_c = _FakePlugin(
        source_type=SourceType.PURR,
        source_version="0.3.0",
        detected=(SourceType.PURR, "0.3.0"),
    )
    monkeypatch.setattr(
        dr,
        "get_registered_materialiser_plugins",
        _plugins_getter([plugin_a, plugin_b, plugin_c]),
    )

    with pytest.raises(SourceValidationError, match="Ambiguous source descriptor"):
        dr.resolve_source_descriptor({})


def test_supersession_ignores_unmatched_superseded_descriptor(monkeypatch):
    plugin_a = _FakePluginWithSupersedes(
        source_type=SourceType.PURR,
        source_version="0.1.0",
        detected=(SourceType.PURR, "0.1.0"),
        supersedes=((SourceType.PURR, "9.9.9"),),
    )
    plugin_b = _FakePlugin(
        source_type=SourceType.PURR,
        source_version="0.2.0",
        detected=(SourceType.PURR, "0.2.0"),
    )
    monkeypatch.setattr(
        dr,
        "get_registered_materialiser_plugins",
        _plugins_getter([plugin_a, plugin_b]),
    )

    with pytest.raises(SourceValidationError, match="Ambiguous source descriptor"):
        dr.resolve_source_descriptor({})


# ── String source-type tests ──────────────────────────────────────────────────


def test_string_source_type_detected_via_plugin_detection(monkeypatch):
    ext_plugin = _FakePlugin(
        source_type="ext-source",
        source_version="1.0.0",
        detected=("ext-source", "1.0.0"),
    )
    monkeypatch.setattr(
        dr,
        "get_registered_materialiser_plugins",
        _plugins_getter([ext_plugin]),
    )

    resolved = dr.resolve_source_descriptor({})

    assert resolved == ("ext-source", "1.0.0")


def test_string_source_type_hint_in_metadata_resolves_correctly(monkeypatch):
    monkeypatch.setattr(
        dr,
        "get_registered_source_versions",
        lambda source_type: ("1.0.0",) if source_type == "ext-source" else (),
    )
    monkeypatch.setattr(
        dr,
        "get_registered_materialiser_plugins",
        _plugins_getter([]),
    )

    def _fake_get_registered_plugins_all(source_type=None):
        if source_type is None:
            return (
                _FakePlugin(
                    source_type="ext-source", source_version="1.0.0", detected=None
                ),
            )
        return ()

    # Also patch the source-type hint resolver to accept "ext-source" as known
    monkeypatch.setattr(
        dr,
        "get_registered_materialiser_plugins",
        _fake_get_registered_plugins_all,
    )

    resolved = dr.resolve_source_descriptor(
        {"metadata": {"source_type": "ext-source", "source_version": "1.0.0"}}
    )

    assert resolved == ("ext-source", "1.0.0")


def test_unknown_string_source_type_hint_raises_unsupported(monkeypatch):
    monkeypatch.setattr(
        dr,
        "get_registered_materialiser_plugins",
        _plugins_getter([]),
    )

    with pytest.raises(UnsupportedSourceError, match="Unsupported source type"):
        dr.resolve_source_descriptor(
            {
                "metadata": {
                    "source_type": "totally-unknown-source",
                    "source_version": "1.0.0",
                }
            }
        )


def test_supported_source_values_deduplicate_external_source_versions(monkeypatch):
    plugins = [
        _FakePlugin(source_type=SourceType.PURR, source_version="0.1.0", detected=None),
        _FakePlugin(source_type="ext-source", source_version="1.0.0", detected=None),
        _FakePlugin(source_type="ext-source", source_version="2.0.0", detected=None),
    ]
    monkeypatch.setattr(
        dr,
        "get_registered_materialiser_plugins",
        _plugins_getter(plugins),
    )

    assert dr._supported_source_values() == ("model", "purr", "ext-source")
