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
