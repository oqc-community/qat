# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Integration tests for the model materialiser boundary flow."""

import pytest

from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.system_data.materialisers.boundary import materialise
from qat.experimental.system_data.materialisers.builder import (
    CanonicalSystemDataBuilder,
    version_structure_hash,
)
from qat.experimental.system_data.materialisers.errors import (
    SourceValidationError,
    UnsupportedSourceVersionError,
)
from qat.experimental.system_data.materialisers.model.materialise import materialise_model
from qat.experimental.system_data.materialisers.model.plugin import DefaultPlugin


def _make_simple_canonical() -> CanonicalSystemData:
    """Return a minimal canonical instance used across boundary flow tests."""
    return (
        CanonicalSystemDataBuilder()
        .with_calibration_id("echo-boundary-test")
        .with_oscillator("osc0", 5_000_000_000)
        .with_port("p0", 1000)
        .with_channel("ch0", "p0", 5_000_000_000)
        .with_qubit("q0", 0)
        .build()
    )


def test_model_materialiser_returns_equal_instance():
    """The model materialiser must reconstruct an equal CanonicalSystemData."""
    canonical = _make_simple_canonical()
    result = materialise_model(
        source_payload=CanonicalSystemDataBuilder()
        .with_calibration_id("echo-boundary-test")
        .with_oscillator("osc0", 5_000_000_000)
        .with_port("p0", 1000)
        .with_channel("ch0", "p0", 5_000_000_000)
        .with_qubit("q0", 0)
        .build_payload()
    )
    assert result == canonical


def test_boundary_materialise_model_round_trip():
    """End-to-end model boundary flow should reconstruct an equal CanonicalSystemData."""
    canonical = _make_simple_canonical()
    payload = (
        CanonicalSystemDataBuilder()
        .with_calibration_id("echo-boundary-test")
        .with_oscillator("osc0", 5_000_000_000)
        .with_port("p0", 1000)
        .with_channel("ch0", "p0", 5_000_000_000)
        .with_qubit("q0", 0)
        .build_payload()
    )
    result = materialise(source_payload=payload)
    assert result == canonical


def test_boundary_materialise_model_via_build_payload():
    """build_payload() should produce a payload the boundary accepts directly."""
    canonical = materialise(
        source_payload=(
            CanonicalSystemDataBuilder()
            .with_calibration_id("payload-method")
            .with_oscillator("osc0", 5_000_000_000)
            .with_port("p0", 1000)
            .with_channel("ch0", "p0", 5_000_000_000)
            .with_qubit("q0", 0)
            .build_payload()
        )
    )
    assert isinstance(canonical, CanonicalSystemData)
    assert canonical.calibration_id == "payload-method"


def test_boundary_materialise_version_auto_detect():
    """Model payloads with only canonical field keys are auto-detected as model source."""
    result = materialise(
        source_payload=(
            CanonicalSystemDataBuilder()
            .with_oscillator("osc0", 5_000_000_000)
            .with_port("p0", 1000)
            .with_channel("ch0", "p0", 5_000_000_000)
            .with_qubit("q0", 0)
            .build_payload()
        )
    )
    assert isinstance(result, CanonicalSystemData)


def test_materialise_model_missing_version_raises():
    """materialise_model raises UnsupportedSourceVersionError when _version is absent."""
    payload = {CanonicalSystemDataBuilder.data_field: {"calibration_id": "test"}}
    with pytest.raises(UnsupportedSourceVersionError, match="versioning hash field"):
        materialise_model(payload)


def test_materialise_model_wrong_version_raises():
    """materialise_model raises UnsupportedSourceVersionError for a stale hash."""
    payload = {
        CanonicalSystemDataBuilder.data_field: {
            CanonicalSystemDataBuilder.versioning_key: "stale-hash-value"
        }
    }
    with pytest.raises(UnsupportedSourceVersionError, match="different model version"):
        materialise_model(payload)


def test_materialise_model_bad_constructor_fields_raises():
    """materialise_model raises SourceValidationError for unknown CanonicalSystemData
    fields."""
    payload = {
        CanonicalSystemDataBuilder.data_field: {
            CanonicalSystemDataBuilder.versioning_key: version_structure_hash,
            "unknown_field_xyz": "boom",
        }
    }
    with pytest.raises(SourceValidationError, match="CanonicalSystemData instance"):
        materialise_model(payload)


def test_default_plugin_resolve_non_dict_payload_returns_none():
    """DefaultPlugin.resolve_type_and_version returns None when source_payload is not a
    dict."""
    plugin = DefaultPlugin()
    assert plugin.resolve_type_and_version("not_a_dict") is None


def test_default_plugin_resolve_non_dict_model_returns_none():
    """DefaultPlugin.resolve_type_and_version returns None when model value is not a
    dict."""
    plugin = DefaultPlugin()
    payload = {CanonicalSystemDataBuilder.data_field: ["not", "a", "dict"]}
    assert plugin.resolve_type_and_version(payload) is None


def test_default_plugin_resolve_wrong_hash_returns_none():
    """DefaultPlugin.resolve_type_and_version returns None when version hash doesn't
    match."""
    plugin = DefaultPlugin()
    payload = {
        CanonicalSystemDataBuilder.data_field: {
            CanonicalSystemDataBuilder.versioning_key: "wrong-hash"
        }
    }
    assert plugin.resolve_type_and_version(payload) is None
