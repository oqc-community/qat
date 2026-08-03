# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Integration tests for the model materialiser boundary flow."""

from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.system_data.materialisers.boundary import materialise
from qat.experimental.system_data.materialisers.builder import CanonicalSystemDataBuilder
from qat.experimental.system_data.materialisers.model.materialise import materialise_model


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


def test_model_materialiser_returns_canonical_system_data_instance():
    """The model materialiser returns a CanonicalSystemData for a valid payload."""
    payload = (
        CanonicalSystemDataBuilder()
        .with_calibration_id("echo-boundary-test")
        .with_oscillator("osc0", 5_000_000_000)
        .with_port("p0", 1000)
        .with_channel("ch0", "p0", 5_000_000_000)
        .with_qubit("q0", 0)
        .build_payload()
    )
    result = materialise_model(source_payload=payload)
    assert isinstance(result, CanonicalSystemData)


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
