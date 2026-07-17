# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from qat.experimental.system_data.materialisers.errors import SourceValidationError
from qat.experimental.system_data.materialisers.purr.adapter import (
    _normalise_default_acquire_mode,
)
from qat.experimental.system_data.materialisers.purr.materialise import (
    _detect_supported_reset_methods,
)


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"default_acquire_mode": []}, None),
        ({"default_acquire_mode": ["integrator"]}, "integrator"),
        ({"default_acquire_mode": "raw"}, "raw"),
    ],
)
def test_normalise_default_acquire_mode_coerces_legacy_shapes(payload, expected):
    """Legacy default-acquire-mode forms should normalize to string-or-none."""

    _normalise_default_acquire_mode(payload)
    assert payload["default_acquire_mode"] == expected


def test_normalise_default_acquire_mode_rejects_invalid_list_shape():
    """Non-string list entries should fail fast at adapter normalisation stage."""

    payload = {"default_acquire_mode": [1]}

    with pytest.raises(SourceValidationError, match="Unsupported default_acquire_mode"):
        _normalise_default_acquire_mode(payload)


def test_detect_supported_reset_methods_uses_structural_qubit_classification():
    """Reset detection should use qubit structure, not ID naming conventions."""

    payload = {
        "quantum_devices": {
            # Resonator-like record with reset key should be ignored.
            "resonator_like": {
                "id": "R1",
                "pulse_channels": {"reset": {}},
            },
            # Qubit by index; should contribute active + ddrop.
            "qubit_by_index": {
                "id": "device_a",
                "index": 3,
                "pulse_channels": {
                    "reset": {},
                    "active_reset": {},
                },
            },
            # Qubit by measure_device fallback; should contribute active.
            "qubit_by_measure_device": {
                "id": "device_b",
                "measure_device": {"id": "R2"},
                "active_reset": {},
            },
        }
    }

    supported = _detect_supported_reset_methods(payload)

    assert supported == ["active", "ddrop"]
