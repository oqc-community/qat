# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from qat.experimental.system_data.materialisers.errors import SourceValidationError
from qat.experimental.system_data.materialisers.purr.adapter import (
    _normalise_default_acquire_mode,
    _normalise_legacy_crc_ownership,
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


def test_normalise_default_acquire_mode_rejects_non_list_non_str():
    """Non-list, non-str, non-None value raises SourceValidationError."""
    payload = {"default_acquire_mode": 42}
    with pytest.raises(SourceValidationError, match="Unsupported default_acquire_mode"):
        _normalise_default_acquire_mode(payload)


def test_normalise_legacy_crc_ownership_returns_early_for_non_dict_quantum_devices():
    """Early return when quantum_devices is absent or not a dict."""
    _normalise_legacy_crc_ownership({})
    _normalise_legacy_crc_ownership({"quantum_devices": "bad"})


def test_normalise_legacy_crc_ownership_skips_non_dict_device_and_pulse_channel():
    """Non-dict device entries and non-dict pulse_channel entries are skipped."""
    payload = {
        "quantum_devices": {
            "bad_device": "not_a_dict",
            "good_device": {
                "id": "good_device",
                "pulse_channels": {
                    42: {"pulse_channel": {}},
                    "Q1.cross_resonance_cancellation": "not_a_dict",
                },
            },
        }
    }
    _normalise_legacy_crc_ownership(payload)


def test_normalise_legacy_crc_ownership_creates_missing_target_device():
    """When the target device doesn't exist, it is created."""
    payload = {
        "quantum_devices": {
            "Q0": {
                "id": "Q0",
                "pulse_channels": {
                    "Q1.cross_resonance_cancellation": {
                        "pulse_channel": {
                            "id": "Q0.Q1.cross_resonance_cancellation",
                        }
                    }
                },
            }
        }
    }
    _normalise_legacy_crc_ownership(payload)
    assert "Q1" in payload["quantum_devices"]


def test_normalise_legacy_crc_ownership_creates_pulse_channels_when_target_has_non_dict():
    """When target device exists but has non-dict pulse_channels, a new dict is created."""
    payload = {
        "quantum_devices": {
            "Q0": {
                "id": "Q0",
                "pulse_channels": {
                    "Q1.cross_resonance_cancellation": {
                        "pulse_channel": {
                            "id": "Q0.Q1.cross_resonance_cancellation",
                        }
                    }
                },
            },
            "Q1": {"id": "Q1", "pulse_channels": "bad"},
        }
    }
    _normalise_legacy_crc_ownership(payload)
    assert isinstance(payload["quantum_devices"]["Q1"]["pulse_channels"], dict)


def test_normalise_legacy_crc_ownership_skips_malformed_crc_entries():
    """CRC entries matching neither canonical nor legacy form are removed (removals
    path)."""
    payload = {
        "quantum_devices": {
            "Q0": {
                "id": "Q0",
                "pulse_channels": {
                    "Q1.cross_resonance_cancellation": {
                        "pulse_channel": {
                            "id": "Q2.Q3.cross_resonance_cancellation",
                        }
                    },
                    # CRC with non-dict pulse_channel → skipped
                    "Q4.cross_resonance_cancellation": {"pulse_channel": "bad"},
                    # CRC with non-str id → skipped
                    "Q5.cross_resonance_cancellation": {"pulse_channel": {"id": 999}},
                    # CRC where id doesn't have 3 parts → skipped
                    "Q6.cross_resonance_cancellation": {
                        "pulse_channel": {"id": "Q0.cross_resonance_cancellation"}
                    },
                    # CRC where suffix != cross_resonance_cancellation → skipped
                    "Q7.cross_resonance_cancellation": {
                        "pulse_channel": {"id": "Q0.Q7.cross_resonance"}
                    },
                },
            }
        }
    }
    _normalise_legacy_crc_ownership(payload)


def test_normalise_legacy_crc_ownership_cr_pairs_with_guard_failures():
    """CR pairs loop guards: missing source device, bad pulse_channels, bad cr_view."""
    payload = {
        "quantum_devices": {
            "Q0": {
                "id": "Q0",
                "pulse_channels": {
                    "Q1.cross_resonance": {
                        "pulse_channel": {"id": "Q0.Q1.cross_resonance"}
                    },
                    # CR where key_peer == source_id → not added to cr_pairs
                    "Q0.cross_resonance": {
                        "pulse_channel": {"id": "Q0.Q0.cross_resonance"}
                    },
                },
            },
        }
    }
    _normalise_legacy_crc_ownership(payload)


def test_normalise_legacy_crc_ownership_cr_pairs_bad_cr_view():
    """CR pairs loop skips when cr_view or cr_channel is not a dict."""
    # Test: cr_view is not a dict
    payload = {
        "quantum_devices": {
            "Q0": {
                "id": "Q0",
                "pulse_channels": {
                    "Q1.cross_resonance": "not_a_dict_view",
                    "Q2.cross_resonance": {"pulse_channel": "not_a_dict_channel"},
                },
            },
            "Q1": {"id": "Q1", "index": 1, "pulse_channels": {}},
            "Q2": {"id": "Q2", "index": 2, "pulse_channels": {}},
        }
    }
    _normalise_legacy_crc_ownership(payload)


def test_detect_supported_reset_methods_uses_structural_qubit_classification():
    """Reset detection should use qubit structure, not ID naming conventions."""

    payload = {
        "quantum_devices": {
            # Resonator-like record with reset key should be ignored.
            "resonator_like": {
                "id": "R1",
                "pulse_channels": {"reset": {}},
            },
            # Qubit by index; should contribute ddrop.
            "qubit_by_index": {
                "id": "device_a",
                "index": 3,
                "pulse_channels": {
                    "reset": {},
                },
            },
            # Qubit by measure_device fallback; no built-in reset method.
            "qubit_by_measure_device": {
                "id": "device_b",
                "measure_device": {"id": "R2"},
            },
        }
    }

    supported = _detect_supported_reset_methods(payload)

    assert supported == ["ddrop"]
