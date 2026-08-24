# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from qat.experimental.system_data.materialisers.purr.plugin import (
    PurrV010AdditionalData,
    PurrV010Plugin,
)
from qat.experimental.system_data.materialisers.types import SourceType


def test_purr_plugin_detection_handles_py_state_and_missing_keys():
    plugin = PurrV010Plugin()

    wrapped_payload = {
        "py/state": {
            "quantum_devices": {},
            "pulse_channels": {},
            "physical_channels": {},
            "basebands": {},
        }
    }
    assert plugin.resolve_type_and_version(wrapped_payload) == (
        SourceType.PURR,
        "0.1.0",
    )

    missing_keys_payload = {
        "quantum_devices": {},
        "pulse_channels": {},
    }
    assert plugin.resolve_type_and_version(missing_keys_payload) is None


def test_purr_plugin_verify_integrity_placeholder_is_noop():
    plugin = PurrV010Plugin()

    plugin.verify_integrity({"any": "payload"})


def test_purr_plugin_materialise_forwards_decoder_options_and_adapted_payload(monkeypatch):
    import qat.experimental.system_data.materialisers.purr.adapter as adapter
    import qat.experimental.system_data.materialisers.purr.materialise as materialise

    calls = {}
    adapted_payload = {"adapted": True}

    def _adapt(payload, **kwargs):
        calls["adapter"] = (payload, kwargs)
        return adapted_payload

    class _FakeMaterialiser:
        def __init__(self, **kwargs):
            calls["constructor"] = kwargs

        def materialise(self, **kwargs):
            calls["materialise"] = kwargs
            return "canonical"

    monkeypatch.setattr(adapter, "adapt_purr_payload", _adapt)
    monkeypatch.setattr(materialise, "PurrMaterialiserV010", _FakeMaterialiser)

    result = PurrV010Plugin().materialise(
        source_payload={"raw": True},
        source_version="0.1.0",
        additional_data=PurrV010AdditionalData(
            decoder_extra_reduce_target_types=["pkg.Type"],
            decoder_extra_reduce_target_suffixes=["Reference"],
        ),
    )

    assert result == "canonical"
    assert calls["adapter"] == (
        {"raw": True},
        {
            "extra_reduce_target_types": {"pkg.Type"},
            "extra_reduce_target_suffixes": {"Reference"},
        },
    )
    assert calls["constructor"] == {
        "target_data": None,
        "supported_acquire_modes": None,
        "native_waveform_shapes": None,
    }
    assert calls["materialise"] == {
        "adapted_payload": adapted_payload,
        "source_version": "0.1.0",
    }
