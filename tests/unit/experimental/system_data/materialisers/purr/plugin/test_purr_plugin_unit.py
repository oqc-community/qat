# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from qat.experimental.system_data.materialisers.purr.plugin import PurrV010Plugin
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
