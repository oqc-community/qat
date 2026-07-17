# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import base64
import struct
import zlib

import pytest

from qat.experimental.system_data.materialisers.errors import SourceValidationError
from qat.experimental.system_data.materialisers.purr.decoder import (
    decode_jsonpickle_payload,
)


def test_decode_jsonpickle_payload_handles_supported_markers_and_references():
    payload = {
        "py/state": {
            "obj": {"py/object": "numpy.float64", "value": 1.25},
            "complex_value": {
                "py/object": "builtins.complex",
                "py/newargs": {"py/tuple": [1.0, 2.0]},
            },
            "enum_value": {
                "py/reduce": [
                    {"py/type": "qat.ir.instruction_basetypes.AcquireMode"},
                    {"py/tuple": ["integrator"]},
                ]
            },
            "container": {
                "py/tuple": [
                    {"py/obj_ref_id": 7, "name": "shared"},
                    {"py/id": 7},
                ]
            },
        }
    }

    decoded = decode_jsonpickle_payload(payload)

    assert decoded["obj"] == pytest.approx(1.25)
    assert decoded["complex_value"] == complex(1.0, 2.0)
    assert decoded["enum_value"] == "integrator"
    assert decoded["container"][0]["name"] == "shared"
    assert decoded["container"][1]["name"] == "shared"


@pytest.mark.parametrize(
    "payload, msg",
    [
        ({"py/id": "bad"}, "Malformed py/id reference"),
        ({"py/object": "numpy.float64"}, "Unsupported jsonpickle marker object"),
        (
            {"py/object": "builtins.complex", "py/newargs": {"py/tuple": [1.0]}},
            "Malformed builtins.complex py/newargs payload",
        ),
        (
            {
                "py/reduce": [
                    {"py/type": "unsupported.Target"},
                    {"py/tuple": ["x"]},
                ]
            },
            "Unsupported py/reduce target",
        ),
        ({"py/reduce": "bad"}, "Malformed py/reduce payload"),
        ({"py/unknown": 1}, "Unsupported jsonpickle marker object"),
        ({"key": object()}, "Unsupported node type in payload"),
    ],
)
def test_decode_jsonpickle_payload_raises_for_invalid_marker_shapes(payload, msg):
    with pytest.raises(SourceValidationError, match=msg):
        decode_jsonpickle_payload(payload)


def test_decode_jsonpickle_payload_raises_for_unresolved_reference():
    with pytest.raises(SourceValidationError, match="Unresolved py/id reference"):
        decode_jsonpickle_payload({"value": {"py/id": 999}})


def test_decode_jsonpickle_payload_rejects_non_mapping_root_after_decode():
    with pytest.raises(
        SourceValidationError, match="Decoded payload root must be a mapping"
    ):
        decode_jsonpickle_payload({"py/tuple": [1, 2]})


def test_decode_jsonpickle_payload_normalises_custom_pulse_and_numpy_array_samples():
    payload = {
        "py/state": {
            "weights": {
                "py/object": "qat.purr.compiler.instructions.CustomPulse",
                "samples": {
                    "py/object": "numpy.ndarray",
                    "value": [
                        {"py/object": "numpy.float64", "value": 1.25},
                        {
                            "py/object": "builtins.complex",
                            "py/newargs": {"py/tuple": [0.5, -0.25]},
                        },
                    ],
                },
                "ignore_channel_scale": False,
            }
        }
    }

    decoded = decode_jsonpickle_payload(payload)

    assert decoded["weights"]["object_type"] == "qat.purr.compiler.instructions.CustomPulse"
    assert decoded["weights"]["ignore_channel_scale"] is False
    assert decoded["weights"]["samples"] == [1.25, complex(0.5, -0.25)]


def test_decode_jsonpickle_payload_handles_top_level_custom_pulse_fields():
    payload = {
        "py/state": {
            "weights": {
                "py/object": "qat.purr.compiler.instructions.CustomPulse",
                "quantum_targets": [{"py/obj_ref_id": 1, "name": "target"}],
                "samples": {
                    "py/object": "numpy.ndarray",
                    "value": [
                        {"py/object": "numpy.float64", "value": 1.0},
                        {"py/object": "numpy.float64", "value": 2.0},
                    ],
                },
                "ignore_channel_scale": False,
            }
        }
    }

    decoded = decode_jsonpickle_payload(payload)

    assert decoded["weights"]["object_type"] == "qat.purr.compiler.instructions.CustomPulse"
    assert decoded["weights"]["quantum_targets"][0]["name"] == "target"
    assert decoded["weights"]["samples"] == [1.0, 2.0]
    assert decoded["weights"]["ignore_channel_scale"] is False


def test_decode_jsonpickle_payload_decodes_numpy_array_values_blob_for_samples():
    values = [complex(1.0, 2.0), complex(3.5, -4.25)]
    packed = struct.pack(
        "<4d", values[0].real, values[0].imag, values[1].real, values[1].imag
    )
    encoded = base64.b64encode(zlib.compress(packed)).decode("ascii")

    payload = {
        "py/state": {
            "weights": {
                "py/object": "qat.purr.compiler.instructions.CustomPulse",
                "samples": {
                    "py/object": "numpy.ndarray",
                    "values": encoded,
                    "shape": [2],
                    "dtype": "complex128",
                    "byteorder": "<",
                },
            }
        }
    }

    decoded = decode_jsonpickle_payload(payload)

    assert decoded["weights"]["samples"] == values
