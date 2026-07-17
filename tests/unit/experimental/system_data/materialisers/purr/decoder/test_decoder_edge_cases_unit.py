# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import base64
import struct
import zlib

import pytest

from qat.experimental.system_data.materialisers.errors import SourceValidationError
from qat.experimental.system_data.materialisers.purr.decoder import PurrJsonpickleDecoder


def test_decoder_coerce_numeric_sequence_and_wrapper_errors():
    decoder = PurrJsonpickleDecoder()

    assert decoder._coerce_numeric_sequence((1, 2, 3), path="$.values") == [1, 2, 3]

    with pytest.raises(
        SourceValidationError, match="Unsupported numpy array payload shape"
    ):
        decoder._coerce_numeric_sequence("bad", path="$.values")

    with pytest.raises(SourceValidationError, match="Unsupported numpy array element type"):
        decoder._coerce_numeric_sequence([1, "bad"], path="$.values")


def test_decoder_numpy_array_paths_cover_value_values_state_and_malformed_wrapper():
    decoder = PurrJsonpickleDecoder()

    assert decoder._decode_numpy_array({"value": [1, 2]}, path="$.x") == [1, 2]
    assert decoder._decode_numpy_array({"values": [1, 2]}, path="$.x") == [1, 2]
    assert decoder._decode_numpy_array({"py/state": {"data": [1, 2]}}, path="$.x") == [1, 2]

    with pytest.raises(SourceValidationError, match="Malformed numpy array wrapper"):
        decoder._decode_numpy_array({}, path="$.x")


@pytest.mark.parametrize(
    "node, msg",
    [
        (
            {"values": 1, "dtype": "float64", "shape": [1]},
            "Malformed numpy array values payload",
        ),
        (
            {"values": "abc", "dtype": 1, "shape": [1]},
            "Malformed numpy array dtype payload",
        ),
        (
            {"values": "abc", "dtype": "float64", "shape": "bad"},
            "Malformed numpy array shape payload",
        ),
        (
            {"values": "abc", "dtype": "float64", "shape": [1], "byteorder": "x"},
            "Malformed numpy array byteorder payload",
        ),
        (
            {"values": "abc", "dtype": "float64", "shape": [1]},
            "Malformed numpy array base64 payload",
        ),
        (
            {
                "values": base64.b64encode(zlib.compress(struct.pack("<d", 1.0))).decode(
                    "ascii"
                ),
                "dtype": "int32",
                "shape": [1],
            },
            "Unsupported numpy array dtype",
        ),
        (
            {
                "values": base64.b64encode(zlib.compress(struct.pack("<d", 1.0))).decode(
                    "ascii"
                ),
                "dtype": "float64",
                "shape": [2],
            },
            "Unexpected numpy array byte length for float64",
        ),
    ],
)
def test_decoder_numpy_array_blob_error_branches(node, msg):
    decoder = PurrJsonpickleDecoder()

    with pytest.raises(SourceValidationError, match=msg):
        decoder._decode_numpy_array_blob(node, path="$.weights")


def test_decoder_numpy_array_blob_supports_raw_bytes_and_numeric_unpacks():
    decoder = PurrJsonpickleDecoder()

    float_values = [1.5, 2.5]
    raw_float64 = struct.pack("<2d", *float_values)
    encoded_float64 = base64.b64encode(raw_float64).decode("ascii")
    assert (
        decoder._decode_numpy_array_blob(
            {"values": encoded_float64, "dtype": "float64", "shape": [2]},
            path="$.weights",
        )
        == float_values
    )

    complex_values = [complex(1.0, 2.0)]
    raw_complex128 = struct.pack("<2d", complex_values[0].real, complex_values[0].imag)
    encoded_complex128 = base64.b64encode(zlib.compress(raw_complex128)).decode("ascii")
    assert (
        decoder._decode_numpy_array_blob(
            {"values": encoded_complex128, "dtype": "complex128", "shape": [1]},
            path="$.weights",
        )
        == complex_values
    )

    with pytest.raises(
        SourceValidationError,
        match="Unexpected numpy array byte length for complex128",
    ):
        decoder._decode_numpy_array_blob(
            {"values": encoded_float64, "dtype": "complex128", "shape": [2]},
            path="$.weights",
        )


def test_decoder_custom_pulse_reduce_and_complex_helpers_cover_edge_paths():
    decoder = PurrJsonpickleDecoder(extra_reduce_target_suffixes={"CustomTarget"})

    decoded = decoder._decode_custom_pulse(
        {"samples": [1.0, 2.0], "ignore_channel_scale": False},
        path="$.weights",
    )
    assert decoded["samples"] == [1.0, 2.0]

    with pytest.raises(
        SourceValidationError, match="Unsupported CustomPulse samples payload shape"
    ):
        decoder._decode_custom_pulse({"samples": "bad"}, path="$.weights")

    assert (
        decoder._decode_reduce(
            {
                "py/reduce": [
                    {"py/type": "qat.ir.instruction_basetypes.AcquireMode"},
                    {"py/tuple": ["integrator"]},
                ]
            },
            path="$.reduce",
        )
        == "integrator"
    )
    assert (
        decoder._decode_reduce(
            {
                "py/reduce": [
                    {"py/type": "pkg.CustomTarget"},
                    {"py/tuple": ["value"]},
                ]
            },
            path="$.reduce",
        )
        == "value"
    )

    with pytest.raises(SourceValidationError, match="Malformed py/reduce payload"):
        decoder._decode_reduce({"py/reduce": "bad"}, path="$.reduce")

    with pytest.raises(SourceValidationError, match="Malformed py/reduce args payload"):
        decoder._decode_reduce(
            {"py/reduce": [{"py/type": "x"}, {"py/tuple": "bad"}]},
            path="$.reduce",
        )

    with pytest.raises(SourceValidationError, match="Unsupported py/reduce arity"):
        decoder._decode_reduce(
            {"py/reduce": [{"py/type": "x"}, {"py/tuple": [1, 2]}]},
            path="$.reduce",
        )

    with pytest.raises(SourceValidationError, match="Unsupported py/reduce target"):
        decoder._decode_reduce(
            {"py/reduce": [{"py/type": "unsupported.Target"}, {"py/tuple": [1]}]},
            path="$.reduce",
        )

    assert decoder._decode_builtin_complex(
        {"py/newargs": {"py/tuple": [1, 2]}},
        path="$.complex",
    ) == complex(1, 2)

    with pytest.raises(SourceValidationError, match="Malformed builtins.complex wrapper"):
        decoder._decode_builtin_complex({}, path="$.complex")

    with pytest.raises(
        SourceValidationError, match="Malformed builtins.complex py/newargs payload"
    ):
        decoder._decode_builtin_complex(
            {"py/newargs": {"py/tuple": [1]}},
            path="$.complex",
        )

    assert decoder._is_numpy_array_wrapper({"py/object": "numpy.ndarray"})
    assert decoder._is_numpy_array_wrapper({"py/object": "pkg.array"})
    assert decoder._is_numpy_array_wrapper({"py/object": "pkg.array", "values": []})
    assert not decoder._is_numpy_array_wrapper({"py/object": 1})


def test_decoder_internal_branches_for_state_shape_tuple_samples_and_target_type():
    decoder = PurrJsonpickleDecoder()

    # Force non-dict decoded state to exercise unsupported state-shape guard.
    decoder._decode_node = lambda _node, path: (
        [1, 2] if path.endswith(".py/state") else _node
    )
    with pytest.raises(
        SourceValidationError, match="Unsupported numpy array state payload shape"
    ):
        decoder._decode_numpy_array({"py/state": {}}, path="$.x")

    # Force tuple samples through the helper to exercise tuple-to-list coercion.
    decoder = PurrJsonpickleDecoder()
    decoder._decode_node = lambda value, path: value
    decoded = decoder._decode_custom_pulse({"samples": (1.0, 2.0)}, path="$.weights")
    assert decoded["samples"] == [1.0, 2.0]

    assert not decoder._is_allowed_reduce_target(123)
