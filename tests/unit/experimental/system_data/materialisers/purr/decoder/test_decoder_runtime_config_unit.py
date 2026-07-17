# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from qat.experimental.system_data.materialisers.errors import SourceValidationError
from qat.experimental.system_data.materialisers.purr.decoder import (
    decode_jsonpickle_payload,
)


def _reduce_payload(target_type: str) -> dict:
    return {
        "node": {
            "py/reduce": [
                {"py/type": target_type},
                {"py/tuple": ["decoded-value"]},
            ]
        }
    }


def test_decode_jsonpickle_payload_rejects_unknown_reduce_target_by_default():
    with pytest.raises(SourceValidationError, match="Unsupported py/reduce target"):
        decode_jsonpickle_payload(_reduce_payload("external.pkg.CustomType"))


def test_decode_jsonpickle_payload_accepts_runtime_extra_reduce_target_type():
    decoded = decode_jsonpickle_payload(
        _reduce_payload("external.pkg.CustomType"),
        extra_reduce_target_types={"external.pkg.CustomType"},
    )

    assert decoded["node"] == "decoded-value"


def test_decode_jsonpickle_payload_accepts_runtime_extra_reduce_target_suffix():
    decoded = decode_jsonpickle_payload(
        _reduce_payload("external.pkg.CustomReference"),
        extra_reduce_target_suffixes={"CustomReference"},
    )

    assert decoded["node"] == "decoded-value"
