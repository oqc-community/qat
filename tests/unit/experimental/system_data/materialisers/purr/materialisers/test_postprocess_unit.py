# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

import pytest

from qat.experimental.system_data.materialisers.errors import SourceValidationError
from qat.experimental.system_data.materialisers.purr.validators.postprocess import (
    _validate_max_likelihood_post_process_method_payload,
)


def _valid_max_likelihood_payload() -> dict:
    return {
        "states": {
            "0": {"location": 1 + 0j, "label": "g"},
            "1": {"location": -1 + 0j, "label": "e"},
        },
        "noise_est": 1.0,
        "p_min": 0.0,
    }


@pytest.mark.parametrize("p_min", [-0.1, 1.1])
def test_validate_max_likelihood_post_process_method_payload_rejects_out_of_range_p_min(
    p_min,
):
    payload = _valid_max_likelihood_payload()
    payload["p_min"] = p_min

    with pytest.raises(SourceValidationError, match=r"must lie in \[0, 1\]"):
        _validate_max_likelihood_post_process_method_payload(
            method_payload=payload,
            path_root="$.quantum_devices.Q0.post_process_method",
        )


def test_validate_max_likelihood_post_process_method_payload_accepts_boundary_p_min_values():
    payload = _valid_max_likelihood_payload()

    for p_min in (0.0, 1.0):
        payload["p_min"] = p_min
        _validate_max_likelihood_post_process_method_payload(
            method_payload=payload,
            path_root="$.quantum_devices.Q0.post_process_method",
        )
