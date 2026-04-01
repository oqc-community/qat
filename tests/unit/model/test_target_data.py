# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import pytest
from pydantic import ValidationError

from qat.model.target_data import TargetData


class TestTargetData:
    def test_file_load(self):
        path = "tests/files/hardware/target_data.yaml"
        target_data = TargetData.from_yaml(path)
        assert isinstance(target_data, TargetData)

    @pytest.mark.parametrize(
        "path,error_message",
        [
            pytest.param(
                "tests/files/hardware/invalid_type_target_data.yaml",
                r"Input should be",
                id="invalid_type",
            ),
            pytest.param(
                "tests/files/hardware/unknown_field_target_data.yaml",
                r"Extra inputs are not permitted",
                id="unknown_field",
            ),
        ],
    )
    def test_invalid_field_throws_error(self, path, error_message):
        with pytest.raises(ValidationError, match=error_message):
            TargetData.from_yaml(path)
