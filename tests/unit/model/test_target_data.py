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
        "path",
        [
            "tests/files/hardware/invalid_type_target_data.yaml",
            "tests/files/hardware/unknown_field_target_data.yaml",
        ],
    )
    def test_invalid_field_throws_error(self, path):
        with pytest.raises(ValidationError):
            TargetData.from_yaml(path)
