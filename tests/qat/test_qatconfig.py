# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import pytest
from pydantic import ValidationError

from qat import qatconfig
from qat.purr.qatconfig import QatConfig


@pytest.mark.parametrize("invalid_argument", ["invalid", {"key": 5}, 5.5])
def test_qatconfig_invalid_assignment(invalid_argument):
    with pytest.raises(ValidationError):
        qatconfig.MAX_REPEATS_LIMIT = invalid_argument


@pytest.mark.parametrize("repeats_limit", [None, 10_000, 16_874, 50_000, 100_000])
def test_change_max_repeats_limit(repeats_limit):
    qatconfig.__init__()  # Reload settings.
    # Test direct change of repeats limit.
    qatconfig.MAX_REPEATS_LIMIT = repeats_limit
    assert qatconfig.MAX_REPEATS_LIMIT == repeats_limit


@pytest.mark.parametrize("repeats_limit", [None, 10_000, 16_874, 50_000, 100_000])
def test_change_env_max_repeats_limit(monkeypatch, repeats_limit):
    NEW_LIMIT = 15_321
    monkeypatch.setenv("QAT_MAX_REPEATS_LIMIT", str(NEW_LIMIT))
    newconfig = QatConfig()
    assert newconfig.MAX_REPEATS_LIMIT == NEW_LIMIT
