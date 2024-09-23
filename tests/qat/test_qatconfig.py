# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import pytest
from pydantic import ValidationError

from qat import qatconfig
from qat.purr.qatconfig import QatConfig

MAX_REPEATS_LIMIT = 100_000  # Default value for qatconfig.MAX_REPEATS_LIMIT.
qatconfig.MAX_REPEATS_LIMIT = MAX_REPEATS_LIMIT


@pytest.mark.parametrize(
    "invalid_argument", ["invalid", {"key": 5}, -10, -1, -0.5, 0, 0.5, 5.5]
)
def test_qatconfig_invalid_assignment(invalid_argument):
    with pytest.raises(ValidationError):
        qatconfig.MAX_REPEATS_LIMIT = invalid_argument


def test_default_repeats_limit(monkeypatch):
    monkeypatch.delenv("QAT_MAX_REPEATS_LIMIT", raising=False)
    newconfig = QatConfig()
    assert (
        isinstance(newconfig.MAX_REPEATS_LIMIT, int)
        and newconfig.MAX_REPEATS_LIMIT == MAX_REPEATS_LIMIT
    )


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
    assert (
        newconfig.MAX_REPEATS_LIMIT == NEW_LIMIT
    )  # New instances of QatConfig should have the updated env variable.
    assert (
        qatconfig.MAX_REPEATS_LIMIT == MAX_REPEATS_LIMIT
    )  # Existing instances should not have the updated env variable.
