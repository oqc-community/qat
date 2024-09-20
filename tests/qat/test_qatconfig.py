# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import pytest

from qat import qatconfig

# @pytest.fixture
# def qatconfig():
#    ''' We import qatconfig as a fixture so we can test mocking the environment. '''
#    from qat import qatconfig
#    return qatconfig


@pytest.mark.parametrize("repeat_limits", [10_000, 50_000, 16_874, 100_000])
def test_change_max_repeats_limit(monkeypatch, repeat_limits):
    # Test direct change of repeats limit.
    qatconfig.MAX_REPEATS_LIMIT = repeat_limits
    assert qatconfig.MAX_REPEATS_LIMIT == repeat_limits

    # Test env change of repeats limit.
    monkeypatch.setenv("MAX_REPEATS_LIMIT", str(repeat_limits))
    assert qatconfig.MAX_REPEATS_LIMIT == repeat_limits


@pytest.mark.parametrize("invalid_argument", ["5", "invalid", {"key": 5}, 5.5])
def test_qatconfig_invalid_assignment(invalid_argument):
    with pytest.raises(TypeError):
        qatconfig.MAX_REPEATS_LIMIT = invalid_argument
