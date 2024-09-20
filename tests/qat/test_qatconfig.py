# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import pytest


# from qat import qatconfig
@pytest.fixture
def qatconfig():
    """We import qatconfig as a fixture so we can test mocking the environment."""
    from qat import qatconfig

    yield qatconfig


@pytest.mark.parametrize("invalid_argument", ["5", "invalid", {"key": 5}, 5.5])
def test_qatconfig_invalid_assignment(qatconfig, invalid_argument):
    with pytest.raises(TypeError):
        qatconfig.MAX_REPEATS_LIMIT = invalid_argument


@pytest.mark.parametrize("repeats_limit", [10_000, 50_000, 16_874, 100_000])
def test_change_max_repeats_limit(qatconfig, repeats_limit):
    qatconfig.__init__()  # Reload settings.
    # Test direct change of repeats limit.
    qatconfig.MAX_REPEATS_LIMIT = repeats_limit
    assert qatconfig.MAX_REPEATS_LIMIT == repeats_limit


@pytest.mark.parametrize("repeats_limit", [10_000, 50_000, 16_874, 100_000])
def test_change_env_max_repeats_limit(qatconfig, monkeypatch, repeats_limit):
    monkeypatch.setenv("QAT_MAX_REPEATS_LIMIT", str(repeats_limit))
    # Default value cannot change if env variable changes after instantiation.
    assert qatconfig.MAX_REPEATS_LIMIT == 100_000
