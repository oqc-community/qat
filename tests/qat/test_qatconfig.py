# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
import pytest
from pydantic import ValidationError

from qat import qatconfig
from qat.purr.qatconfig import QatConfig
from qat.purr.utils.logger import LoggerLevel

MAX_REPEATS_LIMIT = 100_000  # Default value for qatconfig.MAX_REPEATS_LIMIT.


@pytest.mark.parametrize(
    "invalid_argument", ["invalid", {"key": 5}, -10, -1, -0.5, 0, 0.5, 5.5]
)
def test_qatconfig_invalid_assignment(invalid_argument):
    with pytest.raises(ValidationError):
        qatconfig.MAX_REPEATS_LIMIT = invalid_argument


def test_default_disable_pulse_duration_limits(monkeypatch):
    monkeypatch.delenv("QAT_DISABLE_PULSE_DURATION_LIMITS", raising=False)
    newconfig = QatConfig()
    assert (
        isinstance(newconfig.DISABLE_PULSE_DURATION_LIMITS, bool)
        and newconfig.DISABLE_PULSE_DURATION_LIMITS == False
    )


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


@pytest.mark.parametrize("disable_pulse_duration_limits", [False, True])
def test_change_disable_pulse_duration_limits(disable_pulse_duration_limits):
    qatconfig.__init__()  # Reload settings.
    # Test direct change of repeats limit.
    qatconfig.DISABLE_PULSE_DURATION_LIMITS = disable_pulse_duration_limits
    assert qatconfig.DISABLE_PULSE_DURATION_LIMITS == disable_pulse_duration_limits


def test_disable_pulse_duration_limits_throws_warning(caplog):
    qatconfig.__init__()  # Reload settings.
    # Capture if warnings are sent to logger.
    with caplog.at_level(LoggerLevel.WARNING.value):
        qatconfig.DISABLE_PULSE_DURATION_LIMITS = True
        assert "Disabled check for pulse duration limits" in caplog.text


@pytest.mark.parametrize("repeats_limit", [10_000, 16_874, 50_000, 100_000])
def test_change_env_variables(monkeypatch, repeats_limit):
    monkeypatch.setenv("QAT_MAX_REPEATS_LIMIT", str(repeats_limit))
    newconfig = QatConfig()
    assert (
        newconfig.MAX_REPEATS_LIMIT == repeats_limit
    )  # New instances of QatConfig should have the updated env variable.
    assert (
        qatconfig.MAX_REPEATS_LIMIT == MAX_REPEATS_LIMIT
    )  # Existing instances should not have the updated env variable.
