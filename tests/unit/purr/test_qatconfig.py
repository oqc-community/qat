# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023-2024 Oxford Quantum Circuits Ltd
import numpy as np
import pytest
from pydantic import ValidationError

from qat import qatconfig
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.backends.realtime_chip_simulator import get_default_RTCS_hardware
from qat.purr.qatconfig import InstructionValidationConfig, QatConfig
from qat.purr.utils.logger import LoggerLevel

from tests.unit.purr.compiler.test_execution import get_test_model, get_test_runtime

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
    with pytest.warns(DeprecationWarning):
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
    # Test direct change of repeats limit.
    qatconfig.MAX_REPEATS_LIMIT = repeats_limit
    assert qatconfig.MAX_REPEATS_LIMIT == repeats_limit

    qatconfig.__init__()  # Reload settings to default.


@pytest.mark.parametrize("disable_pulse_duration_limits", [True, False])
def test_change_disable_pulse_duration_limits(disable_pulse_duration_limits):
    # Test direct change of repeats limit.
    with pytest.warns(DeprecationWarning):
        qatconfig.DISABLE_PULSE_DURATION_LIMITS = disable_pulse_duration_limits
        assert qatconfig.DISABLE_PULSE_DURATION_LIMITS == disable_pulse_duration_limits

    qatconfig.__init__()  # Reload settings to default.


def test_disable_pulse_duration_limits_throws_warning(caplog):
    newconfig = QatConfig()
    # Capture if warnings are sent to logger.
    with caplog.at_level(LoggerLevel.WARNING.value):
        newconfig.INSTRUCTION_VALIDATION.PULSE_DURATION_LIMITS = False
        assert "Disabled check for pulse duration limits" in caplog.text
    with pytest.warns(DeprecationWarning):
        with caplog.at_level(LoggerLevel.WARNING.value):
            newconfig.DISABLE_PULSE_DURATION_LIMITS = True
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


result_dict = {"q0": np.full((1000, 1000), 1.0 + 0.0j)}
runtimes = [
    pytest.param(get_default_echo_hardware().create_runtime(), id="Echo"),
    pytest.param(get_default_RTCS_hardware().create_runtime(), id="RTCS"),
    pytest.param(get_test_runtime(get_test_model()), id="FakeLive"),
]


@pytest.mark.parametrize("runtime", runtimes)
@pytest.mark.parametrize("validation", [True, False], ids=["enabled", "disabled"])
def test_validate_instructions(monkeypatch, mocker, runtime, validation):
    setting = InstructionValidationConfig()
    if not validation:
        setting.disable()
    monkeypatch.setattr(qatconfig, "INSTRUCTION_VALIDATION", setting)
    engine = runtime.engine
    model = runtime.model
    mocker.patch.object(engine, "validate", autospec=True)
    mocker.patch.object(engine, "execute", autospec=True, return_value=result_dict)

    qubit_0 = model.qubits[0]
    builder = (
        model.create_builder()
        .X(qubit_0, np.pi / 2.0)
        .measure(qubit_0, output_variable="q0")
    )

    runtime.execute(builder)
    assert engine.validate.called == validation


def test_extension_loads():
    qatconfig = QatConfig(EXTENSIONS=["tests.unit.purr.extensions.SomeExtension"])
    from .extensions import SomeExtension, loaded_extensions

    assert "SomeExtension" in loaded_extensions
    assert SomeExtension in qatconfig.EXTENSIONS
    loaded_extensions.clear()


def test_extension_load_multiple():
    qatconfig = QatConfig(
        EXTENSIONS=[
            "tests.unit.purr.extensions.SomeExtension",
            "tests.unit.purr.extensions.AnotherExtension",
        ]
    )
    from .extensions import AnotherExtension, SomeExtension, loaded_extensions

    assert {"SomeExtension", "AnotherExtension"}.issubset(loaded_extensions)
    assert {SomeExtension, AnotherExtension}.issubset(set(qatconfig.EXTENSIONS))
    loaded_extensions.clear()


def test_nonexistant_extension():
    with pytest.raises(ValidationError):
        QatConfig(EXTENSIONS=["tests.unit.doesntexist"])


def test_extension_not_a_QatExtension():
    with pytest.raises(ValidationError):
        QatConfig(EXTENSIONS=["tests.unit.purr.extensions.NotAnExtension"])
