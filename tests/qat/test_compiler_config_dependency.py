# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from os.path import abspath, dirname, join

import pytest
from compiler_config.config import CompilerConfig

from qat.purr.backends.echo import get_default_echo_hardware
from qat.qat import execute_with_metrics
from tests.qat.qasm_utils import ProgramFileType, get_test_file_path

SUPPORTED_CONFIG_VERSIONS = ["v02", "v1"]


def _get_json_path(file_name):
    return join(
        abspath(join(dirname(__file__), "serialised_compiler_config_templates", file_name))
    )


def _get_contents(file_path):
    """Get Json from a file."""
    with open(_get_json_path(file_path)) as ifile:
        return ifile.read()


@pytest.mark.parametrize("version", SUPPORTED_CONFIG_VERSIONS)
def test_runs_successfully_with_config(version):
    program = get_test_file_path(ProgramFileType.QASM2, "ghz.qasm")
    hardware = get_default_echo_hardware()
    serialised_data = _get_contents(f"serialised_full_compiler_config_{version}.json")
    deserialised_conf = CompilerConfig.create_from_json(
        serialised_data
    )  # Test full compiler config v1
    results, metrics = execute_with_metrics(program, hardware, deserialised_conf)
    assert results is not None
    assert metrics is not None
