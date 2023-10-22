# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

from qat.purr.backends.verification import get_verification_model, Lucy, QPUVersion, valid_circuit_length
from qat.qat import execute
from .qasm_utils import get_qasm2

import pytest


class TestFirmwareVerificationEngines:
    def test_latest_lucy(self):
        model = get_verification_model(Lucy.Latest)
        results = execute(get_qasm2("basic.qasm"), model)
        assert len(results['c']) == 2

    def test_unknown_make(self):
        with pytest.raises(NotImplementedError):
            get_verification_model(QPUVersion("something", "123"))

    def test_no_qpu_version_provided(self):
        with pytest.raises(ValueError):
            get_verification_model("something")

    def test_valid_circuit_length(self):
        assert valid_circuit_length(duration=60, max_circuit_duration = 80) is True
        assert valid_circuit_length(duration=80, max_circuit_duration=60) is False






