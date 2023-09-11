# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

from qat.purr.backends.verification import get_verification_model, Lucy, QPUVersion
from qat import execute
from .qasm_utils import get_qasm2

import pytest


class TestFirmwareVerificationEngines:
    def test_latest_lucy(self):
        with pytest.raises(NotImplementedError):
            model = get_verification_model(Lucy.Latest)
            execute(get_qasm2("basic.qasm"), model)

    def test_unknown_make(self):
        model = get_verification_model(QPUVersion("something", "123"))
        assert model is None
