# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from importlib.metadata import version


# TODO: Write tests to verify driver <-> fw compatibility. COMPILER-661
class TestQbloxExecutable:
    def test_qiskit_instruments_compatibility(self):
        assert version("qblox_instruments") == "0.16.0"
