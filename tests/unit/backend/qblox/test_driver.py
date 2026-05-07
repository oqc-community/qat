# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from importlib.metadata import version


# TODO (Follow up from COMPILER-661 and COMPILER-1004) - COMPILER-1005: check against fw version (live testing)
class TestDriverFwCompatibility:
    def test_qblox_instruments_version(self):
        assert version("qblox_instruments") == "1.2.1"
