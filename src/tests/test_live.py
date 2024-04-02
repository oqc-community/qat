# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

import unittest

from qat.purr.backends.live import LiveHardwareModel
from qat.purr.backends.live_devices import Instrument


class TestInstructions(unittest.TestCase):
    def test_module(self):
        """
        Sanity check that module can be loaded as code isn't
        exercised right now.
        """
        model = LiveHardwareModel()
        inst = Instrument("no-ip")
