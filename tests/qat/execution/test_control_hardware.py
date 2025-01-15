# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from os.path import abspath, dirname, join, pardir

from qat.purr.backends.qblox.execution import CompositeControlHardware
from qat.purr.backends.qblox.loader import parse_control_hardware


class TestControlHardware:

    def test_build_control_hardware(self):
        clusters_filepath = abspath(
            join(dirname(__file__), pardir, "files", "config", "clusters.csv")
        )

        control_hardware = parse_control_hardware(clusters_filepath)
        assert control_hardware
        assert isinstance(control_hardware, CompositeControlHardware)
        assert len(control_hardware.components) == 4
