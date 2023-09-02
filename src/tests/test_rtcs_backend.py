# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

import numpy as np
import pytest
from qat.purr.backends.realtime_chip_simulator import (
    get_default_RTCS_hardware,
    qutip_available,
)
from qat.purr.compiler.runtime import execute_instructions, get_builder


@pytest.mark.skipif(not qutip_available, reason="Qutip is not available on this platform")
class TestBaseQuantumQutip:
    def get_simulator(self):
        hw = get_default_RTCS_hardware()
        for i in range(2):
            assert hw.get_qubit(i).is_calibrated
        return hw

    def test_cnot_from_ecr_gate(self):
        def conv(x):
            return 0 if x > 0.0 else 1

        hw = get_default_RTCS_hardware()
        for i in range(2):
            assert hw.get_qubit(i).is_calibrated

        control_q = hw.get_qubit(0)
        target_q = hw.get_qubit(1)
        cr_channel = control_q.get_cross_resonance_channel(target_q)
        sync_channels = [
            cr_channel, control_q.get_drive_channel(), target_q.get_drive_channel()
        ]
        prep = np.linspace(0.0, np.pi, 2)

        # sweep over a time range
        # yapf: disable
        result = [[(
            execute_instructions(
                hw,
                get_builder(hw)
                .X(control_q, c)
                .synchronize(sync_channels)
                .X(target_q, t)
                .ECR(control_q, target_q)
                .X(control_q)
                .X(target_q, -np.pi / 2.0)
                .Z(control_q, -np.pi / 2.0)
                .measure_mean_z(control_q)
                .measure_mean_z(target_q)
            )[0]
        ) for t in prep] for c in prep]
        # yapf: enable

        for i in [0, 1]:
            for j in [0, 1]:
                assert i == conv(result[i][j][0])
                assert (i + j) % 2 == conv(result[i][j][1])
