# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import numpy as np

from qat.engines.zero import ZeroEngine, readout_shape
from qat.purr.compiler.instructions import AcquireMode
from qat.runtime.executables import AcquireData, ChannelData, ChannelExecutable


class TestReadoutShape:
    def test_integrator(self):
        acquire = AcquireData(
            length=254, position=0, mode=AcquireMode.INTEGRATOR, output_variable="test"
        )
        assert readout_shape(acquire, 1234) == (1234,)

    def test_scope(self):
        acquire = AcquireData(
            length=254, position=0, mode=AcquireMode.SCOPE, output_variable="test"
        )
        assert readout_shape(acquire, 1234) == (254,)

    def test_raw(self):
        acquire = AcquireData(
            length=254, position=0, mode=AcquireMode.RAW, output_variable="test"
        )
        assert readout_shape(acquire, 1234) == (
            1234,
            254,
        )


class TestZeroEngine:
    def test_mock_data(self):
        channel1 = ChannelData(
            acquires=[
                AcquireData(
                    length=254, position=0, mode=AcquireMode.SCOPE, output_variable="test0"
                ),
                AcquireData(
                    length=42,
                    position=0,
                    mode=AcquireMode.INTEGRATOR,
                    output_variable="test1",
                ),
            ]
        )
        channel2 = ChannelData(
            acquires=AcquireData(
                length=254, position=0, mode=AcquireMode.RAW, output_variable="test2"
            )
        )
        channel3 = ChannelData(
            acquires=[
                AcquireData(
                    length=254,
                    position=0,
                    mode=AcquireMode.INTEGRATOR,
                    output_variable="test3",
                ),
                AcquireData(
                    length=254,
                    position=1000,
                    mode=AcquireMode.INTEGRATOR,
                    output_variable="test4",
                ),
            ]
        )
        package = ChannelExecutable(
            shots=1000,
            compiled_shots=454,
            channel_data={"CH1": channel1, "CH2": channel2, "CH3": channel3},
        )

        engine = ZeroEngine()
        results = engine.execute(package)
        assert len(results) == 5
        assert np.all(results["test0"] == np.zeros((254,)))
        assert np.all(results["test1"] == np.zeros((454,)))
        assert np.all(results["test2"] == np.zeros((454, 254)))
        assert np.all(results["test3"] == np.zeros((454,)))
        assert np.all(results["test4"] == np.zeros((454,)))
