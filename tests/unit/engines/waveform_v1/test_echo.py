# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import numpy as np
import pytest

from qat.backend.waveform_v1.executable import (
    PositionalAcquireData,
    WaveformV1ChannelData,
    WaveformV1Program,
)
from qat.engines.waveform_v1 import EchoEngine
from qat.purr.compiler.instructions import AcquireMode


class TestEchoEngine:
    @pytest.mark.parametrize(
        "mode", [AcquireMode.RAW, AcquireMode.SCOPE, AcquireMode.INTEGRATOR]
    )
    def test_acquire_mode_gives_expected_results_for_single_acquire(self, mode):
        acquire = PositionalAcquireData(
            mode=mode, position=600, length=100, output_variable="test"
        )
        channel_data = WaveformV1ChannelData(
            buffer=np.linspace(0, 1000, 1001), baseband_frequency=5.0e9, acquires=[acquire]
        )
        program = WaveformV1Program(
            channel_data={"ch1": channel_data}, repetition_time=1e-3, shots=1000
        )

        # Test the shapes and values match up
        engine = EchoEngine()
        results = engine.execute(program)
        assert len(results) == 1
        assert "test" in results
        results = results["test"]

        expected = np.linspace(600, 699, 100)
        if mode == AcquireMode.RAW:
            assert np.shape(results) == (program.shots, acquire.length)
            assert np.allclose(results, np.tile(expected, (program.shots, 1)))
        elif mode == AcquireMode.SCOPE:
            assert np.shape(results) == (acquire.length,)
            assert np.allclose(results, np.linspace(600, 699, 100))
        elif mode == AcquireMode.INTEGRATOR:
            assert np.shape(results) == (program.shots,)
            assert np.allclose(results, np.mean(expected))

    @pytest.mark.parametrize(
        "mode", [AcquireMode.RAW, AcquireMode.SCOPE, AcquireMode.INTEGRATOR]
    )
    def test_acquire_mode_gives_expected_results_for_multiple_acquires_on_same_channel(
        self, mode
    ):
        acquire1 = PositionalAcquireData(
            mode=mode, position=200, length=50, output_variable="test1"
        )
        acquire2 = PositionalAcquireData(
            mode=mode, position=600, length=100, output_variable="test2"
        )
        channel_data = WaveformV1ChannelData(
            buffer=np.linspace(0, 1000, 1001),
            baseband_frequency=5.0e9,
            acquires=[acquire1, acquire2],
        )
        program = WaveformV1Program(
            channel_data={"ch1": channel_data}, repetition_time=1e-3, shots=1000
        )

        # Test the shapes and values match up
        engine = EchoEngine()
        results = engine.execute(program)
        assert len(results) == 2
        for test, start, samples in [("test1", 200, 50), ("test2", 600, 100)]:
            assert test in results
            result = results[test]
            expected = np.linspace(start, start + samples - 1, samples)
            if mode == AcquireMode.RAW:
                assert np.shape(result) == (program.shots, samples)
                assert np.allclose(result, np.tile(expected, (program.shots, 1)))
            elif mode == AcquireMode.SCOPE:
                assert np.shape(result) == (samples,)
                assert np.allclose(result, expected)
            elif mode == AcquireMode.INTEGRATOR:
                assert np.shape(result) == (program.shots,)
                assert np.allclose(result, np.mean(expected))

    @pytest.mark.parametrize(
        "mode", [AcquireMode.RAW, AcquireMode.SCOPE, AcquireMode.INTEGRATOR]
    )
    def test_acquire_mode_gives_expected_results_for_multiple_acquires_on_different_channels(
        self, mode
    ):
        acquire1 = PositionalAcquireData(
            mode=mode, position=200, length=50, output_variable="test1"
        )
        acquire2 = PositionalAcquireData(
            mode=mode, position=600, length=100, output_variable="test2"
        )
        channel_data_1 = WaveformV1ChannelData(
            buffer=np.linspace(0, 1000, 1001), baseband_frequency=5.0e9, acquires=[acquire1]
        )
        channel_data_2 = WaveformV1ChannelData(
            buffer=np.linspace(1000, 2000, 1001),
            baseband_frequency=6.0e9,
            acquires=[acquire2],
        )
        program = WaveformV1Program(
            channel_data={"ch1": channel_data_1, "ch2": channel_data_2},
            repetition_time=1e-3,
            shots=1000,
        )

        # Test the shapes and values match up
        engine = EchoEngine()
        results = engine.execute(program)
        assert len(results) == 2
        for test, start, samples in [("test1", 200, 50), ("test2", 1600, 100)]:
            assert test in results
            result = results[test]
            expected = np.linspace(start, start + samples - 1, samples)
            if mode == AcquireMode.RAW:
                assert np.shape(result) == (program.shots, samples)
                assert np.allclose(result, np.tile(expected, (program.shots, 1)))
            elif mode == AcquireMode.SCOPE:
                assert np.shape(result) == (samples,)
                assert np.allclose(result, expected)
            elif mode == AcquireMode.INTEGRATOR:
                assert np.shape(result) == (program.shots,)
                assert np.allclose(result, np.mean(expected))
