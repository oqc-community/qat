# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from copy import copy

import numpy as np
import pytest

from qat.backend.waveform_v1.codegen import WaveformV1Emitter
from qat.ir.pass_base import QatIR
from qat.purr.backends.echo import get_default_echo_hardware
from qat.purr.compiler.instructions import AcquireMode
from qat.runtime import EchoEngine


class TestEchoEngine:

    @pytest.mark.parametrize(
        "mode", [AcquireMode.RAW, AcquireMode.SCOPE, AcquireMode.INTEGRATOR]
    )
    def test_acquire_mode_gives_expected_results_for_single_acquire(self, mode):
        model = get_default_echo_hardware()
        qubit = model.get_qubit(0)
        measure_channel = qubit.get_measure_channel()
        acquire_channel = qubit.get_acquire_channel()

        # manually add a measure with the desired acquire mode
        builder = model.create_builder()
        builder.pulse(quantum_target=measure_channel, **qubit.pulse_measure)
        measure_acquire = qubit.measure_acquire
        builder.acquire(
            acquire_channel,
            output_variable="test",
            mode=mode,
            delay=measure_acquire["delay"],
            time=measure_acquire["width"],
        )
        package = WaveformV1Emitter(model).emit(QatIR(builder))

        # Test the shapes and values match up
        engine = EchoEngine()
        results = engine.execute(package)
        assert len(results) == 1
        assert "test" in results
        results = results["test"]
        samples = np.ceil(measure_acquire["width"] / acquire_channel.sample_time)
        if mode == AcquireMode.RAW:
            assert np.shape(results) == (package.shots, samples)
        elif mode == AcquireMode.SCOPE:
            assert np.shape(results) == (samples,)
        elif mode == AcquireMode.INTEGRATOR:
            assert np.shape(results) == (package.shots,)
        assert np.all(results == qubit.pulse_measure["amp"])

    @pytest.mark.parametrize(
        "mode", [AcquireMode.RAW, AcquireMode.SCOPE, AcquireMode.INTEGRATOR]
    )
    def test_acquire_mode_gives_expected_results_for_multiple_acquires_on_same_channel(
        self, mode
    ):
        model = get_default_echo_hardware()
        qubit = model.get_qubit(0)
        measure_channel = qubit.get_measure_channel()
        acquire_channel = qubit.get_acquire_channel()

        # Add the acquires: make the second twice as long, add some spacing in between
        builder = model.create_builder()
        builder.pulse(quantum_target=measure_channel, **qubit.pulse_measure)
        measure_acquire = qubit.measure_acquire
        builder.acquire(
            acquire_channel,
            output_variable="test1",
            mode=mode,
            delay=measure_acquire["delay"],
            time=measure_acquire["width"],
        )
        builder.delay(measure_channel, 1e-6)
        builder.synchronize([measure_channel, acquire_channel])
        measure_pulse = copy(qubit.pulse_measure)
        measure_pulse["width"] = 2 * measure_pulse["width"]
        measure_pulse["amp"] = 2.0
        builder.pulse(quantum_target=measure_channel, **measure_pulse)
        builder.acquire(
            acquire_channel,
            output_variable="test2",
            mode=mode,
            delay=measure_acquire["delay"],
            time=2 * measure_acquire["width"],
        )
        package = WaveformV1Emitter(model).emit(QatIR(builder))

        # Test the shapes and values match up
        engine = EchoEngine()
        results = engine.execute(package)
        assert len(results) == 2
        output_vars = ["test1", "test2"]
        amps = [qubit.pulse_measure["amp"], 2 * qubit.pulse_measure["amp"]]
        samples = [
            np.ceil(measure_acquire["width"] / acquire_channel.sample_time),
            2 * np.ceil(measure_acquire["width"] / acquire_channel.sample_time),
        ]
        for i, output_vars in enumerate(output_vars):
            assert output_vars in results
            result = results[output_vars]
            if mode == AcquireMode.RAW:
                assert np.shape(result) == (package.shots, samples[i])
            elif mode == AcquireMode.SCOPE:
                assert np.shape(result) == (samples[i],)
            elif mode == AcquireMode.INTEGRATOR:
                assert np.shape(result) == (package.shots,)
            assert np.all(result == amps[i])

    @pytest.mark.parametrize(
        "mode", [AcquireMode.RAW, AcquireMode.SCOPE, AcquireMode.INTEGRATOR]
    )
    def test_acquire_mode_gives_expected_results_for_multiple_acquires_on_different_channels(
        self, mode
    ):
        model = get_default_echo_hardware()
        qubits = model.qubits[0:2]
        measure_channel = [qubit.get_measure_channel() for qubit in qubits]
        acquire_channel = [qubit.get_acquire_channel() for qubit in qubits]

        # add some measures on different qubits
        builder = model.create_builder()
        for qubit in qubits:
            measure_channel = qubit.get_measure_channel()
            acquire_channel = qubit.get_acquire_channel()
            builder.pulse(quantum_target=measure_channel, **qubit.pulse_measure)
            measure_acquire = qubit.measure_acquire
            builder.acquire(
                acquire_channel,
                output_variable=qubit.full_id(),
                mode=mode,
                delay=measure_acquire["delay"],
                time=measure_acquire["width"],
            )
        package = WaveformV1Emitter(model).emit(QatIR(builder))

        # Test the shapes and values match up
        engine = EchoEngine()
        results = engine.execute(package)
        assert len(results) == 2
        for qubit in qubits:
            assert qubit.full_id() in results
            result = results[qubit.full_id()]
            samples = np.ceil(qubit.measure_acquire["width"] / acquire_channel.sample_time)
            if mode == AcquireMode.RAW:
                assert np.shape(result) == (package.shots, samples)
            elif mode == AcquireMode.SCOPE:
                assert np.shape(result) == (samples,)
            elif mode == AcquireMode.INTEGRATOR:
                assert np.shape(result) == (package.shots,)
            assert np.all(result == qubit.pulse_measure["amp"])
