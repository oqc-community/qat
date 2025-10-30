# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import numpy as np
import pytest

from qat.backend.waveform_v1.purr.codegen import WaveformV1Backend
from qat.engines.waveform_v1 import EchoEngine
from qat.middleend.passes.purr.transform import (
    RepeatTranslation,
)
from qat.model.loaders.purr import EchoModelLoader
from qat.model.target_data import TargetData
from qat.purr.compiler.instructions import AcquireMode, PulseShapeType


class TestEchoEngine:
    @pytest.mark.parametrize(
        "mode", [AcquireMode.RAW, AcquireMode.SCOPE, AcquireMode.INTEGRATOR]
    )
    def test_acquire_mode_gives_expected_results_for_single_acquire(self, mode):
        model = EchoModelLoader().load()
        qubit = model.get_qubit(0)
        measure_channel = qubit.get_measure_channel()
        acquire_channel = qubit.get_acquire_channel()

        # manually add a measure with the desired acquire mode
        builder = model.create_builder()
        builder.repeat(1000, 100e-6)
        builder.pulse(quantum_target=measure_channel, **qubit.pulse_measure)
        measure_acquire = qubit.measure_acquire
        builder.acquire(
            acquire_channel,
            output_variable="test",
            mode=mode,
            delay=measure_acquire["delay"],
            time=measure_acquire["width"],
        )
        builder = RepeatTranslation(TargetData.default()).run(builder)
        package = WaveformV1Backend(model).emit(builder)

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
        model = EchoModelLoader().load()
        qubit = model.get_qubit(0)
        measure_channel = qubit.get_measure_channel()
        acquire_channel = qubit.get_acquire_channel()
        amp = qubit.pulse_measure["amp"]
        width = qubit.pulse_measure["width"]

        # Add the acquires: make the second twice as long, add some spacing in between
        builder = model.create_builder()
        builder.repeat(1000, 100e-6)
        builder.pulse(
            qubit.get_measure_channel(), shape=PulseShapeType.SQUARE, width=width, amp=amp
        )
        builder.acquire(
            qubit.get_acquire_channel(),
            mode=mode,
            delay=0.0,
            time=width,
            output_variable="test1",
        )
        builder.delay(measure_channel, 1e-6)
        builder.delay(acquire_channel, 1e-6)
        builder.pulse(
            qubit.get_measure_channel(),
            shape=PulseShapeType.SQUARE,
            width=2 * width,
            amp=2 * amp,
        )
        builder.acquire(
            qubit.get_acquire_channel(),
            mode=mode,
            delay=0.0,
            time=2 * width,
            output_variable="test2",
        )
        builder = RepeatTranslation(TargetData.default()).run(builder)
        package = WaveformV1Backend(model).emit(builder)

        # Test the shapes and values match up
        engine = EchoEngine()
        results = engine.execute(package)
        assert len(results) == 2
        output_vars = ["test1", "test2"]
        amps = [amp, 2 * amp]
        samples = [
            np.ceil(width / acquire_channel.sample_time),
            2 * np.ceil(width / acquire_channel.sample_time),
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
        model = EchoModelLoader().load()
        qubits = model.qubits[0:2]
        measure_channel = [qubit.get_measure_channel() for qubit in qubits]
        acquire_channel = [qubit.get_acquire_channel() for qubit in qubits]

        # add some measures on different qubits
        builder = model.create_builder()
        builder.repeat(1000, 100e-6)
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
        builder = RepeatTranslation(TargetData.default()).run(builder)
        package = WaveformV1Backend(model).emit(builder)

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
