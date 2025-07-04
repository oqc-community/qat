# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

import random
from contextlib import nullcontext

import pytest

from qat.backend.passes.purr.validation import (
    NCOFrequencyVariability,
    NoAcquiresWithDifferentWeightsValidation,
    NoAcquireWeightsValidation,
    NoMultipleAcquiresValidation,
)
from qat.core.result_base import ResultManager
from qat.ir.lowered import PartitionedIR
from qat.model.loaders.purr import EchoModelLoader
from qat.purr.compiler.instructions import Acquire, CustomPulse, Pulse, PulseShapeType

from tests.unit.utils.builder_nuggets import resonator_spect


class TestValidationPasses:
    def test_nco_freq_pass(self):
        model = EchoModelLoader().load()
        builder = resonator_spect(model)
        res_mgr = ResultManager()

        NCOFrequencyVariability().run(builder, res_mgr, model)

        channel = next(iter(model.pulse_channels.values()))
        channel.fixed_if = True

        with pytest.raises(ValueError):
            NCOFrequencyVariability().run(builder, res_mgr, model)

        channel.fixed_if = False
        NCOFrequencyVariability().run(builder, res_mgr, model)


class TestNoAcquireWeightsValidation:
    def test_acquire_with_filter_raises_error(self):
        model = EchoModelLoader().load()
        res_mgr = ResultManager()
        qubit = model.get_qubit(0)
        channel = qubit.get_acquire_channel()
        builder = model.create_builder()
        builder.acquire(
            channel, delay=0.0, filter=Pulse(channel, PulseShapeType.SQUARE, 1e-6)
        )
        with pytest.raises(NotImplementedError):
            NoAcquireWeightsValidation().run(builder, res_mgr)


class TestNoMultipleAcquiresValidation:
    def test_multiple_acquires_on_same_pulse_channel_raises_error(self):
        model = EchoModelLoader().load()
        res_mgr = ResultManager()
        qubit = model.get_qubit(0)
        channel = qubit.get_acquire_channel()
        builder = model.create_builder()
        builder.acquire(channel, delay=0.0)

        # Test should run as there is only one acquire
        NoMultipleAcquiresValidation().run(builder, res_mgr)

        # Add another acquire and test it breaks it
        builder.acquire(channel, delay=0.0)
        with pytest.raises(NotImplementedError):
            NoMultipleAcquiresValidation().run(builder, res_mgr)

    def test_multiple_acquires_that_share_physical_channel_raises_error(self):
        model = EchoModelLoader().load()
        res_mgr = ResultManager()
        qubit = model.get_qubit(0)
        acquire_channel = qubit.get_acquire_channel()
        # in practice, we shouldn't do this with a measure channel, but convinient for
        # testing
        measure_channel = qubit.get_measure_channel()
        builder = model.create_builder()
        builder.acquire(acquire_channel, delay=0.0)
        builder.acquire(measure_channel, delay=0.0)
        with pytest.raises(NotImplementedError):
            NoMultipleAcquiresValidation().run(builder, res_mgr)


class TestNoAcquiresWithDifferentWeightsValidation:
    @pytest.mark.parametrize("offset,invalid", [(0, False), (0.05, True), (1e-05, True)])
    def test_multiple_acquires_with_weights(self, offset, invalid):
        model = EchoModelLoader().load()
        qubit = model.get_qubit(0)
        acquire_channel = qubit.get_acquire_channel()
        t = acquire_channel.sample_time * 10

        samples1 = [random.uniform(0.0, 1.0) for _ in range(10)]
        samples2 = [sample + offset for sample in samples1]
        custom_pulse1 = CustomPulse(acquire_channel, samples=samples1)
        custom_pulse2 = CustomPulse(acquire_channel, samples=samples2)
        acquires = [
            Acquire(acquire_channel, filter=custom_pulse1, time=t),
            Acquire(acquire_channel, filter=custom_pulse2, time=t),
        ]

        ir = PartitionedIR(acquire_map={acquire_channel: acquires})

        with pytest.raises(ValueError) if invalid else nullcontext():
            NoAcquiresWithDifferentWeightsValidation().run(ir)


# Test passes for the Pydantic hardware model.
