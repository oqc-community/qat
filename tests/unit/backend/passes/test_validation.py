# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

import pytest

from qat.backend.passes.validation import (
    NCOFrequencyVariability,
    NoAcquireWeightsValidation,
    NoMultipleAcquiresValidation,
)
from qat.core.result_base import ResultManager
from qat.model.loaders.legacy import EchoModelLoader
from qat.purr.compiler.instructions import Pulse, PulseShapeType

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


# Test passes for the Pydantic hardware model.
