# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.backend.passes.validation import (
    NoAcquireWeightsValidation,
)
from qat.core.result_base import ResultManager
from qat.ir.instruction_builder import QuantumInstructionBuilder
from qat.ir.measure import Acquire
from qat.ir.waveforms import Pulse, SquareWaveform
from qat.model.loaders.lucy import LucyModelLoader


class TestNoAcquireWeightsValidation:
    def test_acquire_with_filter_raises_error(self):
        model = LucyModelLoader().load()
        res_mgr = ResultManager()
        qubit = model.qubits[0]
        builder = QuantumInstructionBuilder(model)
        builder.acquire(
            target=qubit,
            delay=0.0,
            filter=Pulse(
                target=qubit.acquire_pulse_channel.uuid, waveform=SquareWaveform(width=1e-6)
            ),
            duration=1e-6,
        )
        with pytest.raises(NotImplementedError):
            NoAcquireWeightsValidation().run(builder, res_mgr)

    def test_acquire_without_filter_returns_instructions(self):
        model = LucyModelLoader().load()
        res_mgr = ResultManager()
        qubit = model.qubits[0]
        builder = QuantumInstructionBuilder(model)
        builder.acquire(
            target=qubit,
            delay=0.0,
            duration=1e-6,
        )
        ir = NoAcquireWeightsValidation().run(builder, res_mgr)
        assert isinstance(ir.instructions[0], Acquire)
