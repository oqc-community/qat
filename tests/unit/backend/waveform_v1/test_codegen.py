# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import pytest

from qat.backend.waveform_v1.codegen import WaveformV1Backend, WaveformV1Executable
from qat.middleend.passes.legacy.transform import (
    RepeatTranslation,
)
from qat.middleend.passes.transform import LowerSyncsToDelays
from qat.model.loaders.legacy import EchoModelLoader


class TestWaveformV1Executable:
    def test_same_after_serialize_deserialize_roundtrip(self):
        model = EchoModelLoader(10).load()
        builder = model.create_builder()
        builder.repeat(1000, 100e-6)
        builder.had(model.get_qubit(0))
        for i in range(9):
            builder.cnot(model.get_qubit(i), model.get_qubit(i + 1))
        # backend is not expected to see syncs, so lets remove them using this pass for
        # ease
        builder = LowerSyncsToDelays().run(builder)
        executable = WaveformV1Backend(model).emit(builder)
        blob = executable.serialize()
        new_executable = WaveformV1Executable.deserialize(blob)
        assert executable == new_executable


class TestWaveformV1Backend:
    @pytest.mark.parametrize("repetition_period", [150e-6, None])
    @pytest.mark.parametrize("repeat_translation", [True, False])
    def test_repeat_handeling(self, repetition_period, repeat_translation):
        model = EchoModelLoader(10).load()
        builder = model.create_builder()
        builder.repeat(1000, repetition_period)
        builder.had(model.get_qubit(0))
        for i in range(9):
            builder.cnot(model.get_qubit(i), model.get_qubit(i + 1))
        # backend is not expected to see syncs, so lets remove them using this pass for
        # ease
        builder = LowerSyncsToDelays().run(builder)
        if repeat_translation:
            builder = RepeatTranslation(model).run(builder)
        executable = WaveformV1Backend(model).emit(builder)
        assert isinstance(executable, WaveformV1Executable)
        assert executable.shots == 1000
        if repetition_period is not None:
            assert executable.repetition_time == repetition_period
        else:
            assert executable.repetition_time == model.default_repetition_period
