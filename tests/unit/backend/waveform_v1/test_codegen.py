# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd


from qat.backend.waveform_v1.codegen import WaveformV1Backend, WaveformV1Executable
from qat.model.loaders.legacy import EchoModelLoader


class TestWaveformV1Executable:
    def test_same_after_serialize_deserialize_roundtrip(self):
        model = EchoModelLoader(10).load()
        builder = model.create_builder()
        builder.repeat(1000, 100e-6)
        builder.had(model.get_qubit(0))
        for i in range(9):
            builder.cnot(model.get_qubit(i), model.get_qubit(i + 1))

        executable = WaveformV1Backend(model).emit(builder)
        blob = executable.serialize()
        new_executable = WaveformV1Executable.deserialize(blob)
        assert executable == new_executable
