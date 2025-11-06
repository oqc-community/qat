# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd
import numpy as np
import pytest
from pydantic_core import from_json, to_json

from qat.backend.waveform_v1.executable import (
    PositionalAcquireData,
    WaveformV1ChannelData,
    WaveformV1Program,
)
from qat.ir.measure import AcquireMode


class TestWaveformV1Program:
    @pytest.fixture
    def program(self):
        channel_data_1 = WaveformV1ChannelData(
            buffer=np.random.rand(254) + 1j * np.random.rand(254),
            baseband_frequency=5.5e9,
            acquires=[
                PositionalAcquireData(
                    mode=AcquireMode.INTEGRATOR,
                    position=0,
                    length=1000,
                    output_variable="acq1",
                ),
                PositionalAcquireData(
                    mode=AcquireMode.RAW,
                    position=200,
                    length=51,
                    output_variable="acq2",
                ),
            ],
        )
        channel_data_2 = WaveformV1ChannelData(
            buffer=np.random.rand(128) + 1j * np.random.rand(128),
            baseband_frequency=6.0e9,
            acquires=[
                PositionalAcquireData(
                    mode=AcquireMode.SCOPE,
                    position=64,
                    length=64,
                    output_variable="acq3",
                ),
            ],
        )

        return WaveformV1Program(
            channel_data={
                "channel_1": channel_data_1,
                "channel_2": channel_data_2,
            },
            repetition_time=1e-3,
            shots=1024,
        )

    def test_same_after_serialize_deserialize_roundtrip(self, program):
        blob = to_json(program)
        deserialized = WaveformV1Program(**from_json(blob))
        assert deserialized == program

    def test_acquire_shapes(self, program):
        shapes = program.acquire_shapes
        assert shapes == {
            "acq1": (1024,),
            "acq2": (1024, 51),
            "acq3": (64,),
        }
