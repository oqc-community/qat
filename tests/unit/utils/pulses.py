# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np

from qat.ir.waveforms import (
    BlackmanWaveform,
    CosWaveform,
    ExtraSoftSquareWaveform,
    GaussianWaveform,
    GaussianZeroEdgeWaveform,
    RoundedSquareWaveform,
    SechWaveform,
    SetupHoldWaveform,
    SinWaveform,
    SofterGaussianWaveform,
    SofterSquareWaveform,
    SoftSquareWaveform,
    SquareWaveform,
)
from qat.purr.compiler.instructions import PulseShapeType

# Different pulse shapes with properties chosen to give reasonable waveforms
pulse_attributes = [
    {"shape": PulseShapeType.GAUSSIAN, "rise": 1 / 3},
    {"shape": PulseShapeType.SOFTER_GAUSSIAN, "rise": 1 / 3},
    {
        "shape": PulseShapeType.GAUSSIAN_DRAG,
        "std_dev": 1 / 3,
        "zero_at_edges": True,
        "beta": 0.5,
    },
    {
        "shape": PulseShapeType.GAUSSIAN_ZERO_EDGE,
        "std_dev": 1 / 3,
        "zero_at_edges": True,
    },
    {"shape": PulseShapeType.SOFT_SQUARE, "rise": 50e-9},
    {"shape": PulseShapeType.SOFTER_SQUARE, "rise": 100e-9},
    {"shape": PulseShapeType.EXTRA_SOFT_SQUARE, "rise": 100e-9},
    {"shape": PulseShapeType.ROUNDED_SQUARE, "rise": 10e-9, "std_dev": 50e-9},
    {
        "shape": PulseShapeType.BLACKMAN,
    },
    {
        "shape": PulseShapeType.SETUP_HOLD,
        "rise": 100e-9,
        "amp_setup": 0.5,
        "std_dev": 100e-9,
    },
    {"shape": PulseShapeType.SECH, "std_dev": 100e-9},
    {
        "shape": PulseShapeType.SIN,
        "frequency": 2 / 400e-9,
        "internal_phase": np.pi / 2,
    },
    {
        "shape": PulseShapeType.COS,
        "frequency": 2 / 400e-9,
        "internal_phase": np.pi / 2,
    },
]

test_waveforms = [
    SquareWaveform(amp=0.5, width=400e-9),
    GaussianWaveform(width=400e-9, amp=0.5, rise=1 / 3),
    SofterGaussianWaveform(width=400e-9, amp=0.5, rise=1 / 3),
    GaussianZeroEdgeWaveform(width=400e-9, amp=0.5, std_dev=1 / 3, zero_at_edges=True),
    SoftSquareWaveform(amp=0.5, width=400e-9, rise=50e-9),
    SofterSquareWaveform(amp=0.5, width=400e-9, rise=100e-9, std_dev=200e-9),
    ExtraSoftSquareWaveform(amp=0.5, width=400e-9, rise=100e-9, std_dev=200e-9),
    RoundedSquareWaveform(amp=0.5, width=400e-9, rise=10e-9, std_dev=50e-9),
    BlackmanWaveform(amp=0.5, width=400e-9),
    SetupHoldWaveform(amp=0.5, width=400e-9, rise=100e-9, amp_setup=0.5, std_dev=100e-9),
    SechWaveform(amp=0.5, width=400e-9, std_dev=100e-9),
    SinWaveform(amp=0.5, width=400e-9, frequency=2 / 400e-9, internal_phase=np.pi / 2),
    CosWaveform(amp=0.5, width=400e-9, frequency=2 / 400e-9, internal_phase=np.pi / 2),
]
