# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import numpy as np

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
