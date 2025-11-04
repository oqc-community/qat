# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import pytest

from qat.purr.compiler.devices import PhysicalBaseband, PhysicalChannel, PulseChannel
from qat.purr.compiler.instructions import PulseShapeType
from qat.purr.compiler.waveforms import (
    BlackmanWaveform,
    ExtraSoftSquareWaveform,
    GaussianWaveform,
    SetupHoldWaveform,
    SofterGaussianWaveform,
    SofterSquareWaveform,
    SoftSquareWaveform,
    SquareWaveform,
)


@pytest.mark.parametrize("width", [80e-9, 160e-9])
@pytest.mark.parametrize("amp", [1e-2, 1e-1])
@pytest.mark.parametrize("ignore_channel_scale", [True, False])
@pytest.mark.parametrize(
    "waveform_class, expected_shape",
    [
        (SquareWaveform, PulseShapeType.SQUARE),
        (GaussianWaveform, PulseShapeType.GAUSSIAN),
        (SoftSquareWaveform, PulseShapeType.SOFT_SQUARE),
        (BlackmanWaveform, PulseShapeType.BLACKMAN),
        (SetupHoldWaveform, PulseShapeType.SETUP_HOLD),
        (SofterSquareWaveform, PulseShapeType.SOFTER_SQUARE),
        (ExtraSoftSquareWaveform, PulseShapeType.EXTRA_SOFT_SQUARE),
        (SofterGaussianWaveform, PulseShapeType.SOFTER_GAUSSIAN),
    ],
)
def test_abstract_waveform(
    waveform_class, expected_shape, width, amp, ignore_channel_scale
):
    """Regression test for wrong constructors in waveform subclasses."""
    baseband = PhysicalBaseband("BB1", 6e9)
    physical_channel = PhysicalChannel("CH1", 1e-9, baseband)
    pulse_channel = PulseChannel("test", physical_channel, frequency=5.5e9)
    waveform = waveform_class(pulse_channel, width, amp, ignore_channel_scale)

    # properties that are being set
    assert waveform.channel == pulse_channel
    assert waveform.shape == expected_shape
    assert waveform.width == width
    assert waveform.amp == amp
    assert waveform.ignore_channel_scale == ignore_channel_scale

    # misc other properties that might have been wrong before
    assert waveform.drag == 0.0
    assert waveform.phase == 0.0
    assert waveform.frequency == 0.0
    assert waveform.std_dev == 0.0
