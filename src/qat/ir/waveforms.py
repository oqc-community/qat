# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from typing import Literal

from pydantic import BaseModel

from qat.ir.instructions import Pulse
from qat.purr.compiler.devices import PulseShapeType


class WaveformDef(BaseModel):
    # from what I can tell, type and udescription are never used...
    name: str
    type: str = "Unknown"
    description: str = "Empty"


class AbstractWaveform(Pulse):
    inst: Literal["AbstractWaveform"] = "AbstractWaveform"
    waveform_definition: WaveformDef = None


class SquareWaveform(AbstractWaveform):
    inst: Literal["SquareWaveform"] = "SquareWaveform"
    waveform_definition: WaveformDef = WaveformDef(name="Gaussian")

    def __init__(self, quantum_targets, width, amp, ignore_channel_scale, **kwargs):
        super().__init__(
            quantum_targets,
            PulseShapeType.SQUARE,
            width,
            amp=amp,
            ignore_channel_scale=ignore_channel_scale,
        )


class GaussianWaveform(AbstractWaveform):
    inst: Literal["Gaussianform"] = "GaussianWaveform"
    waveform_definition: WaveformDef = WaveformDef(name="Gaussian")

    def __init__(self, quantum_targets, width, amp, ignore_channel_scale, **kwargs):
        super().__init__(
            quantum_targets,
            PulseShapeType.GAUSSIAN,
            width,
            amp=amp,
            ignore_channel_scale=ignore_channel_scale,
        )


class SoftSquareWaveform(AbstractWaveform):
    inst: Literal["SoftSquareWaveform"] = "SoftSquareWaveform"
    waveform_definition: WaveformDef = WaveformDef(name="Soft square")

    def __init__(self, quantum_targets, width, amp, ignore_channel_scale, **kwargs):
        super().__init__(
            quantum_targets,
            PulseShapeType.SOFT_SQUARE,
            width,
            amp=amp,
            ignore_channel_scale=ignore_channel_scale,
        )


class BlackmanWaveform(AbstractWaveform):
    inst: Literal["BlackmanWaveform"] = "BlackmanWaveform"
    waveform_definition: WaveformDef = WaveformDef(name="Blackman")

    def __init__(self, quantum_targets, width, amp, ignore_channel_scale, **kwargs):
        super().__init__(
            quantum_targets,
            PulseShapeType.BLACKMAN,
            width,
            amp=amp,
            ignore_channel_scale=ignore_channel_scale,
        )


class SetupHoldWaveform(AbstractWaveform):
    inst: Literal["SetupHoldWaveform"] = "SetupHoldWaveform"
    waveform_definition: WaveformDef = WaveformDef(name="Setup hold")

    def __init__(self, quantum_targets, width, amp, ignore_channel_scale, **kwargs):
        super().__init__(
            quantum_targets,
            PulseShapeType.SETUP_HOLD,
            width,
            amp=amp,
            ignore_channel_scale=ignore_channel_scale,
        )


class SofterSquareWaveform(AbstractWaveform):
    inst: Literal["SofterSquareWaveform"] = "SofterSquareWaveform"
    waveform_definition: WaveformDef = WaveformDef(name="Softer square")

    def __init__(self, quantum_targets, width, amp, ignore_channel_scale, **kwargs):
        super().__init__(
            quantum_targets,
            PulseShapeType.SOFTER_SQUARE,
            width,
            amp=amp,
            ignore_channel_scale=ignore_channel_scale,
        )


class ExtraSoftSquareWaveform(AbstractWaveform):
    inst: Literal["ExtraSoftSquareWaveform"] = "ExtraSoftSquareWaveform"
    waveform_definition: WaveformDef = WaveformDef(name="Extra soft square")

    def __init__(self, quantum_targets, width, amp, ignore_channel_scale, **kwargs):
        super().__init__(
            quantum_targets,
            PulseShapeType.EXTRA_SOFT_SQUARE,
            width,
            amp=amp,
            ignore_channel_scale=ignore_channel_scale,
        )


class SofterGaussianWaveform(AbstractWaveform):
    inst: Literal["SofterGaussianWaveform"] = "SofterGaussianWaveform"
    waveform_definition: WaveformDef = WaveformDef(name="Softer Gaussian")

    def __init__(self, quantum_targets, width, amp, ignore_channel_scale, **kwargs):
        super().__init__(
            quantum_targets,
            PulseShapeType.SOFTER_GAUSSIAN,
            width,
            amp=amp,
            ignore_channel_scale=ignore_channel_scale,
        )
