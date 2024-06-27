# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd
from dataclasses import dataclass
from typing import Dict

from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.instructions import Pulse


@dataclass
class WaveformDef:
    name: str
    type: str = "Unknown"
    description: str = "Empty"


class AbstractWaveform(Pulse):
    actual_waveforms: Dict[str, "AbstractWaveform"] = dict()
    waveform_definition: WaveformDef = None

    def __init_subclass__(cls: "AbstractWaveform"):
        if cls.waveform_definition is None:
            raise ValueError(
                f"Subclass of {AbstractWaveform.__name__} doesn't have a "
                "waveform_definition."
            )

        normalized_name = cls.waveform_definition.name.lower().replace(" ", "_")
        AbstractWaveform.actual_waveforms[normalized_name] = cls


class SquareWaveform(AbstractWaveform):
    waveform_definition: WaveformDef = WaveformDef(name="Square")

    def __init__(self, channel, width, amp, ignore_channel_scale):
        super().__init__(channel, PulseShapeType.SQUARE, width, amp, ignore_channel_scale)


class GaussianWaveform(AbstractWaveform):
    waveform_definition: WaveformDef = WaveformDef(name="Gaussian")

    def __init__(self, channel, width, amp, ignore_channel_scale):
        super().__init__(channel, PulseShapeType.GAUSSIAN, width, amp, ignore_channel_scale)


class SoftSquareWaveform(AbstractWaveform):
    waveform_definition: WaveformDef = WaveformDef(name="Soft square")

    def __init__(self, channel, width, amp, ignore_channel_scale):
        super().__init__(
            channel, PulseShapeType.SOFT_SQUARE, width, amp, ignore_channel_scale
        )


class BlackmanWaveform(AbstractWaveform):
    waveform_definition: WaveformDef = WaveformDef(name="Blackman")

    def __init__(self, channel, width, amp, ignore_channel_scale):
        super().__init__(channel, PulseShapeType.BLACKMAN, width, amp, ignore_channel_scale)


class SetupHoldWaveform(AbstractWaveform):
    waveform_definition: WaveformDef = WaveformDef(name="Setup hold")

    def __init__(self, channel, width, amp, ignore_channel_scale):
        super().__init__(
            channel, PulseShapeType.SETUP_HOLD, width, amp, ignore_channel_scale
        )


class SofterSquareWaveform(AbstractWaveform):
    waveform_definition: WaveformDef = WaveformDef(name="Softer square")

    def __init__(self, channel, width, amp, ignore_channel_scale):
        super().__init__(
            channel, PulseShapeType.SOFTER_SQUARE, width, amp, ignore_channel_scale
        )


class ExtraSoftSquareWaveform(AbstractWaveform):
    waveform_definition: WaveformDef = WaveformDef(name="Extra soft square")

    def __init__(self, channel, width, amp, ignore_channel_scale):
        super().__init__(
            channel, PulseShapeType.EXTRA_SOFT_SQUARE, width, amp, ignore_channel_scale
        )


class SofterGaussianWaveform(AbstractWaveform):
    waveform_definition: WaveformDef = WaveformDef(name="Softer Gaussian")

    def __init__(self, channel, width, amp, ignore_channel_scale):
        super().__init__(
            channel, PulseShapeType.SOFTER_GAUSSIAN, width, amp, ignore_channel_scale
        )


def get_waveform_type(wf_name: str):
    return AbstractWaveform.actual_waveforms.get(wf_name, None)


def build_waveform(wf_name: str, *args, **kwargs):
    wf = AbstractWaveform.actual_waveforms.get(wf_name, None)
    if wf is not None:
        return wf(*args, **kwargs)
    return None
