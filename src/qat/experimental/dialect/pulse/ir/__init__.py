# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""The Pulse dialect represents the operation set to interact with quantum mechanical
systems through microwave interactions.

It is modelled by waveforms that are used to drive the systems, and can be captured as a
response from the systems. Pulses must be played down a reference frame, which encodes a
frequency that typically matches some resonant frequency of the system. The frame tracks
phase accumulations, including offsets from explicit phase manipulation instructions. They
also implicitly track time, as pulse interactions must be played at the right time.

The dialect includes a number of operations to define waveforms and related properties, and
arithmetic on those types. The waveforms and related types are used to manipulate the state
of a frame, and play / acquire waveforms on a physical channel in that reference frame.
"""

from xdsl.ir import Dialect

from .attributes import (
    AmplitudeAttr,
    FrequencyAttr,
    FrequencyUnitsData,
    PhaseAttr,
    PulseNumericTypedAttr,
    SampledWaveformAttr,
    TimeAttr,
    TimeUnitsData,
    WaveformData,
)
from .interfaces import IsAnalyticalWaveformInterface
from .ops import (
    AcquireOp,
    AddOp,
    BinaryOp,
    BlackmanWaveformOp,
    ConstantOp,
    CosWaveformOp,
    CreateFrameOp,
    DragGaussianWaveformOp,
    ExtraSoftSquareWaveformOp,
    GaussianSquareWaveformOp,
    GaussianWaveformOp,
    GaussianZeroEdgeWaveformOp,
    IntegrateOp,
    InternalBinaryOp,
    MaxTimeOp,
    ModulateOp,
    ModuloOp,
    PhaseOp,
    PhaseSetOp,
    PhaseShiftOp,
    PulseOp,
    RoundedSquareWaveformOp,
    ScaleOp,
    SechWaveformOp,
    SetupHoldWaveformOp,
    SinWaveformOp,
    SofterGaussianWaveformOp,
    SofterSquareWaveformOp,
    SoftSquareWaveformOp,
    SquareWaveformOp,
    StartContinuousWaveformOp,
    StopContinuousWaveformOp,
    SubOp,
    SynchronizeOp,
    WaitOp,
)
from .traits import (
    AdvancesTimeTrait,
    FrameCanonicalizationPatternsTrait,
    PulseTypesCanonicalizationPatternsTrait,
)
from .types import (
    AcquisitionType,
    AmplitudeType,
    FrameType,
    FrequencyType,
    IQResultType,
    PhaseType,
    TimeType,
    WaveformType,
)

_ops = [
    AcquireOp,
    AddOp,
    BlackmanWaveformOp,
    ConstantOp,
    CosWaveformOp,
    CreateFrameOp,
    DragGaussianWaveformOp,
    ExtraSoftSquareWaveformOp,
    GaussianSquareWaveformOp,
    GaussianWaveformOp,
    GaussianZeroEdgeWaveformOp,
    IntegrateOp,
    MaxTimeOp,
    ModulateOp,
    ModuloOp,
    PhaseSetOp,
    PhaseShiftOp,
    PulseOp,
    RoundedSquareWaveformOp,
    ScaleOp,
    SechWaveformOp,
    SetupHoldWaveformOp,
    SinWaveformOp,
    SofterGaussianWaveformOp,
    SofterSquareWaveformOp,
    SoftSquareWaveformOp,
    SquareWaveformOp,
    StartContinuousWaveformOp,
    StopContinuousWaveformOp,
    SubOp,
    SynchronizeOp,
    WaitOp,
]

_interfaces = [IsAnalyticalWaveformInterface]

_traits = [
    AdvancesTimeTrait,
    FrameCanonicalizationPatternsTrait,
    PulseTypesCanonicalizationPatternsTrait,
]

_types = [
    AcquisitionType,
    AmplitudeType,
    FrequencyType,
    FrameType,
    IQResultType,
    PhaseType,
    TimeType,
    WaveformType,
]

_data_attributes = [WaveformData, TimeUnitsData, FrequencyUnitsData]

_attributes = [AmplitudeAttr, FrequencyAttr, PhaseAttr, TimeAttr, SampledWaveformAttr]

Pulse = Dialect(
    "pulse",
    _ops,
    _types + _data_attributes + _attributes,
    _traits + _interfaces,
)

_all_classes = _ops + _interfaces + _traits + _types + _data_attributes + _attributes

ANALYTICAL_WAVEFORM_OPS = tuple(IsAnalyticalWaveformInterface.__subclasses__())

__all__ = [
    "ANALYTICAL_WAVEFORM_OPS",
    "BinaryOp",
    "InternalBinaryOp",
    "PhaseOp",
    "Pulse",
    "PulseNumericTypedAttr",
] + [cls_.__name__ for cls_ in _all_classes]
