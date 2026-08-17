# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod
from typing import ClassVar

from xdsl.irdl import Operation, SSAValue

from qat.experimental.waveforms.shapes.base import WaveformShape


class IsAnalyticalWaveformInterface(Operation, ABC):
    """Marks operations that produce waveforms via an analytical definition.

    Operations implementing this interface know how to construct the
    :class:`~qat.experimental.waveforms.shapes.base.WaveformShape` they represent from
    their own shape-specific operands and properties.
    """

    # By convention and xDSL enforcement, this class variable name has to be capitalised
    WAVEFORM_NAME: ClassVar[str]
    """The string representation of the waveform which acts as a hook for waveform
    information that lives outside the IR."""

    @abstractmethod
    def build_shape(self) -> WaveformShape | None:
        """Build the waveform shape for this op from shape-specific operands.

        Amplitude, duration, and DRAG coefficients are handled by the waveform evaluation
        pass. Returns ``None`` if any shape-defining operand is not a compile-time
        constant.

        :returns: The waveform shape instance, or ``None`` if it cannot be built.
        """
        ...

    @property
    @abstractmethod
    def amplitude(self) -> SSAValue:
        """The amplitude of the waveform produced by this operation."""
        ...

    @property
    @abstractmethod
    def width(self) -> SSAValue:
        """The width of the waveform produced by this operation."""
        ...

    @property
    @abstractmethod
    def drag_coefficients(self) -> tuple[SSAValue, ...]:
        """Optional DRAG coefficient operands for this waveform."""
        ...
