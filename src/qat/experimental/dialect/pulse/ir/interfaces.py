# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod

from xdsl.irdl import Operation, SSAValue

from qat.ir.waveforms import Waveform


class IsAnalyticalWaveformInterface(Operation, ABC):
    """Marks operations that produce waveforms as an analytical definition.

    Operations with this interface need to implement a method that returns the waveform type
    that can be used to evaluate the shape of the waveform.
    """

    @property
    @abstractmethod
    def waveform_type(self) -> type[Waveform]:
        """The type of the waveform shape that this operation produces.

        This is used to determine how to evaluate the shape of the waveform.
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
