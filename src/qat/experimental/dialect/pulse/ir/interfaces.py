# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd

from abc import ABC, abstractmethod
from typing import ClassVar

from xdsl.dialects.builtin import FloatAttr
from xdsl.irdl import Operation, SSAValue
from xdsl.traits import ConstantLike

from qat.experimental.dialect.pulse.ir.attributes import PulseNumericTypedAttr
from qat.ir.waveforms import Waveform


def extract_constant_scalar(ssa: SSAValue) -> float | complex | None:
    """Return the Python scalar behind ``ssa`` if it is a compile-time constant.

    Handles both pulse-dialect ``ConstantOp`` values (which fold to a
    :class:`PulseNumericTypedAttr`) and standard ``arith.constant`` values (which
    fold to a :class:`FloatAttr`). Returns ``None`` otherwise.

    Complex values whose imaginary part is exactly zero are narrowed to ``float``,
    so waveform fields typed strictly as ``float`` accept scalars extracted from an
    :class:`AmplitudeAttr`, which always stores its literal value as ``complex``.
    """

    attr = ConstantLike.get_constant_value(ssa)
    if isinstance(attr, PulseNumericTypedAttr):
        value = attr.literal_value
    elif isinstance(attr, FloatAttr):
        value = attr.value.data
    else:
        return None
    if isinstance(value, complex) and value.imag == 0:
        return value.real
    return value


class IsAnalyticalWaveformInterface(Operation, ABC):
    """Marks operations that produce waveforms via an analytical definition.

    Operations implementing this interface know how to construct the pydantic
    :class:`Waveform` they represent from their own operands and properties by
    extracting compile-time-constant scalars from their SSA operands.
    """

    # By convention and xDSL enforcement, this class variable name has to be capitalised
    WAVEFORM_NAME: ClassVar[str]
    """The string representation of the waveform which acts as a hook for waveform
    information that lives outside the IR."""

    @abstractmethod
    def build_waveform(self) -> Waveform | None:
        """Build the pydantic :class:`Waveform` this op represents.

        Returns ``None`` if at least one of the op's SSA operands is not a
        compile-time constant, in which case the waveform must be left for runtime
        evaluation.

        :returns: The pydantic waveform instance, or ``None`` if it cannot be built.
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
