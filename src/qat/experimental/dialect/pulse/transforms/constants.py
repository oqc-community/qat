# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
from abc import ABC, abstractmethod
from enum import Enum
from math import isclose

import numpy as np
from xdsl.dialects.builtin import FloatAttr, IntegerAttr
from xdsl.dialects.complex import ComplexNumberAttr
from xdsl.ir import TypeAttribute
from xdsl.irdl import Attribute
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern, op_type_rewrite_pattern
from xdsl.traits import ConstantLike
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.pulse.ir import (
    AmplitudeAttr,
    AmplitudeType,
    BinaryOp,
    ConstantOp,
    FrequencyAttr,
    FrequencyType,
    MaxTimeOp,
    PhaseAttr,
    PhaseType,
    PulseNumericTypedAttr,
    SampledWaveformAttr,
    TimeAttr,
    TimeType,
    WaveformType,
)
from qat.experimental.dialect.pulse.units import (
    FREQUENCY_UNIT_EXPONENTS,
    TIME_UNIT_EXPONENTS,
)


class PulseConstantFoldAdapter(ABC):
    """Abstract base class for constant folding adapters."""

    @abstractmethod
    def fold(self, op: BinaryOp, lhs: Attribute, rhs: Attribute) -> Attribute | None:
        """Using the context of the binary operation, folds two attributes from constant-
        like operations into a single constant attribute if possible, returning None if the
        attributes cannot be folded."""
        ...

    def _unpack_value(self, attr: Attribute) -> np.ndarray | int | float | complex | None:
        """Helper function to unpack the value from a supported attribute, returning it as a
        native Python type that can be used for folding operations.

        Returns None if the attribute type is not supported.
        """

        match attr:
            case PulseNumericTypedAttr() | SampledWaveformAttr():
                return attr.literal_value
            case ComplexNumberAttr():
                return complex(attr.real.data, attr.imag.data)
            case FloatAttr() | IntegerAttr():
                return attr.value.data
        return None


class ScalarConstantFoldAdapter(PulseConstantFoldAdapter):
    """Base class for folding binary operations on scalar attributes in the pulse dialect.

    Unpacks the values from the attributes, promotes units if necessary, applies the binary
    operation, and then constructs a new attribute with the result.
    """

    def __init__(
        self, attr_constructor: type[Attribute], exponent_map: dict[Enum, int] | None = None
    ):
        """
        :param attr_constructor: A callable that takes the result of the binary operation
            (and unit if applicable) and returns a new attribute.
        :param exponent_map: An optional mapping from unit enum values to decimal exponents
            If provided, the adapter will align the units of the two attributes using these
            exponents before performing the binary operation.
        """
        self.attr_constructor = attr_constructor
        self.exponent_map = exponent_map

    def _extract(self, attr: Attribute):
        """Helper function to extract the value and unit from an attribute, if
        applicable."""
        if self.exponent_map and isinstance(attr, self.attr_constructor):
            return attr.value.data, attr.unit.data
        return self._unpack_value(attr), None

    def _align_units(self, lhs_value, lhs_unit, rhs_value, rhs_unit):
        """Helper function to align the units of two attributes based on the provided
        UnitSpec, returning the potentially modified values and the resulting unit to use
        for the folded attribute."""

        if not self.exponent_map:
            return lhs_value, rhs_value, None

        exp = self.exponent_map

        if rhs_unit is None:
            return lhs_value, rhs_value, lhs_unit
        if lhs_unit is None:
            return lhs_value, rhs_value, rhs_unit
        if lhs_unit == rhs_unit:
            return lhs_value, rhs_value, lhs_unit

        if exp[lhs_unit] < exp[rhs_unit]:
            rhs_value *= 10 ** (exp[rhs_unit] - exp[lhs_unit])
            return lhs_value, rhs_value, lhs_unit
        else:
            lhs_value *= 10 ** (exp[lhs_unit] - exp[rhs_unit])
            return lhs_value, rhs_value, rhs_unit

    def fold(self, op: BinaryOp, lhs: Attribute, rhs: Attribute) -> Attribute | None:
        """Folds two scalar attributes, potentially with units, into a single attribute if
        possible, returning None if the attributes cannot be folded."""

        lhs_value, lhs_unit = self._extract(lhs)
        rhs_value, rhs_unit = self._extract(rhs)

        if lhs_value is None or rhs_value is None:
            return None

        lhs_value, rhs_value, out_unit = self._align_units(
            lhs_value, lhs_unit, rhs_value, rhs_unit
        )
        result = op.py_operation(lhs_value, rhs_value)

        if self.exponent_map:
            return self.attr_constructor(result, out_unit)
        return self.attr_constructor(result)


class WaveformConstantFoldAdapter(PulseConstantFoldAdapter):
    """Adapter for folding binary operations on SampledWaveformAttrs, which requires special
    handling to ensure that the widths and sample times of the waveforms are compatible."""

    def fold(self, op: BinaryOp, lhs: Attribute, rhs: Attribute) -> Attribute | None:
        """Folds two attributes, one of which being a SampledWaveformAttr, into a single
        SampledWaveformAttr if possible, returning None if the attributes cannot be
        folded."""

        if isinstance(lhs, SampledWaveformAttr) and isinstance(rhs, SampledWaveformAttr):
            if not isclose(lhs.width.literal_value, rhs.width.literal_value):
                raise PassFailedException(
                    "Cannot fold two SampledWaveformAttrs with different widths."
                )
            if not isclose(lhs.sample_time.literal_value, rhs.sample_time.literal_value):
                return None

            # Pick the width and sample time ambiguously from either operand
            width = lhs.width
            sample_time = lhs.sample_time
        else:
            waveform_attr = lhs if isinstance(lhs, SampledWaveformAttr) else rhs
            width = waveform_attr.width
            sample_time = waveform_attr.sample_time

        lhs_value = self._unpack_value(lhs)
        rhs_value = self._unpack_value(rhs)
        if lhs_value is None or rhs_value is None:
            return None

        return SampledWaveformAttr(
            op.py_operation(lhs_value, rhs_value), width, sample_time
        )


_ADAPTERS: dict[TypeAttribute, PulseConstantFoldAdapter] = {
    AmplitudeType(): ScalarConstantFoldAdapter(AmplitudeAttr),
    PhaseType(): ScalarConstantFoldAdapter(PhaseAttr),
    TimeType(): ScalarConstantFoldAdapter(TimeAttr, TIME_UNIT_EXPONENTS),
    FrequencyType(): ScalarConstantFoldAdapter(FrequencyAttr, FREQUENCY_UNIT_EXPONENTS),
    WaveformType(): WaveformConstantFoldAdapter(),
}


class FoldConstantConstantOp(RewritePattern):
    """Finds arithmetic operations whose operands are both constant-like and folds them into
    a single :class:`ConstantOp` with the result of the operation as its value.

    This is hooked in from canonicalization passes, and eliminates redundant arithmetic
    operations in the pulse dialect.

    For example,

    .. code-block:: mlir

        %frequency1 = pulse.constant<5e9> : !pulse.frequency
        %frequency2 = pulse.constant<1e9> : !pulse.frequency
        %result = pulse.add(%frequency1, %frequency2) : !pulse.frequency

    can be replaced with

    .. code-block ::mlir

        %result = pulse.constant<6e9> : !pulse.frequency
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: BinaryOp, rewriter: PatternRewriter):
        """Matches and applies the rewrite if an appropriate condition is found."""

        lhs = op.lhs.owner
        rhs = op.rhs.owner

        if not (lhs.has_trait(ConstantLike) and rhs.has_trait(ConstantLike)):
            return

        result_type = op.result.type
        lhs_value = lhs.fold()
        rhs_value = rhs.fold()

        if not lhs_value:
            return
        if not rhs_value:
            return

        lhs_value = lhs_value[0]
        rhs_value = rhs_value[0]

        adapter = _ADAPTERS.get(result_type)
        if adapter is None:
            return

        folded_attr = adapter.fold(op, lhs_value, rhs_value)
        if folded_attr is None:
            return

        folded_op = ConstantOp(folded_attr, result_type)
        rewriter.replace_op(op, folded_op)


class FoldMaxTimeOp(RewritePattern):
    """Finds :class:`MaxTimeOp` that has constant operands and folds it into the constant
    with maximum time."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: MaxTimeOp, rewriter: PatternRewriter):
        """Matches and applies the rewrite if an appropriate condition is found."""

        operands = [operand.owner for operand in op.operands]
        if not all(isinstance(operand, ConstantOp) for operand in operands):
            return

        operand_values = [operand.fold()[0].literal_value for operand in operands]
        max_index = np.argmax(operand_values)
        rewriter.replace_op(op, [], operands[max_index].results)
