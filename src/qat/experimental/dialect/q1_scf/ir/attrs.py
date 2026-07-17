# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Attributes for the ``q1_scf`` structured control flow dialect.

The dialect reuses the ``q1_cf`` predicate enumerations for its conditions (see
:mod:`qat.experimental.dialect.q1_cf.ir.attrs`) and defines one attribute of its
own:

- :class:`IterDomainAttr` is optional metadata on
  :class:`~qat.experimental.dialect.q1_scf.ir.ops.ForOp` describing the linear
  iteration domain of a counted loop. It records the ``start``, ``stop`` and
  ``step`` of the varied quantity, the integer ``count`` of iterations, and an
  :class:`IterParameter` tag naming the varied quantity and its unit. The attribute
  carries no control-flow semantics.

Reference: https://docs.qblox.com/en/main/products/qblox_instruments/q1/index.html
"""

from __future__ import annotations

import math
from enum import auto

from xdsl.dialects.builtin import FloatAttr, IntAttr
from xdsl.ir import (
    EnumAttribute,
    ParametrizedAttribute,
    SpacedOpaqueSyntaxAttribute,
    StrEnum,
)
from xdsl.irdl import irdl_attr_definition, param_def
from xdsl.utils.exceptions import VerifyException


class IterParameter(StrEnum):
    """The quantity varied across iterations, which also fixes the domain unit."""

    frequency = auto()  # Hz
    phase = auto()  # rad
    gain = auto()  # dimensionless, normalised DAC scale
    duration = auto()  # ns


@irdl_attr_definition
class IterParameterAttr(EnumAttribute[IterParameter], SpacedOpaqueSyntaxAttribute):
    """Attribute wrapper carrying an :class:`IterParameter`."""

    name = "q1_scf.iter_parameter"


@irdl_attr_definition
class IterDomainAttr(ParametrizedAttribute):
    """Linear iteration domain of a :class:`~qat.experimental.dialect.q1_scf.ir.ops.ForOp`.

    The domain is the affine sequence ``start, start + step, ...`` truncated to
    ``count`` points, all measured in the unit implied by ``parameter``. It records
    the physical meaning of a counted loop and does not alter its control-flow
    semantics.

    :param start: First value of the varied quantity.
    :param stop: Exclusive upper bound of the varied quantity.
    :param step: Increment between successive iterations; must be non-zero.
    :param count: Number of loop iterations, which must equal ``ceil((stop - start) / step)``.
    :param parameter: The varied quantity and, implicitly, its unit.
    """

    name = "q1_scf.iter_domain"

    start: FloatAttr = param_def(FloatAttr)
    stop: FloatAttr = param_def(FloatAttr)
    step: FloatAttr = param_def(FloatAttr)
    count: IntAttr = param_def(IntAttr)
    parameter: IterParameterAttr = param_def(IterParameterAttr)

    def __init__(
        self,
        start: float,
        stop: float,
        step: float,
        count: int,
        parameter: IterParameter,
    ):
        super().__init__(
            FloatAttr(start, 64),
            FloatAttr(stop, 64),
            FloatAttr(step, 64),
            IntAttr(count),
            IterParameterAttr(parameter),
        )

    def verify(self) -> None:
        """Verify the domain is a well-formed linear iteration.

        :raises VerifyException: If ``step`` is zero, if ``count`` is negative, or
            if ``count`` does not equal ``ceil((stop - start) / step)``.
        """
        start = self.start.value.data
        stop = self.stop.value.data
        step = self.step.value.data
        if step == 0.0:
            raise VerifyException("q1_scf.iter_domain step must be non-zero")

        if self.count.data < 0:
            raise VerifyException(
                f"q1_scf.iter_domain count {self.count.data} must be non-negative"
            )

        expected = math.ceil((stop - start) / step)
        if self.count.data != expected:
            raise VerifyException(
                f"q1_scf.iter_domain count {self.count.data} does not equal"
                f" ceil((stop - start) / step) = {expected}"
            )
