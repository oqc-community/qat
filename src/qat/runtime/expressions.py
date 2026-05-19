# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Variables and Expressions for Quantum Program Runtime Arguments.

Provides a Pydantic-serializable representation of typed variables and expressions
(unary/binary) over those variables and literals. Intended for use in variational quantum
circuits, software-iterated sweeps, and general runtime argument injection via virtual
registers.

This module adds support for different quantum variable types, such as Phase and Frequency,
which are important for correctly interpreting their meaning in quantum programs, in
addition to standard numeric types. It also supports a variety of unary and binary
operators over variables (and literals), allowing for the construction of complex
expressions that can be evaluated at runtime. The top-level node for these expressions used
in compiled parameterised programs is :class:`RuntimeExpression`, which wraps an expression
tree and includes the necessary context for evaluation. Similarly, the
:class:`ParameterisedWaveform` class allows us to represent waveforms that are parameterised
by these expressions, which can be used to represent parameterised analytical pulse shapes
in compiled programs.
"""

from __future__ import annotations

import math
from enum import Enum
from functools import cache
from typing import Annotated, Any, TypeAlias

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    PositiveFloat,
    field_serializer,
    field_validator,
)

from qat.ir.waveforms import SampledWaveform, Waveform, sample_waveform
from qat.utils.pydantic import NoExtraFieldsFrozenModel, find_all_subclasses


class VariableType(str, Enum):
    """Supported types for program variables.

    This includes standard numeric types, but also domain-specific types such as Phase and
    Frequency, which are contextually needed to correctly interpret their meaning in quantum
    programs.
    """

    INT = "int"
    FLOAT = "float"
    COMPLEX = "complex"
    BOOL = "bool"
    PHASE = "phase"
    FREQUENCY = "frequency"
    AMPLITUDE = "amplitude"
    TIME = "time"


_PYTHON_TYPE: dict[VariableType, type] = {
    VariableType.INT: int,
    VariableType.FLOAT: float,
    VariableType.COMPLEX: complex,
    VariableType.BOOL: bool,
    VariableType.PHASE: float,
    VariableType.FREQUENCY: float,
    VariableType.AMPLITUDE: complex,
    VariableType.TIME: float,
}


class UnaryOp(str, Enum):
    """Unary operators supported in expressions.

    This only includes a basic subset of operators, but we can easily expand this in the
    future.

    :var NEG: Negation operator, does :code:`-x`
    :var ABS: Absolute value operator, does :code:`abs(x)`
    :var SIN: Sine operator, does :code:`sin(x)`
    :var COS: Cosine operator, does :code:`cos(x)`
    :var SQRT: Square root operator, does :code:`sqrt(x)`
    """

    NEG = "neg"
    ABS = "abs"
    SIN = "sin"
    COS = "cos"
    SQRT = "sqrt"


_UNARY_FN: dict[UnaryOp, Any] = {
    UnaryOp.NEG: lambda x: -x,
    UnaryOp.ABS: abs,
    UnaryOp.SIN: math.sin,
    UnaryOp.COS: math.cos,
    UnaryOp.SQRT: math.sqrt,
}


class BinaryOp(str, Enum):
    """Binary operators supported in expressions.

    For now, this only includes basic arithmetic operators, but we can easily expand this
    in the future.

    :var ADD: Addition operator, does :code:`a + b`
    :var SUB: Subtraction operator, does :code:`a - b`
    :var MUL: Multiplication operator, does :code:`a * b`
    :var DIV: Division operator, does :code:`a / b`
    :var POW: Power operator, does :code:`a ** b`
    :var MOD: Modulo operator, does :code:`a % b`
    """

    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    MOD = "mod"


_BINARY_FN: dict[BinaryOp, Any] = {
    BinaryOp.ADD: lambda a, b: a + b,
    BinaryOp.SUB: lambda a, b: a - b,
    BinaryOp.MUL: lambda a, b: a * b,
    BinaryOp.DIV: lambda a, b: a / b,
    BinaryOp.POW: lambda a, b: a**b,
    BinaryOp.MOD: lambda a, b: a % b,
}


class Literal(BaseModel):
    """A concrete numeric constant wrapped for use inside an expression tree."""

    value: int | float | complex | bool

    def evaluate(self, _params: dict[str, Any] | None = None) -> Any:
        """Evaluate the literal, which just returns its value."""
        return self.value

    def simplify(self, _params: dict[str, Any] | None = None) -> Literal:
        """Does nothing, as a literal is already as simple as it can be."""
        return self

    def __neg__(self) -> UnaryExpression:
        return UnaryExpression(operator=UnaryOp.NEG, operand=self)

    def __abs__(self) -> UnaryExpression:
        return UnaryExpression(operator=UnaryOp.ABS, operand=self)

    def __add__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.ADD, left=self, right=_wrap(other))

    def __radd__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.ADD, left=_wrap(other), right=self)

    def __sub__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.SUB, left=self, right=_wrap(other))

    def __rsub__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.SUB, left=_wrap(other), right=self)

    def __mul__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.MUL, left=self, right=_wrap(other))

    def __rmul__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.MUL, left=_wrap(other), right=self)

    def __truediv__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.DIV, left=self, right=_wrap(other))

    def __rtruediv__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.DIV, left=_wrap(other), right=self)

    def __pow__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.POW, left=self, right=_wrap(other))

    def __rpow__(self, other) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.POW, left=_wrap(other), right=self)

    def __mod__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.MOD, left=self, right=_wrap(other))

    def __rmod__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.MOD, left=_wrap(other), right=self)

    def __repr__(self) -> str:
        return f"Literal({self.value!r})"


class Variable(BaseModel):
    """A named and typed variable representing a runtime argument.

    :ivar name: Unique identifier for the variable.
    :ivar var_type: The :class:`VariableType` of this variable.
    """

    name: str
    var_type: VariableType

    @field_validator("name")
    @classmethod
    def name_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Variable name must not be empty.")
        return v

    def evaluate(self, params: dict[str, Any] | None = None) -> Any:
        """Evaluate the variable, fetching the value from params.

        :param params: Optional dict of variable values to use for evaluation.
        """
        params = params or {}
        if self.name not in params:
            raise ValueError(f"Variable '{self.name}' is not in params.")
        return _PYTHON_TYPE[self.var_type](params[self.name])

    def simplify(self, params: dict[str, Any] | None = None) -> Variable | Literal:
        """Simplifies a variable by replacing it with a literal if the value is known.

        :param params: Optional dict of variable values to use for simplification.
        """

        if params is not None and self.name in params:
            return Literal(value=self.evaluate(params))
        return self

    def __neg__(self) -> UnaryExpression:
        return UnaryExpression(operator=UnaryOp.NEG, operand=self)

    def __abs__(self) -> UnaryExpression:
        return UnaryExpression(operator=UnaryOp.ABS, operand=self)

    def __add__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.ADD, left=self, right=_wrap(other))

    def __radd__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.ADD, left=_wrap(other), right=self)

    def __sub__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.SUB, left=self, right=_wrap(other))

    def __rsub__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.SUB, left=_wrap(other), right=self)

    def __mul__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.MUL, left=self, right=_wrap(other))

    def __rmul__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.MUL, left=_wrap(other), right=self)

    def __truediv__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.DIV, left=self, right=_wrap(other))

    def __rtruediv__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.DIV, left=_wrap(other), right=self)

    def __pow__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.POW, left=self, right=_wrap(other))

    def __rpow__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.POW, left=_wrap(other), right=self)

    def __mod__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.MOD, left=self, right=_wrap(other))

    def __rmod__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.MOD, left=_wrap(other), right=self)

    def __repr__(self) -> str:
        return f"Variable(name={self.name!r}, var_type={self.var_type!r})"


class UnaryExpression(BaseModel):
    """An expression that applies a unary operator to a single operand.

    :ivar operator: The unary operator to apply.
    :ivar operand: The operand, which can be a Variable, Literal, or another expression.
    """

    operator: UnaryOp
    operand: Node

    def evaluate(self, params: dict[str, Any] | None = None) -> Any:
        """Evaluate the expression, resolving free variables from params.

        :param params: Optional dict of variable values to use for evaluation.
        """
        val = self.operand.evaluate(params or {})
        return _UNARY_FN[self.operator](val)

    def simplify(self, params: dict[str, Any] | None = None) -> Node:
        """Recursively simplify the expression.

        :param params: Dictionary of variable values used for simplification, optional.
        """

        simplified_operand = self.operand.simplify(params)
        if isinstance(simplified_operand, Literal):
            return Literal(value=_UNARY_FN[self.operator](simplified_operand.value))
        return UnaryExpression(operator=self.operator, operand=simplified_operand)

    def __neg__(self) -> UnaryExpression:
        return UnaryExpression(operator=UnaryOp.NEG, operand=self)

    def __abs__(self) -> UnaryExpression:
        return UnaryExpression(operator=UnaryOp.ABS, operand=self)

    def __add__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.ADD, left=self, right=_wrap(other))

    def __radd__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.ADD, left=_wrap(other), right=self)

    def __sub__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.SUB, left=self, right=_wrap(other))

    def __rsub__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.SUB, left=_wrap(other), right=self)

    def __mul__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.MUL, left=self, right=_wrap(other))

    def __rmul__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.MUL, left=_wrap(other), right=self)

    def __truediv__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.DIV, left=self, right=_wrap(other))

    def __rtruediv__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.DIV, left=_wrap(other), right=self)

    def __pow__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.POW, left=self, right=_wrap(other))

    def __rpow__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.POW, left=_wrap(other), right=self)

    def __mod__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.MOD, left=self, right=_wrap(other))

    def __rmod__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.MOD, left=_wrap(other), right=self)

    def __repr__(self) -> str:
        return f"UnaryExpression({self.operator.value}, {self.operand!r})"


class BinaryExpression(BaseModel):
    """A representation of a binary operation on two nodes, which might be literals,
    variables, or nested expressions.

    :ivar operator: The binary operator to apply.
    :ivar left: The left operand, which can be a Variable, Literal, or another expression.
    :ivar right: The right operand, which can be a Variable, Literal, or another expression.
    """

    operator: BinaryOp
    left: Node
    right: Node

    def evaluate(self, params: dict[str, Any] | None = None) -> Any:
        """Evaluate the expression, resolving free variables from params.

        :param params: Dictionary of variable values to use for evaluation.
        """

        params = params or {}
        lv = self.left.evaluate(params)
        rv = self.right.evaluate(params)
        return _BINARY_FN[self.operator](lv, rv)

    def simplify(self, params: dict[str, Any] | None = None) -> Node:
        """Recursively simplify the expression. If both operands reduce to :class:`Literal`
        values, evaluate immediately and return a :class:`Literal`.

        :param params: Dictionary of variable values used for simplification, optional.
        """

        params = params or {}
        lhs = self.left.simplify(params)
        rhs = self.right.simplify(params)

        if isinstance(lhs, Literal) and isinstance(rhs, Literal):
            return Literal(value=_BINARY_FN[self.operator](lhs.value, rhs.value))

        # Algebraic identities
        op = self.operator
        if isinstance(rhs, Literal):
            if op == BinaryOp.ADD and rhs.value == 0:
                return lhs
            if op == BinaryOp.SUB and rhs.value == 0:
                return lhs
            if op == BinaryOp.MUL and rhs.value == 1:
                return lhs
            if op == BinaryOp.MUL and rhs.value == 0:
                return Literal(value=0)
            if op == BinaryOp.DIV and rhs.value == 1:
                return lhs
            if op == BinaryOp.POW and rhs.value == 1:
                return lhs
            if op == BinaryOp.POW and rhs.value == 0:
                return Literal(value=1)

        if isinstance(lhs, Literal):
            if op == BinaryOp.ADD and lhs.value == 0:
                return rhs
            if op == BinaryOp.SUB and lhs.value == 0:
                return UnaryExpression(operator=UnaryOp.NEG, operand=rhs)
            if op == BinaryOp.MUL and lhs.value == 1:
                return rhs
            if op == BinaryOp.MUL and lhs.value == 0:
                return Literal(value=0)

        return BinaryExpression(operator=op, left=lhs, right=rhs)

    def __neg__(self) -> UnaryExpression:
        return UnaryExpression(operator=UnaryOp.NEG, operand=self)

    def __abs__(self) -> UnaryExpression:
        return UnaryExpression(operator=UnaryOp.ABS, operand=self)

    def __add__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.ADD, left=self, right=_wrap(other))

    def __radd__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.ADD, left=_wrap(other), right=self)

    def __sub__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.SUB, left=self, right=_wrap(other))

    def __rsub__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.SUB, left=_wrap(other), right=self)

    def __mul__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.MUL, left=self, right=_wrap(other))

    def __rmul__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.MUL, left=_wrap(other), right=self)

    def __truediv__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.DIV, left=self, right=_wrap(other))

    def __rtruediv__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.DIV, left=_wrap(other), right=self)

    def __pow__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.POW, left=self, right=_wrap(other))

    def __rpow__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.POW, left=_wrap(other), right=self)

    def __mod__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.MOD, left=self, right=_wrap(other))

    def __rmod__(self, other: ExpressionLike) -> BinaryExpression:
        return BinaryExpression(operator=BinaryOp.MOD, left=_wrap(other), right=self)

    def __repr__(self) -> str:
        return f"BinaryExpression({self.left!r} {self.operator.value} {self.right!r})"


Node = Variable | Literal | UnaryExpression | BinaryExpression
ExpressionLike = Node | int | float | complex | bool

# Update forward references now that all models are defined
Variable.model_rebuild()
Literal.model_rebuild()
UnaryExpression.model_rebuild()
BinaryExpression.model_rebuild()


def _wrap(value: ExpressionLike) -> Node:
    """Coerce a plain Python scalar into a :class:`Literal`."""
    if isinstance(value, Variable | Literal | UnaryExpression | BinaryExpression):
        return value
    return Literal(value=value)


def sin(operand: ExpressionLike) -> UnaryExpression:
    """Construct a ``sin`` :class:`UnaryExpression`."""
    return UnaryExpression(operator=UnaryOp.SIN, operand=_wrap(operand))


def cos(operand: ExpressionLike) -> UnaryExpression:
    """Construct a ``cos`` :class:`UnaryExpression`."""
    return UnaryExpression(operator=UnaryOp.COS, operand=_wrap(operand))


def sqrt(operand: ExpressionLike) -> UnaryExpression:
    """Construct a ``sqrt`` :class:`UnaryExpression`."""
    return UnaryExpression(operator=UnaryOp.SQRT, operand=_wrap(operand))


class RuntimeExpression(NoExtraFieldsFrozenModel):
    """Represents an expression that can be evaluated at runtime.

    It is expected to return a value of a given type when evaluated.

    The :class:`RuntimeExpression` is generically typed to allow it be used for
    expressions that return different types, e.g.
    :code:`RuntimeExpression[VariableType.TIME]` returns this class with an annotation to
    validate the evaluated type. This can be used in other Pydantic models to enforce
    expressions of the correct type.

    It allows us to represent an expression tree that is unknown at compile time, but can
    be evaluated at runtime given a set of variable bindings. This is used to allow
    parameterised runtime arguments in programs for backends that lack runtime argument
    support. :class:`RuntimeExpression` will be used within the representation of compiled
    programs to represent these unknown values.

    For example, there might be a runtime expression to represent a variable number of shots
    in a program, which would only contain a variable node. Similarly, if the frequency of
    a program is being varied, we can compile a program with the frequency provided at
    runtime, representing that as a :class:`RuntimeExpression` containing a variable node
    for the frequency variable.

    A more sophisticated example would be using runtime expressions within variational
    quantum circuits, where parameters of gates become phases within phase shift
    operations. After optimizations and compressions, those phase shifts would be
    represented as expressions over those parameters (in the most simple case, sums of
    parameters). This allows us to represent the parameterised program in a way that can be
    compiled and optimized without knowing the parameter values, but still allows us to
    evaluate the final program with specific parameter values at runtime.

    :ivar evaluated_type: The type that this expression is expected to evaluate to. This is
        used for bitcasting the result to the correct type when assembling the program.
    :ivar expression: The expression or variable to be evaluated given the runtime
        variables.
    """

    evaluated_type: VariableType
    expression: Node

    def evaluate(self, params: dict[str, Any] | None = None) -> Any:
        """Evaluate the expression, resolving free variables from params.

        Does not handle details of the evaluated_type, such as bitcasting and legalisation.
        This is to be done by the consumer.

        :param params: Dictionary of variable values to use for evaluation, optional.
        :return: The result of evaluating the expression with the given params.
        """
        return self.expression.evaluate(params or {})

    @classmethod
    @cache
    def __class_getitem__(cls, expression_type: VariableType) -> TypeAlias:
        if not isinstance(expression_type, VariableType):
            raise TypeError("RuntimeExpression must be subscripted with a VariableType.")

        def _check(v: RuntimeExpression) -> RuntimeExpression:
            if v.evaluated_type != expression_type:
                raise ValueError(
                    f"Expected type {expression_type}, got {v.evaluated_type}."
                )
            return v

        return Annotated[cls, AfterValidator(_check)]


_WAVEFORM_BY_NAME: dict[str, type[Waveform]] = {
    cls.__name__: cls for cls in find_all_subclasses(Waveform)
}


class ParameterisedWaveform(NoExtraFieldsFrozenModel):
    """Represents an analytical waveform that is parameterised by a set of parameters,
    including runtime expressions.

    :ivar waveform_type: The type of the waveform, which determines the functional form of
        the waveform.
    :ivar sample_time: The sample time of the waveform in seconds, which determines the time
        resolution of the waveform.
    :ivar amplitude: The amplitude of the waveform, which can be given as a complex number
        or a runtime expression with evaluated type :class:`VariableType.AMPLITUDE`.
    :ivar width: The width of the waveform, which can be given as a float or a runtime
        expression that evaluates to a float.
    :ivar parameters: A dictionary of additional parameters for the waveform, which can be
        boolean, float, complex, or runtime expressions.
    """

    waveform_type: type[Waveform]
    sample_time: PositiveFloat
    amplitude: complex | RuntimeExpression[VariableType.AMPLITUDE]
    width: PositiveFloat | RuntimeExpression[VariableType.TIME]
    parameters: dict[str, bool | float | complex | RuntimeExpression] = Field(
        default_factory=dict
    )

    def evaluate(self, params: dict[str, Any] | None = None) -> SampledWaveform:
        """Evaluate the parameterised waveform, resolving any runtime expressions from
        params.

        Returns a sampled representation of the waveform.

        :param params: Dictionary of variable values to use for evaluation, optional.
        :return: A SampledWaveform instance representing the evaluated waveform.
        """
        evaluated_amplitude = (
            self.amplitude.evaluate(params)
            if isinstance(self.amplitude, RuntimeExpression)
            else self.amplitude
        )
        evaluated_width = (
            self.width.evaluate(params)
            if isinstance(self.width, RuntimeExpression)
            else self.width
        )
        evaluated_parameters = {
            k: v.evaluate(params) if isinstance(v, RuntimeExpression) else v
            for k, v in self.parameters.items()
        }

        analytical_waveform = self.waveform_type(
            amp=evaluated_amplitude, width=evaluated_width, **evaluated_parameters
        )
        return sample_waveform(analytical_waveform, self.sample_time)

    @field_serializer("waveform_type", when_used="json")
    def _serialize_waveform_type(self, value: type[Waveform]) -> str:
        return value.__name__

    @field_validator("waveform_type", mode="before")
    @classmethod
    def _deserialize_waveform_type(cls, value):
        if not isinstance(value, str):
            return value
        if not (waveform_cls := _WAVEFORM_BY_NAME.get(value)):
            raise ValueError(f"Unknown waveform_type: {value}")
        return waveform_cls


__all__ = [
    # Types
    "VariableType",
    "UnaryOp",
    "BinaryOp",
    # Models
    "Variable",
    "Literal",
    "UnaryExpression",
    "BinaryExpression",
    "RuntimeExpression",
    "ParameterisedWaveform",
    # Type alias
    "Node",
    "ExpressionLike",
    # Factories
    "sin",
    "cos",
    "sqrt",
]
