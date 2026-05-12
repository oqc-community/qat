# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for qat.runtime.expressions."""

import math
from math import isclose, pi

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from qat.ir.waveforms import GaussianWaveform, SampledWaveform
from qat.runtime.expressions import (
    BinaryExpression,
    BinaryOp,
    Literal,
    ParameterisedWaveform,
    RuntimeExpression,
    UnaryExpression,
    UnaryOp,
    Variable,
    VariableType,
    cos,
    sin,
    sqrt,
)


class TestLiteral:
    def test_create_literal(self):
        value = Literal(value=3.14)
        assert value.value == 3.14

    def test_evaluate(self):
        assert isclose(Literal(value=3.14).evaluate({}), 3.14)

    def test_simplify_is_noop(self):
        value = Literal(value=7)
        value2 = value.simplify()
        assert value2 is value

    def test_repr(self):
        value = Literal(value=2.718)
        assert repr(value) == "Literal(2.718)"

    def test_serialization_roundtrip(self):
        value = Literal(value=1.618)
        data = value.model_dump()
        value2 = Literal(**data)
        assert value2 == value

    def test_json_serializable(self):
        value = Literal(value=0.577)
        json_str = value.model_dump_json()
        value2 = Literal.model_validate_json(json_str)
        assert value2 == value

    @pytest.mark.parametrize(
        "op, python_op, expected",
        [
            (UnaryOp.NEG, lambda x: -x, -1.7),
            (UnaryOp.ABS, lambda x: abs(x), 1.7),
            (UnaryOp.SIN, lambda x: sin(x), math.sin(1.7)),
            (UnaryOp.COS, lambda x: cos(x), math.cos(1.7)),
            (UnaryOp.SQRT, lambda x: sqrt(x), math.sqrt(1.7)),
        ],
    )
    def test_unary_operations(self, op, python_op, expected):
        value = Literal(value=1.7)
        result = python_op(value)
        assert isinstance(result, UnaryExpression)
        assert result.operator == op
        assert result.operand == value
        assert isclose(result.evaluate(), expected)

    @pytest.mark.parametrize(
        "op, python_op",
        [
            (BinaryOp.ADD, lambda x, y: x + y),
            (BinaryOp.SUB, lambda x, y: x - y),
            (BinaryOp.MUL, lambda x, y: x * y),
            (BinaryOp.DIV, lambda x, y: x / y),
            (BinaryOp.POW, lambda x, y: x**y),
            (BinaryOp.MOD, lambda x, y: x % y),
        ],
    )
    def test_binary_operations(self, op, python_op):
        l1 = Literal(value=2.0)
        l2 = Literal(value=3.0)
        result = python_op(l1, l2)
        assert isinstance(result, BinaryExpression)
        assert result.operator == op
        assert result.left == l1
        assert result.right == l2
        assert isclose(result.evaluate(), python_op(2.0, 3.0))


class TestVariable:
    def test_create_typed_variable(self):
        v = Variable(name="n", var_type=VariableType.FREQUENCY)
        assert v.var_type == VariableType.FREQUENCY
        assert v.name == "n"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="not be empty"):
            Variable(name="  ", var_type=VariableType.PHASE)

    def test_serialization_roundtrip(self):
        v = Variable(name="alpha", var_type=VariableType.COMPLEX)
        data = v.model_dump()
        v2 = Variable(**data)
        assert v2 == v

    def test_json_serializable(self):
        v = Variable(name="x", var_type=VariableType.FLOAT)
        json_str = v.model_dump_json()
        v2 = Variable.model_validate_json(json_str)
        assert v2 == v

    def test_simplification(self):
        v = Variable(name="beta", var_type=VariableType.PHASE)
        bound = v.simplify({"beta": 0.5})
        assert isinstance(bound, Literal)

    def test_evaluate(self):
        v = Variable(name="beta", var_type=VariableType.PHASE)
        bound = v.evaluate({"beta": 0.5})
        assert isclose(bound, 0.5)

    def test_evaluate_without_binding_raises(self):
        v = Variable(name="gamma", var_type=VariableType.COMPLEX)
        with pytest.raises(ValueError, match="not in params"):
            v.evaluate()

    def test_repr(self):
        v = Variable(name="gamma", var_type=VariableType.COMPLEX)
        assert (
            repr(v) == "Variable(name='gamma', var_type=<VariableType.COMPLEX: 'complex'>)"
        )

    @pytest.mark.parametrize(
        "op, python_op",
        [
            (UnaryOp.NEG, lambda x: -x),
            (UnaryOp.ABS, lambda x: abs(x)),
            (UnaryOp.SIN, lambda x: sin(x)),
            (UnaryOp.COS, lambda x: cos(x)),
            (UnaryOp.SQRT, lambda x: sqrt(x)),
        ],
    )
    def test_unary_operators(self, op, python_op):
        v = Variable(name="x", var_type=VariableType.FLOAT)
        expr = python_op(v)
        assert isinstance(expr, UnaryExpression)
        assert expr.operator == op
        assert expr.operand == v

    @pytest.mark.parametrize(
        "lhs, rhs",
        [
            (Variable(name="x", var_type=VariableType.FLOAT), Literal(value=2.0)),
            (Literal(value=3.0), Variable(name="y", var_type=VariableType.FLOAT)),
            (
                Variable(name="a", var_type=VariableType.FLOAT),
                Variable(name="b", var_type=VariableType.FLOAT),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "op, python_op",
        [
            (BinaryOp.ADD, lambda x, y: x + y),
            (BinaryOp.SUB, lambda x, y: x - y),
            (BinaryOp.MUL, lambda x, y: x * y),
            (BinaryOp.DIV, lambda x, y: x / y),
            (BinaryOp.POW, lambda x, y: x**y),
            (BinaryOp.MOD, lambda x, y: x % y),
        ],
    )
    def test_binary_operators(self, lhs, rhs, op, python_op):
        expr = python_op(lhs, rhs)
        assert isinstance(expr, BinaryExpression)
        assert expr.operator == op
        assert expr.left == lhs
        assert expr.right == rhs


class TestUnaryExpression:
    def test_simplify_folds_literal(self):
        expr = -Literal(value=3.0)
        simplified = expr.simplify()
        assert isinstance(simplified, Literal)
        assert isclose(simplified.value, -3.0)

    def test_simplify_with_free_variable(self):
        v = Variable(name="theta", var_type=VariableType.PHASE)
        expr = sin(v)
        simplified = expr.simplify()
        assert isinstance(simplified, UnaryExpression)

    def test_evaluate(self):
        expr = -Literal(value=2.0)
        assert isclose(expr.evaluate(), -2.0)

    def test_serialization_roundtrip(self):
        expr = cos(Variable(name="theta", var_type=VariableType.PHASE))
        data = expr.model_dump()
        expr2 = UnaryExpression(**data)
        assert expr2.evaluate({"theta": 0.0}) == pytest.approx(1.0)

    def test_serialization_json_roundtrip(self):
        expr = sqrt(Literal(value=16.0))
        json_str = expr.model_dump_json()
        expr2 = UnaryExpression.model_validate_json(json_str)
        assert expr2.evaluate() == pytest.approx(4.0)

    def test_repr(self):
        expr = sin(Variable(name="alpha", var_type=VariableType.PHASE))
        assert (
            repr(expr)
            == "UnaryExpression(sin, Variable(name='alpha', var_type=<VariableType.PHASE: 'phase'>))"
        )

    @pytest.mark.parametrize(
        "op, python_op",
        [
            (UnaryOp.NEG, lambda x: -x),
            (UnaryOp.ABS, lambda x: abs(x)),
            (UnaryOp.SIN, lambda x: sin(x)),
            (UnaryOp.COS, lambda x: cos(x)),
            (UnaryOp.SQRT, lambda x: sqrt(x)),
        ],
    )
    def test_unary_operations(self, op, python_op):
        expr = sin(Variable(name="x", var_type=VariableType.FLOAT))
        result = python_op(expr)
        assert isinstance(result, UnaryExpression)
        assert result.operator == op
        assert result.operand == expr

    @pytest.mark.parametrize(
        "lhs, rhs",
        [
            (-Variable(name="x", var_type=VariableType.FLOAT), Literal(value=2.0)),
            (Literal(value=3.0), -Variable(name="y", var_type=VariableType.FLOAT)),
            (
                -Variable(name="a", var_type=VariableType.FLOAT),
                cos(Variable(name="b", var_type=VariableType.FLOAT)),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "op, python_op",
        [
            (BinaryOp.ADD, lambda x, y: x + y),
            (BinaryOp.SUB, lambda x, y: x - y),
            (BinaryOp.MUL, lambda x, y: x * y),
            (BinaryOp.DIV, lambda x, y: x / y),
            (BinaryOp.POW, lambda x, y: x**y),
            (BinaryOp.MOD, lambda x, y: x % y),
        ],
    )
    def test_binary_operations(self, lhs, rhs, op, python_op):
        expr = python_op(lhs, rhs)
        assert isinstance(expr, BinaryExpression)
        assert expr.operator == op
        assert expr.left == lhs
        assert expr.right == rhs


class TestBinaryExpression:
    def test_create_binary_expression(self):
        v1 = Variable(name="x", var_type=VariableType.FLOAT)
        v2 = Variable(name="y", var_type=VariableType.FLOAT)
        expr = BinaryExpression(operator=BinaryOp.ADD, left=v1, right=v2)
        assert expr.operator == BinaryOp.ADD
        assert expr.left == v1
        assert expr.right == v2

    def test_simplify_folds_literals(self):
        expr = Literal(value=2.0) + Literal(value=3.0)
        simplified = expr.simplify()
        assert isinstance(simplified, Literal)
        assert isclose(simplified.value, 5.0)

    def test_addition_simplification(self):
        v = Variable(name="x", var_type=VariableType.FLOAT)
        const = Literal(value=0)
        expr = v + const
        simplified = expr.simplify()
        assert isinstance(simplified, Variable)
        assert simplified.name == "x"

        expr = const + v
        simplified = expr.simplify()
        assert isinstance(simplified, Variable)
        assert simplified.name == "x"

    def test_subtraction_simplification(self):
        v = Variable(name="x", var_type=VariableType.FLOAT)
        const = Literal(value=0)
        expr = v - const
        simplified = expr.simplify()
        assert isinstance(simplified, Variable)
        assert simplified.name == "x"

        expr = const - v
        simplified = expr.simplify()
        assert isinstance(simplified, UnaryExpression)
        assert simplified.operator == UnaryOp.NEG
        assert simplified.operand == v

    def test_multiplication_simplification(self):
        v = Variable(name="x", var_type=VariableType.FLOAT)
        one = Literal(value=1)
        zero = Literal(value=0)

        expr = v * one
        simplified = expr.simplify()
        assert isinstance(simplified, Variable)
        assert simplified.name == "x"

        expr = one * v
        simplified = expr.simplify()
        assert isinstance(simplified, Variable)
        assert simplified.name == "x"

        expr = v * zero
        simplified = expr.simplify()
        assert isinstance(simplified, Literal)
        assert isclose(simplified.value, 0.0)

        expr = zero * v
        simplified = expr.simplify()
        assert isinstance(simplified, Literal)
        assert isclose(simplified.value, 0.0)

    def test_div_simplification(self):
        v = Variable(name="x", var_type=VariableType.FLOAT)
        one = Literal(value=1)

        expr = v / one
        simplified = expr.simplify()
        assert isinstance(simplified, Variable)
        assert simplified.name == "x"

    def test_pow_simplification(self):
        v = Variable(name="x", var_type=VariableType.FLOAT)
        one = Literal(value=1)
        zero = Literal(value=0)

        expr = v**one
        simplified = expr.simplify()
        assert isinstance(simplified, Variable)
        assert simplified.name == "x"

        expr = v**zero
        simplified = expr.simplify()
        assert isinstance(simplified, Literal)
        assert isclose(simplified.value, 1.0)

    @pytest.mark.parametrize(
        "op, python_op",
        [
            (UnaryOp.NEG, lambda x: -x),
            (UnaryOp.ABS, lambda x: abs(x)),
            (UnaryOp.SIN, lambda x: sin(x)),
            (UnaryOp.COS, lambda x: cos(x)),
            (UnaryOp.SQRT, lambda x: sqrt(x)),
        ],
    )
    def test_unary_operations(self, op, python_op):
        v1 = Variable(name="x", var_type=VariableType.FLOAT)
        v2 = Variable(name="y", var_type=VariableType.FLOAT)
        expr = v1 + v2

        result = python_op(expr)
        assert isinstance(result, UnaryExpression)
        assert result.operator == op
        assert result.operand == expr

    @pytest.mark.parametrize(
        "op, python_op",
        [
            (BinaryOp.ADD, lambda x, y: x + y),
            (BinaryOp.SUB, lambda x, y: x - y),
            (BinaryOp.MUL, lambda x, y: x * y),
            (BinaryOp.DIV, lambda x, y: x / y),
            (BinaryOp.POW, lambda x, y: x**y),
            (BinaryOp.MOD, lambda x, y: x % y),
        ],
    )
    def test_binary_operations(self, op, python_op):
        v1 = Variable(name="x", var_type=VariableType.FLOAT)
        v2 = Variable(name="y", var_type=VariableType.FLOAT)
        expr = v1 + v2
        new_expr = python_op(expr, Literal(value=2.0))
        assert isinstance(new_expr, BinaryExpression)
        assert new_expr.operator == op
        assert new_expr.left == expr
        assert new_expr.right == Literal(value=2.0)


class TestComposedExpressions:
    def test_variational_rotation_angle(self):
        """Typical variational circuit: angle = 2 * theta + phi"""
        theta = Variable(name="theta", var_type=VariableType.FLOAT)
        phi = Variable(name="phi", var_type=VariableType.FLOAT)
        angle = Literal(value=2.0) * theta + phi
        result = angle.evaluate({"theta": 0.5, "phi": 0.1})
        assert result == pytest.approx(1.1)

    def test_deep_tree_simplify(self):
        """All leaves bound → whole tree folds to Literal."""
        a = Variable(name="a", var_type=VariableType.FLOAT)
        b = Variable(name="b", var_type=VariableType.FLOAT)
        expr = (a + b) * Literal(value=2.0)
        simplified = expr.simplify({"a": 3.0, "b": 4.0})
        assert isinstance(simplified, Literal)
        assert isclose(simplified.value, 14.0)

    def test_partial_inject_then_evaluate(self):
        a = Variable(name="a", var_type=VariableType.FLOAT)
        b = Variable(name="b", var_type=VariableType.FLOAT)
        c = Variable(name="c", var_type=VariableType.FLOAT)
        expr = a * b + c
        partial = expr.simplify({"a": 2.0, "b": 3.0})
        result = partial.evaluate({"c": 1.0})
        assert isclose(result, 7.0)

    def test_sin_of_binary(self):
        theta = Variable(name="theta", var_type=VariableType.FLOAT)
        expr = sin(Literal(value=2.0) * theta)
        result = expr.evaluate({"theta": pi / 4})
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_full_json_roundtrip_nested(self):
        theta = Variable(name="theta", var_type=VariableType.FLOAT)
        expr = sin(theta) ** Literal(value=2.0) + cos(theta) ** Literal(value=2.0)
        data = expr.model_dump_json()
        expr2 = BinaryExpression.model_validate_json(data)
        # sin²θ + cos²θ == 1  for any θ
        for angle in [0.0, 0.5, 1.0, pi]:
            assert isclose(expr2.evaluate({"theta": angle}), 1.0, abs_tol=1e-9)


class TestRuntimeExpression:
    """Tests that Runtime expressions can be created with each of the different variable
    types.

    Evaluates that subtyping within a Pydantic basemodel implements the correct validation.
    Tests serialization as that is a fundamental part of how this is used.
    """

    @pytest.mark.parametrize("expected_type", [val for val in VariableType])
    def test_runtime_expression_with_different_types(self, expected_type):
        """Checks instantiation of RuntimeExpression with different variable types."""

        expr = RuntimeExpression(
            evaluated_type=expected_type,
            expression=Variable(name="test", var_type=expected_type),
        )
        assert expr.evaluated_type == expected_type

    @pytest.mark.parametrize("expected_type", [val for val in VariableType])
    def test_subtyped_runtime_expression(self, expected_type):
        """Tests subtyping of RuntimeExpression within a Pydantic BaseModel, ensuring that
        the correct type validation is enforced."""

        class TestModel(BaseModel):
            value: RuntimeExpression[expected_type]

        test_value = RuntimeExpression(
            evaluated_type=expected_type,
            expression=Variable(name="test", var_type=expected_type),
        )
        model_instance = TestModel(value=test_value)
        assert model_instance.value.evaluated_type == expected_type

        with pytest.raises(ValidationError, match="Expected type"):
            type_ = next(t for t in VariableType if t != expected_type)
            TestModel(
                value=RuntimeExpression(
                    evaluated_type=type_,
                    expression=Variable(name="test", var_type=type_),
                )
            )

    @pytest.mark.parametrize("expected_type", [val for val in VariableType])
    def test_serialization_roundtrip(self, expected_type):
        """Tests that a RuntimeExpression with a specific evaluated type can be serialized
        to a dict and deserialized back, preserving the structure and types."""

        expr = RuntimeExpression(
            evaluated_type=expected_type,
            expression=Variable(name="test", var_type=expected_type),
        )
        data = expr.model_dump()
        loaded_expr = RuntimeExpression(**data)
        assert loaded_expr == expr

    @pytest.mark.parametrize("expected_type", [val for val in VariableType])
    def test_json_serialization_roundtrip(self, expected_type):
        """Tests that a RuntimeExpression with a specific evaluated type can be serialized
        to JSON and deserialized back, preserving the structure and types."""

        expr = RuntimeExpression(
            evaluated_type=expected_type,
            expression=Variable(name="test", var_type=expected_type),
        )
        json_str = expr.model_dump_json()
        loaded_expr = RuntimeExpression.model_validate_json(json_str)
        assert loaded_expr == expr

    def test_invalid_subtyping(self):
        """Tests that subtyping of RuntimeExpression enforces that the subscripted type is a
        VariableType, and that using an invalid type raises a TypeError."""

        with pytest.raises(
            TypeError, match="RuntimeExpression must be subscripted with a VariableType."
        ):
            RuntimeExpression["AMPLITUDE"]


class TestParameterisedWaveform:
    """Tests for ParameterisedWaveform, which includes RuntimeExpressions as parameters.

    Tests validation on instantiation, evaluation of parameters, and serialization. This is
    a key part of how parameterised waveforms are represented in the runtime, so ensuring
    that this works correctly is crucial.
    """

    def test_serialization_roundtrip(self):
        """Tests that a ParameterisedWaveform with RuntimeExpressions can be serialized to a
        dict and deserialized back, preserving the structure and types."""

        waveform = ParameterisedWaveform(
            waveform_type=GaussianWaveform,
            sample_time=5e-9,
            amplitude=RuntimeExpression(
                evaluated_type=VariableType.AMPLITUDE,
                expression=Variable(name="amp", var_type=VariableType.AMPLITUDE),
            ),
            width=RuntimeExpression(
                evaluated_type=VariableType.TIME,
                expression=BinaryExpression(
                    operator=BinaryOp.ADD,
                    left=Variable(name="width", var_type=VariableType.TIME),
                    right=Literal(value=80e-9),
                ),
            ),
            parameters={"rise": 1 / 3},
        )

        dump = waveform.model_dump()
        loaded_waveform = ParameterisedWaveform(**dump)
        assert loaded_waveform == waveform

    def test_json_serialization_roundtrip(self):
        """Tests that a ParameterisedWaveform with RuntimeExpressions can be serialized to
        JSON and deserialized back, preserving the structure and types."""

        waveform = ParameterisedWaveform(
            waveform_type=GaussianWaveform,
            sample_time=5e-9,
            amplitude=RuntimeExpression(
                evaluated_type=VariableType.AMPLITUDE,
                expression=Variable(name="amp", var_type=VariableType.AMPLITUDE),
            ),
            width=RuntimeExpression(
                evaluated_type=VariableType.TIME,
                expression=BinaryExpression(
                    operator=BinaryOp.ADD,
                    left=Variable(name="width", var_type=VariableType.TIME),
                    right=Literal(value=80e-9),
                ),
            ),
            parameters={"rise": 1 / 3},
        )
        json_str = waveform.model_dump_json()
        loaded_waveform = ParameterisedWaveform.model_validate_json(json_str)
        assert loaded_waveform == waveform

    def test_invalid_waveform_type(self):
        """Tests that instantiating a ParameterisedWaveform with an invalid waveform type
        raises a ValidationError."""
        with pytest.raises(ValidationError, match="Unknown waveform_type"):
            data = {
                "waveform_type": "triangle",
                "sample_time": 5e-9,
                "amplitude": 0.5,
                "width": 100e-9,
                "parameters": {},
            }
            ParameterisedWaveform(**data)

    def test_evaluate_parameters(self):
        """Tests that the parameters of a ParameterisedWaveform can be evaluated
        correctly."""

        sample_time = 5e-9
        offset_time = 80e-9
        waveform = ParameterisedWaveform(
            waveform_type=GaussianWaveform,
            sample_time=sample_time,
            amplitude=RuntimeExpression(
                evaluated_type=VariableType.AMPLITUDE,
                expression=Variable(name="amp", var_type=VariableType.AMPLITUDE),
            ),
            width=RuntimeExpression(
                evaluated_type=VariableType.TIME,
                expression=BinaryExpression(
                    operator=BinaryOp.ADD,
                    left=Variable(name="width", var_type=VariableType.TIME),
                    right=Literal(value=offset_time),
                ),
            ),
            parameters={"rise": 1 / 3},
        )

        params = {"amp": 0.5, "width": 160e-9}
        sampled_waveform = waveform.evaluate(params)
        assert isinstance(sampled_waveform, SampledWaveform)
        assert sampled_waveform.sample_time == sample_time
        assert len(sampled_waveform.samples) == int(
            (params["width"] + offset_time) / sample_time
        )
        # Doesn't catch the center so it's slightly lower
        max_amp = np.max(sampled_waveform.samples)
        assert max_amp <= params["amp"]
        assert np.isclose(max_amp, params["amp"], rtol=1e-2)
