# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Oxford Quantum Circuits Ltd
"""Tests the waveform evaluation pass, which converts analytical waveform ops into constant
sampled waveforms via :class:`ConstantOp` and :class:`SampledWaveformAttr`.

The tests are data-driven: a :class:`_WaveformSpec` per analytical shape declares the op
class, the values used for each operand, and the boolean property values. Generic tests
iterate over every spec, so adding a new analytical waveform op only requires appending
one entry to :data:`_WAVEFORM_SPECS`.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import BoolAttr, FloatAttr, StringAttr, f64, i1
from xdsl.ir import SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def
from xdsl.utils.exceptions import PassFailedException

from qat.experimental.dialect.pulse.ir import (
    AddOp,
    AmplitudeAttr,
    ConstantOp,
    CreateFrameOp,
    FrequencyAttr,
    PhaseAttr,
    Pulse,
    TimeAttr,
)
from qat.experimental.dialect.pulse.ir.attributes import SampledWaveformAttr
from qat.experimental.dialect.pulse.ir.ops import (
    BlackmanWaveformOp,
    GaussianSquareWaveformOp,
    GaussianWaveformOp,
    PulseOp,
    RoundedSquareWaveformOp,
    SechWaveformOp,
    SetupHoldWaveformOp,
    SinusoidalWaveformOp,
    SoftSquareWaveformOp,
    SquareWaveformOp,
    extract_constant_scalar,
)
from qat.experimental.dialect.pulse.ir.types import TimeType, WaveformType
from qat.experimental.dialect.pulse.transforms.constants import OrderedCanonicalizePass
from qat.experimental.dialect.pulse.transforms.waveform_evaluation import (
    EvaluateWaveformsAsSamples,
    _seconds_to_picoseconds,
)
from qat.experimental.system_data.pulse.constraints import (
    PortConstraints,
    PulseLevelConstraints,
)
from qat.experimental.waveforms.evaluate import evaluate_waveform

from tests.unit.utils.ir import (
    build_module_from_ops,
    create_context,
    get_operations_with_type,
)

_CONTEXT = create_context(Pulse)

PORT_CONTROL = "channel_1"
PORT_READOUT = "channel_2"


def _create_pulse_constraints(
    port_sample_times: dict[str, float],
    native_waveform_shapes: tuple[type, ...] = (SquareWaveformOp,),
) -> PulseLevelConstraints:
    """Create a PulseLevelConstraints object for testing.

    :param port_sample_times: Dictionary mapping port IDs to sample times in seconds.
    :param native_waveform_shapes: Tuple of native waveform shapes supported by the
        hardware.
    :returns: A PulseLevelConstraints object.
    """
    port_constraints = {}
    for port_id, sample_time in port_sample_times.items():
        # Convert from seconds to picoseconds
        sample_time_ps = int(round(sample_time * 1e12))
        port_constraints[port_id] = PortConstraints(
            sample_time_ps=sample_time_ps,
            min_duration_ps=0,
            max_duration_ps=None,
            acquire_allowed=True,
        )

    return PulseLevelConstraints(
        ports=port_constraints,
        granularity_ps=8000,
        native_waveform_shapes=native_waveform_shapes,
    )


@irdl_op_definition
class _ProducerOp(IRDLOperation):
    """A dummy op producing a non-constant SSA value for regression tests."""

    name = "test.producer"
    result = result_def()

    def __init__(self, result_type):
        super().__init__(result_types=[result_type])


@dataclass(frozen=True)
class _WaveformSpec:
    """Describes how to build one analytical waveform op for the rewrite tests.

    :ivar op_cls: The xDSL waveform op class under test.
    :ivar operands: Ordered map from pydantic kwarg name to a
        ``(attribute class, python value)`` pair, in the same order the op
        constructor expects the SSA operands.
    :ivar bool_props: Map from property name to boolean value for trailing
        ``BoolAttr`` arguments.
    """

    op_cls: type
    operands: dict[str, tuple[type[Any], Any]]
    bool_props: dict[str, bool] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.op_cls.__name__


_SQUARE_SPEC = _WaveformSpec(
    op_cls=SquareWaveformOp,
    operands={
        "width": (TimeAttr, 80e-9),
        "amp": (AmplitudeAttr, 0.5),
    },
)

_GAUSSIAN_SPEC = _WaveformSpec(
    op_cls=GaussianWaveformOp,
    operands={
        "width": (TimeAttr, 80e-9),
        "amp": (AmplitudeAttr, 0.5),
        "fractional_breadth": (FloatAttr, 0.47),
    },
    bool_props={"regularize": False},
)

_WAVEFORM_SPECS: list[_WaveformSpec] = [
    _SQUARE_SPEC,
    _WaveformSpec(
        op_cls=SoftSquareWaveformOp,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
            "fractional_top_width": (FloatAttr, 0.5),
            "fractional_rise": (FloatAttr, 0.1),
        },
        bool_props={"regularize": False},
    ),
    _WaveformSpec(
        op_cls=GaussianSquareWaveformOp,
        operands={
            "width": (TimeAttr, 160e-9),
            "amp": (AmplitudeAttr, 0.5),
            "fractional_rise": (FloatAttr, 0.25),
            "fractional_top_width": (FloatAttr, 0.5),
        },
        bool_props={"regularize": True},
    ),
    _GAUSSIAN_SPEC,
    _WaveformSpec(
        op_cls=BlackmanWaveformOp,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
        },
    ),
    _WaveformSpec(
        op_cls=SetupHoldWaveformOp,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
            "setup": (FloatAttr, 0.5),
            "fractional_rise": (FloatAttr, 0.1),
        },
    ),
    _WaveformSpec(
        op_cls=RoundedSquareWaveformOp,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
            "fractional_top_width": (FloatAttr, 0.5),
            "fractional_rise": (FloatAttr, 0.1),
        },
    ),
    _WaveformSpec(
        op_cls=SinusoidalWaveformOp,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
            "number_of_periods": (FloatAttr, 0.5),
            "internal_phase": (PhaseAttr, 0.0),
        },
    ),
    _WaveformSpec(
        op_cls=SechWaveformOp,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
            "fractional_breadth": (FloatAttr, 1.0 / 3.0),
        },
        bool_props={"regularize": False},
    ),
]

_BOOL_PROP_SPECS: list[_WaveformSpec] = [
    spec for spec in _WAVEFORM_SPECS if spec.bool_props
]


def _spec_id(spec: _WaveformSpec) -> str:
    """Test ID formatter so the parametrise output shows the op class name."""
    return spec.id


def _build_module_with_pulse(
    spec: _WaveformSpec,
    port: str = PORT_CONTROL,
    operand_overrides: dict[str, IRDLOperation] | None = None,
    bool_prop_overrides: dict[str, bool] | None = None,
    drag_coefficients: list[float | IRDLOperation] | None = None,
) -> tuple:
    """Build a minimal module containing one instance of ``spec``'s waveform op feeding a
    :class:`PulseOp` on a frame with the requested ``port``.

    :param operand_overrides: Replace the auto-generated :class:`ConstantOp` for the
        named pydantic operand with the supplied op (used to inject e.g. sweep-time
        producers for the non-constant-operand test).
    :param bool_prop_overrides: Override the spec's boolean property values by pydantic
        name (used to exercise both branches of ``zero_at_edges``).
    :returns: A ``(module, waveform_op)`` pair. ``waveform_op`` is the analytical op
        before the pass rewrites it.
    """
    overrides = operand_overrides or {}
    bool_overrides = bool_prop_overrides or {}
    drag_coefficients = drag_coefficients or []

    freq = ConstantOp(FrequencyAttr(5e9))
    frame = CreateFrameOp(freq, StringAttr(port))
    ops_in_order: list[IRDLOperation] = [freq, frame]

    ctor_args: list[Any] = []
    for pyd_name, (attr_cls, value) in spec.operands.items():
        if pyd_name in overrides:
            operand_op = overrides[pyd_name]
        else:
            if attr_cls is FloatAttr:
                operand_op = ArithConstantOp(FloatAttr(value, 64), f64)
            else:
                operand_op = ConstantOp(attr_cls(value))
        ops_in_order.append(operand_op)
        ctor_args.append(operand_op)

    for prop_name, prop_default in spec.bool_props.items():
        ctor_args.append(BoolAttr(bool_overrides.get(prop_name, prop_default), i1))

    for drag_coefficient in drag_coefficients:
        if isinstance(drag_coefficient, IRDLOperation):
            drag_coefficient_op = drag_coefficient
        else:
            drag_coefficient_op = ArithConstantOp(FloatAttr(drag_coefficient, 64), f64)
        ops_in_order.append(drag_coefficient_op)
        ctor_args.append(drag_coefficient_op)

    waveform = spec.op_cls(*ctor_args)
    pulse = PulseOp(frame, waveform)
    ops_in_order.extend([waveform, pulse])

    module = build_module_from_ops(ops_in_order)
    return module, waveform


def _get_sampled_constants(module) -> list[ConstantOp]:
    """Return every :class:`ConstantOp` in ``module`` whose value is a
    :class:`SampledWaveformAttr`."""
    return [
        op
        for op in get_operations_with_type(module, ConstantOp)
        if isinstance(op.value, SampledWaveformAttr)
    ]


class TestSecondToPicosecondConversion:
    @pytest.mark.parametrize("seconds", [0.0, -1e-9])
    def test_non_positive_value_raises(self, seconds):
        with pytest.raises(PassFailedException, match="must be positive"):
            _seconds_to_picoseconds(seconds, value_name="Sample time")

    def test_sub_picosecond_value_raises(self):
        with pytest.raises(PassFailedException, match="at least 1 ps"):
            _seconds_to_picoseconds(0.4e-12, value_name="Sample time")

    def test_non_representable_value_raises(self):
        with pytest.raises(
            PassFailedException,
            match="cannot be represented as integer picoseconds",
        ):
            _seconds_to_picoseconds(1.5e-12, value_name="Sample time")


@pytest.mark.parametrize("control_sample_time", [1e-9, 2e-9])
@pytest.mark.parametrize("spec", _WAVEFORM_SPECS, ids=_spec_id)
class TestWaveformShapeCoverage:
    """Runs the pass over every analytical waveform shape and checks the rewrite outcome.

    Covers replacement, PulseOp re-wiring, and sample fidelity for every op in
    :data:`_WAVEFORM_SPECS`. Any new analytical waveform op is tested here for free by
    adding a new spec entry.
    """

    def test_analytical_waveform_is_replaced_with_sampled_constant(
        self, spec, control_sample_time
    ):
        module, _ = _build_module_with_pulse(spec)

        assert get_operations_with_type(module, spec.op_cls) != []
        assert _get_sampled_constants(module) == []

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: control_sample_time},
            native_waveform_shapes=(),
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

        assert get_operations_with_type(module, spec.op_cls) == []
        sampled_constants = _get_sampled_constants(module)
        assert len(sampled_constants) == 1

        sampled_constant = sampled_constants[0]
        assert isinstance(sampled_constant.result.type, WaveformType)
        assert sampled_constant.value.sample_time.literal_value == control_sample_time

    def test_pulse_op_reads_the_sampled_constant(self, spec, control_sample_time):
        module, _ = _build_module_with_pulse(spec)

        assert get_operations_with_type(module, spec.op_cls) != []
        assert _get_sampled_constants(module) == []

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: control_sample_time},
            native_waveform_shapes=(),
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

        pulse_ops = get_operations_with_type(module, PulseOp)
        assert len(pulse_ops) == 1
        sampled_constants = _get_sampled_constants(module)
        assert pulse_ops[0].waveform is sampled_constants[0].result

    def test_samples_match_reference_sampling(self, spec, control_sample_time):
        module, waveform_op = _build_module_with_pulse(spec)

        assert get_operations_with_type(module, spec.op_cls) != []
        assert _get_sampled_constants(module) == []

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: control_sample_time},
            native_waveform_shapes=(),
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)
        # Compute expected samples inline
        width = extract_constant_scalar(waveform_op.width)
        amplitude = extract_constant_scalar(waveform_op.amplitude)
        assert width is not None and amplitude is not None
        width_ps = int(round(width * 1e12))
        sample_time_ps = int(round(control_sample_time * 1e12))
        expected = evaluate_waveform(
            width=width_ps,
            sample_time=sample_time_ps,
            shape=waveform_op.build_shape(),
            amplitude=amplitude,
            drag_coefficients=[],
        )
        sampled_constant = _get_sampled_constants(module)[0]
        assert_array_equal(sampled_constant.value.samples.data, expected)


@pytest.mark.parametrize("control_sample_time", [1e-9, 2e-9])
class TestPulseOpRewrite:
    """Shape-independent behaviour of the pulse-op rewrite pattern.

    Uses :data:`_GAUSSIAN_SPEC` as a representative analytical waveform; the shape does
    not matter for these assertions (they are about port selection, missing-port
    handling, and non-constant operands).
    """

    @pytest.mark.parametrize("readout_sample_time", [2e-9, 4e-9])
    @pytest.mark.parametrize(
        "port, expected_selector",
        [
            (PORT_CONTROL, "control"),
            (PORT_READOUT, "readout"),
        ],
    )
    def test_sample_time_is_selected_by_port(
        self, control_sample_time, readout_sample_time, port, expected_selector
    ):
        module, _ = _build_module_with_pulse(_GAUSSIAN_SPEC, port=port)
        expected_sample_time = (
            control_sample_time if expected_selector == "control" else readout_sample_time
        )

        assert get_operations_with_type(module, GaussianWaveformOp) != []
        assert _get_sampled_constants(module) == []

        constraints = _create_pulse_constraints(
            port_sample_times={
                PORT_CONTROL: control_sample_time,
                PORT_READOUT: readout_sample_time,
            },
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

        sampled_constants = _get_sampled_constants(module)
        assert len(sampled_constants) == 1
        assert sampled_constants[0].value.sample_time.literal_value == expected_sample_time

    def test_unknown_port_leaves_waveform_untouched(self, control_sample_time):
        module, _ = _build_module_with_pulse(_GAUSSIAN_SPEC, port="unknown")

        assert len(get_operations_with_type(module, GaussianWaveformOp)) == 1
        assert _get_sampled_constants(module) == []

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: control_sample_time},
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

        assert len(get_operations_with_type(module, GaussianWaveformOp)) == 1
        assert _get_sampled_constants(module) == []

    def test_non_constant_operand_leaves_waveform_untouched(self, control_sample_time):
        sweep_width = _ProducerOp(TimeType())
        module, _ = _build_module_with_pulse(
            _GAUSSIAN_SPEC, operand_overrides={"width": sweep_width}
        )

        assert len(get_operations_with_type(module, GaussianWaveformOp)) == 1
        assert _get_sampled_constants(module) == []

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: control_sample_time},
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

        assert len(get_operations_with_type(module, GaussianWaveformOp)) == 1
        assert _get_sampled_constants(module) == []

    def test_non_constant_shape_parameter_leaves_waveform_untouched(
        self, control_sample_time
    ):
        sweep_fractional_breadth = _ProducerOp(f64)
        module, _ = _build_module_with_pulse(
            _GAUSSIAN_SPEC,
            operand_overrides={"fractional_breadth": sweep_fractional_breadth},
        )

        assert len(get_operations_with_type(module, GaussianWaveformOp)) == 1
        assert _get_sampled_constants(module) == []

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: control_sample_time},
            native_waveform_shapes=(),
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

        assert len(get_operations_with_type(module, GaussianWaveformOp)) == 1
        assert _get_sampled_constants(module) == []

    def test_non_constant_drag_coefficient_leaves_waveform_untouched(
        self, control_sample_time
    ):
        """Tests that a producer not registered as a constant leaves the waveform untouched,
        even if the other operands are constant."""
        sweep_drag = _ProducerOp(f64)
        module, _ = _build_module_with_pulse(
            _GAUSSIAN_SPEC,
            drag_coefficients=[sweep_drag],
        )

        assert len(get_operations_with_type(module, GaussianWaveformOp)) == 1
        assert _get_sampled_constants(module) == []

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: control_sample_time},
            native_waveform_shapes=(),
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

        assert len(get_operations_with_type(module, GaussianWaveformOp)) == 1
        assert _get_sampled_constants(module) == []

    @pytest.mark.parametrize("regularize", [True, False])
    @pytest.mark.parametrize("spec", _BOOL_PROP_SPECS, ids=_spec_id)
    def test_bool_property_is_extracted(self, spec, regularize, control_sample_time):
        module, waveform_op = _build_module_with_pulse(
            spec, bool_prop_overrides={"regularize": regularize}
        )

        assert get_operations_with_type(module, spec.op_cls) != []
        assert _get_sampled_constants(module) == []

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: control_sample_time},
            native_waveform_shapes=(),
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

        assert get_operations_with_type(module, spec.op_cls) == []
        sampled_constants = _get_sampled_constants(module)
        assert len(sampled_constants) == 1
        # Compute expected samples inline
        width = extract_constant_scalar(waveform_op.width)
        amplitude = extract_constant_scalar(waveform_op.amplitude)
        assert width is not None and amplitude is not None
        width_ps = int(round(width * 1e12))
        sample_time_ps = int(round(control_sample_time * 1e12))
        expected = evaluate_waveform(
            width=width_ps,
            sample_time=sample_time_ps,
            shape=waveform_op.build_shape(),
            amplitude=amplitude,
            drag_coefficients=[],
        )
        assert_array_equal(sampled_constants[0].value.samples.data, expected)

    def test_incompatible_width_raises_pass_failed_exception(self, control_sample_time):
        module, _ = _build_module_with_pulse(_GAUSSIAN_SPEC)

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: 3e-9},
            native_waveform_shapes=(),
        )

        with pytest.raises(
            PassFailedException,
            match="Width .* is not an integer multiple of sample time",
        ):
            EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

    def test_drag_coefficients_are_included_in_sampling(self, control_sample_time):
        module, waveform_op = _build_module_with_pulse(
            _GAUSSIAN_SPEC,
            drag_coefficients=[0.1, 0.2],
        )

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: control_sample_time},
            native_waveform_shapes=(),
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)
        # Compute expected samples inline with and without DRAG
        width = extract_constant_scalar(waveform_op.width)
        amplitude = extract_constant_scalar(waveform_op.amplitude)
        assert width is not None and amplitude is not None
        drag_coefficients = [
            extract_constant_scalar(drag_coefficient)
            for drag_coefficient in waveform_op.drag_coefficients
        ]
        assert all(coefficient is not None for coefficient in drag_coefficients)
        width_ps = int(round(width * 1e12))
        sample_time_ps = int(round(control_sample_time * 1e12))
        shape = waveform_op.build_shape()
        expected_with_drag = evaluate_waveform(
            width=width_ps,
            sample_time=sample_time_ps,
            shape=shape,
            amplitude=amplitude,
            drag_coefficients=[
                coefficient for coefficient in drag_coefficients if coefficient is not None
            ],
        )
        expected_without_drag = evaluate_waveform(
            width=width_ps,
            sample_time=sample_time_ps,
            shape=shape,
            amplitude=amplitude,
            drag_coefficients=[],
        )

        sampled_constant = _get_sampled_constants(module)[0]
        assert_array_equal(sampled_constant.value.samples.data, expected_with_drag)
        assert not np.allclose(expected_with_drag, expected_without_drag)

    def test_two_pulses_are_rewritten_independently(self, control_sample_time):
        freq = ConstantOp(FrequencyAttr(5e9))
        frame = CreateFrameOp(
            freq,
            StringAttr(PORT_CONTROL),
        )
        width_a = ConstantOp(TimeAttr(80e-9))
        amp_a = ConstantOp(AmplitudeAttr(0.5))
        fractional_breadth_a = ArithConstantOp(FloatAttr(0.4, 64), f64)
        wf_a = GaussianWaveformOp(width_a, amp_a, fractional_breadth_a, BoolAttr(False, i1))
        pulse_a = PulseOp(frame, wf_a)

        width_b = ConstantOp(TimeAttr(120e-9))
        amp_b = ConstantOp(AmplitudeAttr(0.25))
        fractional_breadth_b = ArithConstantOp(FloatAttr(0.3, 64), f64)
        wf_b = GaussianWaveformOp(width_b, amp_b, fractional_breadth_b, BoolAttr(False, i1))
        pulse_b = PulseOp(pulse_a, wf_b)

        module = build_module_from_ops(
            [
                freq,
                frame,
                width_a,
                amp_a,
                fractional_breadth_a,
                wf_a,
                pulse_a,
                width_b,
                amp_b,
                fractional_breadth_b,
                wf_b,
                pulse_b,
            ],
        )

        assert len(get_operations_with_type(module, GaussianWaveformOp)) == 2
        assert _get_sampled_constants(module) == []

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: control_sample_time},
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

        assert get_operations_with_type(module, GaussianWaveformOp) == []
        assert len(_get_sampled_constants(module)) == 2

    def test_shared_waveform_is_sampled_once_when_all_pulses_share_a_port(
        self, control_sample_time
    ):
        freq = ConstantOp(FrequencyAttr(5e9))
        frame_a = CreateFrameOp(freq, StringAttr(PORT_CONTROL))
        frame_b = CreateFrameOp(freq, StringAttr(PORT_CONTROL))
        width = ConstantOp(TimeAttr(80e-9))
        amp = ConstantOp(AmplitudeAttr(0.5))
        fractional_breadth = ArithConstantOp(FloatAttr(0.4, 64), f64)
        wf = GaussianWaveformOp(width, amp, fractional_breadth, BoolAttr(False, i1))
        pulse_a = PulseOp(frame_a, wf)
        pulse_b = PulseOp(frame_b, wf)

        module = build_module_from_ops(
            [
                freq,
                frame_a,
                frame_b,
                width,
                amp,
                fractional_breadth,
                wf,
                pulse_a,
                pulse_b,
            ],
        )

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: control_sample_time},
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

        assert get_operations_with_type(module, GaussianWaveformOp) == []
        sampled_constants = _get_sampled_constants(module)
        assert len(sampled_constants) == 1
        pulse_ops = get_operations_with_type(module, PulseOp)
        assert pulse_ops[0].waveform is sampled_constants[0].result
        assert pulse_ops[1].waveform is sampled_constants[0].result

    def test_shared_waveform_is_sampled_per_port(self, control_sample_time):
        readout_sample_time = 4e-9
        freq = ConstantOp(FrequencyAttr(5e9))
        frame_control = CreateFrameOp(freq, StringAttr(PORT_CONTROL))
        frame_readout = CreateFrameOp(freq, StringAttr(PORT_READOUT))
        width = ConstantOp(TimeAttr(80e-9))
        amp = ConstantOp(AmplitudeAttr(0.5))
        fractional_breadth = ArithConstantOp(FloatAttr(0.4, 64), f64)
        wf = GaussianWaveformOp(width, amp, fractional_breadth, BoolAttr(False, i1))
        pulse_control = PulseOp(frame_control, wf)
        pulse_readout = PulseOp(frame_readout, wf)

        module = build_module_from_ops(
            [
                freq,
                frame_control,
                frame_readout,
                width,
                amp,
                fractional_breadth,
                wf,
                pulse_control,
                pulse_readout,
            ],
        )

        constraints = _create_pulse_constraints(
            port_sample_times={
                PORT_CONTROL: control_sample_time,
                PORT_READOUT: readout_sample_time,
            },
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

        assert get_operations_with_type(module, GaussianWaveformOp) == []
        sampled_constants = _get_sampled_constants(module)
        assert len(sampled_constants) == 2
        sample_times = {c.value.sample_time.literal_value for c in sampled_constants}
        assert sample_times == {control_sample_time, readout_sample_time}
        pulse_ops = get_operations_with_type(module, PulseOp)
        assert pulse_ops[0].waveform is not pulse_ops[1].waveform

    def test_waveform_with_non_pulse_consumer_is_ignored(self, control_sample_time):
        """Mocks a non-pulse op that takes a waveform operand, to test that such waveforms
        with non pulse consumers are ignored."""

        @irdl_op_definition
        class NonPulseOp(IRDLOperation):
            name = "test.non_pulse"
            waveform = operand_def(WaveformType)

            def __init__(self, waveform: SSAValue[WaveformType]):
                super().__init__(operands=[waveform])

        freq = ConstantOp(FrequencyAttr(5e9))
        frame = CreateFrameOp(freq, StringAttr(PORT_CONTROL))
        width = ConstantOp(TimeAttr(80e-9))
        amp = ConstantOp(AmplitudeAttr(0.5))
        fractional_breadth = ArithConstantOp(FloatAttr(0.4, 64), f64)
        wf = GaussianWaveformOp(width, amp, fractional_breadth, BoolAttr(False, i1))
        non_pulse = NonPulseOp(wf)

        module = build_module_from_ops(
            [freq, frame, width, amp, fractional_breadth, wf, non_pulse],
        )

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: control_sample_time},
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

        assert get_operations_with_type(module, GaussianWaveformOp) == [wf]
        assert _get_sampled_constants(module) == []


@pytest.mark.parametrize("control_sample_time", [1e-9, 2e-9])
class TestNativeWaveformShapes:
    """Tests that waveforms whose shape is listed as natively-supported are left as-is."""

    def test_square_is_ignored_by_default(self, control_sample_time):
        module, _ = _build_module_with_pulse(_SQUARE_SPEC)

        assert len(get_operations_with_type(module, SquareWaveformOp)) == 1
        assert _get_sampled_constants(module) == []

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: control_sample_time},
            native_waveform_shapes=(SquareWaveformOp,),
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

        assert len(get_operations_with_type(module, SquareWaveformOp)) == 1
        assert _get_sampled_constants(module) == []

    @pytest.mark.parametrize("spec", _WAVEFORM_SPECS, ids=_spec_id)
    def test_shape_is_left_untouched_when_ignored(self, spec, control_sample_time):
        module, _ = _build_module_with_pulse(spec)

        assert len(get_operations_with_type(module, spec.op_cls)) == 1
        assert _get_sampled_constants(module) == []

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: control_sample_time},
            native_waveform_shapes=(spec.op_cls,),
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

        assert len(get_operations_with_type(module, spec.op_cls)) == 1
        assert _get_sampled_constants(module) == []

    def test_square_is_sampled_when_ignored_shapes_is_empty(self, control_sample_time):
        module, _ = _build_module_with_pulse(_SQUARE_SPEC)

        assert get_operations_with_type(module, SquareWaveformOp) != []
        assert _get_sampled_constants(module) == []

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: control_sample_time},
            native_waveform_shapes=(),
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

        assert get_operations_with_type(module, SquareWaveformOp) == []
        assert len(_get_sampled_constants(module)) == 1


@pytest.mark.parametrize("control_sample_time", [1e-9, 2e-9])
class TestConstantFoldedOperands:
    """Behaviour around operands that only become constant after constant propagation.

    Waveform evaluation no longer folds constants itself; it relies on
    :class:`~qat.experimental.dialect.pulse.transforms.constants.OrderedCanonicalizePass`
    having run first. These tests pin that separation of concerns down.
    """

    def _build_module_with_folded_width(self):
        """Build a Gaussian whose width operand is ``add(40ns, 40ns)`` before folding."""
        freq = ConstantOp(FrequencyAttr(5e9))
        frame = CreateFrameOp(freq, StringAttr(PORT_CONTROL))
        width_lhs = ConstantOp(TimeAttr(40e-9))
        width_rhs = ConstantOp(TimeAttr(40e-9))
        width = AddOp(width_lhs, width_rhs, TimeType())
        amp = ConstantOp(AmplitudeAttr(0.5))
        fractional_breadth = ArithConstantOp(FloatAttr(0.47, f64), f64)
        wf = GaussianWaveformOp(width, amp, fractional_breadth, BoolAttr(False, i1))
        pulse = PulseOp(frame, wf)
        module = build_module_from_ops(
            [
                freq,
                frame,
                width_lhs,
                width_rhs,
                width,
                amp,
                fractional_breadth,
                wf,
                pulse,
            ]
        )
        return module, wf

    def test_folded_arith_operand_is_evaluated_after_constant_propagation(
        self, control_sample_time
    ):
        module, wf = self._build_module_with_folded_width()

        assert len(get_operations_with_type(module, AddOp)) == 1
        assert _get_sampled_constants(module) == []

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: control_sample_time},
        )
        OrderedCanonicalizePass().apply(_CONTEXT, module)
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

        assert get_operations_with_type(module, GaussianWaveformOp) == []
        assert get_operations_with_type(module, AddOp) == []
        sampled_constants = _get_sampled_constants(module)
        assert len(sampled_constants) == 1

        width = extract_constant_scalar(wf.width)
        amplitude = extract_constant_scalar(wf.amplitude)
        assert width is not None and amplitude is not None
        width_ps = int(round(width * 1e12))
        sample_time_ps = int(round(control_sample_time * 1e12))
        expected = evaluate_waveform(
            width=width_ps,
            sample_time=sample_time_ps,
            shape=wf.build_shape(),
            amplitude=amplitude,
            drag_coefficients=[],
        )
        assert_array_equal(sampled_constants[0].value.samples.data, expected)

    def test_folded_operand_is_skipped_without_constant_propagation(
        self, control_sample_time
    ):
        module, _ = self._build_module_with_folded_width()

        constraints = _create_pulse_constraints(
            port_sample_times={PORT_CONTROL: control_sample_time},
        )
        EvaluateWaveformsAsSamples(constraints=constraints).apply(_CONTEXT, module)

        assert len(get_operations_with_type(module, GaussianWaveformOp)) == 1
        assert len(get_operations_with_type(module, AddOp)) == 1
        assert _get_sampled_constants(module) == []
