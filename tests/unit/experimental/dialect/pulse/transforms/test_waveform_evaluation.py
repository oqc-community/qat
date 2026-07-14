# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Oxford Quantum Circuits Ltd
"""Tests the waveform evaluation pass, which converts analytical waveform ops into constant
sampled waveforms via :class:`ConstantOp` and :class:`SampledWaveformAttr`.

The tests are data-driven: a :class:`_WaveformSpec` per analytical shape declares the op
class, the corresponding pydantic waveform, and the values used for each operand and
boolean property. Generic tests iterate over every spec, so adding a new analytical
waveform op only requires appending one entry to :data:`_ALL_SPECS`.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest
from numpy.testing import assert_array_equal
from xdsl.dialects.builtin import BoolAttr, StringAttr, i1
from xdsl.irdl import IRDLOperation, irdl_op_definition, result_def

from qat.experimental.dialect.pulse.ir import (
    AmplitudeAttr,
    ConstantOp,
    CreateFrameOp,
    FrequencyAttr,
    PhaseAttr,
    Pulse,
    TimeAttr,
)
from qat.experimental.dialect.pulse.ir.attributes import (
    PulseNumericTypedAttr,
    SampledWaveformAttr,
)
from qat.experimental.dialect.pulse.ir.ops import (
    BlackmanWaveformOp,
    CosWaveformOp,
    DragGaussianWaveformOp,
    ExtraSoftSquareWaveformOp,
    GaussianSquareWaveformOp,
    GaussianWaveformOp,
    GaussianZeroEdgeWaveformOp,
    PulseOp,
    RoundedSquareWaveformOp,
    SechWaveformOp,
    SetupHoldWaveformOp,
    SinWaveformOp,
    SofterGaussianWaveformOp,
    SofterSquareWaveformOp,
    SoftSquareWaveformOp,
    SquareWaveformOp,
)
from qat.experimental.dialect.pulse.ir.types import TimeType, WaveformType
from qat.experimental.dialect.pulse.transforms.waveform_evaluation import (
    EvaluateWaveformsAsSamples,
)
from qat.ir.waveforms import (
    BlackmanWaveform,
    CosWaveform,
    DragGaussianWaveform,
    ExtraSoftSquareWaveform,
    GaussianSquareWaveform,
    GaussianWaveform,
    GaussianZeroEdgeWaveform,
    RoundedSquareWaveform,
    SechWaveform,
    SetupHoldWaveform,
    SinWaveform,
    SofterGaussianWaveform,
    SofterSquareWaveform,
    SoftSquareWaveform,
    SquareWaveform,
    Waveform,
    sample_waveform,
)

from tests.unit.utils.ir import (
    build_module_from_ops,
    create_context,
    get_operations_with_type,
)

_CONTEXT = create_context(Pulse)

PORT_CONTROL = "channel_1"
PORT_READOUT = "channel_2"


@irdl_op_definition
class _SweepTimeOp(IRDLOperation):
    """A dummy op producing a non-constant ``pulse.time`` SSA value used to model a sweep
    variable feeding the width operand of a waveform."""

    name = "test.sweep_time"
    result = result_def(TimeType)

    def __init__(self):
        super().__init__(result_types=[TimeType()])


@dataclass(frozen=True)
class _WaveformSpec:
    """Describes how to build one analytical waveform op and its pydantic reference.

    :ivar op_cls: The xDSL waveform op class under test.
    :ivar pydantic_cls: The QAT pydantic :class:`Waveform` subclass the op maps to.
    :ivar operands: Ordered map from pydantic kwarg name to a
        ``(PulseNumericTypedAttr subclass, python value)`` pair, in the same order the op
        constructor expects the SSA operands.
    :ivar bool_props: Map from pydantic kwarg name to boolean value, for properties
        (currently only ``zero_at_edges``) that the op constructor takes as trailing
        ``BoolAttr`` arguments.
    """

    op_cls: type
    pydantic_cls: type[Waveform]
    operands: dict[str, tuple[type[PulseNumericTypedAttr], Any]]
    bool_props: dict[str, bool] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.op_cls.__name__

    def build_pydantic_waveform(self) -> Waveform:
        """Return the reference pydantic waveform built from this spec's values."""
        kwargs: dict[str, Any] = {name: value for name, (_, value) in self.operands.items()}
        kwargs.update(self.bool_props)
        return self.pydantic_cls(**kwargs)


_SQUARE_SPEC = _WaveformSpec(
    op_cls=SquareWaveformOp,
    pydantic_cls=SquareWaveform,
    operands={
        "width": (TimeAttr, 80e-9),
        "amp": (AmplitudeAttr, 0.5),
    },
)

_GAUSSIAN_SPEC = _WaveformSpec(
    op_cls=GaussianWaveformOp,
    pydantic_cls=GaussianWaveform,
    operands={
        "width": (TimeAttr, 80e-9),
        "amp": (AmplitudeAttr, 0.5),
        "rise": (TimeAttr, 10e-9),
    },
)

_WAVEFORM_SPECS: list[_WaveformSpec] = [
    _SQUARE_SPEC,
    _WaveformSpec(
        op_cls=SoftSquareWaveformOp,
        pydantic_cls=SoftSquareWaveform,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
            "rise": (TimeAttr, 10e-9),
        },
    ),
    _WaveformSpec(
        op_cls=SofterSquareWaveformOp,
        pydantic_cls=SofterSquareWaveform,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
            "std_dev": (TimeAttr, 20e-9),
            "rise": (TimeAttr, 10e-9),
        },
    ),
    _WaveformSpec(
        op_cls=ExtraSoftSquareWaveformOp,
        pydantic_cls=ExtraSoftSquareWaveform,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
            "std_dev": (TimeAttr, 20e-9),
            "rise": (TimeAttr, 10e-9),
        },
    ),
    _WaveformSpec(
        op_cls=GaussianSquareWaveformOp,
        pydantic_cls=GaussianSquareWaveform,
        operands={
            "width": (TimeAttr, 160e-9),
            "amp": (AmplitudeAttr, 0.5),
            "std_dev": (TimeAttr, 20e-9),
            "square_width": (TimeAttr, 80e-9),
        },
        bool_props={"zero_at_edges": True},
    ),
    _GAUSSIAN_SPEC,
    _WaveformSpec(
        op_cls=SofterGaussianWaveformOp,
        pydantic_cls=SofterGaussianWaveform,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
            "rise": (TimeAttr, 10e-9),
        },
    ),
    _WaveformSpec(
        op_cls=BlackmanWaveformOp,
        pydantic_cls=BlackmanWaveform,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
        },
    ),
    _WaveformSpec(
        op_cls=SetupHoldWaveformOp,
        pydantic_cls=SetupHoldWaveform,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
            "amp_setup": (AmplitudeAttr, 0.25),
            "rise": (TimeAttr, 10e-9),
        },
    ),
    _WaveformSpec(
        op_cls=RoundedSquareWaveformOp,
        pydantic_cls=RoundedSquareWaveform,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
            "rise": (TimeAttr, 10e-9),
            "std_dev": (TimeAttr, 20e-9),
        },
    ),
    _WaveformSpec(
        op_cls=DragGaussianWaveformOp,
        pydantic_cls=DragGaussianWaveform,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
            "std_dev": (TimeAttr, 20e-9),
            "beta": (TimeAttr, 0.1),
        },
        bool_props={"zero_at_edges": True},
    ),
    _WaveformSpec(
        op_cls=GaussianZeroEdgeWaveformOp,
        pydantic_cls=GaussianZeroEdgeWaveform,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
            "std_dev": (TimeAttr, 20e-9),
        },
        bool_props={"zero_at_edges": True},
    ),
    _WaveformSpec(
        op_cls=CosWaveformOp,
        pydantic_cls=CosWaveform,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
            "frequency": (FrequencyAttr, 5e6),
            "internal_phase": (PhaseAttr, 0.0),
        },
    ),
    _WaveformSpec(
        op_cls=SinWaveformOp,
        pydantic_cls=SinWaveform,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
            "frequency": (FrequencyAttr, 5e6),
            "internal_phase": (PhaseAttr, 0.0),
        },
    ),
    _WaveformSpec(
        op_cls=SechWaveformOp,
        pydantic_cls=SechWaveform,
        operands={
            "width": (TimeAttr, 80e-9),
            "amp": (AmplitudeAttr, 0.5),
            "std_dev": (TimeAttr, 20e-9),
        },
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

    freq = ConstantOp(FrequencyAttr(5e9))
    frame = CreateFrameOp(freq, StringAttr(port))
    ops_in_order: list[IRDLOperation] = [freq, frame]

    ctor_args: list[Any] = []
    for pyd_name, (attr_cls, value) in spec.operands.items():
        if pyd_name in overrides:
            operand_op = overrides[pyd_name]
        else:
            operand_op = ConstantOp(attr_cls(value))
        ops_in_order.append(operand_op)
        ctor_args.append(operand_op)

    for prop_name, prop_default in spec.bool_props.items():
        ctor_args.append(BoolAttr(bool_overrides.get(prop_name, prop_default), i1))

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


@pytest.mark.parametrize("control_sample_time", [1e-9, 2e-9])
@pytest.mark.parametrize("spec", _WAVEFORM_SPECS, ids=_spec_id)
class TestWaveformShapeCoverage:
    """Runs the pass over every analytical waveform shape and checks the rewrite outcome.

    Covers replacement, PulseOp re-wiring, and sample fidelity for every op in
    :data:`_ALL_SPECS`. Any new analytical waveform op is tested here for free by adding
    a new spec entry.
    """

    def test_analytical_waveform_is_replaced_with_sampled_constant(
        self, spec, control_sample_time
    ):
        module, _ = _build_module_with_pulse(spec)

        assert get_operations_with_type(module, spec.op_cls) != []
        assert _get_sampled_constants(module) == []

        EvaluateWaveformsAsSamples(
            port_sample_times={PORT_CONTROL: control_sample_time},
            ignored_shapes=(),
        ).apply(_CONTEXT, module)

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

        EvaluateWaveformsAsSamples(
            port_sample_times={PORT_CONTROL: control_sample_time},
            ignored_shapes=(),
        ).apply(_CONTEXT, module)

        pulse_ops = get_operations_with_type(module, PulseOp)
        assert len(pulse_ops) == 1
        sampled_constants = _get_sampled_constants(module)
        assert pulse_ops[0].waveform is sampled_constants[0].result

    def test_samples_match_reference_sampling(self, spec, control_sample_time):
        module, _ = _build_module_with_pulse(spec)

        assert get_operations_with_type(module, spec.op_cls) != []
        assert _get_sampled_constants(module) == []

        EvaluateWaveformsAsSamples(
            port_sample_times={PORT_CONTROL: control_sample_time},
            ignored_shapes=(),
        ).apply(_CONTEXT, module)

        expected = sample_waveform(spec.build_pydantic_waveform(), control_sample_time)
        sampled_constant = _get_sampled_constants(module)[0]
        assert_array_equal(sampled_constant.value.samples.data, expected.samples)


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

        EvaluateWaveformsAsSamples(
            port_sample_times={
                PORT_CONTROL: control_sample_time,
                PORT_READOUT: readout_sample_time,
            },
        ).apply(_CONTEXT, module)

        sampled_constants = _get_sampled_constants(module)
        assert len(sampled_constants) == 1
        assert sampled_constants[0].value.sample_time.literal_value == expected_sample_time

    def test_unknown_port_leaves_waveform_untouched(self, control_sample_time):
        module, _ = _build_module_with_pulse(_GAUSSIAN_SPEC, port="unknown")

        assert len(get_operations_with_type(module, GaussianWaveformOp)) == 1
        assert _get_sampled_constants(module) == []

        EvaluateWaveformsAsSamples(
            port_sample_times={PORT_CONTROL: control_sample_time},
        ).apply(_CONTEXT, module)

        assert len(get_operations_with_type(module, GaussianWaveformOp)) == 1
        assert _get_sampled_constants(module) == []

    def test_non_constant_operand_leaves_waveform_untouched(self, control_sample_time):
        sweep_width = _SweepTimeOp()
        module, _ = _build_module_with_pulse(
            _GAUSSIAN_SPEC, operand_overrides={"width": sweep_width}
        )

        assert len(get_operations_with_type(module, GaussianWaveformOp)) == 1
        assert _get_sampled_constants(module) == []

        EvaluateWaveformsAsSamples(
            port_sample_times={PORT_CONTROL: control_sample_time},
        ).apply(_CONTEXT, module)

        assert len(get_operations_with_type(module, GaussianWaveformOp)) == 1
        assert _get_sampled_constants(module) == []

    @pytest.mark.parametrize("zero_at_edges", [True, False])
    @pytest.mark.parametrize("spec", _BOOL_PROP_SPECS, ids=_spec_id)
    def test_bool_property_is_extracted(self, spec, zero_at_edges, control_sample_time):
        module, _ = _build_module_with_pulse(
            spec, bool_prop_overrides={"zero_at_edges": zero_at_edges}
        )

        assert get_operations_with_type(module, spec.op_cls) != []
        assert _get_sampled_constants(module) == []

        EvaluateWaveformsAsSamples(
            port_sample_times={PORT_CONTROL: control_sample_time},
        ).apply(_CONTEXT, module)

        assert get_operations_with_type(module, spec.op_cls) == []
        sampled_constants = _get_sampled_constants(module)
        assert len(sampled_constants) == 1

        pydantic_kwargs: dict[str, Any] = {
            name: value for name, (_, value) in spec.operands.items()
        }
        pydantic_kwargs["zero_at_edges"] = zero_at_edges
        expected = sample_waveform(
            spec.pydantic_cls(**pydantic_kwargs), control_sample_time
        )
        assert_array_equal(sampled_constants[0].value.samples.data, expected.samples)

    def test_two_pulses_are_rewritten_independently(self, control_sample_time):
        freq = ConstantOp(FrequencyAttr(5e9))
        frame = CreateFrameOp(
            freq,
            StringAttr(PORT_CONTROL),
        )
        width_a = ConstantOp(TimeAttr(80e-9))
        amp_a = ConstantOp(AmplitudeAttr(0.5))
        rise_a = ConstantOp(TimeAttr(10e-9))
        wf_a = GaussianWaveformOp(width_a, amp_a, rise_a)
        pulse_a = PulseOp(frame, wf_a)

        width_b = ConstantOp(TimeAttr(120e-9))
        amp_b = ConstantOp(AmplitudeAttr(0.25))
        rise_b = ConstantOp(TimeAttr(20e-9))
        wf_b = GaussianWaveformOp(width_b, amp_b, rise_b)
        pulse_b = PulseOp(pulse_a, wf_b)

        module = build_module_from_ops(
            [
                freq,
                frame,
                width_a,
                amp_a,
                rise_a,
                wf_a,
                pulse_a,
                width_b,
                amp_b,
                rise_b,
                wf_b,
                pulse_b,
            ],
        )

        assert len(get_operations_with_type(module, GaussianWaveformOp)) == 2
        assert _get_sampled_constants(module) == []

        EvaluateWaveformsAsSamples(
            port_sample_times={PORT_CONTROL: control_sample_time},
        ).apply(_CONTEXT, module)

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
        rise = ConstantOp(TimeAttr(10e-9))
        wf = GaussianWaveformOp(width, amp, rise)
        pulse_a = PulseOp(frame_a, wf)
        pulse_b = PulseOp(frame_b, wf)

        module = build_module_from_ops(
            [freq, frame_a, frame_b, width, amp, rise, wf, pulse_a, pulse_b],
        )

        EvaluateWaveformsAsSamples(
            port_sample_times={PORT_CONTROL: control_sample_time},
        ).apply(_CONTEXT, module)

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
        rise = ConstantOp(TimeAttr(10e-9))
        wf = GaussianWaveformOp(width, amp, rise)
        pulse_control = PulseOp(frame_control, wf)
        pulse_readout = PulseOp(frame_readout, wf)

        module = build_module_from_ops(
            [
                freq,
                frame_control,
                frame_readout,
                width,
                amp,
                rise,
                wf,
                pulse_control,
                pulse_readout,
            ],
        )

        EvaluateWaveformsAsSamples(
            port_sample_times={
                PORT_CONTROL: control_sample_time,
                PORT_READOUT: readout_sample_time,
            },
        ).apply(_CONTEXT, module)

        assert get_operations_with_type(module, GaussianWaveformOp) == []
        sampled_constants = _get_sampled_constants(module)
        assert len(sampled_constants) == 2
        sample_times = {c.value.sample_time.literal_value for c in sampled_constants}
        assert sample_times == {control_sample_time, readout_sample_time}
        pulse_ops = get_operations_with_type(module, PulseOp)
        assert pulse_ops[0].waveform is not pulse_ops[1].waveform


@pytest.mark.parametrize("control_sample_time", [1e-9, 2e-9])
class TestIgnoredShapes:
    """Tests that waveforms whose shape is listed as natively-supported are left as-is."""

    def test_square_is_ignored_by_default(self, control_sample_time):
        module, _ = _build_module_with_pulse(_SQUARE_SPEC)

        assert len(get_operations_with_type(module, SquareWaveformOp)) == 1
        assert _get_sampled_constants(module) == []

        EvaluateWaveformsAsSamples(
            port_sample_times={PORT_CONTROL: control_sample_time},
        ).apply(_CONTEXT, module)

        assert len(get_operations_with_type(module, SquareWaveformOp)) == 1
        assert _get_sampled_constants(module) == []

    @pytest.mark.parametrize("spec", _WAVEFORM_SPECS, ids=_spec_id)
    def test_shape_is_left_untouched_when_ignored(self, spec, control_sample_time):
        module, _ = _build_module_with_pulse(spec)

        assert len(get_operations_with_type(module, spec.op_cls)) == 1
        assert _get_sampled_constants(module) == []

        EvaluateWaveformsAsSamples(
            port_sample_times={PORT_CONTROL: control_sample_time},
            ignored_shapes=(spec.pydantic_cls,),
        ).apply(_CONTEXT, module)

        assert len(get_operations_with_type(module, spec.op_cls)) == 1
        assert _get_sampled_constants(module) == []

    def test_square_is_sampled_when_ignored_shapes_is_empty(self, control_sample_time):
        module, _ = _build_module_with_pulse(_SQUARE_SPEC)

        assert get_operations_with_type(module, SquareWaveformOp) != []
        assert _get_sampled_constants(module) == []

        EvaluateWaveformsAsSamples(
            port_sample_times={PORT_CONTROL: control_sample_time},
            ignored_shapes=(),
        ).apply(_CONTEXT, module)

        assert get_operations_with_type(module, SquareWaveformOp) == []
        assert len(_get_sampled_constants(module)) == 1
