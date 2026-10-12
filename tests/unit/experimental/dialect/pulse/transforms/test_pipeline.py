# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for :mod:`qat.experimental.dialect.pulse.transforms.pipeline`."""

from __future__ import annotations

from numpy.testing import assert_array_equal
from xdsl.context import Context
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import BoolAttr, FloatAttr, StringAttr, f64, i1
from xdsl.passes import PassPipeline
from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination

from qat.experimental.conversion.pulse.lower_kernels_to_arrays import (
    LowerKernelsToResultsArrays,
)
from qat.experimental.dialect.pulse.ir import (
    AmplitudeAttr,
    ConstantOp,
    CreateFrameOp,
    FrequencyAttr,
    TimeAttr,
)
from qat.experimental.dialect.pulse.ir.attributes import SampledWaveformAttr
from qat.experimental.dialect.pulse.ir.ops import GaussianWaveformOp, PulseOp
from qat.experimental.dialect.pulse.transforms.constants import OrderedCanonicalizePass
from qat.experimental.dialect.pulse.transforms.dead_frame_elimination import (
    DeadFrameEliminationPass,
)
from qat.experimental.dialect.pulse.transforms.granularity_sanitisation import (
    ApplyGranularitySanitisation,
)
from qat.experimental.dialect.pulse.transforms.optimize_contiguous_squashable_instructions import (  # noqa: E501
    ApplySquashContiguousOptimizations,
)
from qat.experimental.dialect.pulse.transforms.pipeline import PulsePipelineManager
from qat.experimental.dialect.pulse.transforms.timeline_normalization import (
    TimelineNormalization,
)
from qat.experimental.dialect.pulse.transforms.waveform_evaluation import (
    EvaluateWaveformsAsSamples,
)
from qat.experimental.passes.pass_ordering import OrderedPassPipeline
from qat.experimental.system_data.canonical.schema import CanonicalSystemData, PortData
from qat.experimental.system_data.pulse.constraints import (
    PortConstraints,
    PulseLevelConstraints,
)
from qat.experimental.waveforms.evaluate import evaluate_waveform

from tests.unit.utils.ir import build_module_from_ops, get_operations_with_type


def _constraints() -> PulseLevelConstraints:
    return PulseLevelConstraints(
        ports={
            "channel_1": PortConstraints(
                sample_time_ps=1000,
                min_duration_ps=0,
                max_duration_ps=None,
                acquire_allowed=True,
            )
        },
        granularity_ps=8000,
        native_waveform_shapes=(),
    )


def _canonical_data() -> CanonicalSystemData:
    return CanonicalSystemData(
        ports=(
            PortData(
                id="channel_1",
                sample_time=1000,
                block_size=8,
                min_blocks=0,
                max_blocks=-1,
                acquire_allowed=True,
                native_waveform_shapes=(),
            ),
        ),
    )


def test_default_pulse_pipeline_runs_constant_propagation_before_evaluation():
    pipeline = OrderedPassPipeline(
        (
            OrderedCanonicalizePass(),
            EvaluateWaveformsAsSamples(constraints=_constraints()),
        )
    )

    assert isinstance(pipeline, PassPipeline)
    assert [type(p) for p in pipeline.passes] == [
        OrderedCanonicalizePass,
        EvaluateWaveformsAsSamples,
    ]


def test_default_pulse_pipeline_carries_forward_constraints():
    constraints = _constraints()
    pipeline = OrderedPassPipeline(
        (
            OrderedCanonicalizePass(),
            EvaluateWaveformsAsSamples(constraints=constraints),
        )
    )

    evaluation = pipeline.passes[-1]
    assert isinstance(evaluation, EvaluateWaveformsAsSamples)
    assert evaluation.constraints is constraints


def test_pipeline_manager_produces_correct_pass_order():
    manager = PulsePipelineManager(constraints=_constraints())
    pipeline = manager.build_default_pipeline()

    assert [type(p) for p in pipeline.passes] == [
        OrderedCanonicalizePass,
        ApplyGranularitySanitisation,
        CommonSubexpressionElimination,
        EvaluateWaveformsAsSamples,
        TimelineNormalization,
        DeadFrameEliminationPass,
        ApplySquashContiguousOptimizations,
        LowerKernelsToResultsArrays,
        OrderedCanonicalizePass,
    ]


def test_pipeline_manager_carries_constraints_to_passes():
    constraints = _constraints()
    manager = PulsePipelineManager(constraints=constraints)
    pipeline = manager.build_default_pipeline()

    passes_by_type = {type(p): p for p in pipeline.passes}
    assert passes_by_type[EvaluateWaveformsAsSamples].constraints is constraints
    assert passes_by_type[ApplyGranularitySanitisation].constraints is constraints


def test_pipeline_manager_returns_ordered_pass_pipeline():
    manager = PulsePipelineManager(constraints=_constraints())
    pipeline = manager.build_default_pipeline()

    assert isinstance(pipeline, OrderedPassPipeline)
    assert isinstance(pipeline, PassPipeline)


def test_pipeline_manager_from_canonical_data():
    manager = PulsePipelineManager.from_canonical_data(_canonical_data())

    assert isinstance(manager, PulsePipelineManager)
    assert isinstance(manager.constraints, PulseLevelConstraints)
    assert manager.constraints.granularity_ps == 8000
    assert "channel_1" in manager.constraints.ports


def test_pipeline_manager_from_canonical_data_builds_valid_pipeline():
    manager = PulsePipelineManager.from_canonical_data(_canonical_data())
    pipeline = manager.build_default_pipeline()

    assert [type(p) for p in pipeline.passes] == [
        OrderedCanonicalizePass,
        ApplyGranularitySanitisation,
        CommonSubexpressionElimination,
        EvaluateWaveformsAsSamples,
        TimelineNormalization,
        DeadFrameEliminationPass,
        ApplySquashContiguousOptimizations,
        LowerKernelsToResultsArrays,
        OrderedCanonicalizePass,
    ]


def test_pipeline_manager_reuses_canonicalize_as_cleanup():
    manager = PulsePipelineManager(constraints=_constraints())
    pipeline = manager.build_default_pipeline()

    canonicalize_positions = [
        index
        for index, p in enumerate(pipeline.passes)
        if isinstance(p, OrderedCanonicalizePass)
    ]
    assert canonicalize_positions == [0, 8]


def test_pipeline_manager_lowers_kernels_before_final_cleanup():
    manager = PulsePipelineManager(constraints=_constraints())
    pipeline = manager.build_default_pipeline()

    pass_types = [type(p) for p in pipeline.passes]
    lowering_index = pass_types.index(LowerKernelsToResultsArrays)

    # Kernel lowering must run after the core pulse passes and immediately before the
    # final constant-propagation cleanup.
    assert pass_types[lowering_index - 1] is ApplySquashContiguousOptimizations
    assert pass_types[lowering_index + 1] is OrderedCanonicalizePass
    assert lowering_index == len(pass_types) - 2


def test_default_pipeline_samples_identical_waveforms_once_per_sample_time():
    """Identical waveforms on ports with different sample times are merged, then sampled
    once per sample time, so each pulse gets the samples for its own port."""
    sample_times = {"channel_1": 1_000, "channel_2": 2_000}
    constraints = PulseLevelConstraints(
        ports={
            port: PortConstraints(
                sample_time_ps=sample_time,
                min_duration_ps=0,
                max_duration_ps=None,
                acquire_allowed=True,
            )
            for port, sample_time in sample_times.items()
        },
        granularity_ps=8000,
        native_waveform_shapes=(),
    )

    width, amplitude, breadth = 80_000, 0.5, 0.47
    operands = (
        ConstantOp(TimeAttr(width)),
        ConstantOp(AmplitudeAttr(amplitude)),
        ArithConstantOp(FloatAttr(breadth, 64), f64),
    )
    ops = list(operands)
    pulses = {}
    waveforms = []
    for port in sample_times:
        frame = CreateFrameOp(ConstantOp(FrequencyAttr(5e9)), StringAttr(port))
        waveform = GaussianWaveformOp(*operands, BoolAttr(False, i1))
        waveforms.append(waveform)
        pulses[port] = PulseOp(frame, waveform)
        ops.extend([frame.frequency.owner, frame, waveform, pulses[port]])
    module = build_module_from_ops(ops)
    assert len(get_operations_with_type(module, GaussianWaveformOp)) == 2

    PulsePipelineManager(constraints).build_default_pipeline().apply(Context(), module)

    sampled = [
        op
        for op in get_operations_with_type(module, ConstantOp)
        if isinstance(op.value, SampledWaveformAttr)
    ]
    assert get_operations_with_type(module, GaussianWaveformOp) == []
    assert sorted(op.value.sample_time.literal_value for op in sampled) == [1_000, 2_000]

    shape = waveforms[0].build_shape()
    for port, sample_time in sample_times.items():
        attr = pulses[port].waveform.owner.value
        assert attr.sample_time.literal_value == sample_time
        assert_array_equal(
            attr.samples.data,
            evaluate_waveform(
                width=width, sample_time=sample_time, shape=shape, amplitude=amplitude
            ),
        )
