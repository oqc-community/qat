# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for :mod:`qat.experimental.dialect.pulse.transforms.pipeline`."""

from __future__ import annotations

from xdsl.passes import PassPipeline

from qat.experimental.dialect.pulse.transforms.constants import OrderedCanonicalizePass
from qat.experimental.dialect.pulse.transforms.granularity_sanitisation import (
    ApplyGranularitySanitisation,
)
from qat.experimental.dialect.pulse.transforms.optimize_contiguous_squashable_instructions import (
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
        EvaluateWaveformsAsSamples,
        TimelineNormalization,
        ApplySquashContiguousOptimizations,
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
        EvaluateWaveformsAsSamples,
        TimelineNormalization,
        ApplySquashContiguousOptimizations,
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
    assert canonicalize_positions == [0, 5]
