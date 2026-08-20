# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for :mod:`qat.experimental.dialect.pulse.transforms.pipeline`."""

from __future__ import annotations

import pytest
from xdsl.passes import PassPipeline
from xdsl.utils.exceptions import VerifyException

from qat.experimental.dialect.pulse.transforms.constants import OrderedCanonicalizePass
from qat.experimental.dialect.pulse.transforms.waveform_evaluation import (
    EvaluateWaveformsAsSamples,
)
from qat.experimental.passes.pass_ordering import OrderedPassPipeline
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


def test_pipeline_rejects_evaluation_before_constant_propagation():
    with pytest.raises(VerifyException, match="requires 'constant-propagation'"):
        OrderedPassPipeline(
            (
                EvaluateWaveformsAsSamples(constraints=_constraints()),
                OrderedCanonicalizePass(),
            )
        )


def test_pipeline_rejects_missing_constant_propagation():
    with pytest.raises(VerifyException, match="requires 'constant-propagation'"):
        OrderedPassPipeline((EvaluateWaveformsAsSamples(constraints=_constraints()),))
