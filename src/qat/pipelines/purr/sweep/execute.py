# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from warnings import warn

from qat.pipelines.base import AbstractPipeline
from qat.pipelines.pipeline import ExecutePipeline


def ExecuteSweepPipeline(base_pipeline: AbstractPipeline):
    """Runtimes now support execution of batched programs in an executable, so the old
    ExecuteSweepPipeline is no longer necessary. This function is kept for backwards
    compatibility, and simply returns the base pipeline after checking its type."""

    warn(
        "The behaviour of ExecuteSweepPipeline is now supported the SimpleRuntime, and an "
        "expclicit ExecuteSweepPipeline is no longer required. This is considered "
        "and will be removed in a future release. ",
        DeprecationWarning,
        stacklevel=2,
    )

    if not base_pipeline.is_subtype_of(ExecutePipeline):
        raise TypeError("The base pipeline must be an ExecutePipeline.")
    return base_pipeline
