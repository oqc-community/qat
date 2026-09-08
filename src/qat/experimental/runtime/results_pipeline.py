# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Experimental Qblox result processing into structured result records."""

from __future__ import annotations

from qat.core.pass_base import PassManager
from qat.runtime.passes.transform import (
    AssignResultsTransform,
    InlineResultsProcessingTransform,
    QBloxAcquisitionPostProcessing,
    ResultTransform,
)


def get_qblox_results_pipeline() -> PassManager:
    """Build the experimental Qblox results pipeline for :class:`SimpleRuntime`.

    ``SimpleRuntime`` already owns execution batching, connection handling, and the
    standard ``QBloxAggregator``. The pipeline converts aggregated Qblox acquisitions into
    processed result records, including inline processing, assignments, and result
    formatting.
    """

    # This pipeline does not include error mitigation passes as they rely on the legacy
    # QuantumHardwareModel. Future work will implement the results processing passes using
    # the new xDSL IR and system data views.

    pipeline = (
        PassManager()
        | QBloxAcquisitionPostProcessing()
        | InlineResultsProcessingTransform()
        | AssignResultsTransform()
        | ResultTransform()
    )
    return pipeline
