# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Experimental QBlox backend stub for the M1 pipeline.

TODO(COMPILER-1422): Implement the experimental QBlox backend for the M1 pipeline.
TODO(COMPILER-1443): Replace with new pipeline infrastructure.

.. warning::

    Q1 lowering and QBlox code generation from the new pulse/Q1 IR stack are not yet
    implemented — COMPILER-1422 ("Write an experimental Qblox Backend pipeline to compile
    for M1") is still In Progress as of Sprint 36. This class exists so
    :class:`~qat.experimental.pipelines.compile.ExperimentalQbloxCompilePipeline`
    has a concrete, structurally-correct backend slot to swap the real implementation into
    once it lands, rather than leaving the pipeline unbuildable in the meantime.
"""

from xdsl.dialects.builtin import ModuleOp

from qat.backend.base import BaseBackend
from qat.backend.qblox.execution import QbloxProgram
from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.executables import Executable
from qat.experimental.system_data.canonical.schema import CanonicalSystemData


class ExperimentalQbloxBackend(BaseBackend[QbloxProgram]):
    """Placeholder backend for Q1 lowering + QBlox code generation (COMPILER-1422).

    :param model: The canonical system data the eventual backend will lower against.
    """

    def __init__(self, model: CanonicalSystemData):
        super().__init__(model=model)

    def emit(
        self,
        ir: ModuleOp,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        **kwargs,
    ) -> Executable[QbloxProgram]:
        raise NotImplementedError(
            "Q1 lowering and QBlox code generation for the experimental M1 pipeline are "
            "not yet implemented (COMPILER-1422, in progress). Replace "
            "ExperimentalQbloxBackend with the real implementation once it lands."
        )
