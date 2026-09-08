# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Experimental pulse-level middleend for the Qblox pipeline.

TODO(COMPILER-1421): Update as part of the implementation of the experimental pulse-level
middleend.

Wraps :class:`~qat.experimental.dialect.pulse.transforms.pipeline.PulsePipelineManager`
— the pulse-level pass pipeline already implemented in ``qat.experimental`` — as a
:class:`~qat.middleend.base.BaseMiddleend` so it can be slotted into a
:class:`~qat.pipelines.pipeline.CompilePipeline`.

.. warning::

    Experimental. This only covers the pulse-level passes described in COMPILER-1421's
    acceptance criteria (passes that run before Q1 lowering); Q1 lowering and code
    generation are handled by the backend (COMPILER-1422), which is not yet implemented.
"""

from compiler_config.config import CompilerConfig
from xdsl.context import Context
from xdsl.dialects.builtin import ModuleOp

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.experimental.dialect.pulse.transforms.pipeline import PulsePipelineManager
from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.middleend.base import BaseMiddleend


class PulseLevelMiddleend(BaseMiddleend):
    """Runs the default pulse-level pass pipeline ahead of Q1 lowering.

    :param model: The canonical system data used to derive pulse-level constraints (timing
        granularity, per-port sample times, native waveform shapes).
    """

    # TODO(COMPILER-1443): Replace with new pipeline infrastructure.

    def __init__(self, model: CanonicalSystemData):
        super().__init__(model=model)
        self._pipeline = PulsePipelineManager.from_canonical_data(
            model
        ).build_default_pipeline()

    def emit(
        self,
        ir: ModuleOp,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        compiler_config: CompilerConfig | None = None,
        **kwargs,
    ) -> ModuleOp:
        """Apply the pulse-level pass pipeline to *ir* in place.

        :param ir: The pulse-level IR emitted by the frontend.
        :returns: The same :class:`ModuleOp`, mutated by the pass pipeline.
        """

        self._pipeline.apply(Context(), ir)
        return ir
