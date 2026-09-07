# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Experimental M1 compile pipeline (COMPILER-1409 / COMPILER-1421 / COMPILER-1422).

Wires together the experimental frontend, middleend, and backend for the M1 QBlox target,
using a :class:`~qat.experimental.system_data.canonical.schema.CanonicalSystemData`
instance rather than the legacy ``QuantumHardwareModel``/``PhysicalHardwareModel`` used by
production pipelines such as :class:`~qat.pipelines.purr.qblox.compile.QbloxCompilePipeline1`.

There is no :class:`~qat.model.loaders.base.BaseModelLoader` that produces
``CanonicalSystemData`` (it can be materialised from a supported source payload through
:func:`~qat.experimental.system_data.materialisers.boundary.materialise`), so this pipeline
cannot be wired up through a qatconfig ``HARDWARE``/``hardware_loader`` entry. Instantiate it
directly instead::

    from qat.experimental.pipelines.compile import (
        ExperimentalQbloxCompilePipeline,
        ExperimentalQbloxCompilePipelineConfig,
    )

    pipeline = ExperimentalQbloxCompilePipeline(
        config=ExperimentalQbloxCompilePipelineConfig(),
        model=canonical_system_data,
    )

.. warning::

    Experimental. :meth:`compile` will raise ``NotImplementedError`` at the backend stage
    until COMPILER-1422 (QBlox code generation from the new IR stack) is implemented. The
    frontend (COMPILER-1409) and middleend (COMPILER-1421 pulse-level passes) already run
    against real components. The ``program`` passed to ``compile`` must be a
    ``QuantumInstructionBuilder`` (PuRR IR) — the experimental frontend only exposes the
    PuRR importer, per COMPILER-1409's acceptance criteria.
"""

from qat.backend.qblox.target_data import TARGET_DATA, QbloxTargetData
from qat.experimental.backend.qblox.backend import ExperimentalQbloxBackend
from qat.experimental.frontend.purr import PurrFrontend
from qat.experimental.middleend.middleend import PulseLevelMiddleend
from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.pipelines.pipeline import CompilePipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline


class ExperimentalQbloxCompilePipelineConfig(PipelineConfig):
    """Configuration for :class:`ExperimentalQbloxCompilePipeline`.

    :param name: The name of the pipeline, defaults to "experimental_qblox".
    """

    name: str = "experimental_qblox"


class ExperimentalQbloxCompilePipeline(UpdateablePipeline):
    # TODO(COMPILER-1443): Replace with new pipeline infrastructure.
    """Compiles programs for M1 using the experimental frontend/middleend/backend stack.

    .. warning::

        This pipeline is for compilation purposes only and does not execute programs.
        Backend code generation is not yet implemented (COMPILER-1422); compilation will
        raise ``NotImplementedError`` once it reaches that stage.
    """

    @staticmethod
    def _build_pipeline(
        config: ExperimentalQbloxCompilePipelineConfig,
        model: CanonicalSystemData,
        target_data: QbloxTargetData | None = None,
        engine: None = None,
    ) -> CompilePipeline:
        target_data = target_data if target_data is not None else TARGET_DATA
        return CompilePipeline(
            name=config.name,
            model=model,
            target_data=target_data,
            frontend=PurrFrontend(model=model),
            middleend=PulseLevelMiddleend(model=model),
            backend=ExperimentalQbloxBackend(model=model),
        )
