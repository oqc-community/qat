# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Experimental Qblox compile pipeline.

Wires together the experimental frontend, middleend, and backend for the experimental Qblox
target, using a
:class:`~qat.experimental.system_data.canonical.schema.CanonicalSystemData`
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

The ``program`` passed to :meth:`compile` must be a ``QuantumInstructionBuilder`` (PuRR IR)
because the experimental frontend currently exposes only the PuRR importer.

The inherited ``target_data`` argument carries the Qblox limits used by the backend and
lowering pipeline.
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
    """Compiles programs for the experimental Qblox target using the frontend, middleend,
    and backend stack.

    This pipeline is for compilation purposes only and does not execute programs.
    """

    @staticmethod
    def _build_pipeline(
        config: ExperimentalQbloxCompilePipelineConfig,
        model: CanonicalSystemData,
        target_data: QbloxTargetData | None = None,
        engine: None = None,
    ) -> CompilePipeline:
        target_data = TARGET_DATA if target_data is None else target_data
        return CompilePipeline(
            name=config.name,
            model=model,
            target_data=target_data,
            frontend=PurrFrontend(model=model),
            middleend=PulseLevelMiddleend(model=model),
            backend=ExperimentalQbloxBackend(model=model, target_data=target_data),
        )
