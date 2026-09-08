# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Experimental Qblox execute pipeline.

.. note:: Related work (TODOs)

        * COMPILER-1417: Ensure QBlox payloads are compatible with the bSLAM specification.
        * COMPILER-1418: Expand bSLAM zero-engine integration testing.

Executes compiled QBlox programs on a live Qblox cluster using
:class:`~qat.experimental.system_data.canonical.schema.CanonicalSystemData` as the hardware
model.

As with the compile pipeline, there is no ``BaseModelLoader`` for ``CanonicalSystemData``,
so this pipeline must be instantiated directly rather than wired up via qatconfig::

    from qat.experimental.pipelines.execute import (
        ExperimentalQbloxExecutePipeline,
        ExperimentalQbloxExecutePipelineConfig,
    )

    pipeline = ExperimentalQbloxExecutePipeline(
        config=ExperimentalQbloxExecutePipelineConfig(host="127.0.0.1"),
        model=canonical_system_data,
    )
"""

from qat.backend.qblox.target_data import TARGET_DATA, QbloxTargetData
from qat.engines import NativeEngine
from qat.engines.qblox.execution import QbloxEngine
from qat.engines.qblox.live import QbloxLeafInstrument
from qat.experimental.runtime.results_pipeline import get_qblox_results_pipeline
from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.utils.logging import get_logger
from qat.pipelines.pipeline import ExecutePipeline
from qat.pipelines.updateable import PipelineConfig, UpdateablePipeline
from qat.runtime import SimpleRuntime
from qat.runtime.aggregator import QBloxAggregator

log = get_logger(__name__)


class ExperimentalQbloxExecutePipelineConfig(PipelineConfig):
    """Configuration for :class:`ExperimentalQbloxExecutePipeline`.

    :param name: The name of the pipeline, defaults to "experimental_qblox_bslam".
    :param host: The host address for the Qblox cluster.
    """

    name: str = "experimental_qblox_bslam"
    host: str


class ExperimentalQbloxExecutePipeline(UpdateablePipeline):
    # TODO(COMPILER-1443): Replace with new pipeline infrastructure.
    """Executes compiled ``QbloxProgram`` objects on a live Qblox cluster.

    .. warning::

        This pipeline executes compiled programs only. Select an appropriate experimental
        compilation pipeline when compilation is required beforehand. This is an
        experimental feature and may change in future releases.
    """

    @staticmethod
    def _build_pipeline(
        config: ExperimentalQbloxExecutePipelineConfig,
        model: CanonicalSystemData,
        target_data: QbloxTargetData | None = None,
        engine: NativeEngine | None = None,
    ) -> ExecutePipeline:
        if engine is not None:
            log.warning(
                "An engine was provided to the ExperimentalQbloxExecutePipeline, but it is "
                "intended to be built from the configured Qblox cluster. The provided engine "
                "will be "
                "ignored."
            )

        target_data = target_data if target_data is not None else TARGET_DATA
        instrument = QbloxLeafInstrument(
            id=config.name,
            name=config.name,
            address=config.host,
        )
        engine = QbloxEngine(instrument)
        return ExecutePipeline(
            name=config.name,
            model=model,
            target_data=target_data,
            runtime=SimpleRuntime(
                engine=engine,
                results_pipeline=get_qblox_results_pipeline(),
                aggregator=QBloxAggregator(),
            ),
        )
