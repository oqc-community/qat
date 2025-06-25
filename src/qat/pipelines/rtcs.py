# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.pipelines.pipeline import Pipeline
from qat.pipelines.updateable import UpdateablePipeline


class RTCSPipeline(UpdateablePipeline):
    """A pipeline that compiles programs using the real time chip simulator."""

    @staticmethod
    def _build_pipeline(config, model, target_data=None, engine=None) -> Pipeline:
        raise NotImplementedError("This pipeline is not implemented yet")
