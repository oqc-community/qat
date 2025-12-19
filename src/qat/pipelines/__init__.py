# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
def get_default_pipelines():
    """Returns a list of descriptions for default full pipelines, compile pipelines, and
    execute pipelines."""

    from qat.core.config.descriptions import PipelineInstanceDescription

    from .echo import echo8, echo16, echo32

    full_pipelines = [
        PipelineInstanceDescription(name=echo8.name, pipeline=echo8),
        PipelineInstanceDescription(name=echo16.name, pipeline=echo16),
        PipelineInstanceDescription(name=echo32.name, pipeline=echo32, default=True),
    ]
    compile_pipelines = []
    execute_pipelines = []
    return full_pipelines, compile_pipelines, execute_pipelines
