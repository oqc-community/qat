def get_default_pipelines():
    from qat.core.pipeline import PipelineInstanceDescription
    from .echo import echo8, echo16, echo32

    return [
        PipelineInstanceDescription(name=echo8.name, pipeline=echo8),
        PipelineInstanceDescription(name=echo16.name, pipeline=echo16),
        PipelineInstanceDescription(name=echo32.name, pipeline=echo32, default=True),
    ]
