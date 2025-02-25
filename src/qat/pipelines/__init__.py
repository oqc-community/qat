def get_default_pipelines():
    from qat.core.pipeline import PipelineImportDescription
    from .echo import echo8, echo16, echo32

    return [
        PipelineImportDescription(name=echo8.name, pipeline=echo8),
        PipelineImportDescription(name=echo16.name, pipeline=echo16),
        PipelineImportDescription(name=echo32.name, pipeline=echo32, default=True),
    ]
