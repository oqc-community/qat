from qat.backend.validation_passes import HardwareConfigValidity
from qat.compiler.analysis_passes import InputAnalysis
from qat.compiler.transform_passes import (
    InputOptimisation,
    Parse,
    PhaseOptimisation,
    PostProcessingOptimisation,
)
from qat.compiler.validation_passes import InstructionValidation, ReadoutValidation
from qat.ir.pass_base import PassManager
from qat.runtime.analysis_passes import CalibrationAnalysis
from qat.runtime.transform_passes import ErrorMitigation, ResultTransform


def DefaultCompile(hardware_model):
    pipeline = PassManager()
    return (
        pipeline
        | InputAnalysis()
        | InputOptimisation(hardware_model)
        | Parse(hardware_model)
    )


def DefaultExecute(hardware_model, engine=None):
    if engine is None:
        engine = hardware_model.create_engine()
    pipeline = PassManager()
    return (
        pipeline
        | HardwareConfigValidity(hardware_model)
        | CalibrationAnalysis()
        | PhaseOptimisation()
        | PostProcessingOptimisation()
        | InstructionValidation(engine)
        | ReadoutValidation(hardware_model)
    )


def DefaultPostProcessing(hardware_model):
    pipeline = PassManager()
    return pipeline | ResultTransform() | ErrorMitigation(hardware_model)
