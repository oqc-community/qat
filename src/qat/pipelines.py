from qat.compiler.analysis_passes import InputAnalysis
from qat.compiler.transform_passes import InputOptimisation, Parse
from qat.ir.pass_base import PassManager, TransformPass
from qat.ir.result_base import ResultManager
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.builders import InstructionBuilder


def DefaultCompile(hardware_model):
    pipeline = PassManager()
    return (
        pipeline
        | InputAnalysis()
        | InputOptimisation(hardware_model)
        | Parse(hardware_model)
    )


def DefaultExecute(hardware_model):

    class ExecutionPass(TransformPass):
        """
        This is temporary hack that just holds a hardware model
        """

        def __init__(self, hardware: QuantumHardwareModel):
            self.hardware = hardware

        def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
            pass

    pipeline = PassManager()
    return pipeline | ExecutionPass(hardware_model)
