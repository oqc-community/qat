from qat.ir.pass_base import ValidationPass
from qat.ir.result_base import ResultManager
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Repeat, Return
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class RepeatSanitisationValidation(ValidationPass):
    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        Checks if the builder has a repeat instruction and warns if none exists.
        """

        repeats = [inst for inst in builder.instructions if isinstance(inst, Repeat)]
        if not repeats:
            log.warning("Could not find any repeat instructions")


class ReturnSanitisationValidation(ValidationPass):
    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        Every builder must have a single return instruction
        """

        returns = [inst for inst in builder.instructions if isinstance(inst, Return)]

        if not returns:
            raise ValueError("Could not find any return instructions")
        elif len(returns) > 1:
            raise ValueError("Found multiple return instructions")


class NCOFrequencyVariability(ValidationPass):
    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        model = next((a for a in args if isinstance(a, QuantumHardwareModel)), None)

        if not model:
            model = kwargs.get("model", None)

        if not model or not isinstance(model, QuantumHardwareModel):
            raise ValueError(
                f"Expected to find an instance of {QuantumHardwareModel} in arguments list, but got {model} instead"
            )

        for channel in model.pulse_channels.values():
            if channel.fixed_if:
                raise ValueError("Cannot allow constance of the NCO frequency")


# TODO - bring in stuff from verification.py in here in the form of a pass (or a bunch of passes)
