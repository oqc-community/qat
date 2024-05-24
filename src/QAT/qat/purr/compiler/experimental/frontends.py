from qat.purr.backends.calibrations.remote import find_calibration
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.config import CompilerConfig

from qat.purr.compiler.runtime import execute_instructions_with_interrupt
import qat.purr.compiler.frontends as core_frontends
from qat.purr.compiler.interrupt import Interrupt, NullInterrupt
from qat.purr.utils.logging_utils import log_duration


class InterruptableExecutingMixin:
    """
    Support experimental wait interrupt capabilities
    """
    def _execute(
        self,
        hardware,
        compiler_config: CompilerConfig,
        instructions,
        interrupt: Interrupt = NullInterrupt()
    ):

        calibrations = [
            find_calibration(arg) for arg in compiler_config.active_calibrations
        ]

        return execute_instructions_with_interrupt(
            hardware,
            instructions,
            compiler_config.results_format,
            calibrations,
            compiler_config.repeats,
            interrupt
        )

    def execute_with_interrupt(
        self,
        instructions: InstructionBuilder,
        hardware=None,
        compiler_config: CompilerConfig = None,
        interrupt=NullInterrupt(),
    ):

        hardware, compiler_config = self._default_common_args(hardware, compiler_config)

        with log_duration("Execution completed, took {} seconds."):
            return self._execute(hardware, compiler_config, instructions, interrupt)


class QIRFrontend(InterruptableExecutingMixin, core_frontends.QIRFrontend):
    """
    Static decorated QIR frontend for interruptibility
    """
    pass


class QASMFrontend(InterruptableExecutingMixin, core_frontends.QASMFrontend):
    """
    Static decorated QASM frontend for interruptibility
    """
    pass
