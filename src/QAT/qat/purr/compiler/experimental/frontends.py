from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.config import CompilerConfig

from qat.purr.compiler.runtime import execute_instructions_with_interrupt_via_config
import qat.purr.compiler.frontends as core_frontends
from qat.purr.compiler.interrupt import NullInterrupt


class InterruptableExecutingMixin:
    """
    Support experimental wait interrupt capabilities
    """

    def execute_with_interrupt(
        self,
        instructions: InstructionBuilder,
        hardware=None,
        compiler_config: CompilerConfig = None,
        interrupt=NullInterrupt(),
    ):

        return execute_instructions_with_interrupt_via_config(
            hardware, instructions, compiler_config, interrupt
        )


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
