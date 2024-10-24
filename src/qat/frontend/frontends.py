from compiler_config.config import CompilerConfig

import qat.purr.compiler.frontends as core_frontends
from qat.purr.backends.calibrations.remote import find_calibration
from qat.purr.backends.qiskit_simulator import QiskitEngine, QiskitHardwareModel
from qat.purr.compiler.interrupt import Interrupt, NullInterrupt
from qat.purr.compiler.runtime import (
    _execute_instructions_with_interrupt,
    execute_instructions,
)


class InterruptableExecutingMixin:
    """
    Support experimental wait interrupt capabilities
    """

    def _execute(
        self,
        hardware,
        compiler_config: CompilerConfig,
        instructions,
        *args,
        interrupt: Interrupt = NullInterrupt(),
        **kwargs,
    ):

        calibrations = [
            find_calibration(arg) for arg in compiler_config.active_calibrations
        ]

        exe_method = _execute_instructions_with_interrupt
        if isinstance(hardware, (QiskitHardwareModel, QiskitEngine)):
            exe_method = execute_instructions

        return exe_method(hardware, instructions, compiler_config, calibrations, interrupt)


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
