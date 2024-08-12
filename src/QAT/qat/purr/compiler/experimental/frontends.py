from qat.purr.compiler.frontends import QASMFrontend, QIRFrontend
from qat.purr.compiler.interrupt import NullInterrupt


class InterruptableFrontend:
    ...


class InterruptableQIRFrontend(InterruptableFrontend, QIRFrontend):
    """
    Static decorated QIR frontend for interruptibility
    """
    def __init__(self, interruption=None, *args, **kwargs):
        super().__init__(*args, **kwargs, interruption=NullInterrupt() if interruption is None else interruption)


class InterruptableQASMFrontend(InterruptableFrontend, QASMFrontend):
    """
    Static decorated QASM frontend for interruptibility
    """
    def __init__(self, interruption=None, *args, **kwargs):
        super().__init__(*args, **kwargs, interruption=NullInterrupt() if interruption is None else interruption)
