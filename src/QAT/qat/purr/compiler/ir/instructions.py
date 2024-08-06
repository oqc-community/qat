from qat.purr.compiler.instructions import Instruction


class EndSweep(Instruction):
    """
    Basic scoping. Marks the end of the most recent sweep
    """

    pass


class EndRepeat(Instruction):
    """
    Basic scoping. Marks the end of the most recent repeat
    """

    pass
