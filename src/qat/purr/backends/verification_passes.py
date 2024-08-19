from qat.ir.pass_base import ValidationPass
from qat.purr.backends.qblox.instructions import EndRepeat, EndSweep
from qat.purr.compiler.instructions import Repeat, Return, Sweep
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class ScopeSanitisationValidation(ValidationPass):
    def do_run(self, builder, *args, **kwargs):
        """
        Repeat and Sweep scopes are valid if they have a start and end delimiters and if the delimiters
        are balanced.
        """

        stack = []
        for inst in builder.instructions:
            # Scope stacks
            if isinstance(inst, (Sweep, Repeat)):
                stack.append(inst)
            elif isinstance(inst, (EndSweep, EndRepeat)):
                type = Sweep if isinstance(inst, EndSweep) else Repeat
                try:
                    if not isinstance(stack.pop(), type):
                        raise ValueError(f"Unbalanced {type} scope. Found orphan {inst}")
                except IndexError:
                    raise ValueError(f"Unbalanced {type} scope. Found orphan {inst}")

        if stack:
            raise ValueError(f"Unbalanced scopes. Found orphans {stack}")


class RepeatSanitisationValidation(ValidationPass):
    def do_run(self, builder, *args, **kwargs):
        """
        Repeat and Sweep scopes are valid if they have a start and end delimiters and if the delimiters
        are balanced.
        """

        repeats = [inst for inst in builder.instructions if isinstance(inst, Repeat)]
        if not repeats:
            log.warning("Could not find any repeat instructions")


class ReturnSanitisationValidation(ValidationPass):
    def do_run(self, builder, *args, **kwargs):
        """
        Every builder must have a single return instruction
        """

        returns = [inst for inst in builder.instructions if isinstance(inst, Return)]

        if not returns:
            raise ValueError("Could not find any return instructions")
        elif len(returns) > 1:
            raise ValueError("Found multiple return instructions")


# TODO - bring in stuff in verification.py in here in the form of a pass (or a bunch of passes)
