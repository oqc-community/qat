import itertools

import numpy as np

from qat.ir.pass_base import QatIR, TransformPass
from qat.ir.result_base import ResultManager
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    Acquire,
    EndRepeat,
    EndSweep,
    Repeat,
    Return,
    Sweep,
)
from qat.utils.algorithm import stable_partition


class ScopeSanitisation(TransformPass):

    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        """
        Bubbles up all sweeps and repeats to the beginning of the list.
        Adds delimiter instructions to the repeats and sweeps signifying the end of their scopes.

        Intended for legacy existing builders and the relative order of instructions guarantees backwards
        compatibility.
        """

        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        head, tail = stable_partition(
            builder.instructions, lambda inst: isinstance(inst, (Sweep, Repeat))
        )

        delimiters = [
            EndSweep() if isinstance(inst, Sweep) else EndRepeat() for inst in head
        ]

        builder.instructions = head + tail + delimiters[::-1]


class RepeatSanitisation(TransformPass):
    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        """
        Fixes repeat instructions if any with default values from the HW model
        Adds a repeat instructions if none is found
        """

        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        model = next((a for a in args if isinstance(a, QuantumHardwareModel)), None)

        if not model:
            model = kwargs.get("model", None)

        if not model or not isinstance(model, QuantumHardwareModel):
            raise ValueError(
                f"Expected to find an instance of {QuantumHardwareModel} in arguments list, but got {model} instead"
            )

        repeats = [inst for inst in builder.instructions if isinstance(inst, Repeat)]
        if repeats:
            for rep in repeats:
                if rep.repeat_count is None:
                    rep.repeat_count = model.default_repeat_count
                if rep.repetition_period is None:
                    rep.repetition_period = model.default_repetition_period
        else:
            builder.repeat(model.default_repeat_count, model.default_repetition_period)


class ReturnSanitisation(TransformPass):
    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        returns = [inst for inst in builder.instructions if isinstance(inst, Return)]
        acquires = [inst for inst in builder.instructions if isinstance(inst, Acquire)]

        if returns:
            unique_variables = set(itertools.chain(*[ret.variables for ret in returns]))
            for ret in returns:
                builder.instructions.remove(ret)
        else:
            # If we don't have an explicit return, imply all results.
            unique_variables = set(acq.output_variable for acq in acquires)

        builder.returns(list(unique_variables))


class DesugaringPass(TransformPass):
    """
    Transforms syntactic sugars and implicit effects. Not that it applies here, but a classical
    example of a sugar syntax is the ternary expression "cond ? val1 : val2" which is nothing more
    than an if else in disguise.

    The goal of this pass is to desugar any syntactic and semantic sugars. One of these sugars
    is iteration constructs such as Sweep and Repeat.
    """

    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        for inst in builder.instructions:
            if isinstance(inst, Sweep):
                count = len(next(iter(inst.variables.values())))
                iter_name = f"sweep_{hash(inst)}"
                inst.variables[iter_name] = np.linspace(1, count, count, dtype=int).tolist()
