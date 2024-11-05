import itertools

from qat.ir.pass_base import TransformPass
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
    SweepValue,
)
from qat.utils.algorithm import stable_partition


class SweepDecomposition(TransformPass):

    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        Decomposes complex multi-dim sweeps into simpler one-dim sweeps.
        """
        result = []
        for i, inst in enumerate(builder.instructions):
            if isinstance(inst, Sweep) and len(inst.variables) > 1:
                for name, value in inst.variables.items():
                    result.append(Sweep(SweepValue(name, value)))
            else:
                result.append(inst)
        builder.instructions = result


class ScopeSanitisation(TransformPass):

    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        Bubbles up all sweeps and repeats to the beginning of the list.
        Adds delimiter instructions to the repeats and sweeps signifying the end of their scopes.

        Intended for legacy existing builders and the relative order of instructions guarantees backwards
        compatibility.
        """

        head, tail = stable_partition(
            builder.instructions, lambda inst: isinstance(inst, (Sweep, Repeat))
        )

        delimiters = [
            EndSweep() if isinstance(inst, Sweep) else EndRepeat() for inst in head
        ]

        builder.instructions = head + tail + delimiters[::-1]


class RepeatSanitisation(TransformPass):
    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        Fixes repeat instructions if any with default values from the HW model
        Adds a repeat instructions if none is found
        """

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
    def run(self, builder: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
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
