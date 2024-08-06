from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.instructions import (
    DeviceUpdate,
    Repeat,
    Sweep,
    SweepValue,
)
from qat.purr.utils.algorithm import stable_partition


class TransformPass:
    def run(self, builder: InstructionBuilder, *args):
        pass


class SweepDecomposition(TransformPass):
    def run(self, builder: InstructionBuilder, *args):
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
        return result


class ScopeBalancing(TransformPass):
    def run(self, builder: InstructionBuilder, *args):
        """
        Bubbles up all sweeps and repeats to the beginning of the list.
        Adds delimiter instructions to the repeats and sweeps signifying the end of their scopes.
        Collects targets AOT.
        """

        head, tail = stable_partition(
            builder.instructions, lambda inst: isinstance(inst, (Sweep, Repeat))
        )

        # Bubble up DeviceUpdates just after their associated sweeps
        new_head = []
        to_exclude = []
        for h in head:
            new_head.append(h)
            if isinstance(h, Sweep):
                name = next(iter(h.variables.keys()))
                for i, b in enumerate(body):
                    if isinstance(b, DeviceUpdate) and b.value.name == name:
                        new_head.append(b)
                        to_exclude.append(i)
        body = [b for i, b in enumerate(body) if i not in to_exclude]

        return new_head + body + tail[::-1]
