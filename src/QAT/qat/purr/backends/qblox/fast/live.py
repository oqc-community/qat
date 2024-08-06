from typing import Dict, List

import numpy as np

from qat.purr.backends.qblox.fast.codegen import FastQbloxEmitter
from qat.purr.backends.qblox.live import QbloxLiveEngine
from qat.purr.compiler.control_flow.instructions import EndRepeat, EndSweep
from qat.purr.compiler.instructions import (
    DeviceUpdate,
    Instruction,
    Repeat,
    Sweep,
    SweepValue,
)
from qat.purr.compiler.interrupt import Interrupt, NullInterrupt
from qat.purr.utils.logger import get_default_logger
from qat.purr.utils.logging_utils import log_duration

log = get_default_logger()


class FastQbloxLiveEngine(QbloxLiveEngine):
    def optimize(self, instructions: List[Instruction]) -> List[Instruction]:
        """
        Bubbles up all sweeps and repeats to the beginning of the list.
        Decomposes complex multi-dim sweeps into simple one-dim sweeps.
        Batches big repeats into repeats that can run on the hardware.
        Adds delimiter instructions to the repeats and sweeps signifying the end of their scopes.
        Collects targets AOT.
        """
        instructions = super().optimize(instructions)

        # Decompose complex Sweeps and bubble them up along with Repeats to the top
        head = []
        body = []
        tail = []
        for i, inst in enumerate(instructions):
            if isinstance(inst, Sweep):
                for name, value in inst.variables.items():
                    head.append(Sweep(SweepValue(name, value)))
                    tail.append(EndSweep())
            elif isinstance(inst, Repeat):
                head.append(inst)
                tail.append(EndRepeat())
            else:
                body.append(inst)

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

    def validate(self, instructions: List[Instruction]):
        """
        Repeat and Sweep scopes are valid if they have a start and end delimiters and if the delimiters
        are balanced.
        """
        super().validate(instructions)

        stack = []
        for inst in instructions:
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

    def _common_execute(self, instructions, interrupt: Interrupt = NullInterrupt()):
        """Executes this qat file against this current hardware."""
        self._model_exists()

        with log_duration("QPU returned results in {} seconds."):
            instructions = self.optimize(instructions)
            packages = FastQbloxEmitter(instructions).emit_packages()
            self.model.control_hardware.set_data(packages)
            playback_results: Dict[str, np.ndarray] = (
                self.model.control_hardware.start_playback(None, None)
            )

            # Process metadata assign/return values to make sure the data is in the
            # right form.
            # results = self._process_results(results, qat_file)
            # results = self._process_assigns(results, qat_file)

            # return results

            return playback_results
