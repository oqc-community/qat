# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd

from dataclasses import dataclass

import numpy as np

from qat.core.pass_base import AnalysisPass, ResultManager
from qat.core.result_base import ResultInfoMixin
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Repeat


@dataclass
class BatchedShotsResult(ResultInfoMixin):
    total_shots: int
    batched_shots: int


class BatchedShots(AnalysisPass):
    """Determines how shots should be grouped when the total number exceeds that maximum
    allowed.

    The target machine might have an allowed number of shots that can be executed by a
    single execution call. To execute a number of shots greater than this value, shots can
    be batched, with each batch executed by its own "execute" call on the target machine. For
    example, if the maximum number of shots for a target machine is 2000, but you required 4000
    shots, then this could be done as [2000, 2000] shots.

    Now consider the more complex scenario where  4001 shots are required. Clearly this can
    be done in three batches. While it is tempting to do this in batches of [2000, 2000, 1],
    for some target machines, specification of the number of shots can only be achieved at
    compilation (as opposed to runtime). Batching as described above would result in us
    needing to compile two separate programs. Instead, it makes more sense to batch the
    shots as three lots of 1334 shots, which gives a total of 4002 shots. The extra two
    shots can just be discarded at run time.
    """

    def __init__(self, model: QuantumHardwareModel):
        """Instantiate the pass with a hardware model.

        :param model: The hardware model that contains the total number of shots.
        """
        # TODO: replace the hardware model with whatever structures will contain the allowed
        # number of shots in the future.
        # TODO: determine if this should be fused with `RepeatSanitisation`.
        self.model = model

    def run(self, ir: InstructionBuilder, res_mgr: ResultManager, *args, **kwargs):
        """
        :param ir: The list of instructions stored in an :class:`InstructionBuilder`.
        :param res_mgr: The result manager to store the analysis results.
        """

        repeats = [inst for inst in ir.instructions if isinstance(inst, Repeat)]
        if len(repeats) > 0:
            shots = repeats[0].repeat_count
        else:
            shots = self.model.default_repeat_count

        if shots < 0 or not isinstance(shots, int):
            raise ValueError("The number of shots must be a non-negative integer.")

        max_shots = self.model.repeat_limit
        num_batches = int(np.ceil(shots / max_shots))
        if num_batches == 0:
            shots_per_batch = 0
        else:
            shots_per_batch = int(np.ceil(shots / num_batches))
        res_mgr.add(BatchedShotsResult(total_shots=shots, batched_shots=shots_per_batch))
        return ir
