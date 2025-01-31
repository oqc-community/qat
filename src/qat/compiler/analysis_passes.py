# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Oxford Quantum Circuits Ltd

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from compiler_config.config import Languages

from qat.ir.pass_base import AnalysisPass, QatIR, ResultManager
from qat.ir.result_base import ResultInfoMixin
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import Repeat

path_regex = re.compile(r"^.+\.(qasm|ll|bc)$")
string_regex = re.compile(
    r"((?P<qasm>OPENQASM) (?P<version>[0-9]+)(?:.[0-9])?;)|(?P<qiskit>@__quantum__qis)"
)


@dataclass
class InputAnalysisResult(ResultInfoMixin):
    language: Languages = Languages.Empty
    raw_input: Union[str, bytes] = None


class InputAnalysis(AnalysisPass):
    def run(self, ir: QatIR, res_mgr: ResultManager, *args, **kwargs):
        result = InputAnalysisResult()
        path_or_str = ir.value
        if isinstance(path_or_str, bytes) and path_or_str.startswith(b"BC"):
            result.language = Languages.QIR
            result.raw_input = path_or_str
        elif path_regex.match(path_or_str) is not None:
            result.language, result.raw_input = self._process_path_string(path_or_str)
        else:
            result.language, result.raw_input = (
                self._process_string(path_or_str),
                path_or_str,
            )
        res_mgr.add(result)
        if result.language is Languages.Empty:
            raise ValueError("Unable to determine input language.")

    def _process_path_string(self, path_string):
        path = Path(path_string)
        if path.suffix in (".qasm", ".ll"):
            with path.open() as file:
                string = file.read()
            return self._process_string(string), string
        elif path.suffix == ".bc":
            with path.open("rb") as file:
                bytes_ = file.read()
            return Languages.QIR, bytes_

    def _process_string(self, string):
        match = string_regex.search(string)
        if match is not None:
            if match.group("qasm"):
                version = match.group("version")[0]
                if version == "2":
                    return Languages.Qasm2
                elif version == "3":
                    return Languages.Qasm3
            elif match.group("qiskit"):
                return Languages.QIR
        return Languages.Empty


@dataclass
class BatchedShotsResult(ResultInfoMixin):
    total_shots: int
    batched_shots: int


class BatchedShots(AnalysisPass):
    """
    Determines how shots should be grouped when the total number exceeds that maximum allowed.

    The target backend might have an allowed number of shots that can be executed by a single
    execution call. To execute a number of shots greater than this value, shots can be
    batched, with each batch executed by its own "execute" call on the backend. For example,
    if the maximum number of shots for a backend is 2000, but you required 4000 shots, then
    this could be done as [2000, 2000] shots.

    Now consider the more complex scenario where  4001 shots are required. Clearly this can
    be done in three batches. While it is tempting to do this in batches of [2000, 2000, 1],
    for some backends, specification of the number of shots can only be achieved at
    compilation (as opposed to runtime). Batching as described above would result in us
    needing to compile two separate programs. Instead, it makes more sense to batch the shots
    as three lots of 1334 shots, which gives a total of 4002 shots. The extra two shots can
    just be discarded at run time.
    """

    def __init__(self, model: QuantumHardwareModel):
        """
        Instantiate the pass with a hardware model.

        :param QuantumHardwareModel model: The hardware model that contains the total number
            of shots.
        """
        # TODO: replace the hardware model with whatever structures will contain the allowed
        # number of shots in the future.
        # TODO: determine if this should be fused with `RepeatSanitisation`.
        self.model = model

    def run(
        self,
        ir: QatIR,
        res_mgr: ResultManager,
        *args,
        **kwargs,
    ):
        """
        :param QatIR ir: The :class:`InstructionBuilder` wrapped in :class:`QatIR`.
        :param ResultManager res_mgr: The result manager to store the analysis results.
        """
        builder = ir.value
        if not isinstance(builder, InstructionBuilder):
            raise ValueError(f"Expected InstructionBuilder, got {type(builder)}")

        repeats = [inst for inst in builder.instructions if isinstance(inst, Repeat)]
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
