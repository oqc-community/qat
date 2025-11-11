# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import re
from dataclasses import dataclass
from pathlib import Path

from compiler_config.config import Languages

from qat.core.pass_base import AnalysisPass, ResultManager
from qat.core.result_base import ResultInfoMixin

path_regex = re.compile(r"^.+\.(qasm|ll|bc)$")
string_regex = re.compile(
    r"((?P<qasm>OPENQASM) (?P<version>[0-9]+)(?:.[0-9])?;)|(?P<qiskit>@__quantum__qis)"
)


@dataclass
class InputAnalysisResult(ResultInfoMixin):
    language: Languages = Languages.Empty
    raw_input: str | bytes = None


class InputAnalysis(AnalysisPass):
    """Determines the language of the input circuit."""

    def run(self, program: str, res_mgr: ResultManager, *args, **kwargs):
        """
        :param program: A circuit (e.g. QASM or QIR) or the file path for a circuit.
        :param res_mgr: The results manager to save the analysis.
        """

        result = InputAnalysisResult()
        if isinstance(program, bytes) and program.startswith(b"BC"):
            result.language = Languages.QIR
            result.raw_input = program
        elif path_regex.match(program) is not None:
            result.language, result.raw_input = self._process_path_string(program)
        else:
            result.language, result.raw_input = (
                self._process_string(program),
                program,
            )
        res_mgr.add(result)
        if result.language is Languages.Empty:
            raise ValueError("Unable to determine input language.")
        return program

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
