import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from compiler_config.config import Languages

from qat.ir.pass_base import AnalysisPass, QatIR, ResultManager
from qat.ir.result_base import ResultInfoMixin

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
