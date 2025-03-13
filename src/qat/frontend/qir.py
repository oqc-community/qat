# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from abc import ABC
from pathlib import Path

from compiler_config.config import CompilerConfig

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.frontend.base import BaseFrontend
from qat.frontend.parsers.qir import QIRParser as PydQIRParser
from qat.ir.instruction_builder import (
    QuantumInstructionBuilder as PydQuantumInstructionBuilder,
)
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.integrations.qir import QIRParser
from qat.utils.hardware_model import check_type_legacy_or_pydantic

string_regex = r"@__quantum__qis"


def is_qir_path(src: str) -> bool:
    """Determines if the string corresponds to a path, and has a `.ll` or `.rb`
    extension."""

    return src.endswith(".ll") or src.endswith(".bc")


def is_qir_str(src: str) -> bool:
    """Determines if the string contains QIR semantics."""

    if isinstance(src, str) and string_regex in src:
        return True
    return False


def load_qir_file(path: str) -> str | bytes:
    """Loads a QIR file from a given file path.

    :param path: The file path.
    :raises ValueError: Raises an error if the the file has an invalid extension.
    :returns: The QIR program as a string or bytes.
    """

    path = Path(path)
    if path.suffix == ".ll":
        with path.open() as file:
            src = file.read()
    elif path.suffix == ".bc":
        with path.open("rb") as file:
            src = file.read()
    else:
        raise ValueError("String expected to end in `.ll` or `.bc`.")
    return src


class QIRFrontend(BaseFrontend, ABC):
    """A frontend for handling QIR programs.

    Handles the parsing of QIR programs, and optionally implements a pipeline for validation
    and transformation.
    """

    def __init__(self, model: QuantumHardwareModel, pipeline: None | PassManager = None):
        """
        :param model: The hardware model can be required for the pipeline and is used in
            parsing.
        :param pipeline: A pipeline for validation and optimising QASM, defaults to a
            predefined pipeline that optimizes the QASM file.
        """

        self.model = check_type_legacy_or_pydantic(model)
        if not pipeline:
            pipeline = self.build_pass_pipeline()
        self.pipeline = pipeline

        if isinstance(self.model, QuantumHardwareModel):  # legacy hardware model
            self.parser = QIRParser(model)
        elif isinstance(model, PydHardwareModel):  # pydantic hardware model
            self.parser = PydQIRParser(model)

    @staticmethod
    def build_pass_pipeline() -> PassManager:
        """Creates an empty pipeline for QIR compilation.

        :return: The pipeline as a pass manager.
        """

        return PassManager()

    def check_and_return_source(self, src: str | bytes) -> bool | str:
        """Checks that the source program (or file path) can be interpreted as a QIR file
        by checking against the language.

        :param src: The QIR program, or the file path to the program.
        :returns: If the program is determined to not be valid, False is returned.
            Otherwise, the program is returned (and loaded if required).
        """
        if not isinstance(src, (str, bytes)):
            return False

        if isinstance(src, str):
            if src.endswith(".ll") or src.endswith(".bc"):
                src = load_qir_file(src)

        if isinstance(src, bytes):
            if src.startswith(b"BC"):
                return src
            else:
                return False

        if is_qir_str(src):
            return src

        return False

    def emit(
        self,
        src: str,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        compiler_config: CompilerConfig | None = None,
    ) -> InstructionBuilder:
        """Compiles the source QIR program into QAT's intermediate representation.

        :param src: The source program, or path to the source program.
        :param res_mgr: The result manager to store results from passes within the pipeline,
            defaults to an empty :class:`ResultManager`.
        :param met_mgr: The metrics manager to store metrics, such as the optimized QIR
            circuit, defaults to an empty :class:`MetricsManager`.
        :param compiler_config: The compiler config is used in both the pipeline and for
            parsing.
        :raises ValueError: An error is thrown if the the program is not detected to be a
            QIR program.
        :returns: The program as an :class:`InstructionBuilder`.
        """

        res_mgr = res_mgr or ResultManager()
        met_mgr = met_mgr or MetricsManager()
        compiler_config = compiler_config or CompilerConfig()

        src = self.check_and_return_source(src)
        if not src:
            raise ValueError(
                "Source program is not a QIR program, or a path to a QIR program."
            )

        src = self.pipeline.run(src, res_mgr, met_mgr, compiler_config=compiler_config)
        if compiler_config.results_format.format is not None:
            self.parser.results_format = compiler_config.results_format.format
        builder = self.parser.parse(src)

        if isinstance(self.model, QuantumHardwareModel):
            return (
                self.model.create_builder()
                .repeat(compiler_config.repeats, compiler_config.repetition_period)
                .add(builder)
            )
        elif isinstance(self.model, PydHardwareModel):
            return (
                PydQuantumInstructionBuilder(self.model)
                .repeat(compiler_config.repeats, compiler_config.repetition_period)
                .__add__(builder)
            )
