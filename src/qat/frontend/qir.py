# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from pathlib import Path

from compiler_config.config import CompilerConfig, Tket, TketOptimizations

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
from qat.purr.integrations.tket import run_tket_optimizations_qir

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


class QIRFrontend(BaseFrontend):
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
        super().__init__(model)
        self.pipeline = pipeline if pipeline is not None else self.build_pass_pipeline()
        self.parser = self._init_parser()

    @staticmethod
    def build_pass_pipeline() -> PassManager:
        """Creates an empty pipeline for QIR compilation.

        :return: The pipeline as a pass manager.
        """

        return PassManager()

    def _init_parser(self) -> QIRParser | None:
        if isinstance(self.model, QuantumHardwareModel):  # legacy hardware model
            return QIRParser(self.model)
        elif isinstance(self.model, PydHardwareModel):  # pydantic hardware model
            return PydQIRParser(self.model)

    def check_and_return_source(self, src: str | bytes) -> bool | str | bytes:
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

    def _check_source(self, src: str) -> str:
        src = self.check_and_return_source(src)
        if not src:
            raise ValueError(
                "Source program is not a QIR program, or a path to a QIR program."
            )
        return src

    def _get_builder(
        self, compiler_config: CompilerConfig, src: str
    ) -> InstructionBuilder | None:
        builder = None
        if isinstance(self.model, QuantumHardwareModel):
            optimizations = compiler_config.optimizations
            if (
                isinstance(optimizations, Tket)
                and optimizations.tket_optimizations != TketOptimizations.Empty
            ):
                builder = run_tket_optimizations_qir(
                    src,
                    optimizations.tket_optimizations,
                    self.model,
                    compiler_config.results_format.format,
                )
            else:
                builder = self.parser.parse(src)

            builder = (
                self.model.create_builder()
                .repeat(
                    compiler_config.repeats,
                    repetition_period=compiler_config.repetition_period,
                    passive_reset_time=compiler_config.passive_reset_time,
                )
                .add(builder)
            )
        elif isinstance(self.model, PydHardwareModel):
            builder = self.parser.parse(src)
            builder = (
                PydQuantumInstructionBuilder(self.model)
                .repeat(compiler_config.repeats)
                .__add__(builder)
            )
        return builder

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
        res_mgr, met_mgr, compiler_config = self._check_metrics_and_config(
            res_mgr, met_mgr, compiler_config
        )

        src = self._check_source(src)

        src = self.pipeline.run(src, res_mgr, met_mgr, compiler_config=compiler_config)

        reset_parser_format: bool = False
        old_parser_results_format = None

        if compiler_config.results_format.format is not None:
            # TODO: Fix the need to reset parser attributes after parsing COMPILER-531
            reset_parser_format = True
            old_parser_results_format = self.parser.results_format
            self.parser.results_format = compiler_config.results_format.format

        builder = self._get_builder(compiler_config, src)

        if reset_parser_format:
            self.parser.results_format = old_parser_results_format

        return builder
