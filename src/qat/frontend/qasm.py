# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import re
from abc import ABC, abstractmethod
from pathlib import Path

from compiler_config.config import CompilerConfig

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.frontend.base import BaseFrontend
from qat.frontend.parsers.qasm import CloudQasmParser as PydCloudQasmParser
from qat.frontend.parsers.qasm import Qasm3Parser as PydQasm3Parser
from qat.frontend.passes.analysis import InputAnalysis
from qat.frontend.passes.transform import InputOptimisation, PydInputOptimisation
from qat.ir.instruction_builder import (
    QuantumInstructionBuilder as PydQuantumInstructionBuilder,
)
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.integrations.qasm import CloudQasmParser, Qasm3Parser
from qat.utils.hardware_model import check_type_legacy_or_pydantic

path_regex = re.compile(r"^.+\.(qasm)$")
string_regex = re.compile(r"OPENQASM (?P<version>[0-9]+)(?:.[0-9])?;")


def is_qasm_path(src: str) -> bool:
    """Determines if the string corresponds to a path, and has a `.qasm` extension.

    :param src: The path to the source program.
    """

    if path_regex.match(src) is not None:
        return True
    return False


def is_qasm_str(src: str) -> bool:
    """Determines if the string contains OPENQASM headers.

    :param src: The source program as a string.
    """

    if string_regex.search(src) is not None:
        return True
    return False


def get_qasm_version(src: str) -> int:
    """Extracts the QASM version from the source file.

    :param: The source program as a string.
    :returns: The major version number.
    """

    return int(string_regex.search(src).group("version"))


def load_qasm_file(path: str) -> str:
    """Loads a Qasm file from a given file path.

    :param path: The file path.
    :raises ValueError: Raises an error if the the file has an invalid extension.
    :returns: The Qasm program as a string or bytes.
    """

    path = Path(path)
    if path.suffix == ".qasm":
        with path.open() as file:
            src = file.read()
    else:
        raise ValueError("String expected to end in `.qasm`.")
    return src


class BaseQasmFrontend(BaseFrontend, ABC):
    """A base frontend for QASM programs.

    Handles the parsing of QASM into QAT's intermediate representation (IR). Optionally, it
    can also run pipelines before to optimize and validate QASM files. The QASM2 and QASM3
    frontends are identical up to the language and its respective parser. This class
    implements the base functionality."""

    def __init__(
        self,
        model: QuantumHardwareModel | PydHardwareModel = None,
        pipeline: None | PassManager = None,
    ):
        """
        :param model: The hardware model can be required for the pipeline and is used in
            parsing.
        :param pipeline: A pipeline for validation and optimising QASM, defaults to a
            predefined pipeline that optimizes the QASM file.
        """
        self.model = check_type_legacy_or_pydantic(model)
        self.pipeline = pipeline or self.build_pass_pipeline(model)

    @property
    @abstractmethod
    def version(self) -> int: ...

    @staticmethod
    def build_pass_pipeline(model: QuantumHardwareModel) -> PassManager:
        """Creates a pipeline to optimize QASM files.

        :param model: The hardware model is needed for optimizations.
        :return: The pipeline as a pass manager.
        """

        # TODO: replace input analysis + input optimisation with a qasm specific pass?
        # (COMPILER-340)
        if isinstance(model, QuantumHardwareModel):  # legacy hardware model
            return PassManager() | InputAnalysis() | InputOptimisation(model)
        elif isinstance(model, PydHardwareModel):  # pydantic hardware model
            return PassManager() | InputAnalysis() | PydInputOptimisation(model)

    def check_and_return_source(self, src: str) -> bool | str:
        """Checks that the source program (or file path) can be interpreted as a QASM file
        by checking against the language.

        :param src: The QASM program, or the file path to the program.
        :returns: If the program is determined to not be valid, False is returned.
            Otherwise, the program is returned (and loaded if required).
        """

        if not isinstance(src, str):
            return False

        if is_qasm_path(src):
            if Path(src).is_file():
                src = load_qasm_file(src)

        if not is_qasm_str(src):
            return False

        if get_qasm_version(src) != self.version:
            return False

        return src

    def emit(
        self,
        src: str,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        compiler_config: CompilerConfig | None = None,
    ) -> InstructionBuilder:
        """Compiles the source QASM program into QAT's intermediate representation.

        :param src: The source program, or path to the source program.
        :param res_mgr: The result manager to store results from passes within the pipeline,
            defaults to an empty :class:`ResultManager`.
        :param met_mgr: The metrics manager to store metrics, such as the optimized QASM
            circuit, defaults to an empty :class:`MetricsManager`.
        :param compiler_config: The compiler config is used in both the pipeline and for
            parsing.
        :raises ValueError: An error is thrown if the the program is not detected to be a
            QASM program.
        :returns: The program as an :class:`InstructionBuilder`.
        """

        res_mgr = res_mgr or ResultManager()
        met_mgr = met_mgr or MetricsManager()
        compiler_config = compiler_config or CompilerConfig()

        src = self.check_and_return_source(src)
        if not src:
            raise ValueError(
                f"Source program is not a QASM{self.version} program, or a path "
                f"to a QASM{self.version} program."
            )

        src = self.pipeline.run(src, res_mgr, met_mgr, compiler_config=compiler_config)

        if isinstance(self.model, QuantumHardwareModel):  # legacy hardware model
            builder = self.model.create_builder()
        elif isinstance(self.model, PydHardwareModel):  # pydantic hardware model
            builder = PydQuantumInstructionBuilder(hardware_model=self.model)

        if compiler_config.results_format.format is not None:
            self.parser.results_format = compiler_config.results_format.format
        builder = self.parser.parse(builder, src)

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


class Qasm2Frontend(BaseQasmFrontend):
    """A frontend for QASM2 programs.

    Handles the parsing of QASM2 into QATs intermediate representation. Optionally, it can
    also run pipelines before to optimize and validate QASM2 files.
    """

    version = 2

    def __init__(self, model: QuantumHardwareModel, pipeline: None | PassManager = None):
        """
        :param model: The hardware model can be required for the pipeline and is used in
            parsing.
        :param pipeline: A pipeline for validation and optimising QASM, defaults to a
            predefined pipeline that optimizes the QASM file.
        """

        super().__init__(model, pipeline)
        if isinstance(self.model, QuantumHardwareModel):
            self.parser = CloudQasmParser()
        elif isinstance(self.model, PydHardwareModel):
            self.parser = PydCloudQasmParser()


class Qasm3Frontend(BaseQasmFrontend):
    """A frontend for QASM3 programs.

    Handles the parsing of QASM3 into QATs intermediate representation. Optionally, it can
    also run pipelines before to optimize and validate QASM3 files.
    """

    version = 3

    def __init__(self, model: QuantumHardwareModel, pipeline: None | PassManager = None):
        """
        :param model: The hardware model can be required for the pipeline and is used in
            parsing.
        :param pipeline: A pipeline for validation and optimising QASM, defaults to a
            predefined pipeline that optimizes the QASM file.
        """

        super().__init__(model, pipeline)
        if isinstance(self.model, QuantumHardwareModel):
            self.parser = Qasm3Parser()
        elif isinstance(self.model, PydHardwareModel):
            self.parser = PydQasm3Parser()
