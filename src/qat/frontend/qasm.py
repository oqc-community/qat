# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
import re
from abc import ABC, abstractmethod
from pathlib import Path

from compiler_config.config import CompilerConfig, InlineResultsProcessing

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.frontend.base import BaseFrontend
from qat.frontend.parsers.qasm import CloudQasmParser as PydCloudQasmParser
from qat.frontend.parsers.qasm import Qasm3Parser as PydQasm3Parser
from qat.frontend.parsers.qasm.base import AbstractParser as PydAbstractParser
from qat.frontend.passes.analysis import InputAnalysis
from qat.frontend.passes.purr.transform import InputOptimisation
from qat.frontend.passes.transform import PydInputOptimisation
from qat.ir.builder_factory import BuilderFactory
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.integrations.qasm import AbstractParser, CloudQasmParser, Qasm3Parser

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
        super().__init__(model)
        self.pipeline = (
            pipeline if pipeline is not None else self.build_pass_pipeline(model)
        )

    @property
    @abstractmethod
    def version(self) -> int: ...

    @abstractmethod
    def create_parser(
        self, config: CompilerConfig
    ) -> AbstractParser | PydAbstractParser: ...

    @staticmethod
    def build_pass_pipeline(model: QuantumHardwareModel) -> PassManager | None:
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

    def _check_source(self, src: str) -> str:
        """
        Checks that the source program (or file path) can be interpreted as a QASM file and raises and error if not.
        """
        src = self.check_and_return_source(src)
        if not src:
            raise ValueError(
                f"Source program is not a QASM{self.version} program, or a path "
                f"to a QASM{self.version} program."
            )
        return src

    def _generate_builder(
        self,
        src: str,
        compiler_config: CompilerConfig,
    ) -> InstructionBuilder | None:
        """
        Creates the instruction builder for execution from the base builder and compiler config.
        :param builder: The base instruction builder.
        :param compiler_config: The compiler config is used in both the pipeline and for
            parsing.
        """
        builder = BuilderFactory.create_builder(self.model)
        if isinstance(self.model, PydHardwareModel):
            builder.repeat(compiler_config.repeats)
        elif isinstance(self.model, QuantumHardwareModel):
            builder = builder.repeat(
                compiler_config.repeats,
                repetition_period=compiler_config.repetition_period,
                passive_reset_time=compiler_config.passive_reset_time,
            )

        parser = self.create_parser(compiler_config)
        builder = parser.parse(builder, src)
        return builder

    def emit(
        self,
        src: str,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        compiler_config: CompilerConfig | None = None,
        **kwargs,
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
        res_mgr, met_mgr, compiler_config = self._check_metrics_and_config(
            res_mgr, met_mgr, compiler_config
        )

        src = self._check_source(src)
        src = self.pipeline.run(
            src, res_mgr, met_mgr, compiler_config=compiler_config, **kwargs
        )
        return self._generate_builder(src, compiler_config)


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

    def create_parser(self, config: CompilerConfig) -> CloudQasmParser | PydCloudQasmParser:
        if isinstance(self.model, QuantumHardwareModel):
            parser = CloudQasmParser()
        elif isinstance(self.model, PydHardwareModel):
            parser = PydCloudQasmParser()
        else:
            raise TypeError("Invalid hardware model type set.")

        parser.results_format = (
            config.results_format.format
            if config.results_format.format is not None
            else InlineResultsProcessing.Program
        )
        return parser


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

    def create_parser(self, config: CompilerConfig) -> Qasm3Parser | PydQasm3Parser:
        if isinstance(self.model, QuantumHardwareModel):
            parser = Qasm3Parser()
        elif isinstance(self.model, PydHardwareModel):
            parser = PydQasm3Parser()
        else:
            raise TypeError("Invalid hardware model type set.")

        parser.results_format = (
            config.results_format.format
            if config.results_format.format is not None
            else InlineResultsProcessing.Program
        )
        return parser
