# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from compiler_config.config import CompilerConfig

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.frontend.base import BaseFrontend
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.qat import QATInput


class CustomFrontend(BaseFrontend):
    """
    Frontend that uses a custom pipeline to compile the input to an IR.
    While it is not equipped with a specific parser, it allows the user
    to specify custom compilation requirements via a pipeline.
    """

    # TODO: Allow the custom frontend to be equipped with an optional parser. Requires
    # the move over to pydantic hardware parsers with a shared API. (COMPILER-334)

    def __init__(
        self,
        model: None | QuantumHardwareModel | PydHardwareModel,
        pipeline: None | PassManager = None,
    ):
        """
        :param model: The hardware model that holds calibrated information on the qubits on the QPU,
                    defaults to None.
        :param pipeline: The custom pipeline, defaults to None.
        """
        super().__init__(model)
        self.pipeline = pipeline

    def check_and_return_source(self, src):
        """Custom frontends are too flexible to have any type checking, so just return the
        source file."""
        return src

    def emit(
        self,
        src: QATInput,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        compiler_config: CompilerConfig | None = None,
        **kwargs,
    ):
        """
        Compiles an input :class:`QatInput` down to :class:`QatIR` with the custom
        pipeline and emits it.
        :param src: The high-level input.
        :param res_mgr: Collection of analysis results with caching and aggregation
                        capabilities, defaults to None.
        :param met_mgr: Stores useful intermediary metrics that are generated during
                        compilation, defaults to None.
        :param compiler_config: Compiler settings, defaults to None.
        :return: An intermediate representation as an :class:`InstructionBuilder` which
                holds a list of instructions to be executed on the QPU.
        """

        res_mgr = res_mgr or ResultManager()
        met_mgr = met_mgr or MetricsManager()
        compiler_config = compiler_config or CompilerConfig()

        ir = self.pipeline.run(
            src, res_mgr, met_mgr, compiler_config=compiler_config, **kwargs
        )
        return ir
