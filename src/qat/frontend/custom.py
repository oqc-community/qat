# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from compiler_config.config import CompilerConfig

from qat.frontend.base import BaseFrontend
from qat.passes.metrics_base import MetricsManager
from qat.passes.pass_base import PassManager
from qat.passes.result_base import ResultManager
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.qat import QATInput


class CustomFrontend(BaseFrontend):
    """The custom frontend allows for more general compilation of source programs.

    While it is not equipped with a specific parser (although we could allow for parsers
    that fit a required contract to be passed?), it allows the user to specify custom
    compilation requirements via a pipeline.
    """

    # TODO: Allow the custom frontend to be equipped with an optional parser. Requires
    # the move over to pydantic hardware parsers with a shared API. (COMPILER-334)

    def __init__(
        self,
        model: None | QuantumHardwareModel,
        pipeline: None | PassManager = None,
    ):
        """
        :param model: The hardware model is required for pipeline validation.
        :param pipeline: The custom pipeline defines the compilation process, defaults to
            None.
        """

        self.model = model
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
    ):
        """Executes the custom pipeline and returns the modified program.

        :param src: The source program
        :param res_mgr: A results manager for the pipeline, if not provided, defaults to an
            empty :class:`ResultManager`
        :param met_mgr: A metrics manager for the pipeline, if not provided, defaults to an
            empty :class:`MetricsManager`
        :param compiler_config: The compiler config, if not provided, defaults to the
            default compiler config
        :return: The potentially modified source program
        """

        res_mgr = res_mgr or ResultManager()
        met_mgr = met_mgr or MetricsManager()
        compiler_config = compiler_config or CompilerConfig()

        ir = self.pipeline.run(src, res_mgr, met_mgr, compiler_config=compiler_config)
        return ir
