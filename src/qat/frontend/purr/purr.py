# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from compiler_config.config import CompilerConfig

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.frontend.base import BaseFrontend
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.instructions import DeviceUpdate, Sweep
from qat.purr.utils.logger import get_default_logger

logger = get_default_logger()


class PurrFrontend(BaseFrontend):
    """A frontend for processing purr :class:`InstructionBuilder` source programs.

    The instruction builder will be accepted if it has a model that matches the provided
    model (or if no model is provided), and if it has no sweeps or device injections. These
    are handled by the SweepPipeline.

    :param model: The hardware model to use for compilation.
    """

    def __init__(self, model):
        self.model = model

    def check_and_return_source(self, src):
        """Checks if the source is a valid purr instruction builder.

        :param src: The source program, or path to the program.
        :returns: If the program is determined to not be valid, False is returned.
            Otherwise, the program is returned (and loaded if required).
        """
        if not isinstance(src, InstructionBuilder):
            return False

        if self.model and src.model and self.model != src.model:
            logger.info("InstructionBuilder model does not match the frontend model.")
            return False

        for instr in src.instructions:
            if isinstance(instr, (Sweep, DeviceUpdate)):
                logger.info(
                    "InstructionBuilder contains sweeps or device updates, which are not "
                    "supported by the PurrFrontend. Please use sweep pipelines to handle "
                    "programs with sweeping."
                )
                return False
        return src

    def emit(
        self,
        src: InstructionBuilder,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        compiler_config: CompilerConfig | None = None,
        **kwargs,
    ) -> InstructionBuilder:
        """Passes through the instruction builder without modifying it."""

        src = self.check_and_return_source(src)
        if src is False:
            raise ValueError("Source is not a valid purr InstructionBuilder.")
        return src
