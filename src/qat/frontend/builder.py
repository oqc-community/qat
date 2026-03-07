# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd


from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.frontend.base import BaseFrontend
from qat.frontend.converters.purr import PurrConverter
from qat.frontend.passes.transform import PostProcessingSanitisation
from qat.ir.instruction_builder import InstructionBuilder
from qat.model.hardware_model import PhysicalHardwareModel
from qat.purr.compiler.builders import InstructionBuilder as PurrInstructionBuilder


class BuilderFrontend(BaseFrontend):
    """Allows a :class:`PurrInstructionBuilder` to be used as an input to the compiler.

    Parses the :class:`PurrInstructionBuilder` into a :class:`InstructionBuilder` containing
    QAT IR instructions, and then runs a pipeline to sanitise the IR.
    """

    def __init__(self, model: PhysicalHardwareModel, pipeline: PassManager | None = None):
        """
        :param model: The hardware model is used to validate that the instruction builder
            is compatible with the target hardware.
        """
        self.model = model
        self._pipeline = pipeline if pipeline is not None else self._build_pipeline()

    @staticmethod
    def _build_pipeline():
        """Builds the pipeline of passes to run on the IR after parsing."""
        return PassManager() | PostProcessingSanitisation()

    def emit(
        self,
        src: PurrInstructionBuilder,
        res_mgr: ResultManager,
        met_mgr: MetricsManager,
        *args,
        **kwargs,
    ) -> InstructionBuilder:
        """Parses the PuRR instruction builder and returns it as QAT IR.

        :param src: The source instruction builder to parse. Must be a
            :class:`PurrInstructionBuilder`.
        """

        parser = PurrConverter(self.model)
        ir = parser.convert(src)
        ir = self._pipeline.run(ir, res_mgr, met_mgr, **kwargs)
        return ir

    def check_and_return_source(self, src) -> bool | PurrInstructionBuilder:
        """Checks that the source is a :class:`PurrInstructionBuilder` and returns it."""

        if not isinstance(src, PurrInstructionBuilder):
            return False
        return src
