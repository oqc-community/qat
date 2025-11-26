# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from compiler_config.config import CompilerConfig

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.frontend.base import BaseFrontend
from qat.purr.compiler.builders import InstructionBuilder


class FallthroughFrontend(BaseFrontend):
    """
    A frontend that passes through an input :class:`InstructionBuilder` and does not modify it.
    Used in situations where a frontend is not required, but is used to make a pipeline complete.
    """

    # TODO: add support for OPTIONAL type checking. We do not always want anything to
    # "fall through" here, we might only want particular types, e.g., Qat IR (COMPILER-333)

    def emit(
        self,
        src: InstructionBuilder,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        compiler_config: CompilerConfig | None = None,
        **kwargs,
    ) -> InstructionBuilder:
        return src

    def check_and_return_source(self, src):
        """All source files are valid for the :class:`FallthroughFrontend`."""
        return src
