# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import abc

from compiler_config.config import CompilerConfig

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.purr.compiler.hardware_models import QuantumHardwareModel


class BaseMiddleend(abc.ABC):
    """
    Base class for a middle end that takes an intermediate representation (IR) :class:`QatIR`
    and alters it based on optimisation and/or validation passes.
    """

    def __init__(self, model: None | QuantumHardwareModel):
        """
        :param model: The hardware model that holds calibrated information on the qubits on the QPU.
        """
        self.model = model

    @abc.abstractmethod
    def emit(
        self,
        ir,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        compiler_config: CompilerConfig | None = None,
        **kwargs,
    ):
        """
        Converts an IR :class:`QatIR` to an optimised IR.
        :param ir: The intermediate representation.
        :param res_mgr: Collection of analysis results with caching and aggregation
            capabilities, defaults to None.
        :param met_mgr: Stores useful intermediary metrics that are generated during
            compilation, defaults to None.
        :param compiler_config: Compiler settings, defaults to None.
        """
        ...


class CustomMiddleend(BaseMiddleend):
    """
    Middle end that uses a custom pipeline to convert the IR to an (optimised) IR.
    """

    def __init__(
        self, model: None | QuantumHardwareModel, pipeline: None | PassManager = None
    ):
        self.pipeline = pipeline
        super().__init__(model=model)

    def emit(
        self,
        ir,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        compiler_config: CompilerConfig | None = None,
        **kwargs,
    ):
        """
        Converts an IR :class:`QatIR` to an optimised IR with a custom pipeline.
        :param ir: The intermediate representation.
        :param res_mgr: Collection of analysis results with caching and aggregation
            capabilities, defaults to None.
        :param met_mgr: Stores useful intermediary metrics that are generated during
            compilation, defaults to None.
        :param compiler_config: Compiler settings, defaults to None.
        """

        res_mgr = res_mgr if res_mgr is not None else ResultManager()
        met_mgr = met_mgr if met_mgr is not None else MetricsManager()
        compiler_config = (
            compiler_config if compiler_config is not None else CompilerConfig()
        )

        ir = self.pipeline.run(
            ir, res_mgr, met_mgr, compiler_config=compiler_config, **kwargs
        )
        return ir
