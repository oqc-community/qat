# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import abc
from typing import Optional

from compiler_config.config import CompilerConfig

from qat.backend.passes.validation import HardwareConfigValidity, PydHardwareConfigValidity
from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.middleend.passes.analysis import ActivePulseChannelAnalysis
from qat.middleend.passes.transform import (
    AcquireSanitisation,
    EndOfTaskResetSanitisation,
    InactivePulseChannelSanitisation,
    InstructionGranularitySanitisation,
    PhaseOptimisation,
    PostProcessingSanitisation,
    PydPhaseOptimisation,
    SynchronizeTask,
)
from qat.middleend.passes.validation import (
    PydNoMidCircuitMeasurementValidation,
    ReadoutValidation,
)
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime.passes.analysis import CalibrationAnalysis


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
        res_mgr: Optional[ResultManager] = None,
        met_mgr: Optional[MetricsManager] = None,
        compiler_config: Optional[CompilerConfig] = None,
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
        res_mgr: Optional[ResultManager] = None,
        met_mgr: Optional[MetricsManager] = None,
        compiler_config: Optional[CompilerConfig] = None,
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

        res_mgr = res_mgr or ResultManager()
        met_mgr = met_mgr or MetricsManager()

        ir = self.pipeline.run(ir, res_mgr, met_mgr, compiler_config=compiler_config)
        return ir


class FallthroughMiddleend(CustomMiddleend):
    """
    A middle end that passes through an input :class:`InstructionBuilder` and does not alter it.
    """

    def __init__(self, model: None = None):
        super().__init__(model=None, pipeline=PassManager())


class DefaultMiddleend(CustomMiddleend):
    """
    Validates the compiler settings against the hardware model, finds calibrations within the IR,
    compresses contiguous :class:`PhaseShift` instructions, checks that the :class:`PostProcessing`
    instructions that follow an acquisition are suitable for the acquisition mode, validates that
    there are no mid-circuit measurements.
    """

    def __init__(self, model: QuantumHardwareModel, clock_cycle: float = 8e-9):
        """
        :param model: The hardware model that holds calibrated information on the qubits on
            the QPU.
        :param clock_cycle: The period for a single sequencer clock cycle.
        """
        pipeline = self.build_pass_pipeline(model, clock_cycle)
        super().__init__(model=model, pipeline=pipeline)

    @staticmethod
    def build_pass_pipeline(model, clock_cycle: float = 8e-9) -> PassManager:
        """
        Builds the default middle end pass pipeline.
        :param model: The hardware model that holds calibrated information on the qubits on
            the QPU.
        :return: A :class:`PassManager` containing a sequence of passes.
        """
        return (
            PassManager()
            | HardwareConfigValidity(model)
            | CalibrationAnalysis()
            | ActivePulseChannelAnalysis(model)
            | InactivePulseChannelSanitisation()
            | EndOfTaskResetSanitisation()
            | PhaseOptimisation()
            | PostProcessingSanitisation()
            | ReadoutValidation(model)
            | AcquireSanitisation()
            | InstructionGranularitySanitisation(clock_cycle)
            | SynchronizeTask()
        )


class PydDefaultMiddleend(CustomMiddleend):
    def __init__(self, model: PydHardwareModel):
        pipeline = self.build_pass_pipeline(model)
        super().__init__(model=model, pipeline=pipeline)

    @staticmethod
    def build_pass_pipeline(model) -> PassManager:
        return (
            PassManager()
            | PydHardwareConfigValidity(model)
            | CalibrationAnalysis()
            | PydPhaseOptimisation()
            | PydNoMidCircuitMeasurementValidation(model)
        )
