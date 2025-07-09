# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import abc
from typing import Optional

from compiler_config.config import CompilerConfig

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.ir.conversion import ConvertToPydanticIR
from qat.middleend.passes.purr.analysis import (
    ActivePulseChannelAnalysis,
)
from qat.middleend.passes.purr.transform import (
    AcquireSanitisation,
    BatchedShots,
    EndOfTaskResetSanitisation,
    EvaluatePulses,
    FreqShiftSanitisation,
    InactivePulseChannelSanitisation,
    InitialPhaseResetSanitisation,
    InstructionGranularitySanitisation,
    InstructionLengthSanitisation,
    LowerSyncsToDelays,
    MeasurePhaseResetSanitisation,
    PhaseOptimisation,
    PostProcessingSanitisation,
    RepeatSanitisation,
    RepeatTranslation,
    ResetsToDelays,
    ReturnSanitisation,
    ScopeSanitisation,
    SquashDelaysOptimisation,
    SynchronizeTask,
)
from qat.middleend.passes.purr.validation import (
    FrequencyValidation,
    HardwareConfigValidity,
    ReadoutValidation,
)
from qat.middleend.passes.transform import (
    PydBatchedShots,
    PydEndOfTaskResetSanitisation,
    PydEvaluateWaveforms,
    PydFreqShiftSanitisation,
    PydInactivePulseChannelSanitisation,
    PydInitialPhaseResetSanitisation,
    PydInstructionGranularitySanitisation,
    PydInstructionLengthSanitisation,
    PydLowerSyncsToDelays,
    PydPhaseOptimisation,
    PydRepeatTranslation,
    PydResetsToDelays,
    PydReturnSanitisation,
    PydScopeSanitisation,
    PydSquashDelaysOptimisation,
)
from qat.middleend.passes.validation import (
    PydHardwareConfigValidity,
    PydNoMidCircuitMeasurementValidation,
)
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.model.target_data import TargetData
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime.passes.purr.analysis import CalibrationAnalysis


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

        res_mgr = res_mgr if res_mgr is not None else ResultManager()
        met_mgr = met_mgr if met_mgr is not None else MetricsManager()
        compiler_config = (
            compiler_config if compiler_config is not None else CompilerConfig()
        )

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

    def __init__(
        self,
        model: QuantumHardwareModel,
        target_data: TargetData = TargetData.default(),
    ):
        """
        :param model: The hardware model that holds calibrated information on the qubits on
            the QPU.
        :param clock_cycle: The period for a single sequencer clock cycle.
        """
        pipeline = self.build_pass_pipeline(model, target_data)
        self.target_data = target_data
        super().__init__(model=model, pipeline=pipeline)

    @staticmethod
    def build_pass_pipeline(
        model: QuantumHardwareModel, target_data: TargetData = TargetData.default()
    ) -> PassManager:
        """
        Builds the default middle end pass pipeline.
        :param model: The hardware model that holds calibrated information on the qubits on
            the QPU.
        :return: A :class:`PassManager` containing a sequence of passes.
        """
        return (
            PassManager()
            | HardwareConfigValidity(model)
            | FrequencyValidation(model, target_data)
            | ActivePulseChannelAnalysis(model)
            # Sanitising input IR to make it complete
            | RepeatSanitisation(model, target_data)
            | ReturnSanitisation()
            | SynchronizeTask()
            # Corrections / optimisations to the IR
            | PostProcessingSanitisation()
            | ReadoutValidation(model)
            | AcquireSanitisation()
            | MeasurePhaseResetSanitisation()
            | InstructionGranularitySanitisation(model, target_data)
            # Preparing for codegen
            | EvaluatePulses()
            | LowerSyncsToDelays()
            | InactivePulseChannelSanitisation()
            | FreqShiftSanitisation(model)
            | InitialPhaseResetSanitisation()
            | PhaseOptimisation()
            | EndOfTaskResetSanitisation()
            | ResetsToDelays(target_data)
            | SquashDelaysOptimisation()
            | InstructionLengthSanitisation(target_data)
            | BatchedShots(target_data)
            | ScopeSanitisation()
            | RepeatTranslation(target_data)
        )


class ExperimentalDefaultMiddleend(CustomMiddleend):
    """
    Validates the compiler settings against the hardware model, finds calibrations within the IR,
    compresses contiguous :class:`PhaseShift` instructions, checks that the :class:`PostProcessing`
    instructions that follow an acquisition are suitable for the acquisition mode, validates that
    there are no mid-circuit measurements.
    """

    def __init__(
        self,
        model: QuantumHardwareModel,
        pyd_model: PydHardwareModel,
        target_data: TargetData = TargetData.default(),
    ):
        """
        :param model: The hardware model that holds calibrated information on the qubits on
            the QPU.
        :param clock_cycle: The period for a single sequencer clock cycle.
        """
        pipeline = self.build_pass_pipeline(model, pyd_model, target_data)
        self.target_data = target_data
        self.pyd_model = pyd_model
        super().__init__(model=model, pipeline=pipeline)

    @staticmethod
    def build_pass_pipeline(
        legacy_model: QuantumHardwareModel,
        pyd_model: PydHardwareModel,
        target_data: TargetData = TargetData.default(),
    ) -> PassManager:
        """
        Builds the default middle end pass pipeline.
        :param model: The hardware model that holds calibrated information on the qubits on
            the QPU.
        :return: A :class:`PassManager` containing a sequence of passes.
        """
        return (
            PassManager()
            | HardwareConfigValidity(legacy_model)
            | FrequencyValidation(legacy_model, target_data)  # TODO: COMPILER-380
            | ActivePulseChannelAnalysis(legacy_model)  # TODO: COMPILER-393
            # Sanitising input IR to make it complete
            | RepeatSanitisation(
                legacy_model, target_data
            )  # TODO: COMPILER-553, COMPILER-347
            | ReturnSanitisation()
            | SynchronizeTask()  # TODO: COMPILER-549
            # Corrections / optimisations to the IR
            | PostProcessingSanitisation()  # TODO: COMPILER-540
            | ReadoutValidation(legacy_model)  # TODO: COMPILER-556
            | AcquireSanitisation()  # TODO: COMPILER-292
            | MeasurePhaseResetSanitisation()  # TODO: COMPILER-547
            | ConvertToPydanticIR(legacy_model, pyd_model)
            | PydInstructionGranularitySanitisation(pyd_model, target_data)
            # Preparing for codegen
            | PydEvaluateWaveforms(pyd_model, target_data)
            | PydLowerSyncsToDelays()
            | PydInactivePulseChannelSanitisation()
            | PydFreqShiftSanitisation(pyd_model)
            | PydInitialPhaseResetSanitisation()
            | PydPhaseOptimisation()
            | PydEndOfTaskResetSanitisation(pyd_model)
            | PydResetsToDelays(pyd_model, target_data)
            | PydSquashDelaysOptimisation()
            | PydInstructionLengthSanitisation(target_data)
            | PydBatchedShots(target_data)
            | PydScopeSanitisation()
            | PydRepeatTranslation(target_data)
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
            | PydReturnSanitisation()
            | PydNoMidCircuitMeasurementValidation(model)
        )
