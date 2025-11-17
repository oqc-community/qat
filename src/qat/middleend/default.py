# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd


from qat.core.pass_base import PassManager
from qat.middleend.base import CustomMiddleend
from qat.middleend.passes.analysis import ActivePulseChannelAnalysis
from qat.middleend.passes.transform import (
    BatchedShots,
    EndOfTaskResetSanitisation,
    EvaluateWaveforms,
    FreqShiftSanitisation,
    InactivePulseChannelSanitisation,
    InitialPhaseResetSanitisation,
    InstructionGranularitySanitisation,
    InstructionLengthSanitisation,
    LowerSyncsToDelays,
    MeasurePhaseResetSanitisation,
    PhaseOptimisation,
    PopulateWaveformSampleTime,
    PostProcessingSanitisation,
    RepeatSanitisation,
    RepeatTranslation,
    ResetsToDelays,
    ReturnSanitisation,
    ScopeSanitisation,
    SquashDelaysOptimisation,
    SynchronizeTask,
)
from qat.middleend.passes.validation import (
    DynamicFrequencyValidation,
    FrequencySetupValidation,
    HardwareConfigValidity,
    ReadoutValidation,
)
from qat.model.hardware_model import PhysicalHardwareModel as HardwareModel
from qat.model.target_data import TargetData


class DefaultMiddleend(CustomMiddleend):
    """The standard middle end used for antic pipelines.

    Implements a number of passes, including validation and sanitisation of IR, and
    optimizations. Also implements a number of lowering passes, e.g., lowering repeats to
    more explicit control flow instructions.
    """

    def __init__(
        self,
        model: HardwareModel,
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
        model: HardwareModel,
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
            | PopulateWaveformSampleTime(model, target_data)
            | HardwareConfigValidity(model)
            | ActivePulseChannelAnalysis(model)
            | InactivePulseChannelSanitisation()
            | FrequencySetupValidation(model, target_data)
            | DynamicFrequencyValidation(model, target_data)
            # Sanitising input IR to make it complete
            | RepeatSanitisation(target_data)
            | ReturnSanitisation()
            | SynchronizeTask()
            | PostProcessingSanitisation()
            | ReadoutValidation()
            | MeasurePhaseResetSanitisation(model)
            | InstructionGranularitySanitisation(target_data)
            # Preparing for codegen
            | EvaluateWaveforms(model, target_data)
            | LowerSyncsToDelays()
            | FreqShiftSanitisation(model)
            | InitialPhaseResetSanitisation()
            | PhaseOptimisation()
            | EndOfTaskResetSanitisation(model)
            | ResetsToDelays(model, target_data)
            | SquashDelaysOptimisation()
            | InstructionLengthSanitisation(model, target_data)
            | BatchedShots(target_data)
            | ScopeSanitisation()
            | RepeatTranslation(target_data)
        )


PydDefaultMiddleend = DefaultMiddleend
