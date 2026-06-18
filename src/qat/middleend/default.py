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
    InsertPreSelectionMeasurement,
    InstructionGranularitySanitisation,
    InstructionLengthSanitisation,
    LowerSyncsToDelays,
    MeasurePhaseResetSanitisation,
    PhaseOptimisation,
    PopulateWaveformSampleTime,
    RepeatSanitisation,
    RepeatTranslation,
    ResetTransformation,
    ReturnSanitisation,
    ScopeSanitisation,
    SquashDelaysOptimisation,
    SynchronizeTask,
)
from qat.middleend.passes.validation import (
    DynamicFrequencyValidation,
    FrequencySetupValidation,
    HardwareConfigValidity,
    InstructionValidation,
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
        target_data: TargetData | None = None,
    ):
        """
        :param model: The hardware model that holds calibrated information on the qubits on
            the QPU.
        :param clock_cycle: The period for a single sequencer clock cycle.
        """
        if target_data is None:
            target_data = TargetData()
        pipeline = self.build_pass_pipeline(model, target_data)
        self.target_data = target_data
        super().__init__(model=model, pipeline=pipeline)

    @staticmethod
    def build_pass_pipeline(
        model: HardwareModel,
        target_data: TargetData | None = None,
    ) -> PassManager:
        """Builds the default middle end pass pipeline.

        :param model: The hardware model that holds calibrated information on the qubits on
            the QPU.
        :return: A :class:`PassManager` containing a sequence of passes.
        """
        if target_data is None:
            target_data = TargetData()
        return (
            PassManager()
            | HardwareConfigValidity(model)
            | ActivePulseChannelAnalysis(model)
            # Preselection must run for every shot, so sanitise repeats before preselection
            | RepeatSanitisation(target_data)
            | InsertPreSelectionMeasurement(model)
            # Must run after InsertPreSelectionMeasurement so injected filter waveforms
            # (SampledWaveform with use_weights=True) get their sample_time populated
            # before InstructionGranularitySanitisation divides by it.
            | PopulateWaveformSampleTime(model, target_data)
            | InactivePulseChannelSanitisation()
            | FrequencySetupValidation(model, target_data)
            | DynamicFrequencyValidation(model, target_data)
            # Sanitising input IR to make it complete
            | ReturnSanitisation()
            | ReadoutValidation()
            | MeasurePhaseResetSanitisation(model)
            | InstructionValidation(model, target_data)
            | InstructionGranularitySanitisation(target_data)
            # Preparing for codegen
            | EvaluateWaveforms(model, target_data)
            | EndOfTaskResetSanitisation(model)
            | SynchronizeTask()
            | ResetTransformation(model, target_data)
            | LowerSyncsToDelays()
            | FreqShiftSanitisation(model)
            | InitialPhaseResetSanitisation()
            | PhaseOptimisation()
            | SquashDelaysOptimisation()
            | InstructionLengthSanitisation(model, target_data)
            | BatchedShots(target_data)
            | ScopeSanitisation()
            | RepeatTranslation(target_data)
        )


PydDefaultMiddleend = DefaultMiddleend
