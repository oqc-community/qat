# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.core.pass_base import PassManager
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.model.target_data import TargetData
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime.passes.analysis import IndexMappingAnalysis as PydIndexMappingAnalysis
from qat.runtime.passes.purr.analysis import IndexMappingAnalysis
from qat.runtime.passes.transform import (
    AcquisitionPostprocessing,
    AssignResultsTransform,
    ErrorMitigation,
    InlineResultsProcessingTransform,
    ResultTransform,
)


def get_default_results_pipeline(model: QuantumHardwareModel) -> PassManager:
    """Factory for creating the default results pipeline.

    :param model: The quantum hardware model to use for the pipeline.
    """

    return (
        PassManager()
        | AcquisitionPostprocessing()
        | InlineResultsProcessingTransform()
        | AssignResultsTransform()
        | ResultTransform()
        | IndexMappingAnalysis(model)
        | ErrorMitigation(model)
    )


def get_results_pipeline(model: PydHardwareModel, target_data: TargetData) -> PassManager:
    """Factory for creating the results pipeline.

    :param model: The quantum hardware model.
    """

    return (
        PassManager()
        | AcquisitionPostprocessing(target_data)
        | InlineResultsProcessingTransform()
        | AssignResultsTransform()
        | ResultTransform()
        | PydIndexMappingAnalysis(model)
        | ErrorMitigation(model)
    )
