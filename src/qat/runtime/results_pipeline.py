# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

from qat.core.pass_base import PassManager
from qat.model.hardware_model import PhysicalHardwareModel as PydHardwareModel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.runtime.passes.analysis import IndexMappingAnalysis as PydIndexMappingAnalysis
from qat.runtime.passes.legacy.analysis import IndexMappingAnalysis
from qat.runtime.passes.transform import (
    AssignResultsTransform,
    ErrorMitigation,
    InlineResultsProcessingTransform,
    PostProcessingTransform,
    ResultTransform,
)


def get_default_results_pipeline(model: QuantumHardwareModel) -> PassManager:
    """Factory for creating the default results pipeline.

    :param model: The quantum hardware model to use for the pipeline.
    """

    return (
        PassManager()
        | PostProcessingTransform()
        | InlineResultsProcessingTransform()
        | AssignResultsTransform()
        | ResultTransform()
        | IndexMappingAnalysis(model)
        | ErrorMitigation(model)
    )


def get_experimental_results_pipeline(
    model: QuantumHardwareModel, pyd_model: PydHardwareModel
) -> PassManager:
    """Factory for creating the experimental results pipeline.

    This pipeline includes additional experimental features.

    :param model: The quantum hardware model to use for the pipeline.
    :param pyd_model: The quantum hardware model converted to Pydantic.
    """

    return (
        PassManager()
        | PostProcessingTransform()
        | InlineResultsProcessingTransform()
        | AssignResultsTransform()
        | ResultTransform()
        | PydIndexMappingAnalysis(pyd_model)
        | ErrorMitigation(model)  # TODO: COMPILER-607
    )
