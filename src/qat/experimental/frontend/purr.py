# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Expose PuRR-to-IR import as a QAT frontend.

This module provides a frontend that normalises legacy PuRR instruction builders with a
sanitisation pass pipeline, then imports the result into pulse IR.
"""

from typing import Any

from compiler_config.config import CompilerConfig
from xdsl.dialects.builtin import ModuleOp

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import PassManager
from qat.core.result_base import ResultManager
from qat.experimental.frontend.importer.pulse.post_processing import PostSelectionBuilder
from qat.experimental.frontend.importer.pulse.purr import PurrImporter
from qat.experimental.system_data.canonical.schema import CanonicalSystemData
from qat.experimental.system_data.pulse.post_processing import PostProcessing
from qat.frontend.base import BaseFrontend
from qat.middleend.passes.purr.analysis import ActivePulseChannelAnalysis
from qat.middleend.passes.purr.transform import (
    AcquireSanitisation,
    EndOfTaskResetSanitisation,
    InitialPhaseResetSanitisation,
    RepeatSanitisation,
    ResetsToDelays,
    ReturnSanitisation,
    SynchronizeTask,
)
from qat.model.target_data import QubitDescription, TargetData
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.hardware_models import QuantumHardwareModel

_PICOSECONDS_AS_SECONDS = 1e-12


def _extract_passive_reset_time(
    model: CanonicalSystemData | None,
    fallback_reset_time: float,
) -> float:
    """Resolve the passive reset duration for target data.

    Canonical reset-method durations are stored in picoseconds. If the canonical model
    defines a passive reset duration, convert it back to seconds and return it. Otherwise,
    fall back to the builder model's default repetition period.
    """

    if model is None:
        return fallback_reset_time

    for reset_method in model.reset_methods:
        if reset_method.type != "passive":
            continue

        for attribute in reset_method.attributes:
            if attribute.key == "duration" and isinstance(attribute.value, int | float):
                return float(attribute.value) * _PICOSECONDS_AS_SECONDS

    return fallback_reset_time


def _build_pass_manager(
    hardware_model: QuantumHardwareModel,
    passive_reset_time: float,
) -> PassManager:
    """Build the PuRR sanitisation pipeline used before the PuRR import.

    :param hardware_model: Hardware model used by analysis and sanitisation passes.
    :returns: A pass manager configured to materialise legacy implicit PuRR behaviour before
        import.
    """

    target_data = TargetData(
        default_shots=hardware_model.default_repeat_count,
        QUBIT_DATA=QubitDescription(passive_reset_time=passive_reset_time),
    )
    return (
        PassManager()
        | ActivePulseChannelAnalysis(hardware_model)
        | RepeatSanitisation(hardware_model, target_data)
        | ReturnSanitisation()
        | AcquireSanitisation()
        | InitialPhaseResetSanitisation()
        | EndOfTaskResetSanitisation()
        | ResetsToDelays(target_data)
        | SynchronizeTask()
    )


class PurrFrontend(BaseFrontend):
    """Import PuRR builders into pulse-level IR after legacy sanitisation.

    PuRR IR built using :class:`QuantumInstructionBuilder` can omit structure that legacy
    execution paths insert implicitly. This frontend reconstructs that structure before
    importing to the new IR stack. The sanitisation pipeline can add:

    * the shots loop via a Repeat instruction, if not provided,
    * end-of-line resets on the qubits used,
    * a return instruction if not provided,
    * start-of-line phase accumulation resets.

    This keeps the importer focused on translation while preserving legacy program
    behaviour expected by existing PuRR workloads.
    """

    def __init__(
        self, model: CanonicalSystemData | None = None, run_purr_pipeline: bool = True
    ):
        """
        :param model: The model as a :class:`CanonicalSystemData` representation. Optional,
            but required if post-selection is enabled. The frontend derives a
            post-selection view from this model and applies it to the mapped results.
        :param run_purr_pipeline: Can be used to specify if the pipeline should run before
            importing.
        """
        self.model = model
        self._run_pipeline = run_purr_pipeline

    def emit(
        self,
        src: QuantumInstructionBuilder,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        compiler_config: CompilerConfig | None = None,
        **kwargs,
    ) -> ModuleOp:
        """Build a :class:`ModuleOp` from a PuRR instruction builder.

        :param src: The legacy PuRR representation of quantum programs.
        :param res_mgr: Collection of analysis results with caching and aggregation
            capabilities, defaults to None.
        :param met_mgr: Stores useful intermediary metrics that are generated during
            compilation, defaults to None.
        :param compiler_config: Compiler settings, defaults to None.
        :returns: A compiler :class:`ModuleOp` that holds the program in pulse-level
            representation.
        :raises TypeError: If *src* is not a :class:`QuantumInstructionBuilder`.
        :raises ValueError: If post-selection is enabled but no canonical model was
            provided at frontend construction.
        """

        if not isinstance(src, QuantumInstructionBuilder):
            raise TypeError(
                f"PuRR frontend does not support object of type `{type(src).__name__}`."
            )

        res_mgr, met_mgr, compiler_config = self._check_metrics_and_config(
            res_mgr, met_mgr, compiler_config
        )

        # Use the builder's legacy hardware model for sanitisation. The canonical model,
        # when provided, supplies passive reset metadata for target data and post-selection.
        hardware_model = src.model
        passive_reset_time = _extract_passive_reset_time(
            self.model, hardware_model.default_repetition_period
        )

        if self._run_pipeline:
            pipeline = _build_pass_manager(hardware_model, passive_reset_time)
            src = pipeline.run(
                src,
                res_mgr,
                met_mgr,
                compiler_config=compiler_config,
                **kwargs,
            )

        if compiler_config.post_selection:
            if self.model is None:
                raise ValueError(
                    "Canonical system data description must be provided if post-selection "
                    "is enabled."
                )

            pp_derived_view = PostProcessing.derive(self.model)
            post_selection_builder = PostSelectionBuilder(pp_derived_view, enabled=True)
        else:
            post_selection_builder = None

        return PurrImporter(post_selection_builder).build(src)

    def check_and_return_source(self, src: Any) -> QuantumInstructionBuilder | bool:
        """Validate that the source can be processed by this frontend.

        :param src: The input program.
        :returns: The original source when valid, otherwise ``False``.
        """

        if not isinstance(src, QuantumInstructionBuilder):
            return False
        return src
