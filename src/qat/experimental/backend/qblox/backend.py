# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Experimental Qblox backend for configured Q1 lowering and code generation."""

from compiler_config.config import CompilerConfig
from pydantic import JsonValue
from xdsl.context import Context
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.scf import ForOp
from xdsl.utils.exceptions import PassFailedException

from qat.backend.base import BaseBackend
from qat.backend.qblox.execution import QbloxProgram
from qat.backend.qblox.target_data import TARGET_DATA, QbloxTargetData
from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.executables import Executable
from qat.experimental.analysis.post_processing import extract_post_processing_instructions
from qat.experimental.backend.qblox.codegen import emit_qblox_program
from qat.experimental.conversion.pulse_to_q1.passes import (
    create_qblox_configured_q1_pipeline,
)
from qat.experimental.dialect.pulse.analysis.locate_kernels import locate_kernels
from qat.experimental.dialect.results.ir import ResultsArrayType
from qat.experimental.system_data.canonical.schema import CanonicalSystemData


class ExperimentalQbloxBackend(BaseBackend[QbloxProgram]):
    """Lower pulse IR to configured Q1 IR and emit a Qblox compiler payload.

    :param model: Canonical system data used for physical binding.
    :param target_data: Qblox target limits used during lowering.
    """

    def __init__(
        self,
        model: CanonicalSystemData,
        target_data: QbloxTargetData = TARGET_DATA,
    ):
        super().__init__(model=model)
        self.target_data = target_data

    def emit(
        self,
        ir: ModuleOp,
        res_mgr: ResultManager | None = None,
        met_mgr: MetricsManager | None = None,
        compiler_config: CompilerConfig | None = None,
        metadata: dict[str, JsonValue] | None = None,
        **kwargs,
    ) -> Executable[QbloxProgram]:
        """Lower pulse IR and package the emitted program for the runtime.

        Pulse-level preprocessing is owned by the experimental middleend. The backend
        applies configured Q1 lowering, then translates the verified target IR into the
        shared Qblox runtime payload.

        :param ir: Pulse IR produced by the experimental middleend.
        :param res_mgr: Compilation result manager.
        :param met_mgr: Compilation metrics manager, currently unused.
        :param compiler_config: Compiler settings accepted for pipeline compatibility. The
            effective shot count has already been materialised in ``ir`` by the frontend.
        :param metadata: Optional runtime metadata copied into the emitted program.
        :returns: Executable containing the emitted Qblox program.
        """

        res_mgr = ResultManager() if res_mgr is None else res_mgr
        shots = _extract_shot_count(ir)
        post_processing = extract_post_processing_instructions(ir, (shots,))
        res_mgr.add(post_processing)

        create_qblox_configured_q1_pipeline(self.model, self.target_data).apply(
            Context(), ir
        )
        program = emit_qblox_program(ir, self.target_data, metadata=metadata)
        return Executable[QbloxProgram](
            programs=[program],
            acquires=post_processing.acquire_data,
            assigns=post_processing.assigns,
            returns=post_processing.returns,
            shots=shots,
            calibration_id=self.model.calibration_id,
        )


def _static_loop_trip_count(loop: ForOp, kernel_name: str) -> int:
    """Return the trip count of a statically bounded kernel shot loop."""

    constant_bounds = []
    for bound in (loop.lb, loop.ub, loop.step):
        owner = bound.owner
        if not isinstance(owner, ArithConstantOp):
            raise PassFailedException(
                f"Kernel '{kernel_name}' shot loop must have static bounds."
            )
        constant_bounds.append(owner.value.value.data)

    lower, upper, step = constant_bounds
    if step == 0:
        raise PassFailedException(
            f"Kernel '{kernel_name}' shot loop must have a non-zero step."
        )

    shots = len(range(lower, upper, step))
    if shots < 1:
        raise PassFailedException(
            f"Kernel '{kernel_name}' shot loop must execute at least once."
        )
    return shots


def _extract_shot_count(ir: ModuleOp) -> int:
    """Read the effective shot count from each kernel's materialised control flow.

    The frontend resolves the legacy ``Repeat`` before importing it, then
    :class:`PulseKernelBuilder` represents repeated execution with one top-level static
    :class:`scf.ForOp`. Its no-loop branch represents a single execution. Lowered
    result-array sizes are derived from the same value, so they are checked for consistency
    but are not the source of truth.

    :param ir: Pulse module after middleend processing.
    :returns: The common materialised kernel shot count.
    :raises PassFailedException: If shot loops are ambiguous, dynamic, inconsistent, or
        disagree with lowered result-array sizes.
    """

    kernels = locate_kernels(ir)
    if not kernels:
        raise PassFailedException("Cannot determine executable shots: no kernels found.")

    shot_counts = set()
    for kernel in kernels:
        loops = [
            op
            for block in kernel.operation.body.blocks
            for op in block.ops
            if isinstance(op, ForOp)
        ]
        if len(loops) > 1:
            raise PassFailedException(
                f"Kernel '{kernel.symbol_name}' must contain at most one top-level shot "
                f"loop, found {len(loops)}."
            )

        shots = 1 if not loops else _static_loop_trip_count(loops[0], kernel.symbol_name)
        array_sizes = {
            result_type.size.data
            for result_type in kernel.operation.function_type.outputs
            if isinstance(result_type, ResultsArrayType)
        }
        if array_sizes and array_sizes != {shots}:
            raise PassFailedException(
                f"Kernel '{kernel.symbol_name}' result-array sizes "
                f"{sorted(array_sizes)} do not match its shot-loop count {shots}."
            )
        shot_counts.add(shots)

    if len(shot_counts) != 1:
        raise PassFailedException(
            f"Kernels have inconsistent shot-loop counts {sorted(shot_counts)}."
        )
    return next(iter(shot_counts))
