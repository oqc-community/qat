# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for the experimental Qblox compile pipeline."""

import pytest
from compiler_config.config import CompilerConfig
from xdsl.dialects import scf
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import IndexType, IntAttr, ModuleOp
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import PassFailedException

from qat.backend.qblox.execution import QbloxProgram
from qat.backend.qblox.target_data import TARGET_DATA, QbloxTargetData
from qat.core.result_base import ResultManager
from qat.executables import AcquireData, Executable
from qat.experimental.analysis.post_processing import PostProcessingAnalysis
from qat.experimental.backend.qblox.backend import (
    ExperimentalQbloxBackend,
    _extract_shot_count,
)
from qat.experimental.dialect.pulse.ir import AcquisitionType, KernelOp
from qat.experimental.dialect.results.ir import ResultsArrayType
from qat.experimental.frontend.importer.pulse.builder import PulseKernelBuilder
from qat.experimental.frontend.purr import PurrFrontend
from qat.experimental.middleend.middleend import PulseLevelMiddleend
from qat.experimental.pipelines.compile import (
    ExperimentalQbloxCompilePipeline,
    ExperimentalQbloxCompilePipelineConfig,
)
from qat.ir.measure import AcquireMode, PostSelect
from qat.pipelines.pipeline import CompilePipeline

pytest_plugins = ("tests.unit.experimental.utils.canonical",)


def _kernel_with_shot_loop(
    shots: int,
    *,
    name: str = "program",
    result_sizes: tuple[int, ...] = (),
    additional_shot_loops: tuple[int, ...] = (),
) -> KernelOp:
    body_ops = []
    index_type = IndexType()
    for loop_shots in (shots, *additional_shot_loops):
        lower = ArithConstantOp.from_int_and_width(0, index_type)
        upper = ArithConstantOp.from_int_and_width(loop_shots, index_type)
        step = ArithConstantOp.from_int_and_width(1, index_type)
        loop = scf.ForOp(
            lower,
            upper,
            step,
            [],
            Block(ops=[scf.YieldOp()], arg_types=[index_type]),
        )
        body_ops.extend([lower, upper, step, loop])
    result_types = [
        ResultsArrayType(AcquisitionType(), IntAttr(size)) for size in result_sizes
    ]
    return KernelOp(
        name,
        ((), result_types),
        Region(Block(body_ops)),
    )


def _kernel_with_dynamic_shot_loop() -> KernelOp:
    index_type = IndexType()
    body = Block(arg_types=[index_type])
    lower = ArithConstantOp.from_int_and_width(0, index_type)
    step = ArithConstantOp.from_int_and_width(1, index_type)
    loop = scf.ForOp(
        lower,
        body.args[0],
        step,
        [],
        Block(ops=[scf.YieldOp()], arg_types=[index_type]),
    )
    body.add_ops([lower, step, loop])
    return KernelOp("program", ((index_type,), ()), Region(body))


class TestExperimentalQbloxCompilePipeline:
    def test_can_be_instantiated_with_canonical_model(self, canonical_model):
        """The public pipeline factory accepts canonical system data directly."""
        pipeline = ExperimentalQbloxCompilePipeline(
            config=ExperimentalQbloxCompilePipelineConfig(),
            model=canonical_model,
        )

        assert pipeline.model is canonical_model
        assert isinstance(pipeline.pipeline, CompilePipeline)

    def test_build_pipeline_uses_canonical_model_and_experimental_components(
        self, canonical_model
    ):
        """The compile pipeline wires canonical data into all experimental components."""
        pipeline = ExperimentalQbloxCompilePipeline._build_pipeline(
            config=ExperimentalQbloxCompilePipelineConfig(),
            model=canonical_model,
            target_data=None,
        )

        assert isinstance(pipeline, CompilePipeline)
        assert pipeline.model is canonical_model
        assert pipeline.target_data is TARGET_DATA
        assert isinstance(pipeline.frontend, PurrFrontend)
        assert pipeline.frontend.model is canonical_model
        assert isinstance(pipeline.middleend, PulseLevelMiddleend)
        assert pipeline.middleend.model is canonical_model
        assert isinstance(pipeline.backend, ExperimentalQbloxBackend)
        assert pipeline.backend.model is canonical_model
        assert pipeline.backend.target_data is TARGET_DATA

    def test_build_pipeline_uses_supplied_target_data(self, canonical_model):
        """The pipeline and backend preserve explicitly supplied Qblox target data."""
        target_data = QbloxTargetData()

        pipeline = ExperimentalQbloxCompilePipeline._build_pipeline(
            config=ExperimentalQbloxCompilePipelineConfig(),
            model=canonical_model,
            target_data=target_data,
        )

        assert pipeline.target_data is target_data
        assert pipeline.backend.target_data is target_data

    def test_backend_emits_executable_with_runtime_payload(self, canonical_model, mocker):
        """The backend lowers configured Q1 IR and wraps compiler and result metadata."""
        lowering = mocker.Mock()
        lowering_factory = mocker.patch(
            "qat.experimental.backend.qblox.backend.create_qblox_configured_q1_pipeline",
            return_value=lowering,
        )
        program = QbloxProgram(
            packages={},
            driver_version=TARGET_DATA.driver_version,
            fw_version=TARGET_DATA.fw_version,
            metadata={"revision": 1},
        )
        emitter = mocker.patch(
            "qat.experimental.backend.qblox.backend.emit_qblox_program",
            return_value=program,
        )
        backend = ExperimentalQbloxBackend(canonical_model)
        module = ModuleOp([_kernel_with_shot_loop(1)])

        executable = backend.emit(module, metadata={"revision": 1})

        lowering_factory.assert_called_once_with(canonical_model, TARGET_DATA)
        lowering.apply.assert_called_once()
        emitter.assert_called_once_with(
            module,
            TARGET_DATA,
            metadata={"revision": 1},
        )
        assert isinstance(executable, Executable)
        assert executable.programs == [program]
        assert executable.calibration_id == canonical_model.calibration_id

    def test_backend_attaches_results_processing_metadata(self, canonical_model, mocker):
        """The backend preserves analysis metadata needed by the current runtime."""
        mocker.patch(
            "qat.experimental.backend.qblox.backend.create_qblox_configured_q1_pipeline"
        )
        program = QbloxProgram(
            packages={},
            driver_version=TARGET_DATA.driver_version,
            fw_version=TARGET_DATA.fw_version,
        )
        mocker.patch(
            "qat.experimental.backend.qblox.backend.emit_qblox_program",
            return_value=program,
        )
        acquire = AcquireData(
            mode=AcquireMode.INTEGRATOR,
            shape=(7,),
            physical_channel="",
        )
        post_select = PostSelect(output_variable="readout", additional_disallowed={1})
        analysis = PostProcessingAnalysis(
            acquire_data={"readout": acquire},
            post_selects=[post_select],
            returns={"readout"},
        )
        extractor = mocker.patch(
            "qat.experimental.backend.qblox.backend.extract_post_processing_instructions",
            return_value=analysis,
        )
        result_manager = ResultManager()
        module = ModuleOp([_kernel_with_shot_loop(7)])

        executable = ExperimentalQbloxBackend(canonical_model).emit(
            module,
            res_mgr=result_manager,
            compiler_config=CompilerConfig(repeats=99),
        )

        extractor.assert_called_once_with(module, (7,))
        assert executable.acquires == {"readout": acquire}
        assert executable.acquires["readout"].post_processing == []
        assert executable.returns == {"readout"}
        assert executable.shots == 7
        assert result_manager.lookup_by_type(PostProcessingAnalysis) is analysis
        assert analysis.post_selects == [post_select]

    def test_shot_count_comes_from_materialised_kernel_loop(self):
        """The importer-generated shot loop is authoritative for executable shots."""
        module = ModuleOp([_kernel_with_shot_loop(11, result_sizes=(11,))])

        assert _extract_shot_count(module) == 11

    def test_shot_count_does_not_require_result_arrays(self):
        """Acquisition-free kernels retain their shot count in the materialised loop."""
        module = ModuleOp([_kernel_with_shot_loop(7)])

        assert _extract_shot_count(module) == 7

    def test_kernel_without_shot_loop_is_one_shot(self):
        """The no-loop builder branch represents one execution of the kernel body."""
        module = ModuleOp([PulseKernelBuilder("program").finalize()])

        assert _extract_shot_count(module) == 1

    @pytest.mark.parametrize(
        ("kernels", "expected"),
        [
            pytest.param(
                [],
                "no kernels found",
                id="no-kernels",
            ),
            pytest.param(
                [_kernel_with_shot_loop(7, additional_shot_loops=(7,))],
                "at most one top-level shot loop",
                id="multiple-shot-loops",
            ),
            pytest.param(
                [_kernel_with_dynamic_shot_loop()],
                "must have static bounds",
                id="dynamic-shot-loop",
            ),
            pytest.param(
                [
                    _kernel_with_shot_loop(7, name="first"),
                    _kernel_with_shot_loop(8, name="second"),
                ],
                "inconsistent shot-loop counts",
                id="inconsistent-kernels",
            ),
            pytest.param(
                [_kernel_with_shot_loop(7, result_sizes=(8,))],
                "do not match its shot-loop count",
                id="inconsistent-results",
            ),
        ],
    )
    def test_rejects_ambiguous_shot_count(self, kernels, expected):
        """Malformed Pulse IR must not produce unreliable executable shot metadata."""
        with pytest.raises(PassFailedException, match=expected):
            _extract_shot_count(ModuleOp(kernels))
