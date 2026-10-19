# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for post-selection metadata emitted by the experimental Qblox backend."""

from dataclasses import replace

import numpy as np
import pytest
from xdsl.dialects import func, scf
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.dialects.builtin import IndexType, IntAttr, ModuleOp, StringAttr
from xdsl.ir import Block, Region
from xdsl.irdl import IRDLOperation, irdl_op_definition, result_def
from xdsl.utils.exceptions import PassFailedException

from qat.backend.qblox.acquisition import (
    Acquisition,
    BinnedAcqData,
    BinnedAndScopeAcqData,
    IntegData,
)
from qat.backend.qblox.execution import QbloxProgram
from qat.core.result_base import ResultManager
from qat.experimental.analysis.post_processing import PostProcessingAnalysis
from qat.experimental.backend.qblox.backend import ExperimentalQbloxBackend
from qat.experimental.dialect.pulse.ir import (
    AcquireOp,
    ConstantOp,
    CreateFrameOp,
    DiscriminateOp,
    FrequencyAttr,
    IntegrateOp,
    IQResultType,
    KernelOp,
    RealThresholdPolicyAttr,
    ReturnOp,
    TimeAttr,
)
from qat.experimental.dialect.results.ir import (
    CreateOp,
    ExtractOp,
    IntegerStatePredicateAttr,
    MapOp,
    PostSelectOp,
    RecordFieldAttr,
    RecordSchemaAttr,
    YieldOp,
)
from qat.experimental.dialect.results.ir.types import RecordType, ResultsCollectionType
from qat.experimental.system_data.canonical.schema import AttributeEntry
from qat.ir.measure import Discriminate, PostSelect
from qat.runtime.passes.analysis import PostSelectionResult
from qat.runtime.passes.transform import QBloxAcquisitionPostProcessing

pytest_plugins = ("tests.unit.experimental.utils.canonical",)


@irdl_op_definition
class _ResultsCollectionOp(IRDLOperation):
    """Provide a results collection SSA value for backend tests.

    :param labels: Acquisition labels exposed by the collection.
    :param shots: Number of acquisition records in the collection.
    """

    name = "test.qblox_post_selection_results"
    result = result_def(ResultsCollectionType)

    def __init__(self, labels: tuple[str, ...], shots: int) -> None:
        """Create a typed collection for the requested acquisitions.

        :param labels: Acquisition labels exposed by the collection.
        :param shots: Number of acquisition records in the collection.
        """

        schema = RecordSchemaAttr(
            [RecordFieldAttr(label, IQResultType()) for label in labels]
        )
        super().__init__(result_types=[ResultsCollectionType(schema, IntAttr(shots))])


@pytest.fixture
def canonical_model_qblox(canonical_model):
    """Provide canonical data with the Qblox versions required by the backend."""
    return replace(
        canonical_model,
        metadata=(
            *canonical_model.metadata,
            AttributeEntry(key="driver_version", value="1.0.0"),
            AttributeEntry(key="fw_version", value="0.13.0"),
        ),
    )


def _kernel(shots: int, acquire_label: str | None = None) -> KernelOp:
    """Build a kernel whose static loop materialises the executable shot count.

    :param shots: Number of loop iterations.
    :param acquire_label: Optional label for an integrated acquisition in the shot loop.
    :returns: Kernel containing the static shot loop.
    """

    index_type = IndexType()
    lower = ArithConstantOp.from_int_and_width(0, index_type)
    upper = ArithConstantOp.from_int_and_width(shots, index_type)
    step = ArithConstantOp.from_int_and_width(1, index_type)
    loop_operations = []
    if acquire_label is not None:
        frequency = ConstantOp(FrequencyAttr(4_200_000_000))
        frame = CreateFrameOp(frequency, StringAttr("port0"))
        duration = ConstantOp(TimeAttr(16_000))
        acquire = AcquireOp(frame, duration, label=acquire_label)
        loop_operations.extend(
            [frequency, frame, duration, acquire, IntegrateOp(acquire.acquisition_result)]
        )
    loop = scf.ForOp(
        lower,
        upper,
        step,
        [],
        Block(ops=[*loop_operations, scf.YieldOp()], arg_types=[index_type]),
    )
    return KernelOp(
        "program",
        ((), ()),
        Region(Block([lower, upper, step, loop, ReturnOp()])),
    )


def _results_function(
    labels: tuple[str, ...],
    predicates: tuple[IntegerStatePredicateAttr, ...] | None,
    *,
    discriminate: bool = True,
) -> func.FuncOp:
    """Build results IR with optional discrimination and record-level predicates.

    :param labels: Acquisition labels processed by the results map.
    :param predicates: Post-selection predicates applied after the map, or ``None``.
    :param discriminate: Whether to discriminate each acquisition before record creation.
    :returns: Function containing the results-processing operations.
    """

    collection = _ResultsCollectionOp(labels, shots=4)
    schema = RecordSchemaAttr([RecordFieldAttr(label, IQResultType()) for label in labels])
    body = Block(arg_types=[RecordType(schema)])
    values = []
    for label in labels:
        extract = ExtractOp.value_from_record(body.args[0], label)
        body.add_op(extract)
        if discriminate:
            operation = DiscriminateOp(extract.result, RealThresholdPolicyAttr(0.0))
            body.add_op(operation)
            values.append(operation.result)
        else:
            values.append(extract.result)

    record = CreateOp.for_record(list(labels), values)
    body.add_ops([record, YieldOp(record.result)])
    mapped = MapOp(
        collection.result,
        body,
        ResultsCollectionType(record.result.type.schema, IntAttr(4)),
    )
    post_select = None if predicates is None else PostSelectOp(mapped.result, *predicates)
    operations = [collection, mapped]
    if post_select is not None:
        operations.append(post_select)
    operations.append(func.ReturnOp())
    return func.FuncOp(
        "results",
        ((), ()),
        Region(Block(operations)),
    )


def _emit(
    module: ModuleOp,
    canonical_model_qblox,
    mocker,
    res_mgr: ResultManager | None = None,
):
    """Emit an executable while replacing code generation outside the test scope.

    :param module: Pulse and results IR consumed by the backend.
    :param canonical_model_qblox: Canonical model used to construct the backend.
    :param mocker: Pytest mock fixture.
    :param res_mgr: Optional result manager used to inspect extracted metadata.
    :returns: Executable produced by the backend.
    """

    mocker.patch(
        "qat.experimental.backend.qblox.backend.create_qblox_configured_q1_pipeline"
    )
    mocker.patch(
        "qat.experimental.backend.qblox.backend.emit_qblox_program",
        return_value=QbloxProgram(
            packages={},
            driver_version="1.0.0",
            fw_version="0.13.0",
        ),
    )
    return ExperimentalQbloxBackend(canonical_model_qblox).emit(module, res_mgr=res_mgr)


def test_backend_appends_each_post_select_after_its_discriminate(
    canonical_model_qblox, mocker
):
    """The executable unions duplicate predicates on their matching acquire chain."""
    predicates = (
        IntegerStatePredicateAttr("readout_0", [1]),
        IntegerStatePredicateAttr("readout_1", [0, 2]),
        IntegerStatePredicateAttr("readout_0", [3]),
    )
    module = ModuleOp(
        [
            _kernel(4),
            _results_function(("readout_0", "readout_1"), predicates),
        ]
    )

    executable = _emit(module, canonical_model_qblox, mocker)

    assert executable.acquires["readout_0"].post_processing == [
        Discriminate(output_variable="readout_0", threshold=0.0),
        PostSelect(output_variable="readout_0", additional_disallowed={1, 3}),
    ]
    assert executable.acquires["readout_1"].post_processing == [
        Discriminate(output_variable="readout_1", threshold=0.0),
        PostSelect(output_variable="readout_1", additional_disallowed={0, 2}),
    ]


def test_backend_without_post_select_preserves_acquire_processing(
    canonical_model_qblox, mocker
):
    """A module without post-selection leaves its executable acquire chain unchanged."""
    module = ModuleOp(
        [
            _kernel(4),
            _results_function(("readout",), None),
        ]
    )
    res_mgr = ResultManager()

    executable = _emit(module, canonical_model_qblox, mocker, res_mgr)

    expected = [Discriminate(output_variable="readout", threshold=0.0)]
    assert executable.acquires["readout"].post_processing == expected
    analysis = res_mgr.lookup_by_type(PostProcessingAnalysis)
    assert analysis.acquire_data["readout"].post_processing == expected


def test_backend_rejects_post_select_without_matching_acquire(
    canonical_model_qblox, mocker
):
    """A predicate cannot reference an acquisition absent from results processing."""
    module = ModuleOp(
        [
            _kernel(4),
            _results_function(
                ("readout",),
                (IntegerStatePredicateAttr("missing", [1]),),
            ),
        ]
    )

    mocker.patch(
        "qat.experimental.backend.qblox.backend.create_qblox_configured_q1_pipeline"
    )
    emitter = mocker.patch("qat.experimental.backend.qblox.backend.emit_qblox_program")
    with pytest.raises(
        PassFailedException,
        match="Post-selection output 'missing' has no matching acquisition",
    ):
        ExperimentalQbloxBackend(canonical_model_qblox).emit(module)
    emitter.assert_not_called()


def test_backend_rejects_post_select_without_discriminate(canonical_model_qblox, mocker):
    """A predicate requires integer state keys from an earlier discriminate step."""
    module = ModuleOp(
        [
            _kernel(4),
            _results_function(
                ("readout",),
                (IntegerStatePredicateAttr("readout", [1]),),
                discriminate=False,
            ),
        ]
    )

    with pytest.raises(
        PassFailedException,
        match="Post-selection output 'readout' has no preceding Discriminate",
    ):
        _emit(module, canonical_model_qblox, mocker)


def test_backend_post_select_filters_runtime_playback(canonical_model_qblox, mocker):
    """The emitted acquire label drives runtime filtering of disallowed shots."""
    kernel = _kernel(4, acquire_label="readout")
    acquire = next(op for op in kernel.walk() if isinstance(op, AcquireOp))
    label = acquire.label.data
    module = ModuleOp(
        [
            kernel,
            _results_function(
                (label,),
                (IntegerStatePredicateAttr(label, [1]),),
            ),
        ]
    )
    executable = _emit(module, canonical_model_qblox, mocker)
    iq = np.array([1.0, -1.0, 0.5, -0.5], dtype=complex)
    playback = {
        "port0": {
            label: Acquisition(
                acquisition=BinnedAndScopeAcqData(
                    bins=BinnedAcqData(
                        integration=IntegData(
                            path0=iq.real.tolist(),
                            path1=iq.imag.tolist(),
                        ),
                        threshold=np.zeros(len(iq)).tolist(),
                    )
                )
            )
        }
    }
    res_mgr = ResultManager()

    result = QBloxAcquisitionPostProcessing().run(playback, res_mgr, package=executable)

    assert result[label].tolist() == [0, 0]
    post_selection = res_mgr.lookup_by_type(PostSelectionResult)
    assert post_selection.shots_requested == 4
    assert post_selection.shots_retained == 2
