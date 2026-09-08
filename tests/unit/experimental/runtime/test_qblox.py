# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
"""Tests for experimental Qblox result record processing."""

import numpy as np
import pytest
from compiler_config.config import CompilerConfig, QuantumResultsFormat

from qat.backend.qblox.acquisition import (
    Acquisition,
    BinnedAcqData,
    BinnedAndScopeAcqData,
    IntegData,
    PathData,
    ScopeAcqData,
)
from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.executables import AcquireData, Executable
from qat.experimental.runtime.results_pipeline import get_qblox_results_pipeline
from qat.ir.instruction_basetypes import AcquireMode
from qat.ir.instructions import Assign
from qat.ir.measure import Discriminate, PostSelect
from qat.runtime.aggregator import QBloxAggregator
from qat.runtime.passes.analysis import PostSelectionResult
from qat.runtime.passes.transform import (
    AssignResultsTransform,
    InlineResultsProcessingTransform,
    QBloxAcquisitionPostProcessing,
    ResultTransform,
)
from qat.runtime.simple import SimpleRuntime

from tests.unit.utils.executables import MockProgram

pytestmark = pytest.mark.qblox


def _acquisition(values: list[complex], name: str = "readout") -> Acquisition:
    """Build a Qblox acquisition containing the supplied integrator IQ values."""
    return Acquisition(
        name=name,
        acquisition=BinnedAndScopeAcqData(
            bins=BinnedAcqData(
                integration=IntegData(
                    path0=[value.real for value in values],
                    path1=[value.imag for value in values],
                ),
                threshold=[0] * len(values),
                avg_cnt=[1] * len(values),
            )
        ),
    )


def _scope_acquisition(values: list[complex], name: str = "readout") -> Acquisition:
    """Build a Qblox acquisition containing the supplied scope IQ values."""
    return Acquisition(
        name=name,
        acquisition=BinnedAndScopeAcqData(
            scope=ScopeAcqData(
                path0=PathData(data=[value.real for value in values]),
                path1=PathData(data=[value.imag for value in values]),
            )
        ),
    )


def _discriminate(name: str = "readout") -> Discriminate:
    """Build a zero-threshold discriminator for an acquisition output."""
    return Discriminate(output_variable=name, threshold=0.0)


def _acquire(
    name: str,
    physical_channel: str,
    shape: tuple[int, ...],
    post_processing=None,
    mode: AcquireMode = AcquireMode.INTEGRATOR,
) -> AcquireData:
    """Build an acquire with optional post-processing instructions."""
    return AcquireData(
        mode=mode,
        shape=shape,
        physical_channel=physical_channel,
        post_processing=(
            [_discriminate(name)] if post_processing is None else post_processing
        ),
    )


def _run_pipeline(playback, package, res_mgr: ResultManager | None = None):
    """Run the model-independent Qblox results pipeline with standard test managers."""
    return get_qblox_results_pipeline().run(
        playback,
        res_mgr or ResultManager(),
        MetricsManager(),
        package=package,
        compiler_config=CompilerConfig(),
    )


def _aggregate(*batches, acquires):
    """Aggregate raw Qblox playback batches into named acquisition records."""
    aggregator = QBloxAggregator()
    for batch in batches:
        aggregator.append(batch, acquires)
    return aggregator.finalise()


def _mock_playback(executable: Executable) -> dict[str, list[Acquisition]]:
    """Build raw playback from the executable's acquisition declarations."""
    playback = {}
    for name, acquire in executable.acquires.items():
        sample_count = int(np.prod(acquire.shape))
        channel = acquire.physical_channel
        playback.setdefault(channel, []).append(
            _acquisition(
                [
                    1.0 + 0.0j if index % 2 == 0 else -1.0 + 0.0j
                    for index in range(sample_count)
                ],
                name=name,
            )
        )
    return playback


@pytest.fixture()
def mock_qblox_engine(mocker):
    """Provide an engine mock that returns deterministic alternating IQ values."""
    engine = mocker.MagicMock()
    engine.execute.side_effect = _mock_playback
    return engine


@pytest.fixture()
def simple_qblox_runtime(mock_qblox_engine):
    """Provide a ``SimpleRuntime`` configured with the Qblox aggregator and pipeline."""
    runtime = SimpleRuntime(
        engine=mock_qblox_engine,
        results_pipeline=get_qblox_results_pipeline(),
        aggregator=QBloxAggregator(),
    )
    return runtime


def test_qblox_results_pipeline_has_expected_passes():
    """Verify that the model-independent pipeline contains the expected four passes."""
    pipeline = get_qblox_results_pipeline()

    assert [type(pass_._pass) for pass_ in pipeline.passes] == [
        QBloxAcquisitionPostProcessing,
        InlineResultsProcessingTransform,
        AssignResultsTransform,
        ResultTransform,
    ]


@pytest.mark.parametrize("qblox_model", [{}], indirect=True)
def test_qblox_results_pipeline_applies_processing_post_selection_and_returns(
    qblox_model,
    simple_qblox_runtime,
):
    """Verify post-selection, register assignment, and retained-shot metadata."""
    package = Executable(
        programs=[],
        acquires={
            "readout": _acquire(
                "readout",
                qblox_model.get_qubit(0).get_acquire_channel().physical_channel_id,
                (4,),
                [
                    _discriminate(),
                    PostSelect(output_variable="readout", additional_disallowed={1}),
                ],
            )
        },
        assigns=[Assign(name="register", value=["readout"])],
        returns={"register"},
        shots=4,
    )

    playback = simple_qblox_runtime.engine.execute(package)
    simple_qblox_runtime.aggregator.append(playback, package.acquires)
    res_mgr = ResultManager()
    result = _run_pipeline(simple_qblox_runtime.aggregator.finalise(), package, res_mgr)

    assert set(result) == {"register"}
    assert np.asarray(result["register"]).tolist() == [[0, 0]]

    post_selection = res_mgr.lookup_by_type(PostSelectionResult)
    assert post_selection.shots_requested == 4
    assert post_selection.shots_retained == 2


@pytest.mark.parametrize("qblox_model", [{}], indirect=True)
def test_qblox_aggregated_playback_is_converted_to_expected_result(
    qblox_model,
):
    """Verify that ordered aggregated IQ batches become the expected bit results."""
    acquire_channel = qblox_model.get_qubit(0).get_acquire_channel().physical_channel_id
    package = Executable(
        programs=[],
        acquires={"readout": _acquire("readout", acquire_channel, (4,))},
        returns={"readout"},
        shots=4,
    )
    first_batch = {acquire_channel: [_acquisition([1.0 + 0.0j, -1.0 + 0.0j])]}
    second_batch = {acquire_channel: [_acquisition([0.5 + 0.0j, -0.5 + 0.0j])]}
    aggregated = _aggregate(first_batch, second_batch, acquires=package.acquires)
    assert np.asarray(
        aggregated[acquire_channel]["readout"].acquisition.bins.integration.path0
    ).tolist() == [1.0, -1.0, 0.5, -0.5]

    result = _run_pipeline(aggregated, package)

    assert np.asarray(result["readout"]).tolist() == [0, 1, 0, 1]


@pytest.mark.parametrize("qblox_model", [{}], indirect=True)
def test_qblox_raw_results_use_scope_data(qblox_model):
    """Verify that RAW acquisitions return complex IQ values from scope data."""
    acquire_channel = qblox_model.get_qubit(0).get_acquire_channel().physical_channel_id
    package = Executable(
        programs=[],
        acquires={
            "readout": _acquire(
                "readout",
                acquire_channel,
                (2,),
                post_processing=[],
                mode=AcquireMode.RAW,
            )
        },
        returns={"readout"},
        shots=2,
    )
    playback = {acquire_channel: [_scope_acquisition([1.0 + 0.5j, -0.25 + 2.0j])]}

    result = get_qblox_results_pipeline().run(
        _aggregate(playback, acquires=package.acquires),
        ResultManager(),
        MetricsManager(),
        package=package,
        compiler_config=CompilerConfig(results_format=QuantumResultsFormat().raw()),
    )

    assert np.asarray(result["readout"]).tolist() == [1.0 + 0.5j, -0.25 + 2.0j]


@pytest.mark.parametrize("qblox_model", [{}], indirect=True)
def test_qblox_results_pipeline_routes_multiple_channels_and_acquisitions(qblox_model):
    """Verify that acquisitions remain associated with their channel and output name."""
    channels = [
        qblox_model.get_qubit(index).get_acquire_channel().physical_channel_id
        for index in (0, 1)
    ]
    acquires = {
        name: _acquire(name, channel, (2,))
        for name, channel in zip(("readout_0", "readout_1"), channels, strict=True)
    }
    package = Executable(programs=[], acquires=acquires, returns=set(acquires), shots=2)
    aggregated = _aggregate(
        {
            channels[0]: [_acquisition([1.0 + 0.0j, -1.0 + 0.0j], "readout_0")],
            channels[1]: [_acquisition([-1.0 + 0.0j, 1.0 + 0.0j], "readout_1")],
        },
        acquires=acquires,
    )

    result = _run_pipeline(aggregated, package)

    assert np.asarray(result["readout_0"]).tolist() == [0, 1]
    assert np.asarray(result["readout_1"]).tolist() == [1, 0]


@pytest.mark.parametrize("qblox_model", [{}], indirect=True)
def test_qblox_runtime_executes_programs_before_processing(qblox_model, mocker):
    """Verify that ``SimpleRuntime`` executes programs before processing their playback."""
    acquire_channel = qblox_model.get_qubit(0).get_acquire_channel().physical_channel_id
    engine = mocker.MagicMock()
    engine.execute.return_value = {
        acquire_channel: [_acquisition([1.0 + 0.0j, -1.0 + 0.0j])]
    }
    runtime = SimpleRuntime(
        engine=engine,
        results_pipeline=get_qblox_results_pipeline(),
        aggregator=QBloxAggregator(),
    )
    package = Executable(
        programs=[MockProgram(shapes={"readout": (2,)})],
        acquires={"readout": _acquire("readout", acquire_channel, (2,))},
        returns={"readout"},
        shots=2,
    )

    result = runtime.execute(package, compiler_config=CompilerConfig())

    engine.execute.assert_called_once()
    assert np.asarray(result["readout"]).tolist() == [0, 1]


@pytest.mark.parametrize("qblox_model", [{}], indirect=True)
def test_qblox_runtime_aggregates_multiple_programs_in_order(qblox_model, mocker):
    """Verify that runtime batches are aggregated in execution order before processing."""
    acquire_channel = qblox_model.get_qubit(0).get_acquire_channel().physical_channel_id
    engine = mocker.MagicMock()
    engine.execute.side_effect = [
        {acquire_channel: [_acquisition([1.0 + 0.0j, -1.0 + 0.0j])]},
        {acquire_channel: [_acquisition([0.5 + 0.0j, -0.5 + 0.0j])]},
    ]
    runtime = SimpleRuntime(
        engine=engine,
        results_pipeline=get_qblox_results_pipeline(),
        aggregator=QBloxAggregator(),
    )
    package = Executable(
        programs=[
            MockProgram(shapes={"readout": (2,)}),
            MockProgram(shapes={"readout": (2,)}),
        ],
        acquires={"readout": _acquire("readout", acquire_channel, (4,))},
        returns={"readout"},
        shots=4,
    )

    result = runtime.execute(package, compiler_config=CompilerConfig())

    assert engine.execute.call_count == 2
    assert np.asarray(result["readout"]).tolist() == [0, 1, 0, 1]
