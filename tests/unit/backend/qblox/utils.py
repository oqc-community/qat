# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from collections import namedtuple

import numpy as np
from compiler_config.config import CompilerConfig
from qblox_instruments import ClusterType

from qat.backend.qblox.codegen import QbloxBackend1, QbloxBackend2
from qat.backend.qblox.execution import QbloxProgram
from qat.backend.qblox.target_data import QbloxTargetData
from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.engines.qblox.execution import QbloxEngine
from qat.executables import Executable
from qat.pipelines.purr.qblox.compile import (
    backend_pipeline1,
    backend_pipeline2,
    middleend_pipeline1,
    middleend_pipeline2,
)
from qat.pipelines.purr.qblox.execute import get_results_pipeline
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel
from qat.runtime import SimpleRuntime
from qat.runtime.aggregator import QBloxAggregator

Allocation = namedtuple("Allocation", ["control_slot", "readout_slot"])

_ADDRESSES = [None, None, None, None, None]

_DUMMY_CONFIGS = [
    {
        1: ClusterType.CLUSTER_QRC,
    },
    {
        5: ClusterType.CLUSTER_QRM_RF,
        15: ClusterType.CLUSTER_QRC,
    },
    {
        1: ClusterType.CLUSTER_QCM_RF,
        2: ClusterType.CLUSTER_QCM_RF,
        3: ClusterType.CLUSTER_QRM_RF,
    },
    {
        1: ClusterType.CLUSTER_QCM_RF,
        2: ClusterType.CLUSTER_QCM_RF,
        3: ClusterType.CLUSTER_QRC,
        5: ClusterType.CLUSTER_QRC,
    },
    {
        1: ClusterType.CLUSTER_QRC,
        3: ClusterType.CLUSTER_QRC,
        5: ClusterType.CLUSTER_QRC,
        7: ClusterType.CLUSTER_QRC,
        9: ClusterType.CLUSTER_QCM_RF,
        10: ClusterType.CLUSTER_QCM_RF,
        11: ClusterType.CLUSTER_QCM_RF,
        12: ClusterType.CLUSTER_QCM_RF,
        13: ClusterType.CLUSTER_QRM_RF,
    },
]

_COUNTS = [1, 2, 4, 8, 16]

_ALLOCATIONS = [
    {0: Allocation(1, 1)},
    {0: Allocation(15, 5), 1: Allocation(15, 5)},
    {
        0: Allocation(1, 3),
        1: Allocation(1, 3),
        2: Allocation(2, 3),
        3: Allocation(2, 3),
    },
    {
        0: Allocation(1, 3),
        1: Allocation(1, 3),
        2: Allocation(2, 3),
        3: Allocation(2, 3),
        4: Allocation(3, 3),
        5: Allocation(3, 3),
        6: Allocation(5, 5),
        7: Allocation(5, 5),
    },
    {
        0: Allocation(1, 1),
        1: Allocation(1, 1),
        2: Allocation(3, 1),
        3: Allocation(3, 1),
        4: Allocation(5, 1),
        5: Allocation(5, 1),
        6: Allocation(7, 3),
        7: Allocation(7, 3),
        8: Allocation(9, 3),
        9: Allocation(9, 3),
        10: Allocation(10, 3),
        11: Allocation(10, 3),
        12: Allocation(11, 5),
        13: Allocation(11, 5),
        14: Allocation(12, 5),
        15: Allocation(12, 5),
    },
]

assert len(_ADDRESSES) == len(_DUMMY_CONFIGS) == len(_COUNTS) == len(_ALLOCATIONS)

QBLOX_TARGET_DATA = QbloxTargetData()
QCM_DATA = QBLOX_TARGET_DATA.QCM_DATA
QCM_RF_DATA = QBLOX_TARGET_DATA.QCM_RF_DATA
QRM_DATA = QBLOX_TARGET_DATA.QRM_DATA
QRM_RF_DATA = QBLOX_TARGET_DATA.QRM_RF_DATA
QRC_DATA = QBLOX_TARGET_DATA.QRC_DATA
Q1ASM_DATA = QBLOX_TARGET_DATA.Q1ASM_DATA
CONTROL_SEQUENCER_DATA = QBLOX_TARGET_DATA.CONTROL_SEQUENCER_DATA
READOUT_SEQUENCER_DATA = QBLOX_TARGET_DATA.READOUT_SEQUENCER_DATA


def create_parameters(selections, indices=None):
    params = []
    indices = indices or range(len(_ADDRESSES))
    for i in indices:
        param = []
        for name in selections:
            match name:
                case "model":
                    param.append(
                        {
                            "address": _ADDRESSES[i],
                            "dummy_config": _DUMMY_CONFIGS[i],
                            "qubit_count": _COUNTS[i],
                        }
                    )
                case "instrument":
                    param.append(
                        {
                            "address": _ADDRESSES[i],
                            "dummy_config": _DUMMY_CONFIGS[i],
                        }
                    )
                case "dummy_config":
                    param.append(_DUMMY_CONFIGS[i])
                case "qubit_count":
                    param.append(_COUNTS[i])
                case "allocation":
                    param.append(_ALLOCATIONS[i])
        params.append(tuple(param))
    return params


def do_emit(model: QbloxLiveHardwareModel, backend_type: type, builder, ignore_empty=True):
    if backend_type == QbloxBackend1:
        middleend_pipeline = middleend_pipeline1(model, QbloxTargetData())
        backend_pipeline = backend_pipeline1()
    elif backend_type == QbloxBackend2:
        middleend_pipeline = middleend_pipeline2(model, QbloxTargetData())
        backend_pipeline = backend_pipeline2()
    else:
        raise ValueError(f"Expected QbloxBackend1 or QbloxBackend2, got {backend_type}")

    res_mgr = ResultManager()
    met_mgr = MetricsManager()
    middleend_pipeline.run(builder, res_mgr, met_mgr, enable_hw_averaging=True)
    backend = backend_type(model=model, pipeline=backend_pipeline)
    executable = backend.emit(builder, res_mgr, met_mgr, ignore_empty)
    return executable


def do_execute(model, instrument, executable: Executable[QbloxProgram]):
    engine = QbloxEngine(instrument)
    runtime = SimpleRuntime(
        engine=engine,
        aggregator=QBloxAggregator(),
        results_pipeline=get_results_pipeline(model),
    )
    assert isinstance(runtime.aggregator, QBloxAggregator)
    results = runtime.execute(executable=executable, compiler_config=CompilerConfig())
    return results


def qblox_resource(instrument, seed, type):
    qcm_type = type in [ClusterType.CLUSTER_QCM, ClusterType.CLUSTER_QCM_RF]
    qrm_type = type in [ClusterType.CLUSTER_QRM, ClusterType.CLUSTER_QRM_RF]
    rf_type = type in [
        ClusterType.CLUSTER_QCM_RF,
        ClusterType.CLUSTER_QRM_RF,
        ClusterType.CLUSTER_QRC,
    ]
    qrc_type = type in [ClusterType.CLUSTER_QRC]
    modules = instrument.driver.get_connected_modules(
        filter_fn=lambda mod: mod.is_qcm_type == qcm_type
        and mod.is_qrm_type == qrm_type
        and mod.is_rf_type == rf_type
        and mod.is_qrc_type == qrc_type
    ).values()

    rng = np.random.default_rng(seed)
    module = rng.choice(list(modules))
    sequencer = rng.choice(module.sequencers)

    return module, sequencer
