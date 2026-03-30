# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
import numpy as np
import pytest
from qblox_instruments import ClusterType

from qat.model.loaders.purr import QbloxModelLoader
from qat.model.loaders.purr.qblox import DEFAULT_DUMMY_CONFIG, DEFAULT_QUBIT_COUNT
from qat.model.loaders.qblox import create_instrument
from qat.utils.uuid import temporary_uuid_seed, uuid4


@pytest.fixture(scope="module")
def qblox_instrument(request, module_seed):
    with temporary_uuid_seed(module_seed):
        id = f"test_{uuid4()}".replace("-", "_")

    if not request.param:
        request.param = {}
    address = request.param.get("address", None)
    dummy_config = request.param.get("dummy_config", DEFAULT_DUMMY_CONFIG)

    instrument = create_instrument(id, id, address, dummy_config)
    instrument.connect()
    yield instrument
    instrument.disconnect()


@pytest.fixture(scope="module")
def qblox_model(request, module_seed):
    with temporary_uuid_seed(module_seed):
        id = f"test_{uuid4()}".replace("-", "_")

    if not request.param:
        request.param = {}
    address = request.param.get("address", None)
    dummy_config = request.param.get("dummy_config", DEFAULT_DUMMY_CONFIG)
    qubit_count = request.param.get("qubit_count", DEFAULT_QUBIT_COUNT)

    hw_model = QbloxModelLoader(
        id=id,
        name=id,
        address=address,
        dummy_config=dummy_config,
        qubit_count=qubit_count,
    ).load()
    yield hw_model


@pytest.fixture()
def qblox_resource(request, legacy_qblox_instrument_factory, function_seed):
    with temporary_uuid_seed(function_seed):
        id = f"{request.node.originalname}_{uuid4()}".replace("-", "_")
        name = id
        address = request.param

    instrument = legacy_qblox_instrument_factory(id, name, address)
    instrument.connect()

    def _qblox_resource(type: ClusterType):
        qcm_type = type in [ClusterType.CLUSTER_QCM, ClusterType.CLUSTER_QCM_RF]
        qrm_type = type in [ClusterType.CLUSTER_QRM, ClusterType.CLUSTER_QRM_RF]
        rf_type = type in [
            ClusterType.CLUSTER_QCM_RF,
            ClusterType.CLUSTER_QRM_RF,
            ClusterType.CLUSTER_QRC,
        ]
        qrc_type = type in [ClusterType.CLUSTER_QRC]
        modules = instrument.driver.get_connected_modules(
            filter_fn=lambda mod: (
                mod.is_qcm_type == qcm_type
                and mod.is_qrm_type == qrm_type
                and mod.is_rf_type == rf_type
                and mod.is_qrc_type == qrc_type
            )
        ).values()

        rng = np.random.default_rng(function_seed)
        module = rng.choice(list(modules))

        # TODO - sequencers are not the same on a QRC, need special handling
        sequencer = rng.choice(module.sequencers)

        return module, sequencer

    yield _qblox_resource
    instrument.disconnect()
