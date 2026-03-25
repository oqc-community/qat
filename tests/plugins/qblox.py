# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Oxford Quantum Circuits Ltd
## Qblox fixtures
import numpy as np
import pytest
from qblox_instruments import ClusterType

with pytest.warns(DeprecationWarning):
    from qat.engines.qblox.dummy import DummyQbloxInstrument
from qat.engines.qblox.live import QbloxLeafInstrument
from qat.purr.backends.qblox.device import QbloxControlHardware
from qat.purr.backends.qblox.dummy import DummyQbloxControlHardware, apply_setup_to_hardware
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel
from qat.utils.uuid import temporary_uuid_seed, uuid4

_QBLOX_DUMMY_CONFIG = {
    1: ClusterType.CLUSTER_QCM,
    2: ClusterType.CLUSTER_QCM_RF,
    4: ClusterType.CLUSTER_QCM_RF,
    6: ClusterType.CLUSTER_QRC,  # slots 6-7 are reserved for QRC module
    12: ClusterType.CLUSTER_QCM_RF,
    13: ClusterType.CLUSTER_QRM,
    14: ClusterType.CLUSTER_QRM_RF,
    16: ClusterType.CLUSTER_QRM_RF,
    18: ClusterType.CLUSTER_QRM_RF,
    19: ClusterType.CLUSTER_QRC,  # slots 19-20 are reserved for QRC module
}


@pytest.fixture()
def qblox_instrument_factory():
    def _instrument(id, name, address):
        if address:
            instrument = QbloxLeafInstrument(id, name, address)
        else:
            instrument = DummyQbloxInstrument(id, name, address, _QBLOX_DUMMY_CONFIG)
        return instrument

    return _instrument


@pytest.fixture()
def qblox_instrument(request, qblox_instrument_factory, function_seed):
    with temporary_uuid_seed(function_seed):
        id = f"{request.node.originalname}_{uuid4()}".replace("-", "_")
        name = id
        address = request.param

        instrument = qblox_instrument_factory(id, name, address)
        instrument.connect()
    yield instrument
    instrument.disconnect()


@pytest.fixture()
def legacy_qblox_instrument_factory():
    def _legacy_instrument(id, name, address):
        if not name.startswith("legacy"):
            name = f"legacy_{id}"

        if address:
            instrument = QbloxControlHardware(dev_id=id, name=name, address=address)
        else:
            instrument = DummyQbloxControlHardware(
                dev_id=id, name=name, dummy_cfg=_QBLOX_DUMMY_CONFIG
            )
        return instrument

    return _legacy_instrument


@pytest.fixture()
def qblox_model(request, legacy_qblox_instrument_factory, function_seed):
    with temporary_uuid_seed(function_seed):
        id = f"{request.node.originalname}_{uuid4()}".replace("-", "_")
        name = id
        address = request.param

    hw_model = QbloxLiveHardwareModel()
    instrument = legacy_qblox_instrument_factory(id, name, address)
    instrument.connect()
    apply_setup_to_hardware(hw_model, instrument=instrument)
    yield hw_model
    instrument.disconnect()


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
            filter_fn=lambda mod: mod.is_qcm_type == qcm_type
            and mod.is_qrm_type == qrm_type
            and mod.is_rf_type == rf_type
            and mod.is_qrc_type == qrc_type
        ).values()

        rng = np.random.default_rng(function_seed)
        module = rng.choice(list(modules))

        # TODO - sequencers are not the same on a QRC, need special handling
        sequencer = rng.choice(module.sequencers)

        return module, sequencer

    yield _qblox_resource
    instrument.disconnect()
