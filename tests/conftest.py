# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

import uuid

import numpy as np
import pytest
from qblox_instruments import ClusterType

with pytest.warns(DeprecationWarning):
    from qat.engines.qblox.dummy import DummyQbloxInstrument
from qat.engines.qblox.live import QbloxLeafInstrument
from qat.purr.backends.qblox.device import QbloxControlHardware
from qat.purr.backends.qblox.dummy import (
    DummyQbloxControlHardware,
    apply_setup_to_hardware,
)
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel


@pytest.fixture(scope="session")
def testpath(pytestconfig):
    return pytestconfig.rootpath / "tests"


tests_dir = None


def pytest_addoption(parser):
    parser.addoption(
        "--experimental-enable",
        action="store_const",
        const=0,
        dest="experimental",
        default=-1,
        help="run experimental tests",
    )
    parser.addoption(
        "--experimental-only",
        action="store_const",
        const=1,
        dest="experimental",
        help="run only experimental tests",
    )
    parser.addoption(
        "--legacy-enable",
        action="store_const",
        const=0,
        dest="legacy",
        default=-1,
        help="run legacy tests",
    )
    parser.addoption(
        "--legacy-only",
        action="store_const",
        const=1,
        dest="legacy",
        help="run only legacy tests",
    )


def pytest_configure(config):
    # Set global tests_dir path
    global tests_dir
    tests_dir = config.rootpath / "tests"
    mark_string = config.option.markexpr
    mark_list = [mark_string] if len(mark_string) > 0 else []
    for marker in ["experimental", "legacy"]:
        if marker in mark_string:
            continue
        val = getattr(config.option, marker)
        if val == 1:
            mark_list.append(marker)
        elif val == -1:
            mark_list.append(f"not {marker}")
    setattr(config.option, "markexpr", " and ".join(mark_list))


## Qblox fixtures
_QBLOX_DUMMY_CONFIG = {
    1: ClusterType.CLUSTER_QCM,
    2: ClusterType.CLUSTER_QCM_RF,
    4: ClusterType.CLUSTER_QCM_RF,
    12: ClusterType.CLUSTER_QCM_RF,
    13: ClusterType.CLUSTER_QRM,
    14: ClusterType.CLUSTER_QRM_RF,
    16: ClusterType.CLUSTER_QRM_RF,
    18: ClusterType.CLUSTER_QRM_RF,
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
def qblox_instrument(request, qblox_instrument_factory):
    id = f"{request.node.originalname}_{uuid.uuid4()}".replace("-", "_")
    name = id
    address = request.param

    instrument = qblox_instrument_factory(id, name, address)
    instrument.connect()
    yield instrument
    instrument.disconnect()


@pytest.fixture()
def legacy_qblox_instrument_factory():
    def _legacy_instrument(id, name, address):
        if address:
            instrument = QbloxControlHardware(dev_id=id, name=name, address=address)
        else:
            instrument = DummyQbloxControlHardware(
                dev_id=id, name=name, dummy_cfg=_QBLOX_DUMMY_CONFIG
            )
        return instrument

    return _legacy_instrument


@pytest.fixture()
def qblox_model(request, legacy_qblox_instrument_factory):
    id = f"{request.node.originalname}_{uuid.uuid4()}".replace("-", "_")
    name = id
    address = request.param

    hw_model = QbloxLiveHardwareModel()
    instrument = legacy_qblox_instrument_factory(id, name, address)
    instrument.connect()
    apply_setup_to_hardware(hw_model, instrument=instrument)
    yield hw_model
    instrument.disconnect()


@pytest.fixture()
def qblox_resource(request, legacy_qblox_instrument_factory):
    id = f"{request.node.originalname}_{uuid.uuid4()}".replace("-", "_")
    name = id
    address = request.param

    instrument = legacy_qblox_instrument_factory(id, name, address)
    instrument.connect()

    def _qblox_resource(type: ClusterType):
        qcm_type = type in [ClusterType.CLUSTER_QCM, ClusterType.CLUSTER_QCM_RF]
        qrm_type = type in [ClusterType.CLUSTER_QRM, ClusterType.CLUSTER_QRM_RF]
        rf_type = type in [ClusterType.CLUSTER_QCM_RF, ClusterType.CLUSTER_QRM_RF]
        modules = [
            module
            for module in instrument.driver.get_connected_modules(
                filter_fn=lambda mod: mod.is_qcm_type == qcm_type
                and mod.is_qrm_type == qrm_type
                and mod.is_rf_type == rf_type
            ).values()
        ]
        module = np.random.choice(modules)
        sequencer = np.random.choice(module.sequencers)

        return module, sequencer

    yield _qblox_resource
    instrument.disconnect()
