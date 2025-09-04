# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2025 Oxford Quantum Circuits Ltd

import uuid

import numpy as np
import pytest
from qblox_instruments import Cluster, ClusterType

from qat.engines.qblox.dummy import DummyQbloxInstrument
from qat.engines.qblox.live import QbloxLeafInstrument
from qat.purr.backends.qblox.config import (
    AttConfig,
    ConnectionConfig,
    LoConfig,
    ModuleConfig,
    OffsetConfig,
    QbloxConfig,
    SequencerConfig,
)
from qat.purr.backends.qblox.device import (
    DummyQbloxControlHardware,
    QbloxControlHardware,
    QbloxPhysicalBaseband,
    QbloxPhysicalChannel,
)
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()

DUMMY_CONFIG = {
    1: ClusterType.CLUSTER_QCM,
    2: ClusterType.CLUSTER_QCM_RF,
    4: ClusterType.CLUSTER_QCM_RF,
    12: ClusterType.CLUSTER_QCM_RF,
    13: ClusterType.CLUSTER_QRM,
    14: ClusterType.CLUSTER_QRM_RF,
    16: ClusterType.CLUSTER_QRM_RF,
    18: ClusterType.CLUSTER_QRM_RF,
}


def choose_qcmrf_qrmrf_slots(cluster: Cluster):
    qcmrfs = cluster.get_connected_modules(
        filter_fn=lambda mod: mod.is_qcm_type and mod.is_rf_type
    )
    qrmrfs = cluster.get_connected_modules(
        filter_fn=lambda mod: mod.is_qrm_type and mod.is_rf_type
    )
    qcmrf_slot = next((k for k, v in qcmrfs.items()))
    qrmrf_slot = next((k for k, v in qrmrfs.items()))

    return qcmrf_slot, qrmrf_slot


def create_qcmrf_qrmrf_config():
    qcmrf_config = QbloxConfig(
        module=ModuleConfig(
            lo=LoConfig(out0_en=True, out1_en=True),
            attenuation=AttConfig(out0=0, out1=0),
            offset=OffsetConfig(out0_path0=0, out0_path1=0, out1_path0=0, out1_path1=0),
        ),
        sequencers={
            i: SequencerConfig(
                connection=ConnectionConfig(bulk_value=[f"out{i % 2}"]),
            )
            for i in range(6)
        },
    )

    qrmrf_config = QbloxConfig(
        module=ModuleConfig(lo=LoConfig(out0_in0_en=True)),
        sequencers={
            i: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out0", "in0"]),
            )
            for i in range(6)
        },
    )

    return qcmrf_config, qrmrf_config


def setup_hw_model(hw_model: QbloxLiveHardwareModel, instrument: QbloxControlHardware):
    qcmrf_slot, qrmrf_slot = choose_qcmrf_qrmrf_slots(instrument.driver)

    qcmrf_config, qrmrf_config = create_qcmrf_qrmrf_config()
    bb1 = QbloxPhysicalBaseband(
        "QCM-RF-LO",
        4.024e9,
        250e6,
        instrument=instrument,
        slot_idx=qcmrf_slot,
        config=qcmrf_config,
    )

    bb2 = QbloxPhysicalBaseband(
        "QRM-RF-LO",
        9.7772e9,
        250e6,
        instrument=instrument,
        slot_idx=qrmrf_slot,
        config=qrmrf_config,
    )

    ch1 = QbloxPhysicalChannel("CH1", 1e-9, bb1, 1)
    ch2 = QbloxPhysicalChannel("CH2", 1e-9, bb2, 1, acquire_allowed=True)

    r0 = ch2.build_resonator("R0", frequency=10.2033e9)
    q0 = ch1.build_qubit(
        0,
        r0,
        drive_freq=3.872e9,
        second_state_freq=4.085e9,
        channel_scale=1,
        measure_amp=0.5,
        fixed_drive_if=True,
    )

    r1 = ch2.build_resonator("R1", frequency=9.9731e9)
    q1 = ch1.build_qubit(
        1,
        r1,
        drive_freq=4.111e9,
        second_state_freq=3.7234e9,
        channel_scale=1,
        measure_amp=0.5,
        fixed_drive_if=True,
    )

    # TODO - Add support for mapping cross resonance channels to the right QBlox sequencers
    # add_cross_resonance(q0, q1)

    hw_model.add_physical_baseband(bb1, bb2)
    hw_model.add_physical_channel(ch1, ch2)
    hw_model.add_quantum_device(r0, q0, r1, q1)
    hw_model.control_hardware = instrument


@pytest.fixture()
def instrument_factory():
    def _instrument(id, name, address):
        if address:
            instrument = QbloxLeafInstrument(id, name, address)
        else:
            instrument = DummyQbloxInstrument(id, name, address, DUMMY_CONFIG)
        return instrument

    return _instrument


@pytest.fixture()
def legacy_instrument_factory():
    def _legacy_instrument(id, name, address):
        if address:
            instrument = QbloxControlHardware(dev_id=id, name=name, address=address)
        else:
            instrument = DummyQbloxControlHardware(
                dev_id=id, name=name, dummy_cfg=DUMMY_CONFIG
            )
        return instrument

    return _legacy_instrument


@pytest.fixture()
def model(request, legacy_instrument_factory):
    id = f"{request.node.originalname}_{uuid.uuid4()}".replace("-", "_")
    name = id
    address = request.param

    hw_model = QbloxLiveHardwareModel()
    instrument = legacy_instrument_factory(id, name, address)
    instrument.connect()
    setup_hw_model(hw_model, instrument)
    yield hw_model
    instrument.disconnect()


@pytest.fixture()
def qblox_resource(request, legacy_instrument_factory):
    id = f"{request.node.originalname}_{uuid.uuid4()}".replace("-", "_")
    name = id
    address = request.param

    instrument = legacy_instrument_factory(id, name, address)
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
