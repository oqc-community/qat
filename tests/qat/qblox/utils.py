from dataclasses import dataclass

import numpy as np
from qblox_instruments import ClusterType

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


@dataclass
class ClusterInfo:
    id: int = None
    name: str = None
    sid: str = None
    ip: str = None


@dataclass
class MixerTestValues:
    num_points = 2  # Low value for testing to reduce the size of the cartesian products

    # Module values
    qcm_i_offsets = np.linspace(-2.5, 2.5, num_points)  # I offsets (Volt)
    qcm_q_offsets = np.linspace(-2.5, 2.5, num_points)  # Q offsets (Volt)
    qcm_rf_i_offsets = np.linspace(-84, 73, num_points)  # I offsets (mVolt)
    qcm_rf_q_offsets = np.linspace(-84, 73, num_points)  # Q offsets (mVolt)
    qrm_i_offsets = np.linspace(-0.09, 0.09, num_points)  # I offsets (Volt)
    qrm_q_offsets = np.linspace(-0.09, 0.09, num_points)  # Q offsets (Volt)
    qrm_rf_i_offsets = np.linspace(-0.09, 0.09, num_points)  # I offsets (Volt)
    qrm_rf_q_offsets = np.linspace(-0.09, 0.09, num_points)  # Q offsets (Volt)

    # Sequencer values
    phase_offsets = np.linspace(-45, 45, num_points)  # Phase offsets (Degree)
    gain_ratios = np.linspace(0.5, 2, num_points)  # Gain ratios


def choose_module_slots(cluster_kit: ClusterInfo):
    if cluster_kit.id is None:
        qcmrf_slot = next(
            (k for k, v in DUMMY_CONFIG.items() if v == ClusterType.CLUSTER_QCM_RF)
        )
        qrmrf_slot = next(
            (k for k, v in DUMMY_CONFIG.items() if v == ClusterType.CLUSTER_QRM_RF)
        )
    else:
        raise NotImplementedError(f"Depends on your live physical setup")

    return qcmrf_slot, qrmrf_slot


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


def setup_qblox_hardware_model(
    model: QbloxLiveHardwareModel, info: ClusterInfo = ClusterInfo()
) -> QbloxControlHardware:
    """
    cluster_kit wraps information about a QBlox cluster. When provided, some environment
    variables need to be defined:

    QBLOX_DEV_ID: Cluster Serial ID
    QBLOX_DEV_NAME: Cluster designation name
    QBLOX_DEV_IP: Cluster IP address

    If none is provided then a Dummy cluster is set up
    """
    qcmrf_slot, qrmrf_slot = choose_module_slots(info)

    if info.id is None:
        instrument = DummyQbloxControlHardware(name=info.name, dummy_cfg=DUMMY_CONFIG)
        qcmrf_config, qrmrf_config = dummy_qblox_config()
    else:
        instrument = QbloxControlHardware(name=info.name)
        qcmrf_config, qrmrf_config = live_qblox_config()

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

    model.add_physical_baseband(bb1, bb2)
    model.add_physical_channel(ch1, ch2)
    model.add_quantum_device(r0, q0, r1, q1)
    model.control_hardware = instrument

    return instrument


def live_qblox_config():
    raise NotImplementedError(f"Depends on your live physical setup")


def dummy_qblox_config():
    qcmrf_config = QbloxConfig(
        module=ModuleConfig(
            lo=LoConfig(out0_en=True, out1_en=True),
            attenuation=AttConfig(out0=0, out1=0),
            offset=OffsetConfig(out0_path0=0, out0_path1=0, out1_path0=0, out1_path1=0),
        ),
        sequencers={
            0: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out0"]),
            ),
            1: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out0"]),
            ),
        },
    )

    qrmrf_config = QbloxConfig(
        module=ModuleConfig(lo=LoConfig(out0_in0_en=True)),
        sequencers={
            0: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out0", "in0"]),
            ),
            1: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out0", "in0"]),
            ),
        },
    )

    return qcmrf_config, qrmrf_config
