from qblox_instruments import ClusterType

from qat.purr.backends.qblox.config import (
    AttConfig,
    AwgConfig,
    ConnectionConfig,
    LoConfig,
    ModuleConfig,
    NcoConfig,
    OffsetConfig,
    QbloxConfig,
    ScopeAcqConfig,
    SequencerConfig,
    SquareWeightAcq,
)
from qat.purr.backends.qblox.device import (
    DummyQbloxControlHardware,
    QbloxControlHardware,
    QbloxPhysicalBaseband,
    QbloxPhysicalChannel,
)


def choose_module_slots(cluster_kit):
    if cluster_kit is None:
        qcmrf_slot = next(
            (k for k, v in DUMMY_CONFIG.items() if v == ClusterType.CLUSTER_QCM_RF)
        )
        qrmrf_slot = next(
            (k for k, v in DUMMY_CONFIG.items() if v == ClusterType.CLUSTER_QRM_RF)
        )
    elif cluster_kit == 1:
        qcmrf_slot, qrmrf_slot = 12, 14
    elif cluster_kit == 2:
        qcmrf_slot, qrmrf_slot = 4, 14
    else:
        raise ValueError(f"Expected cluster id 1 or 2, got {cluster_kit} instead")

    return qcmrf_slot, qrmrf_slot


DUMMY_CONFIG = {
    2: ClusterType.CLUSTER_QCM_RF,
    4: ClusterType.CLUSTER_QCM_RF,
    12: ClusterType.CLUSTER_QCM_RF,
    14: ClusterType.CLUSTER_QRM_RF,
    16: ClusterType.CLUSTER_QRM_RF,
    18: ClusterType.CLUSTER_QRM_RF,
}


def setup_qblox_hardware_model(model, cluster_kit=None, name=None) -> QbloxControlHardware:
    """
    cluster_kit is just an integer referring to any live QBlox clusters available
    When provided, some environment variables need to be defined:

    QBLOX_DEV_ID: Cluster Serial ID
    QBLOX_DEV_NAME: Cluster designation name
    QBLOX_DEV_IP: Cluster IP address

    If none is provided then a Dummy cluster is set up
    """
    qcmrf_slot, qrmrf_slot = choose_module_slots(cluster_kit)

    if cluster_kit is None:
        instrument = DummyQbloxControlHardware(name=name, cfg=DUMMY_CONFIG)
        qcmrf_config, qrmrf_config = dummy_qblox_config()
    else:
        instrument = QbloxControlHardware(name=name)
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
    qcmrf_config = QbloxConfig(
        module=ModuleConfig(
            lo=LoConfig(out0_en=True, out1_en=True),
            attenuation=AttConfig(out0=30, out1=30),
            offset=OffsetConfig(out0_path0=0, out0_path1=0, out1_path0=0, out1_path1=0),
        ),
        sequencers={
            0: SequencerConfig(
                sync_en=True,
                connection=ConnectionConfig(bulk_value=["out0"]),
                awg=AwgConfig(mod_en=True),
                nco=NcoConfig(prop_delay_comp_en=True),
            ),
            1: SequencerConfig(
                sync_en=True,
                connection=ConnectionConfig(bulk_value=["out0"]),
                awg=AwgConfig(mod_en=True),
                nco=NcoConfig(prop_delay_comp_en=True),
            ),
            2: SequencerConfig(
                sync_en=True,
                connection=ConnectionConfig(bulk_value=["out0"]),
                awg=AwgConfig(mod_en=True),
                nco=NcoConfig(prop_delay_comp_en=True),
            ),
        },
    )

    qrmrf_config = QbloxConfig(
        module=ModuleConfig(
            lo=LoConfig(out0_in0_en=True),
            attenuation=AttConfig(out0=0, in0=0),
            offset=OffsetConfig(out0_path0=0, out0_path1=0, in0_path0=0, in0_path1=0),
            scope_acq=ScopeAcqConfig(avg_mode_en_path0=True, avg_mode_en_path1=True),
        ),
        sequencers={
            i: SequencerConfig(
                sync_en=True,
                demod_en_acq=True,
                square_weight_acq=SquareWeightAcq(integration_length=4000),
                connection=ConnectionConfig(bulk_value=["out0", "in0"]),
                awg=AwgConfig(mod_en=True),
                nco=NcoConfig(prop_delay_comp_en=True),
            )
            for i in range(6)
        },
    )

    return qcmrf_config, qrmrf_config


def dummy_qblox_config():
    qcmrf_config = QbloxConfig(
        module=ModuleConfig(
            lo=LoConfig(out0_en=True, out1_en=True),
            attenuation=AttConfig(out0=0, out1=0),
            offset=OffsetConfig(out0_path0=0, out0_path1=0, out1_path0=0, out1_path1=0),
        ),
        sequencers={
            0: SequencerConfig(
                sync_en=True,
                connection=ConnectionConfig(bulk_value=["out0"]),
            ),
            1: SequencerConfig(
                sync_en=True,
                connection=ConnectionConfig(bulk_value=["out0"]),
            ),
        },
    )

    qrmrf_config = QbloxConfig(
        module=ModuleConfig(),
        sequencers={
            0: SequencerConfig(
                sync_en=True,
                connection=ConnectionConfig(bulk_value=["out0", "in0"]),
            ),
            1: SequencerConfig(
                sync_en=True,
                connection=ConnectionConfig(bulk_value=["out0", "in0"]),
            ),
        },
    )

    return qcmrf_config, qrmrf_config
