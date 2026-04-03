# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import uuid
from collections import deque

from qblox_instruments import Cluster, ClusterType

from qat.backend.qblox.config.specification import (
    AttConfig,
    ConnectionConfig,
    LoConfig,
    ModuleConfig,
    OffsetConfig,
    QbloxConfig,
    SequencerConfig,
)
from qat.backend.qblox.target_data import QCM_RF_DATA, QRC_DATA, QRM_RF_DATA
from qat.model.loaders.purr.base import BaseLegacyModelLoader
from qat.purr.backends.echo import Connectivity
from qat.purr.backends.qblox.device import (
    QbloxControlHardware,
    QbloxPhysicalBaseband,
    QbloxPhysicalChannel,
)
from qat.purr.backends.qblox.dummy import DummyQbloxControlHardware
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel

# TODO: 32Q support: COMPILER-728
DEFAULT_QUBIT_COUNT = 16
DEFAULT_DUMMY_CONFIG = {
    1: ClusterType.CLUSTER_QRC,
    2: ClusterType.CLUSTER_QCM_RF,
    3: ClusterType.CLUSTER_QCM_RF,
    4: ClusterType.CLUSTER_QCM_RF,
    5: ClusterType.CLUSTER_QCM_RF,
    6: ClusterType.CLUSTER_QCM_RF,
    7: ClusterType.CLUSTER_QRC,  # slots 7-8 are reserved for QRC module
    12: ClusterType.CLUSTER_QCM_RF,
    13: ClusterType.CLUSTER_QRM,
    14: ClusterType.CLUSTER_QRM_RF,
    16: ClusterType.CLUSTER_QRM_RF,
    18: ClusterType.CLUSTER_QRM_RF,
    19: ClusterType.CLUSTER_QRC,  # slots 19-20 are reserved for QRC module
}

_CONTROL_CONFIG_MAP = {
    ClusterType.CLUSTER_QCM_RF: QbloxConfig(
        module=ModuleConfig(
            lo=LoConfig(out0_en=True, out1_en=True),
            attenuation=AttConfig(out0=0, out1=0),
            offset=OffsetConfig(out0_path0=0, out0_path1=0, out1_path0=0, out1_path1=0),
        ),
        sequencers={
            i: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out0", "out1"]),
            )
            for i in range(QCM_RF_DATA.number_of_sequencers)
        },
    ),
    ClusterType.CLUSTER_QRC: QbloxConfig(
        module=ModuleConfig(
            lo=LoConfig(out0_in0_en=True),
            attenuation=AttConfig(out0=0, out1=0, out2=0, out3=0, out4=0, out5=0),
        ),
        sequencers={
            0: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out0", "out1", "out2"])
            ),
            1: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out0", "out1", "out3"])
            ),
            2: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out0", "out1", "out4"])
            ),
            3: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out0", "out1", "out5"])
            ),
            4: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out0", "out1", "out2"])
            ),
            5: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out0", "out1", "out3"])
            ),
            6: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out0", "out1", "out4"])
            ),
            7: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out0", "out1", "out5"])
            ),
            8: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out2", "out3", "out4", "out5"]),
            ),
            9: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out2", "out3", "out4", "out5"]),
            ),
            10: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out2", "out3", "out4", "out5"]),
            ),
            11: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out2", "out3", "out4", "out5"]),
            ),
        },
    ),
}

_READOUT_CONFIG_MAP = {
    ClusterType.CLUSTER_QRM_RF: QbloxConfig(
        module=ModuleConfig(lo=LoConfig(out0_in0_en=True)),
        sequencers={
            i: SequencerConfig(
                connection=ConnectionConfig(bulk_value=["out0", "in0"]),
            )
            for i in range(QRM_RF_DATA.number_of_sequencers)
        },
    ),
    ClusterType.CLUSTER_QRC: QbloxConfig(
        module=ModuleConfig(lo=LoConfig(out0_in0_en=True)),
        sequencers={
            i: SequencerConfig(
                # only use in/out0 as you cannot connect both inputs to a single seq
                connection=ConnectionConfig(bulk_value=["out0", "in0"]),
            )
            for i in range(QRC_DATA.number_of_readout_sequencers)
        },
    ),
}


def create_legacy_instrument(
    id: str,
    name: str,
    address: str = None,
    dummy_config: dict = None,
) -> QbloxControlHardware:
    if address is None:
        return DummyQbloxControlHardware(
            dev_id=id, name=name, dummy_cfg=dummy_config or DEFAULT_DUMMY_CONFIG
        )
    else:
        return QbloxControlHardware(dev_id=id, name=name, address=address)


def apply_setup_to_hardware(
    model: QbloxLiveHardwareModel,
    instrument: QbloxControlHardware,
    qubit_count: int = 2,
    connectivity: Connectivity | list[tuple[int, int]] | None = None,
    add_direction_couplings: bool = False,
):
    if not instrument.is_connected:
        instrument.connect()

    slot_allocator = QbloxSlotAllocator(cluster=instrument.driver)

    if instrument.is_connected:
        instrument.disconnect()

    for index in range(qubit_count):
        control_slot, control_type = slot_allocator.allocate_control()
        readout_slot, readout_type = slot_allocator.allocate_readout()

        control_config = _CONTROL_CONFIG_MAP[control_type]
        readout_config = _READOUT_CONFIG_MAP[readout_type]

        control_type = control_type.value.split(" ")[1]
        readout_type = readout_type.value.split(" ")[1]

        bb_q = QbloxPhysicalBaseband(
            f"Q{index}-LO-{control_type}-{control_slot}",
            4e9,
            250e6,
            instrument=instrument,
            slot_idx=control_slot,
            config=control_config,
        )
        ch_q = QbloxPhysicalChannel(
            f"Q{index}-CH-{control_type}-{control_slot}", 1e-9, bb_q, 1
        )

        bb_r = QbloxPhysicalBaseband(
            f"R{index}-LO-{readout_type}-{readout_slot}",
            9.9e9,
            250e6,
            instrument=instrument,
            slot_idx=readout_slot,
            config=readout_config,
        )
        ch_r = QbloxPhysicalChannel(
            f"R{index}-CH-{readout_type}-{readout_slot}",
            1e-9,
            bb_r,
            1,
            acquire_allowed=True,
        )

        resonator = ch_r.build_resonator(f"R{index}", frequency=10.2033e9)
        qubit = ch_q.build_qubit(
            index,
            resonator,
            drive_freq=3.872e9,
            second_state_freq=4.085e9,
            channel_scale=1,
            measure_amp=0.5,
            fixed_drive_if=True,
        )
        # TODO - Add support for mapping cross resonance channels to the right QBlox sequencers
        # add_cross_resonance(qubit, resonator)
        # COMPILER-685

        model.add_physical_baseband(bb_q, bb_r)
        model.add_physical_channel(ch_q, ch_r)
        model.add_quantum_device(resonator, qubit)
        model.control_hardware = instrument

    return model


class QbloxSlotAllocator:
    """Helper class to allocate QBlox slots for qubits and resonators based on the provided
    hardware configuration."""

    def __init__(
        self,
        cluster: Cluster,
        qubits_per_qcm: int = 2,
        qubits_per_qcmrf: int = 2,
        qubits_per_qrm: int = 6,
        qubits_per_qrmrf: int = 6,
        qubits_per_qrc_control: int = 2,
        qubits_per_qrc_readout: int = 6,
    ):
        self.num_qubits_per_qcm = qubits_per_qcm
        self.num_qubits_per_qcmrf = qubits_per_qcmrf
        self.num_qubits_per_qrm = qubits_per_qrm
        self.num_qubits_per_qrmrf = qubits_per_qrmrf

        # between 4-6 seqs available on four outputs
        self.num_qubits_per_qrc_control = qubits_per_qrc_control
        # up to 8 seqs for readout on two outputs
        self.num_qubits_per_qrc_readout = qubits_per_qrc_readout

        self.control_queue, self.readout_queue, self.slots_map = self._build_queues(cluster)

    def allocate_control(self) -> tuple[int, ClusterType]:
        """Allocates a control line to the next available control slots."""

        control_slot = self.control_queue.popleft() if self.control_queue else None
        if control_slot is None:
            raise ValueError("No more control slots available for allocation.")
        return control_slot, self.slots_map[control_slot]

    def allocate_readout(self) -> tuple[int, ClusterType]:
        """Allocates a readout line to the next available readout slots."""

        readout_slot = self.readout_queue.popleft() if self.readout_queue else None
        if readout_slot is None:
            raise ValueError("No more readout slots available for allocation.")
        return readout_slot, self.slots_map[readout_slot]

    def _build_queues(
        self, cluster: Cluster
    ) -> tuple[deque[int], deque[int], dict[int, ClusterType]]:
        control_queue = deque()
        readout_queue = deque()
        slots_map = {}

        connected_modules = cluster.get_connected_modules()
        for slot, module in connected_modules.items():
            if module.is_qcm_type:
                if module.is_rf_type:
                    control_queue.extend([slot] * self.num_qubits_per_qcmrf)
                    slots_map[slot] = ClusterType.CLUSTER_QCM_RF
                else:
                    slots_map[slot] = ClusterType.CLUSTER_QCM
            elif module.is_qrm_type:
                if module.is_rf_type:
                    readout_queue.extend([slot] * self.num_qubits_per_qrmrf)
                    slots_map[slot] = ClusterType.CLUSTER_QRM_RF
                else:
                    slots_map[slot] = ClusterType.CLUSTER_QRM
            elif module.is_qrc_type:
                control_queue.extend([slot] * self.num_qubits_per_qrc_control)
                readout_queue.extend([slot] * self.num_qubits_per_qrc_readout)
                slots_map[slot] = ClusterType.CLUSTER_QRC

        return control_queue, readout_queue, slots_map


class QbloxModelLoader(BaseLegacyModelLoader):
    def __init__(
        self,
        id: str = None,
        name: str = None,
        address: str = None,
        dummy_config: dict = None,
        qubit_count: int = 4,
        connectivity: Connectivity | list[(int, int)] | None = None,
        add_direction_couplings=True,
    ):
        id = id or f"qblox_model_{qubit_count}_{uuid.uuid4()}".replace("-", "_")
        name = name or id
        self.instrument = create_legacy_instrument(
            id=id, name=name, address=address, dummy_config=dummy_config
        )

        self.qubit_count = qubit_count
        self.connectivity = connectivity
        self.add_direction_couplings = add_direction_couplings

    def load(self):
        model = QbloxLiveHardwareModel()
        apply_setup_to_hardware(
            model=model,
            instrument=self.instrument,
            qubit_count=self.qubit_count,
            connectivity=self.connectivity,
            add_direction_couplings=self.add_direction_couplings,
        )

        return model
