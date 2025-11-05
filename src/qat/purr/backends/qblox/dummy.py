# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import uuid

import numpy as np
import regex
from qblox_instruments import (
    ClusterType,
    DummyBinnedAcquisitionData,
    DummyScopeAcquisitionData,
)
from qblox_instruments.qcodes_drivers.sequencer import Sequencer

from qat.purr.backends.echo import Connectivity
from qat.purr.backends.qblox.codegen import QbloxPackage
from qat.purr.backends.qblox.config import (
    AttConfig,
    ConnectionConfig,
    LoConfig,
    ModuleConfig,
    OffsetConfig,
    QbloxConfig,
    SequencerConfig,
)
from qat.purr.backends.qblox.constants import Constants
from qat.purr.backends.qblox.device import (
    QbloxControlHardware,
    QbloxPhysicalBaseband,
    QbloxPhysicalChannel,
)
from qat.purr.backends.qblox.ir import Sequence
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel
from qat.purr.compiler.devices import ChannelType

# TODO: 32Q support: COMPILER-728
_DUMMY_CONFIG = {
    1: ClusterType.CLUSTER_QCM_RF,
    2: ClusterType.CLUSTER_QCM_RF,
    3: ClusterType.CLUSTER_QCM_RF,
    4: ClusterType.CLUSTER_QCM_RF,
    5: ClusterType.CLUSTER_QCM_RF,
    6: ClusterType.CLUSTER_QCM_RF,
    7: ClusterType.CLUSTER_QCM_RF,
    8: ClusterType.CLUSTER_QCM_RF,
    10: ClusterType.CLUSTER_QCM_RF,
    12: ClusterType.CLUSTER_QCM_RF,
    13: ClusterType.CLUSTER_QRM_RF,
    14: ClusterType.CLUSTER_QRM_RF,
    16: ClusterType.CLUSTER_QRM_RF,
    18: ClusterType.CLUSTER_QRM_RF,
}


class DummyQbloxControlHardware(QbloxControlHardware):
    shot_pattern = regex.compile("jlt( +)R([0-9]+),([0-9]+),@(.*)\n")

    def _setup_dummy_scope_acq_data(self, sequencer: Sequencer, sequence: Sequence):
        shot_match = next(self.shot_pattern.finditer(sequence.program), None)
        avg_count = int(shot_match.group(3)) if shot_match else 1

        dummy_data = np.random.random(
            size=(Constants.MAX_SAMPLE_SIZE_SCOPE_ACQUISITIONS, 2)
        )
        dummy_data = [(iq[0], iq[1]) for iq in dummy_data]
        dummy_scope_acquisition_data = DummyScopeAcquisitionData(
            data=dummy_data, out_of_range=(False, False), avg_cnt=(avg_count, avg_count)
        )
        sequencer.set_dummy_scope_acquisition_data(data=dummy_scope_acquisition_data)

    def _setup_dummy_binned_acq_data(self, sequencer: Sequencer, sequence: Sequence):
        shot_match = next(self.shot_pattern.finditer(sequence.program), None)
        avg_count = int(shot_match.group(3)) if shot_match else 1

        for name, acquisition in sequence.acquisitions.items():
            dummy_binned_acquisition_data = [
                DummyBinnedAcquisitionData(
                    data=(np.random.random(), np.random.random()),
                    thres=np.random.choice(2),
                    avg_cnt=avg_count,
                )
            ] * acquisition["num_bins"]
            sequencer.set_dummy_binned_acquisition_data(
                acq_index_name=name,
                data=dummy_binned_acquisition_data,
            )

    def _delete_acquisitions(self, sequencer):
        sequencer.delete_dummy_scope_acquisition_data()
        sequencer.delete_dummy_binned_acquisition_data()

    def set_data(self, packages: list[QbloxPackage]):
        super().set_data(packages)

        # Stage Scope and Acquisition data
        for pulse_channel_id, sequencer in self._id2seq.items():
            if ChannelType.macq.name in pulse_channel_id:
                package = next(
                    (pkg for pkg in packages if pkg.pulse_channel_id == pulse_channel_id)
                )
                self._setup_dummy_scope_acq_data(sequencer, package.sequence)
                self._setup_dummy_binned_acq_data(sequencer, package.sequence)


def get_default_dummy_control_hardware() -> DummyQbloxControlHardware:
    name = "default_dummy_control"
    return DummyQbloxControlHardware(dev_id=name, name=name, dummy_cfg=_DUMMY_CONFIG)


def _letter_from_int(val: int) -> str:
    letter = chr(ord("A") + (val % 25))
    if val >= 25:
        letter = _letter_from_int(val // 25 - 1) + letter

    return letter


def _get_module_slot_key_iterators(instrument: DummyQbloxControlHardware):
    cluster = instrument.driver
    qcmrfs = iter(
        cluster.get_connected_modules(
            filter_fn=lambda mod: mod.is_qcm_type and mod.is_rf_type
        ).keys()
    )
    qrmrfs = iter(
        cluster.get_connected_modules(
            filter_fn=lambda mod: mod.is_qrm_type and mod.is_rf_type
        ).keys()
    )

    return qcmrfs, qrmrfs


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


def apply_setup_to_hardware(
    model: QbloxLiveHardwareModel,
    qubit_count: int = 2,
    connectivity: Connectivity | list[tuple[int, int]] | None = None,
    add_direction_couplings: bool = False,
    instrument: DummyQbloxControlHardware = None,
):
    instrument = instrument or get_default_dummy_control_hardware()
    disconnect_after = False
    if not instrument.is_connected:
        instrument.connect()
        disconnect_after = True

    qcmrfs, qrmrfs = _get_module_slot_key_iterators(instrument)
    qcmrf_config, qrmrf_config = create_qcmrf_qrmrf_config()

    for index in range(qubit_count):
        if index % 2 == 0:
            qcmrf_slot = next(qcmrfs)
        if index % 6 == 0:
            qrmrf_slot = next(qrmrfs)

        letter = _letter_from_int(index)

        bb_q = QbloxPhysicalBaseband(
            f"{letter}-LO-0-QCM-RF-{qcmrf_slot}",
            4.024e9,
            250e6,
            instrument=instrument,
            slot_idx=qcmrf_slot,
            config=qcmrf_config,
        )

        bb_r = QbloxPhysicalBaseband(
            f"{letter}-LO-0-QRM-RF-{qrmrf_slot}",
            9.7772e9,
            250e6,
            instrument=instrument,
            slot_idx=qrmrf_slot,
            config=qrmrf_config,
        )

        ch_q = QbloxPhysicalChannel(f"{letter}-CH-QCM-RF-{qcmrf_slot}", 1e-9, bb_q, 1)
        ch_r = QbloxPhysicalChannel(
            f"{letter}-CH-QRM-RF-{qrmrf_slot}", 1e-9, bb_r, 1, acquire_allowed=True
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

    if disconnect_after:
        instrument.disconnect()
    return model


def get_default_dummy_hardware(
    qubit_count: int = 2,
    connectivity: Connectivity | list[tuple[int, int]] | None = None,
    add_direction_couplings: bool = False,
    name: str | None = None,
    dummy_cfg: dict | None = None,
):
    name = name or f"dummy_model_{qubit_count}_{uuid.uuid4()}".replace("-", "_")
    instrument = DummyQbloxControlHardware(
        dev_id=name, name=name, dummy_cfg=dummy_cfg or _DUMMY_CONFIG
    )

    model = QbloxLiveHardwareModel()
    apply_setup_to_hardware(
        model,
        qubit_count=qubit_count,
        connectivity=connectivity,
        add_direction_couplings=add_direction_couplings,
        instrument=instrument,
    )

    return model
