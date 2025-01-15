# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import csv
import os
from typing import Dict, List

from qat.purr.backends.qblox.config import (
    AttConfig,
    AwgConfig,
    ConnectionConfig,
    LoConfig,
    ModuleConfig,
    NcoConfig,
    QbloxConfig,
    SequencerConfig,
    SquareWeightAcq,
)
from qat.purr.backends.qblox.device import (
    QbloxControlHardware,
    QbloxPhysicalBaseband,
    QbloxPhysicalChannel,
)
from qat.purr.backends.qblox.execution import CompositeControlHardware
from qat.purr.backends.qblox.instrument_base import InstrumentModel, LeafInstrument
from qat.purr.backends.qblox.live import QbloxLiveHardwareModel
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


def build_qcm_rf_config(row: Dict):
    lo_idx = row["Q-LO-IDX"]

    lo = LoConfig()
    setattr(lo, f"out{lo_idx}_en", True)
    setattr(lo, f"out{lo_idx}_freq", float(row["Q-LO-FREQ"]))

    output = int(row["Q-OUTPUT"])

    att = AttConfig()
    setattr(att, f"out{output}", int(row["Q-OUTPUT-ATT"]))

    connection = ConnectionConfig(bulk_value=[f"out{output}"])

    nco = NcoConfig()

    awg = AwgConfig(
        mod_en=True,
    )

    qcm_rf_config = QbloxConfig(
        module=ModuleConfig(
            lo=lo,
            attenuation=att,
        ),
        sequencers={
            i
            + 3
            * output: SequencerConfig(
                connection=connection,
                nco=nco,
                awg=awg,
            )
            for i in range(3)
        },
    )

    return qcm_rf_config


def build_qrm_rf_config(row: Dict):
    lo_idx = row["R-LO-IDX"]

    lo = LoConfig()
    setattr(lo, f"out{lo_idx}_in{lo_idx}_en", True)
    setattr(lo, f"out{lo_idx}_in{lo_idx}_freq", float(row["R-LO-FREQ"]))

    output = int(row["R-OUTPUT"])
    input = int(row["R-INPUT"])

    att = AttConfig()
    setattr(att, f"out{output}", int(row["R-OUTPUT-ATT"]))
    setattr(att, f"in{input}", int(row["R-INPUT-ATT"]))

    connection = ConnectionConfig(bulk_value=[f"out{output}", f"in{input}"])

    nco = NcoConfig()

    awg = AwgConfig(
        mod_en=True,
    )

    qrm_rf_config = QbloxConfig(
        module=ModuleConfig(
            lo=lo,
            attenuation=att,
        ),
        sequencers={
            i: SequencerConfig(
                connection=connection,
                nco=nco,
                awg=awg,
                demod_en_acq=True,
                square_weight_acq=SquareWeightAcq(integration_length=4000),
            )
            for i in range(6)
        },
    )

    return qrm_rf_config


def build_hardware_model_from_config_csv(file_path: str):
    if not os.path.exists(file_path):
        raise ValueError(f"File '{file_path}' not found!")

    rows: List[Dict] = []
    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    instruments: Dict[str, QbloxControlHardware] = {}
    model = QbloxLiveHardwareModel()

    for row in rows:
        if not row["Q-CLUSTER-SID"] or not row["R-CLUSTER-SID"]:
            log.warning(f"No instrument found for pair {row['PAIR-IDX']}")
            continue

        q_ctl_hw = instruments.setdefault(
            row["Q-CLUSTER-SID"],
            QbloxControlHardware(),
        )
        q_ctl_hw.id = row["Q-CLUSTER-SID"]
        q_ctl_hw.name = row["Q-CLUSTER-NAME"]
        q_ctl_hw.address = row["Q-CLUSTER-ADDR"]

        r_ctl_hw = instruments.setdefault(
            row["R-CLUSTER-SID"],
            QbloxControlHardware(),
        )
        r_ctl_hw.id = row["R-CLUSTER-SID"]
        r_ctl_hw.name = row["R-CLUSTER-NAME"]
        r_ctl_hw.address = row["R-CLUSTER-ADDR"]

        qcm_rf_config = build_qcm_rf_config(row)
        q_bb_id = f"{row['LABEL']}-LO-{row['Q-LO-IDX']}-QCM-RF-{row['Q-SLOT-IDX']}"
        q_bb = QbloxPhysicalBaseband(
            id_=q_bb_id,
            frequency=float(row["Q-LO-FREQ"]),
            if_frequency=250e6,
            instrument=q_ctl_hw,
            slot_idx=int(row["Q-SLOT-IDX"]),
            config=qcm_rf_config,
        )

        qrm_rf_config = build_qrm_rf_config(row)
        r_bb_id = f"{row['LABEL']}-LO-{row['R-LO-IDX']}-QRM-RF-{row['R-SLOT-IDX']}"
        r_bb = QbloxPhysicalBaseband(
            id_=r_bb_id,
            frequency=float(row["R-LO-FREQ"]),
            if_frequency=250e6,
            instrument=r_ctl_hw,
            slot_idx=int(row["R-SLOT-IDX"]),
            config=qrm_rf_config,
        )

        q_ch_id = f"{row['LABEL']}-CH-QCM-RF-{row['Q-SLOT-IDX']}"
        q_ch = QbloxPhysicalChannel(
            q_ch_id,
            1e-9,
            q_bb,
            1,
        )
        r_ch_id = f"{row['LABEL']}-CH-QRM-RF-{row['R-SLOT-IDX']}"
        r_ch = QbloxPhysicalChannel(
            r_ch_id,
            1e-9,
            r_bb,
            1,
            acquire_allowed=True,
        )

        index = int(row["PAIR-IDX"])
        resonator = r_ch.build_resonator(
            resonator_id=f"R{index}",
            frequency=float(row["R"]),
        )
        qubit = q_ch.build_qubit(
            index,
            resonator,
            drive_freq=float(row["Q"]),
            second_state_freq=float(row["F12"]),
            channel_scale=1,
            measure_amp=50e-3,
            fixed_drive_if=False,
        )

        model.add_physical_baseband(q_bb, r_bb)
        model.add_physical_channel(r_ch, q_ch)
        model.add_quantum_device(resonator, qubit)
        if q_ctl_hw == r_ctl_hw:
            # TODO - Control hardware to support multiple clusters
            model.control_hardware = q_ctl_hw

    for row in rows:
        if not row["Q-CLUSTER-SID"] or not row["R-CLUSTER-SID"]:
            log.warning(f"No instrument found for pair {row['PAIR-IDX']}")
            continue

        cq_name = f"Q{row['PAIR-IDX']}"
        if cq_name not in model.quantum_devices:
            continue
        cq = model.quantum_devices[cq_name]
        for tq in row["PAIR-CONNECTIONS"].split(","):
            tq_name = f"Q{tq}"
            if tq_name not in model.quantum_devices:
                continue
            tq = model.quantum_devices[tq_name]
            if cq not in tq.coupled_qubits:
                log.warning("2Q measurements support is coming up in the future")
                # add_cross_resonance(cq, tq)

    # calibration_file_path = f"{os.path.splitext(os.path.basename(file_path))[0]}"
    # model.save_calibration_to_file(calibration_file_path, use_cwd=True)
    return model


def parse_control_hardware(clusters_csv: str):
    """
    Builds a ControlHardware object wrapping an arbitrary fleet of Qblox clusters defined as CSV
    """

    if not os.path.exists(clusters_csv):
        raise ValueError(f"File '{clusters_csv}' not found!")

    with open(clusters_csv) as f:
        reader = csv.DictReader(f)
        clusters = [InstrumentModel.model_validate(row) for row in reader]

    control_hardware = CompositeControlHardware()
    for cluster in clusters:
        control_hardware.add(LeafInstrument(cluster))

    return control_hardware
