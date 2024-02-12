# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

import json
from datetime import datetime
from functools import reduce
from typing import Dict

import pytest
from qat.purr.utils.logging_utils import log_duration
from qblox_instruments import DummyScopeAcquisitionData, ClusterType

import numpy as np

from qat.purr.backends.live import LiveHardwareModel
from qat.purr.backends.qblox import (
    QbloxControlHardware,
    QbloxPhysicalChannel,
    QbloxPhysicalBaseband,
    SequenceFile,
    QbloxLiveEngine,
)
from qat.purr.compiler.devices import PulseShapeType, PulseChannel
from qat.purr.compiler.runtime import get_runtime, get_builder
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


def setup_qblox_hardware_model(model, instrument):
    """Apply the default live hardware setup to the passed-in hardware."""

    bb_12_0 = QbloxPhysicalBaseband(
        "MOD-12-LO-0",
        4.024e9,
        250e6,
        instrument=instrument,
        slot_idx=12,
        lo_idx=0,
        sequencer_connections={
            0: "out0",
            1: "out0",
            2: "out0",
        },  # for drive, second_state, cross_resonance, and cross_resonance_cancellation channels
    )
    bb_12_1 = QbloxPhysicalBaseband(
        "MOD-12-LO-1",
        4.024e9,
        250e6,
        instrument=instrument,
        slot_idx=12,
        lo_idx=1,
        sequencer_connections={
            3: "out1",
            4: "out1",
            5: "out1",
        },  # for drive, second_state, cross_resonance, and cross_resonance_cancellation channels
    )

    bb_14 = QbloxPhysicalBaseband(
        "MOD-14-LO-0",
        9.7772e9,
        250e6,
        instrument=instrument,
        slot_idx=14,
        lo_idx=0,
        sequencer_connections={
            0: "io0",
            1: "io0",
            2: "io0",
            3: "io0",
            4: "io0",
            5: "io0",
        },  # for measure and acquire channels
    )
    bb_18 = QbloxPhysicalBaseband(
        "MOD-18-LO-0",
        7.8891e9,
        250e6,
        instrument=instrument,
        slot_idx=18,
        lo_idx=0,
        sequencer_connections={
            0: "io0",
            1: "io0",
            2: "io0",
            3: "io0",
            4: "io0",
            5: "io0",
        },  # for measure and acquire channels
    )

    ch1 = QbloxPhysicalChannel("CH1", bb_12_0, 1e-9)
    ch2 = QbloxPhysicalChannel("CH2", bb_14, 1e-9, acquire_allowed=True)

    ch3 = QbloxPhysicalChannel("CH3", bb_12_1, 1e-9)
    ch4 = QbloxPhysicalChannel("CH4", bb_14, 1e-9, acquire_allowed=True)

    # Loop back O[1-2] -> I[1-2]
    ch5 = QbloxPhysicalChannel("CH5", bb_18, 1e-9, acquire_allowed=True)
    ch6 = QbloxPhysicalChannel("CH6", bb_18, 1e-9, acquire_allowed=True)

    r0 = ch2.build_resonator("R0", frequency=8.68135e9)
    q0 = ch1.build_qubit(
        0,
        r0,
        drive_freq=4.1855e9,
        second_state_freq=4.085e9,
        channel_scale=1,
        measure_amp=1,
        fixed_drive_if=True,
    )

    r1 = ch4.build_resonator("R1", frequency=8.68135e9)
    q1 = ch3.build_qubit(
        1,
        r1,
        drive_freq=3.9204e9,
        second_state_freq=3.7234e9,
        channel_scale=1,
        measure_amp=1,
        fixed_drive_if=True,
    )

    r2 = ch6.build_resonator("R2", frequency=8.68135e9)
    q2 = ch5.build_qubit(
        2,
        r2,
        drive_freq=3.9204e9,
        second_state_freq=3.7234e9,
        channel_scale=1,
        measure_amp=1,
        fixed_drive_if=True,
    )

    r3 = ch6.build_resonator("R3", frequency=8.68135e9)
    q3 = ch5.build_qubit(
        3,
        r3,
        drive_freq=3.9204e9,
        second_state_freq=3.7234e9,
        channel_scale=1,
        measure_amp=1,
        fixed_drive_if=True,
    )

    # TODO - Add support for mapping cross resonance channels to the right QBlox sequencers
    # add_cross_resonance(q0, q1)

    model.add_physical_baseband(bb_12_0, bb_12_1, bb_14, bb_18)
    model.add_physical_channel(ch1, ch2, ch3, ch4, ch5, ch6)
    model.add_quantum_device(r0, q0, r1, q1, r2, q2, r3, q3)
    model.repeat_limit = 1000000
    model.control_hardware = instrument


def run_instructions(runtime, instructions):
    with log_duration("Execution completed, took {} seconds."):
        runtime.initialize_metrics(None)
        runtime.run_quantum_executable(None)
        results, _ = (
            runtime.execute(instructions, None),
            runtime.compilation_metrics,
        )

    return results


class TestLiveQblox:

    instrument = QbloxControlHardware(name="live_cluster_mm")
    model = LiveHardwareModel()
    setup_qblox_hardware_model(model, instrument)
    engine = QbloxLiveEngine(model)
    active_runtime = get_runtime(engine)

    q2 = model.get_qubit(2)  # configured on looped back module 18
    q3 = model.get_qubit(3)  # configured on looped back module 18

    def test_instruction_execution(self):
        amp = 1
        rise = 1.0 / 3.0
        phase = 0.72
        frequency = 500

        drive_channel = self.q2.get_drive_channel()
        builder = (
            get_builder(self.model)
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .pulse(drive_channel, PulseShapeType.GAUSSIAN, width=150e-9, rise=rise)
            .phase_shift(drive_channel, phase)
            .frequency_shift(drive_channel, frequency)
        )

        results = run_instructions(self.active_runtime, builder)
        assert results is not None

    def test_one_channel_timeline(self):
        amp = 1
        rise = 1.0 / 3.0

        drive_channel = self.q2.get_drive_channel()
        builder = (
            get_builder(self.model)
            .pulse(
                drive_channel,
                PulseShapeType.GAUSSIAN,
                width=100e-9,
                rise=rise,
                amp=amp / 2,
            )
            .delay(drive_channel, 100e-9)
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .delay(drive_channel, 100e-9)
        )

        results = run_instructions(self.active_runtime, builder)
        assert results is not None

    def test_two_channels_timeline(self):
        amp = 1
        rise = 1.0 / 3.0

        drive_channel2 = self.q2.get_drive_channel()
        drive_channel3 = self.q3.get_drive_channel()
        builder = (
            get_builder(self.model)
            .pulse(
                drive_channel2,
                PulseShapeType.GAUSSIAN,
                width=100e-9,
                rise=rise,
                amp=amp / 2,
            )
            .pulse(drive_channel3, PulseShapeType.SQUARE, width=50e-9, amp=amp)
            .delay(drive_channel2, 100e-9)
            .delay(drive_channel3, 100e-9)
            .pulse(drive_channel2, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .pulse(
                drive_channel3,
                PulseShapeType.GAUSSIAN,
                width=50e-9,
                rise=rise,
                amp=amp / 2,
            )
        )

        results = run_instructions(self.active_runtime, builder)
        assert results is not None

    def test_sync_two_channel(self):
        amp = 1
        rise = 1.0 / 3.0

        drive_channel2 = self.q2.get_drive_channel()
        drive_channel3 = self.q3.get_drive_channel()
        builder = (
            get_builder(self.model)
            .pulse(
                drive_channel2,
                PulseShapeType.GAUSSIAN,
                width=100e-9,
                rise=rise,
                amp=amp / 2,
            )
            .delay(drive_channel2, 100e-9)
            .pulse(drive_channel3, PulseShapeType.SQUARE, width=50e-9, amp=amp)
            .synchronize([self.q2, self.q3])
            .pulse(drive_channel2, PulseShapeType.SQUARE, width=100e-9, amp=1j * amp)
            .pulse(
                drive_channel3,
                PulseShapeType.GAUSSIAN,
                width=50e-9,
                rise=rise,
                amp=amp / 2j,
            )
        )

        results = run_instructions(self.active_runtime, builder)
        assert results is not None

    def test_play_very_long_pulse(self):
        drive_channel = self.q2.get_drive_channel()
        builder = get_builder(self.model).pulse(
            drive_channel, PulseShapeType.SOFT_SQUARE, amp=0.1, width=1e-5, rise=1e-8
        )

        with pytest.raises(ValueError):
            run_instructions(self.active_runtime, builder)


class VirtualQbloxControlHardware(QbloxControlHardware):
    def set_data(self, seq_files: Dict[PulseChannel, SequenceFile]):
        self.clear()
        for pulse_channel, seq_file in seq_files.items():
            module, sequencer = self.configure_components(pulse_channel)

            # Save module and sequencer as active for this experiment
            self.active_components.setdefault(pulse_channel, []).append(
                (module, sequencer)
            )

            if module.is_qrm_type:
                waveforms = seq_file.waveforms
                dummy_data = reduce(
                    lambda a, b: a + b, [a["data"] for a in waveforms.values()]
                )
                dummy_data = [(z, z) for z in dummy_data / np.linalg.norm(dummy_data)]
                dummy_scope_acquisition_data = DummyScopeAcquisitionData(
                    data=dummy_data, out_of_range=(False, False), avg_cnt=(0, 0)
                )
                module.set_dummy_scope_acquisition_data(
                    sequencer=sequencer.seq_idx, data=dummy_scope_acquisition_data
                )

            sequence = seq_file.as_dict()
            # Temp dump for investigation of the sequence objects
            with open(
                f"sequence_{module.slot_idx}_{sequencer.seq_idx}_@_{datetime.utcnow()}.json",
                "w",
            ) as f:
                f.write(json.dumps(sequence))

            log.info(f"Uploading sequence to {module}, sequencer {sequencer}")
            sequencer.sequence(sequence)

    def start_playback(self, repetitions: int, repetition_time: float):
        if not any(self.active_components):
            raise ValueError("No active components found")

        results = {"waveforms": {}, "acquisitions": {}}
        waveforms = results["waveforms"]
        acquisitions = results["acquisitions"]
        for pulse_channel, components in self.active_components.items():
            physical_channel = pulse_channel.physical_channel
            context = physical_channel.id
            for module, sequencer in components:
                module.arm_sequencer(sequencer.seq_idx)
                module.start_sequencer(sequencer.seq_idx)
                log.info(module.get_sequencer_state(sequencer.seq_idx, timeout=1))

                if module.is_qcm_type:
                    waveforms[context] = module.get_waveforms(sequencer.seq_idx)
                elif module.is_qrm_type:
                    if context in results:
                        raise ValueError(
                            "Needs support for multiple acquisition handling"
                        )
                    acquisitions[context] = self.get_acquisitions(module, sequencer)

        return results


class TestVirtualQblox:

    DUMMY_CONFIG = {
        "12": ClusterType.CLUSTER_QCM_RF,
        "14": ClusterType.CLUSTER_QRM_RF,
        "16": ClusterType.CLUSTER_QRM_RF,
    }

    instrument = VirtualQbloxControlHardware(
        name="dummy_cluster_mm", address=None, cfg=DUMMY_CONFIG
    )
    model = LiveHardwareModel()
    setup_qblox_hardware_model(model, instrument)
    engine = QbloxLiveEngine(model)
    active_runtime = get_runtime(engine)

    def test_instruction_execution(self):
        amp = 1
        rise = 1.0 / 3.0
        phase = 0.72
        frequency = 500

        drive_channel = self.model.get_qubit(0).get_drive_channel()
        builder = (
            get_builder(self.model)
            .pulse(drive_channel, PulseShapeType.SQUARE, width=100e-9, amp=amp)
            .pulse(drive_channel, PulseShapeType.GAUSSIAN, width=150e-9, rise=rise)
            .phase_shift(drive_channel, phase)
            .frequency_shift(drive_channel, frequency)
        )
        results = run_instructions(self.active_runtime, builder)
        assert results is not None
