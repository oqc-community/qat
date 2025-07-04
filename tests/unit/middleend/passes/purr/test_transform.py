# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import math
import random
from collections import defaultdict
from enum import Enum

import numpy as np
import pytest
from compiler_config.config import MetricsType

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager
from qat.ir.conversion import ConvertToPydanticIR
from qat.ir.instructions import Delay as PydDelay
from qat.ir.instructions import PhaseSet as PydPhaseSet
from qat.ir.instructions import PhaseShift as PydPhaseShift
from qat.ir.instructions import Synchronize as PydSynchronize
from qat.ir.measure import AcquireMode, PostProcessType, ProcessAxis
from qat.middleend.passes.purr.analysis import (
    ActiveChannelResults,
)
from qat.middleend.passes.purr.transform import (
    AcquireSanitisation,
    BatchedShots,
    EndOfTaskResetSanitisation,
    EvaluatePulses,
    FreqShiftSanitisation,
    InactivePulseChannelSanitisation,
    InitialPhaseResetSanitisation,
    InstructionGranularitySanitisation,
    InstructionLengthSanitisation,
    LegacyPhaseOptimisation,
    LoopCount,
    LowerSyncsToDelays,
    MeasurePhaseResetSanitisation,
    PhaseOptimisation,
    PostProcessingSanitisation,
    RepeatSanitisation,
    RepeatTranslation,
    ResetsToDelays,
    ReturnSanitisation,
    ScopeSanitisation,
    SquashDelaysOptimisation,
    SynchronizeTask,
)
from qat.middleend.passes.purr.validation import (
    RepeatSanitisationValidation,
    ReturnSanitisationValidation,
)
from qat.middleend.passes.transform import PydPhaseOptimisation
from qat.model.loaders.converted import EchoModelLoader as PydEchoModelLoader
from qat.model.loaders.purr import EchoModelLoader
from qat.model.target_data import (
    AbstractTargetData,
    QubitDescription,
    ResonatorDescription,
    TargetData,
)
from qat.purr.backends.utilities import evaluate_shape
from qat.purr.compiler.builders import InstructionBuilder, QuantumInstructionBuilder
from qat.purr.compiler.devices import ChannelType, FreqShiftPulseChannel
from qat.purr.compiler.hardware_models import QuantumHardwareModel
from qat.purr.compiler.instructions import (
    Acquire,
    Assign,
    CustomPulse,
    Delay,
    EndRepeat,
    GreaterThan,
    Instruction,
    Jump,
    Label,
    LessThan,
    MeasurePulse,
    PhaseReset,
    PhaseSet,
    PhaseShift,
    Plus,
    PostProcessing,
    Pulse,
    PulseShapeType,
    QuantumInstruction,
    Repeat,
    Reset,
    Return,
    Synchronize,
    Variable,
)

from tests.unit.utils.pulses import pulse_attributes


class TestLegacyPhaseOptimisation:
    hw = EchoModelLoader(qubit_count=4).load()

    def test_empty_constructor(self):
        hw = EchoModelLoader(8).load()
        builder = hw.create_builder()

        builder_optimised = LegacyPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        assert len(builder_optimised.instructions) == 0

    @pytest.mark.parametrize("phase", [0.15, 1.0, 3.14])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_single_phase(self, phase, pulse_enabled):
        hw = EchoModelLoader(8).load()
        builder = hw.create_builder()
        for qubit in hw.qubits:
            builder.phase_shift(qubit, phase)
            if pulse_enabled:
                builder.pulse(
                    qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9
                )

        builder_optimised = LegacyPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        phase_shifts = [
            instr
            for instr in builder_optimised.instructions
            if isinstance(instr, PhaseShift)
        ]

        if pulse_enabled:
            assert len(phase_shifts) == len(hw.qubits)
        else:
            assert len(phase_shifts) == 0

    @pytest.mark.parametrize("phase", [0.5, 0.73, 2.75])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_accumulate_phases(self, phase, pulse_enabled):
        hw = EchoModelLoader(8).load()
        builder = hw.create_builder()
        qubits = hw.qubits
        for qubit in qubits:
            builder.phase_shift(qubit, phase)

        random.shuffle(hw.qubits)
        for qubit in qubits:
            builder.phase_shift(qubit, phase + 0.3)
            if pulse_enabled:
                builder.pulse(
                    qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9
                )

        builder_optimised = LegacyPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        phase_shifts = [
            instr
            for instr in builder_optimised.instructions
            if isinstance(instr, PhaseShift)
        ]

        if pulse_enabled:
            assert len(phase_shifts) == len(hw.qubits)
            for phase_shift in phase_shifts:
                assert math.isclose(phase_shift.phase, 2 * phase + 0.3)
        else:
            assert (
                len(phase_shifts) == 0
            )  # Phase shifts without a pulse/reset afterwards are removed.

    def test_phase_reset(self):
        hw = EchoModelLoader(2).load()
        builder = hw.create_builder()
        qubits = list(hw.qubits)

        for qubit in qubits:
            builder.phase_shift(qubit, 0.5)
            builder.reset(qubit)

        builder_optimised = LegacyPhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )

        phase_shifts = [
            instr
            for instr in builder_optimised.instructions
            if isinstance(instr, PhaseShift)
        ]
        assert len(phase_shifts) == 0


class TestPhaseOptimisation:
    hw = EchoModelLoader(qubit_count=4).load()

    def test_merged_identical_phase_resets(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        phase_reset = PhaseReset(self.hw.qubits)
        builder.add(phase_reset)
        builder.add(phase_reset)
        builder.delay(phase_reset.quantum_targets, 1e-3)  # to stop them being deleted
        assert len(builder.instructions) == 3

        PhaseOptimisation().run(builder, res_mgr=ResultManager(), met_mgr=MetricsManager())
        for inst in builder.instructions[:-1]:
            assert isinstance(inst, PhaseSet)
        channels = set([inst.channel for inst in builder.instructions[:-1]])
        assert len(channels) == len(builder.instructions) - 1

    def test_empty_constructor(self):
        hw = EchoModelLoader(8).load()
        builder = hw.create_builder()

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        assert len(builder_optimised.instructions) == 0

    @pytest.mark.parametrize("phase", [-4 * np.pi, -2 * np.pi, 0.0, 2 * np.pi, 4 * np.pi])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_zero_phase(self, phase, pulse_enabled):
        hw = EchoModelLoader(8).load()
        builder = hw.create_builder()
        for qubit in hw.qubits:
            builder.add(PhaseShift(qubit.get_drive_channel(), phase))
            if pulse_enabled:
                builder.pulse(
                    qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9
                )

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )

        if pulse_enabled:
            assert len(builder_optimised.instructions) == len(hw.qubits)
        else:
            assert len(builder_optimised.instructions) == 0

    @pytest.mark.parametrize("phase", [0.15, 1.0, 3.14])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_single_phase(self, phase, pulse_enabled):
        hw = EchoModelLoader(8).load()
        builder = hw.create_builder()
        for qubit in hw.qubits:
            builder.phase_shift(qubit, phase)
            if pulse_enabled:
                builder.pulse(
                    qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9
                )

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        phase_shifts = [
            instr
            for instr in builder_optimised.instructions
            if isinstance(instr, PhaseShift)
        ]

        if pulse_enabled:
            assert len(phase_shifts) == len(hw.qubits)
        else:
            assert len(phase_shifts) == 0

    @pytest.mark.parametrize("phase", [0.5, 0.73, 2.75])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_accumulate_phases(self, phase, pulse_enabled):
        hw = EchoModelLoader(8).load()
        builder = hw.create_builder()
        qubits = hw.qubits
        for qubit in qubits:
            builder.phase_shift(qubit, phase)

        random.shuffle(hw.qubits)
        for qubit in qubits:
            builder.phase_shift(qubit, phase + 0.3)
            if pulse_enabled:
                builder.pulse(
                    qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9
                )

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        phase_shifts = [
            instr
            for instr in builder_optimised.instructions
            if isinstance(instr, PhaseShift)
        ]

        if pulse_enabled:
            assert len(phase_shifts) == len(hw.qubits)
            for phase_shift in phase_shifts:
                assert math.isclose(phase_shift.phase, 2 * phase + 0.3)
        else:
            assert (
                len(phase_shifts) == 0
            )  # Phase shifts without a pulse/reset afterwards are removed.

    def test_phase_reset(self):
        hw = EchoModelLoader(2).load()
        builder = hw.create_builder()
        qubits = list(hw.qubits)

        for qubit in qubits:
            builder.phase_shift(qubit, 0.5)
            builder.reset(qubit)

        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )

        phase_shifts = [
            instr
            for instr in builder_optimised.instructions
            if isinstance(instr, PhaseShift)
        ]
        assert len(phase_shifts) == 0

    def test_reset_and_shift_become_set(self):
        hw = EchoModelLoader(2).load()
        ir = hw.create_builder()
        chan = hw.qubits[0].get_drive_channel()

        ir.add(PhaseReset(chan))
        ir.add(PhaseShift(chan, np.pi / 4))
        ir.pulse(chan, shape=PulseShapeType.SQUARE, width=80e-9)

        ir = PhaseOptimisation().run(ir, ResultManager(), MetricsManager())

        assert len(ir.instructions) == 2
        assert isinstance(ir.instructions[0], PhaseSet)
        assert np.isclose(ir.instructions[0].phase, np.pi / 4)

    def test_phaseset_does_not_commute_through_delay(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        chan = self.hw.qubits[0].get_drive_channel()
        builder.add(PhaseReset(chan))
        builder.delay(chan, 1e-3)
        builder.add(PhaseSet(chan, np.pi / 2))
        builder.delay(chan, 1e-3)
        builder.add(PhaseShift(chan, np.pi / 4))
        builder.delay(chan, 1e-3)

        builder = PhaseOptimisation().run(
            builder, res_mgr=ResultManager(), met_mgr=MetricsManager()
        )

        assert [type(inst) for inst in builder.instructions] == [
            PhaseSet,
            Delay,
            PhaseSet,
            Delay,
            Delay,
        ]

    def test_phaseset_does_not_commute_through_sync(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        chan1 = self.hw.qubits[0].get_drive_channel()
        chan2 = self.hw.qubits[0].get_measure_channel()
        builder.add(PhaseReset(chan1))
        builder.add(PhaseReset(chan2))
        builder.delay(chan1, 1e-3)
        builder.synchronize([chan1, chan2])
        builder.add(PhaseSet(chan1, np.pi / 2))
        builder.add(PhaseSet(chan2, np.pi / 2))
        builder.delay(chan2, 1e-3)
        builder.synchronize([chan1, chan2])
        builder.add(PhaseShift(chan1, np.pi / 4))
        builder.add(PhaseShift(chan2, np.pi / 4))

        builder = PhaseOptimisation().run(
            builder, res_mgr=ResultManager(), met_mgr=MetricsManager()
        )

        assert [type(inst) for inst in builder.instructions] == [
            PhaseSet,
            Delay,
            PhaseSet,
            Synchronize,
            PhaseSet,
            Delay,
            PhaseSet,
            Synchronize,
        ]
        assert [
            inst.quantum_targets[0]
            for inst in builder.instructions
            if isinstance(inst, PhaseSet)
        ] == [chan1, chan2, chan2, chan1]

    def test_partial_ids_are_used(self):
        """Regression test to ensure the partial ids are used as keys."""

        model = EchoModelLoader(2).load()
        model.qubits[0].get_drive_channel().id = "test"
        model.qubits[1].get_drive_channel().id = "test"
        builder = model.create_builder()
        builder.add(PhaseShift(model.qubits[0].get_drive_channel(), np.pi / 2))
        builder.add(PhaseShift(model.qubits[1].get_drive_channel(), -np.pi / 4))
        builder.pulse(
            model.qubits[0].get_drive_channel(),
            shape=PulseShapeType.SQUARE,
            width=80e-9,
        )

        builder = PhaseOptimisation().run(
            builder, res_mgr=ResultManager(), met_mgr=MetricsManager()
        )
        phase_shifts = [
            inst for inst in builder.instructions if isinstance(inst, PhaseShift)
        ]
        assert len(phase_shifts) == 1
        assert phase_shifts[0].channel.id == "test"
        assert phase_shifts[0].phase == np.pi / 4


class TestPhaseOptimisationParityWithPyd:
    hw = EchoModelLoader(qubit_count=4).load()
    pyd_hw = PydEchoModelLoader(4).load()

    def _get_optimized_builders(self, builder):
        pyd_builder = ConvertToPydanticIR(self.hw, self.pyd_hw).run(
            builder, ResultManager()
        )
        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        pyd_builder = PydPhaseOptimisation().run(
            pyd_builder, res_mgr=ResultManager(), met_mgr=MetricsManager()
        )
        return builder_optimised, pyd_builder

    def test_merged_identical_phase_resets(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)
        target = self.hw.get_qubit(0).get_pulse_channel()
        phase_reset = PhaseReset(target)
        builder.add(phase_reset)
        builder.add(phase_reset)
        builder.delay(target, 1e-3)  # to stop them being deleted
        assert len(builder.instructions) == 3

        builder_optimised, pyd_builder = self._get_optimized_builders(builder)

        assert pyd_builder.number_of_instructions == len(builder_optimised.instructions)

    def test_empty_constructor(self):
        builder = self.hw.create_builder()
        builder_optimised, pyd_builder = self._get_optimized_builders(builder)
        assert len(builder_optimised.instructions) == 0
        assert pyd_builder.number_of_instructions == 0

    @pytest.mark.parametrize("phase", [-4 * np.pi, -2 * np.pi, 0.0, 2 * np.pi, 4 * np.pi])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_zero_phase(self, phase, pulse_enabled):
        builder = self.hw.create_builder()
        for qubit in self.hw.qubits:
            builder.add(PhaseShift(qubit.get_drive_channel(), phase))
            if pulse_enabled:
                builder.pulse(
                    qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9
                )
        builder_optimised, pyd_builder = self._get_optimized_builders(builder)

        if pulse_enabled:
            assert len(builder_optimised.instructions) == len(self.hw.qubits)
            assert pyd_builder.number_of_instructions == len(self.hw.qubits)
        else:
            assert len(builder_optimised.instructions) == 0
            assert pyd_builder.number_of_instructions == 0

    @pytest.mark.parametrize("phase", [0.15, 1.0, 3.14])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_single_phase(self, phase, pulse_enabled):
        builder = self.hw.create_builder()
        for qubit in self.hw.qubits:
            builder.phase_shift(qubit, phase)
            if pulse_enabled:
                builder.pulse(
                    qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9
                )
        builder_optimised, pyd_builder = self._get_optimized_builders(builder)

        phase_shifts = [
            instr
            for instr in builder_optimised.instructions
            if isinstance(instr, PhaseShift)
        ]

        pyd_phase_shifts = [
            instr for instr in pyd_builder.instructions if isinstance(instr, PydPhaseShift)
        ]

        if pulse_enabled:
            assert len(phase_shifts) == len(self.hw.qubits)
            assert len(pyd_phase_shifts) == len(phase_shifts)
        else:
            assert len(phase_shifts) == 0
            assert len(pyd_phase_shifts) == 0

    @pytest.mark.parametrize("phase", [0.5, 0.73, 2.75])
    @pytest.mark.parametrize("pulse_enabled", [False, True])
    def test_accumulate_phases(self, phase, pulse_enabled):
        hw = self.hw
        builder = hw.create_builder()
        qubits = hw.qubits
        for qubit in qubits:
            builder.phase_shift(qubit, phase)

        random.shuffle(hw.qubits)
        for qubit in qubits:
            builder.phase_shift(qubit, phase + 0.3)
            if pulse_enabled:
                builder.pulse(
                    qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9
                )

        pyd_builder = ConvertToPydanticIR(self.hw, self.pyd_hw).run(
            builder, ResultManager()
        )
        builder_optimised = PhaseOptimisation().run(
            builder, ResultManager(), MetricsManager()
        )
        pyd_builder = PydPhaseOptimisation().run(
            pyd_builder, res_mgr=ResultManager(), met_mgr=MetricsManager()
        )
        phase_shifts = [
            instr
            for instr in builder_optimised.instructions
            if isinstance(instr, PhaseShift)
        ]
        pyd_phase_shifts = [
            instr for instr in pyd_builder.instructions if isinstance(instr, PydPhaseShift)
        ]

        if pulse_enabled:
            assert len(phase_shifts) == len(hw.qubits)
            assert len(pyd_phase_shifts) == len(phase_shifts)
            for phase_shift in phase_shifts:
                assert math.isclose(phase_shift.phase, 2 * phase + 0.3)
            for pyd_phase_shift in pyd_phase_shifts:
                assert math.isclose(pyd_phase_shift.phase, 2 * phase + 0.3)
        else:
            assert len(phase_shifts) == 0
            assert len(pyd_phase_shifts) == 0
            # Phase shifts without a pulse/reset afterwards are removed.

    def test_phase_reset(self):
        hw = self.hw
        builder = hw.create_builder()
        qubits = list(hw.qubits)

        for qubit in qubits:
            builder.phase_shift(qubit, 0.5)
            builder.reset(qubit)

        builder_optimised, pyd_builder = self._get_optimized_builders(builder)

        phase_shifts = [
            instr
            for instr in builder_optimised.instructions
            if isinstance(instr, PhaseShift)
        ]

        pyd_phase_shifts = [
            instr for instr in pyd_builder.instructions if isinstance(instr, PydPhaseShift)
        ]
        assert len(phase_shifts) == 0
        assert len(pyd_phase_shifts) == 0

    def test_reset_and_shift_become_set(self):
        hw = EchoModelLoader(2).load()
        ir = hw.create_builder()
        chan = hw.qubits[0].get_drive_channel()

        ir.add(PhaseReset(chan))
        ir.add(PhaseShift(chan, np.pi / 4))
        ir.pulse(chan, shape=PulseShapeType.SQUARE, width=80e-9)

        ir, pyd_ir = self._get_optimized_builders(ir)

        assert len(ir.instructions) == 2
        assert pyd_ir.number_of_instructions == 2
        assert isinstance(ir.instructions[0], PhaseSet)
        assert isinstance(pyd_ir.instructions[0], PydPhaseSet)
        assert np.isclose(ir.instructions[0].phase, np.pi / 4)
        assert np.isclose(pyd_ir.instructions[0].phase, np.pi / 4)

    def test_phaseset_does_not_commute_through_delay(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        chan = self.hw.qubits[0].get_drive_channel()
        builder.add(PhaseReset(chan))
        builder.delay(chan, 1e-3)
        builder.add(PhaseSet(chan, np.pi / 2))
        builder.delay(chan, 1e-3)
        builder.add(PhaseShift(chan, np.pi / 4))
        builder.delay(chan, 1e-3)

        builder_optimised, pyd_builder = self._get_optimized_builders(builder)

        assert [type(inst) for inst in builder.instructions] == [
            PhaseSet,
            Delay,
            PhaseSet,
            Delay,
            Delay,
        ]

        assert [type(inst) for inst in pyd_builder.instructions] == [
            PydPhaseSet,
            PydDelay,
            PydPhaseSet,
            PydDelay,
            PydDelay,
        ]

    def test_phaseset_does_not_commute_through_sync(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        chan1 = self.hw.qubits[0].get_drive_channel()
        chan2 = self.hw.qubits[0].get_measure_channel()
        builder.add(PhaseReset(chan1))
        builder.add(PhaseReset(chan2))
        builder.delay(chan1, 1e-3)
        builder.synchronize([chan1, chan2])
        builder.add(PhaseSet(chan1, np.pi / 2))
        builder.add(PhaseSet(chan2, np.pi / 2))
        builder.delay(chan2, 1e-3)
        builder.synchronize([chan1, chan2])
        builder.add(PhaseShift(chan1, np.pi / 4))
        builder.add(PhaseShift(chan2, np.pi / 4))

        builder_optimised, pyd_builder = self._get_optimized_builders(builder)

        assert [type(inst) for inst in builder.instructions] == [
            PhaseSet,
            Delay,
            PhaseSet,
            Synchronize,
            PhaseSet,
            Delay,
            PhaseSet,
            Synchronize,
        ]
        assert [type(inst) for inst in pyd_builder.instructions] == [
            PydPhaseSet,
            PydDelay,
            PydPhaseSet,
            PydSynchronize,
            PydPhaseSet,
            PydDelay,
            PydPhaseSet,
            PydSynchronize,
        ]

        assert [
            inst.quantum_targets[0]
            for inst in builder.instructions
            if isinstance(inst, PhaseSet)
        ] == [chan1, chan2, chan2, chan1]

        target_1 = self.pyd_hw.qubits[0].drive_pulse_channel.uuid
        target_2 = self.pyd_hw.qubits[0].measure_pulse_channel.uuid

        assert [
            inst.target
            for inst in pyd_builder.instructions
            if isinstance(inst, PydPhaseSet)
        ] == [target_1, target_2, target_2, target_1]


class TestPostProcessingSanitisation:
    hw = EchoModelLoader(32).load()

    @pytest.mark.parametrize(
        "axis, pp_type",
        [
            (ProcessAxis.TIME, PostProcessType.DOWN_CONVERT),
            (ProcessAxis.SEQUENCE, PostProcessType.MEAN),
        ],
    )
    def test_valid_meas_acq_with_pp(self, axis, pp_type):
        builder = self.hw.create_builder()
        qubit = self.hw.get_qubit(0)

        _, acq = builder.measure(qubit=qubit, axis=axis)
        builder.post_processing(acq, pp_type)
        n_instr_before = len(builder.instructions)

        met_mgr = MetricsManager()
        PostProcessingSanitisation().run(builder, ResultManager(), met_mgr)
        assert len(builder.instructions) == met_mgr.get_metric(
            MetricsType.OptimizedInstructionCount
        )
        assert len(builder.instructions) == n_instr_before

    @pytest.mark.parametrize(
        "acq_mode,pp_type,pp_axes",
        [
            (AcquireMode.SCOPE, PostProcessType.MEAN, [ProcessAxis.SEQUENCE]),
            (AcquireMode.INTEGRATOR, PostProcessType.DOWN_CONVERT, [ProcessAxis.TIME]),
            (AcquireMode.INTEGRATOR, PostProcessType.MEAN, [ProcessAxis.TIME]),
        ],
    )
    def test_invalid_acq_pp(self, acq_mode, pp_type, pp_axes):
        builder = self.hw.create_builder()
        qubit = self.hw.get_qubit(0)

        _, acq = builder.measure(qubit=qubit, output_variable="test")
        acq.mode = acq_mode
        builder.post_processing(acq, process=pp_type, axes=pp_axes, qubit=qubit)
        n_instr_before = len(builder.instructions)
        assert isinstance(builder.instructions[-1], PostProcessing)

        met_mgr = MetricsManager()
        PostProcessingSanitisation().run(builder, ResultManager(), met_mgr)

        # Test whether invalid PP gets sanitised from the IR.
        assert not isinstance(builder.instructions[-1], PostProcessing)
        assert (
            met_mgr.get_metric(MetricsType.OptimizedInstructionCount) == n_instr_before - 1
        )

    def test_mid_circuit_measurement_two_diff_post_processing(self):
        builder = self.hw.create_builder()
        qubit = self.hw.get_qubit(2)

        # Mid-circuit measurement with some manual (different) post-processing options.
        _, acq1 = builder.measure(qubit=qubit, axis=ProcessAxis.TIME)
        builder.post_processing(
            acq1,
            process=PostProcessType.DOWN_CONVERT,
        )
        builder.X(target=qubit)
        _, acq2 = builder.measure(qubit=qubit, axis=ProcessAxis.SEQUENCE)
        builder.post_processing(
            acq2,
            process=PostProcessType.MEAN,
        )

        PostProcessingSanitisation().run(builder, ResultManager(), MetricsManager())

        # Make sure no instructions get discarded in the post-processing sanitisation for a mid-circuit measurement.
        pp = [instr for instr in builder.instructions if isinstance(instr, PostProcessing)]
        assert len(pp) == 2
        assert pp[0].output_variable != pp[1].output_variable


class TestReturnSanitisation:
    hw = EchoModelLoader(4).load()

    def test_empty_builder(self):
        builder = self.hw.create_builder()
        res_mgr = ResultManager()

        with pytest.raises(ValueError):
            ReturnSanitisationValidation().run(builder, res_mgr)

        ReturnSanitisation().run(builder, res_mgr)
        ReturnSanitisationValidation().run(builder, res_mgr)

        return_instr: Return = builder.instructions[-1]
        assert len(return_instr.variables) == 0

    def test_single_return(self):
        builder = self.hw.create_builder()
        builder.returns(variables=["test"])
        ref_nr_instructions = len(builder.instructions)

        res_mgr = ResultManager()
        ReturnSanitisationValidation().run(builder, res_mgr)
        ReturnSanitisation().run(builder, res_mgr)

        assert len(builder.instructions) == ref_nr_instructions
        assert builder.instructions[0].variables == ["test"]

    def test_multiple_returns_squashed(self):
        q0 = self.hw.get_qubit(0)
        q1 = self.hw.get_qubit(1)

        builder = self.hw.create_builder()
        builder.measure_single_shot_z(target=q0, output_variable="out_q0")
        builder.measure_single_shot_z(target=q1, output_variable="out_q1")

        output_vars = [
            instr.output_variable
            for instr in builder.instructions
            if isinstance(instr, Acquire)
        ]
        assert len(output_vars) == 2

        builder.returns(variables=[output_vars[0]])
        builder.returns(variables=[output_vars[1]])

        res_mgr = ResultManager()
        # Two returns in a single IR should raise an error.
        with pytest.raises(ValueError):
            ReturnSanitisationValidation().run(builder, res_mgr)

        # Compress the two returns to a single return and validate.
        ReturnSanitisation().run(builder, res_mgr)
        ReturnSanitisationValidation().run(builder, res_mgr)

        return_instr = builder.instructions[-1]
        assert isinstance(return_instr, Return)
        for var in return_instr.variables:
            assert var in output_vars


class TestAcquireSanitisation:
    def test_acquire_with_no_delay_is_unchanged(self):
        # Mock up some channels and a builder
        model = EchoModelLoader(10).load()
        acquire_chan = model.qubits[0].get_acquire_channel()
        acquire_block_time = acquire_chan.physical_channel.block_time
        builder = model.create_builder()

        # Make some instructions to test
        builder.acquire(acquire_chan, time=acquire_block_time * 10, delay=0.0)

        builder == AcquireSanitisation().run(builder)
        assert len(builder.instructions) == 1
        assert isinstance(builder.instructions[0], Acquire)

    def test_acquire_with_delay_is_decomposed(self):
        # Mock up some channels and a builder
        model = EchoModelLoader(10).load()
        acquire_chan = model.qubits[0].get_acquire_channel()
        acquire_block_time = acquire_chan.physical_channel.block_time
        builder = model.create_builder()

        # Make some instructions to test
        builder.acquire(
            acquire_chan, time=acquire_block_time * 10, delay=acquire_block_time
        )

        builder == AcquireSanitisation().run(builder)
        assert len(builder.instructions) == 2
        assert isinstance(builder.instructions[0], Delay)
        assert isinstance(builder.instructions[1], Acquire)

    def test_acquire_with_delay_two_chans_is_decomposed(self):
        model = EchoModelLoader(10).load()
        builder = model.create_builder()
        for qubit in (0, 1):
            acquire_chan = model.qubits[qubit].get_acquire_channel()
            acquire_block_time = acquire_chan.physical_channel.block_time

            # Make some instructions to test
            builder.acquire(
                acquire_chan, time=acquire_block_time * 10, delay=acquire_block_time
            )

        builder == AcquireSanitisation().run(builder)
        assert len(builder.instructions) == 4
        assert isinstance(builder.instructions[0], Delay)
        assert isinstance(builder.instructions[1], Acquire)
        assert isinstance(builder.instructions[2], Delay)
        assert isinstance(builder.instructions[3], Acquire)


class TestInstructionGranularitySanitisation:
    def test_instructions_with_correct_timings_are_unchanged(self):
        # Mock up some channels and a builder
        model = EchoModelLoader(10).load()
        target_data = TargetData(
            max_shots=1000,
            default_shots=10,
            QUBIT_DATA=QubitDescription.random(),
            RESONATOR_DATA=ResonatorDescription.random(),
        )

        drive_chan = model.qubits[0].get_drive_channel()
        acquire_chan = model.qubits[0].get_acquire_channel()
        builder = model.create_builder()

        # Make some instructions to test
        clock_cycle = target_data.clock_cycle
        delay_time = np.random.randint(1, 100) * clock_cycle
        builder.delay(drive_chan, delay_time)
        pulse_time = np.random.randint(1, 100) * clock_cycle
        builder.pulse(
            quantum_target=drive_chan, shape=PulseShapeType.SQUARE, width=pulse_time
        )
        acquire_time = np.random.randint(1, 100) * clock_cycle
        builder.acquire(acquire_chan, time=acquire_time, delay=0.0)

        ir = InstructionGranularitySanitisation(model, target_data).run(builder)

        # compare in units of ns to ensure np.isclose works fine
        assert np.isclose(ir.instructions[0].duration * 1e9, delay_time * 1e9)
        assert np.isclose(ir.instructions[1].duration * 1e9, pulse_time * 1e9)
        assert np.isclose(ir.instructions[2].duration * 1e9, acquire_time * 1e9)

    def test_instructions_are_rounded_down(self):
        # Mock up some channels and a builder
        model = EchoModelLoader(10).load()
        target_data = TargetData(
            max_shots=1000,
            default_shots=10,
            QUBIT_DATA=QubitDescription.random(),
            RESONATOR_DATA=ResonatorDescription.random(),
        )

        drive_chan = model.qubits[0].get_drive_channel()
        acquire_chan = model.qubits[0].get_acquire_channel()
        builder = model.create_builder()

        # Make some instructions to test
        clock_cycle = target_data.clock_cycle
        delay_time = np.random.randint(1, 100) * clock_cycle
        builder.delay(drive_chan, delay_time + np.random.rand() * clock_cycle)
        pulse_time = np.random.randint(1, 100) * clock_cycle
        builder.pulse(
            quantum_target=drive_chan,
            shape=PulseShapeType.SQUARE,
            width=pulse_time + np.random.rand() * clock_cycle,
        )
        acquire_time = np.random.randint(1, 100) * clock_cycle
        builder.acquire(
            acquire_chan,
            time=acquire_time + np.random.rand() * clock_cycle,
            delay=0.0,
        )

        ir = InstructionGranularitySanitisation(model, target_data).run(builder)

        # compare in units of ns to ensure np.isclose works fine
        assert np.isclose(ir.instructions[0].duration * 1e9, delay_time * 1e9)
        assert np.isclose(ir.instructions[1].duration * 1e9, pulse_time * 1e9)
        assert np.isclose(ir.instructions[2].duration * 1e9, acquire_time * 1e9)

    def test_custom_pulses_with_correct_length_are_unchanged(self):
        # Mock up some channels and a builder
        model = EchoModelLoader(10).load()
        target_data = TargetData(
            max_shots=1000,
            default_shots=10,
            QUBIT_DATA=QubitDescription.random(),
            RESONATOR_DATA=ResonatorDescription.random(),
        )

        drive_chan = model.qubits[0].get_drive_channel()
        sample_time = drive_chan.physical_channel.sample_time
        builder = model.create_builder()

        # Make some instructions to test
        clock_cycle = target_data.clock_cycle
        supersampling = int(np.round(clock_cycle / sample_time, 0))
        num_samples = np.random.randint(1, 100) * supersampling
        samples = [1.0] * num_samples
        builder.add(CustomPulse(drive_chan, samples))

        ir = InstructionGranularitySanitisation(model, target_data).run(builder)
        assert ir.instructions[0].samples == samples

    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_custom_pulses_with_invalid_length_are_padded(self, seed):
        # Mock up some channels and a builder
        model = EchoModelLoader(10).load()
        target_data = TargetData(
            max_shots=1000,
            default_shots=10,
            QUBIT_DATA=QubitDescription.random(seed),
            RESONATOR_DATA=ResonatorDescription.random(seed),
        )

        drive_chan = model.qubits[0].get_drive_channel()
        sample_time = target_data.QUBIT_DATA.sample_time
        builder = model.create_builder()

        # Make some instructions to test
        clock_cycle = target_data.QUBIT_DATA.clock_cycle
        supersampling = int(np.round(clock_cycle / sample_time, 0))
        num_samples = np.random.randint(1, 100) * supersampling

        samples = [1.0] * (num_samples + np.random.randint(1, supersampling - 1))
        builder.add(CustomPulse(drive_chan, samples))
        assert len(builder.instructions[0].samples) == len(samples)

        ir = InstructionGranularitySanitisation(model, target_data).run(builder)
        n = num_samples + supersampling

        assert len(ir.instructions[0].samples) == n


class TestPhaseResetSanitisation:
    hw = EchoModelLoader(qubit_count=4).load()

    @pytest.mark.parametrize("reset_qubits", [False, True])
    def test_phase_reset_shot(self, reset_qubits):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        if reset_qubits:
            builder.add(PhaseReset(self.hw.qubits))

        active_targets = {}
        for qubit in self.hw.qubits:
            builder.X(target=qubit)
            drive_pulse_ch = qubit.get_drive_channel()
            active_targets[drive_pulse_ch] = qubit

        n_instr_before = len(builder.instructions)

        res_mgr = ResultManager()
        # Mock some active targets, i.e., the drive pulse channels of the qubits.
        res_mgr.add(ActiveChannelResults(target_map=active_targets))
        InitialPhaseResetSanitisation().run(builder, res_mgr=res_mgr)

        # One `PhaseReset` instruction with possibly multiple targets gets added to the IR,
        # even if there is already a phase reset present.
        assert len(builder.instructions) == n_instr_before + 1
        assert isinstance(builder.instructions[0], PhaseReset)
        assert builder.instructions[0].quantum_targets == list(active_targets.keys())

    def test_phase_reset_shot_no_active_pulse_channels(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        for qubit in self.hw.qubits:
            builder.X(target=qubit)

        n_instr_before = len(builder.instructions)

        res_mgr = ResultManager()
        # Mock some active targets, i.e., the drive pulse channels of the qubits.
        res_mgr.add(ActiveChannelResults())
        InitialPhaseResetSanitisation().run(builder, res_mgr=res_mgr)

        # No phase reset added since there are no active targets.
        assert len(builder.instructions) == n_instr_before
        assert not isinstance(builder.instructions[0], PhaseReset)


class TestMeasurePhaseResetSanitisation:
    hw = EchoModelLoader(qubit_count=4).load()

    def test_measure_phase_reset(self):
        builder = QuantumInstructionBuilder(hardware_model=self.hw)

        for qubit in self.hw.qubits:
            builder.X(target=qubit)
            builder.measure(qubit)

        n_instr_before = len(builder.instructions)

        ir = MeasurePhaseResetSanitisation().run(builder)

        # A phase reset should be added for each measure instruction.
        assert len(ir.instructions) == n_instr_before + len(self.hw.qubits)

        ref_measure_pulse_channels = set(
            [qubit.get_measure_channel() for qubit in self.hw.qubits]
        )
        measure_pulse_channels = set()
        for i, instr in enumerate(ir.instructions):
            if isinstance(instr, MeasurePulse):
                assert isinstance(ir.instructions[i - 1], PhaseReset)
                measure_pulse_channels.update(ir.instructions[i - 1].quantum_targets)

        assert measure_pulse_channels == ref_measure_pulse_channels


class TestInactivePulseChannelSanitisation:
    def test_syncs_are_sanitized(self):
        model = EchoModelLoader(10).load()
        builder = model.create_builder()
        builder.X(model.qubits[0])
        builder.synchronize(model.qubits[0])
        sync_inst = [
            inst for inst in builder.instructions if isinstance(inst, Synchronize)
        ][0]
        assert len(sync_inst.quantum_targets) > 1

        res = ActiveChannelResults(
            target_map={model.qubits[0].get_drive_channel(): model.qubits[0]}
        )
        res_mgr = ResultManager()
        res_mgr.add(res)
        builder = InactivePulseChannelSanitisation().run(builder, res_mgr)
        sync_inst = [
            inst for inst in builder.instructions if isinstance(inst, Synchronize)
        ][0]
        assert len(sync_inst.quantum_targets) == 1
        assert next(iter(sync_inst.quantum_targets)) == model.qubits[0].get_drive_channel()

    def test_delays_are_sanitized(self):
        model = EchoModelLoader(10).load()
        builder = model.create_builder()
        qubit = model.qubits[0]

        builder.pulse(qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9)
        builder.delay(qubit.get_measure_channel(), 80e-9)

        res = ActiveChannelResults(
            target_map={model.qubits[0].get_drive_channel(): model.qubits[0]}
        )
        res_mgr = ResultManager()
        res_mgr.add(res)
        builder = InactivePulseChannelSanitisation().run(builder, res_mgr)
        assert len(builder.instructions) == 1
        assert isinstance(builder.instructions[0], Pulse)

    def test_phase_shifts_are_sanitized(self):
        model = EchoModelLoader(10).load()
        builder = model.create_builder()
        qubit = model.qubits[0]

        builder.pulse(qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9)
        builder.phase_shift(qubit.get_measure_channel(), np.pi)

        res = ActiveChannelResults(
            target_map={model.qubits[0].get_drive_channel(): model.qubits[0]}
        )
        res_mgr = ResultManager()
        res_mgr.add(res)
        builder = InactivePulseChannelSanitisation().run(builder, res_mgr)
        assert len(builder.instructions) == 1
        assert isinstance(builder.instructions[0], Pulse)

    def test_Z_is_sanitized(self):
        model = EchoModelLoader(10).load()
        builder = model.create_builder()
        qubit = model.qubits[0]

        builder.pulse(qubit.get_drive_channel(), shape=PulseShapeType.SQUARE, width=80e-9)
        builder.Z(qubit, np.pi)

        num_phase_shifts_before = len(
            [inst for inst in builder.instructions if isinstance(inst, PhaseShift)]
        )

        res = ActiveChannelResults(
            target_map={model.qubits[0].get_drive_channel(): model.qubits[0]}
        )
        res_mgr = ResultManager()
        res_mgr.add(res)
        builder = InactivePulseChannelSanitisation().run(builder, res_mgr)

        num_phase_shifts_after = len(
            [inst for inst in builder.instructions if isinstance(inst, PhaseShift)]
        )

        assert num_phase_shifts_after < num_phase_shifts_before
        assert num_phase_shifts_after == 1


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
class TestInstructionLengthSanitisation:
    hw = EchoModelLoader(8).load()

    def test_delay_not_sanitised(self, seed):
        builder = QuantumInstructionBuilder(self.hw)
        target_data = TargetData(
            max_shots=1000,
            default_shots=10,
            QUBIT_DATA=QubitDescription.random(seed),
            RESONATOR_DATA=ResonatorDescription.random(seed),
        )
        pulse_ch = self.hw.qubits[0].get_acquire_channel()

        valid_duration = target_data.QUBIT_DATA.pulse_duration_max / 2
        builder.add(Delay(pulse_ch, valid_duration))
        assert len(builder.instructions) == 1

        InstructionLengthSanitisation(target_data).run(builder)
        assert len(builder.instructions) == 1
        assert builder.instructions[0].duration == valid_duration

    def test_delay_sanitised_zero_remainder(self, seed):
        builder = QuantumInstructionBuilder(self.hw)
        target_data = TargetData(
            max_shots=1000,
            default_shots=10,
            QUBIT_DATA=QubitDescription.random(seed),
            RESONATOR_DATA=ResonatorDescription.random(seed),
        )
        pulse_ch = self.hw.qubits[0].get_acquire_channel()

        invalid_duration = target_data.QUBIT_DATA.pulse_duration_max * 2
        builder.add(Delay(pulse_ch, invalid_duration))
        assert len(builder.instructions) == 1

        InstructionLengthSanitisation(target_data).run(builder)
        assert len(builder.instructions) == 2
        for instr in builder.instructions:
            assert instr.duration == target_data.QUBIT_DATA.pulse_duration_max

    def test_delay_sanitised(self, seed):
        builder = QuantumInstructionBuilder(self.hw)
        target_data = TargetData(
            max_shots=1000,
            default_shots=10,
            QUBIT_DATA=QubitDescription.random(seed),
            RESONATOR_DATA=ResonatorDescription.random(seed),
        )
        pulse_ch = self.hw.qubits[0].get_acquire_channel()

        remainder = random.uniform(1e-06, 2e-06)
        invalid_duration = target_data.QUBIT_DATA.pulse_duration_max * 2 + remainder
        builder.add(Delay(pulse_ch, invalid_duration))
        assert len(builder.instructions) == 1

        InstructionLengthSanitisation(target_data).run(builder)
        assert len(builder.instructions) == 3
        for instr in builder.instructions[:-1]:
            assert instr.duration == target_data.QUBIT_DATA.pulse_duration_max
        assert math.isclose(builder.instructions[-1].duration, remainder)

    def test_square_pulse_not_sanitised(self, seed):
        builder = QuantumInstructionBuilder(self.hw)
        target_data = TargetData(
            max_shots=1000,
            default_shots=10,
            QUBIT_DATA=QubitDescription.random(seed),
            RESONATOR_DATA=ResonatorDescription.random(seed),
        )
        pulse_ch = self.hw.qubits[0].get_acquire_channel()

        valid_width = target_data.QUBIT_DATA.pulse_duration_max / 2
        builder.add(Pulse(pulse_ch, shape=PulseShapeType.SQUARE, width=valid_width))
        assert len(builder.instructions) == 1

        InstructionLengthSanitisation(target_data).run(builder)
        assert len(builder.instructions) == 1
        assert builder.instructions[0].width == valid_width

    def test_square_pulse_sanitised(self, seed):
        builder = QuantumInstructionBuilder(self.hw)
        target_data = TargetData(
            max_shots=1000,
            default_shots=10,
            QUBIT_DATA=QubitDescription.random(seed),
            RESONATOR_DATA=ResonatorDescription.random(seed),
        )
        pulse_ch = self.hw.qubits[0].get_acquire_channel()

        remainder = random.uniform(1e-06, 2e-06)
        invalid_width = target_data.QUBIT_DATA.pulse_duration_max * 2 + remainder
        builder.add(Pulse(pulse_ch, shape=PulseShapeType.SQUARE, width=invalid_width))
        assert len(builder.instructions) == 1

        InstructionLengthSanitisation(target_data).run(builder)
        assert len(builder.instructions) == 3

        for instr in builder.instructions[:-1]:
            assert instr.width == target_data.QUBIT_DATA.pulse_duration_max
        assert math.isclose(builder.instructions[-1].width, remainder)


class TestSynchronizeTask:
    def test_synchronize_task_adds_sync(self):
        model = EchoModelLoader().load()
        qubit = model.qubits[0]
        drive_chan = qubit.get_drive_channel()
        measure_chan = qubit.get_measure_channel()

        res_mgr = ResultManager()
        res_mgr.add(
            ActiveChannelResults(target_map={drive_chan: qubit, measure_chan: qubit})
        )

        builder = model.create_builder()
        builder.pulse(drive_chan, shape=PulseShapeType.SQUARE, width=800e-9)
        builder.pulse(measure_chan, shape=PulseShapeType.SQUARE, width=400e-9)

        ir = SynchronizeTask().run(builder, res_mgr)
        assert isinstance(ir.instructions[-1], Synchronize)
        assert set(ir.instructions[-1].quantum_targets) == set([drive_chan, measure_chan])

    def test_synchronize_task_not_adding_if_inactive(self):
        model = EchoModelLoader().load()

        res_mgr = ResultManager()
        res_mgr.add(ActiveChannelResults(target_map={}))

        builder = model.create_builder()

        ir = SynchronizeTask().run(builder, res_mgr)
        assert len(ir.instructions) == 0


class TestEndOfTaskResetSanitisation:
    @pytest.mark.parametrize("reset_q1", [False, True])
    @pytest.mark.parametrize("reset_q2", [False, True])
    def test_resets_added(self, reset_q1, reset_q2):
        model = EchoModelLoader().load()
        qubits = model.qubits[0:2]

        builder = model.create_builder()
        builder.had(qubits[0])
        builder.cnot(qubits[0], qubits[1])
        builder.measure(qubits[0])
        builder.measure(qubits[1])
        if reset_q1:
            builder.reset(qubits[0])
        if reset_q2:
            builder.reset(qubits[1])

        res = ActiveChannelResults()
        for qubit in qubits:
            res.target_map[qubit.get_drive_channel()] = qubit
            res.target_map[qubit.get_measure_channel()] = qubit
            res.target_map[qubit.get_acquire_channel()] = qubit
        res.target_map[qubits[0].get_cross_resonance_channel(qubits[1])] = qubits[0]
        res.target_map[qubits[1].get_cross_resonance_cancellation_channel(qubits[0])] = (
            qubits[1]
        )

        res_mgr = ResultManager()
        res_mgr.add(res)

        before = len(builder.instructions)
        builder = EndOfTaskResetSanitisation().run(builder, res_mgr)
        assert before + (not reset_q1) + (not reset_q2) == len(builder.instructions)

        reset_channels = [
            inst.quantum_targets[0]
            for inst in builder.instructions
            if isinstance(inst, Reset)
        ]
        assert len(reset_channels) == 2
        assert set(reset_channels) == set([qubit.get_drive_channel() for qubit in qubits])

    def test_mid_circuit_reset_is_ignored(self):
        model = EchoModelLoader().load()
        qubit = model.qubits[0]

        builder = model.create_builder()
        builder.had(qubit)
        builder.reset(qubit)
        builder.had(qubit)

        res = ActiveChannelResults()
        res.target_map[qubit.get_drive_channel()] = qubit

        res_mgr = ResultManager()
        res_mgr.add(res)

        before = len(builder.instructions)
        builder = EndOfTaskResetSanitisation().run(builder, res_mgr)
        assert before + 1 == len(builder.instructions)

        reset_channels = [
            inst.quantum_targets[0]
            for inst in builder.instructions
            if isinstance(inst, Reset)
        ]
        assert len(reset_channels) == 2
        assert set(reset_channels) == set([qubit.get_drive_channel()])

    def test_inactive_instruction_after_reset_ignored(self):
        model = EchoModelLoader().load()
        qubit = model.qubits[0]

        builder = model.create_builder()
        builder.had(qubit)
        builder.reset(qubit)
        builder.phase_shift(qubit.get_drive_channel(), np.pi)

        res = ActiveChannelResults()
        res.target_map[qubit.get_drive_channel()] = qubit

        res_mgr = ResultManager()
        res_mgr.add(res)

        before = len(builder.instructions)
        builder = EndOfTaskResetSanitisation().run(builder, res_mgr)
        assert before == len(builder.instructions)

        reset_channels = [
            inst.quantum_targets[0]
            for inst in builder.instructions
            if isinstance(inst, Reset)
        ]
        assert len(reset_channels) == 1
        assert reset_channels[0] == qubit.get_drive_channel()

    def test_reset_with_no_drive_channel(self):
        model = EchoModelLoader().load()
        qubit = model.qubits[0]

        builder = model.create_builder()
        builder.pulse(qubit.get_measure_channel(), shape=PulseShapeType.SQUARE, width=80e-9)

        res = ActiveChannelResults()
        res.target_map[qubit.get_measure_channel()] = qubit

        res_mgr = ResultManager()
        res_mgr.add(res)

        builder = EndOfTaskResetSanitisation().run(builder, res_mgr)
        reset_instrs = [instr for instr in builder.instructions if isinstance(instr, Reset)]
        assert len(reset_instrs) == 1
        assert reset_instrs[0].quantum_targets[0] == qubit.get_measure_channel()


@pytest.mark.parametrize("passive_reset_time", [3.2e-06, 1e-03, 5.0])
class TestResetsToDelays:
    @pytest.mark.parametrize("add_reset", [True, False])
    def test_qubit_reset(self, passive_reset_time: float, add_reset: bool):
        model = EchoModelLoader().load()
        qubit_data = QubitDescription.random().model_copy(
            update={"passive_reset_time": passive_reset_time}
        )
        target_data = TargetData(
            max_shots=1000,
            default_shots=100,
            RESONATOR_DATA=ResonatorDescription.random(),
            QUBIT_DATA=qubit_data,
        )

        qubit = model.qubits[0]

        builder = model.create_builder()
        builder.had(qubit)
        if add_reset:
            builder.reset(qubit)

        res = ActiveChannelResults()
        res.target_map[qubit.get_drive_channel()] = qubit
        res_mgr = ResultManager()
        res_mgr.add(res)

        before = len(builder.instructions)
        builder = ResetsToDelays(target_data).run(builder, res_mgr)
        assert before == len(builder.instructions)

        delays = []
        for instr in builder.instructions:
            assert not isinstance(instr, Reset)

            if isinstance(instr, Delay):
                delays.append(instr)

        if add_reset:
            assert len(delays) == 1
            assert len(delays[0].quantum_targets) == 1
            assert delays[0].time == passive_reset_time

    def test_pulse_channel_reset(self, passive_reset_time: float):
        model = EchoModelLoader().load()
        qubit_data = QubitDescription.random().model_copy(
            update={"passive_reset_time": passive_reset_time}
        )
        target_data = TargetData(
            max_shots=1000,
            default_shots=100,
            RESONATOR_DATA=ResonatorDescription.random(),
            QUBIT_DATA=qubit_data,
        )

        qubit = model.qubits[0]
        pulse_channels = qubit.get_all_channels()

        builder = model.create_builder()
        builder.reset(pulse_channels)

        res = ActiveChannelResults()
        for pulse_ch in pulse_channels:
            res.target_map[pulse_ch] = qubit
        res_mgr = ResultManager()
        res_mgr.add(res)

        builder = ResetsToDelays(target_data).run(builder, res_mgr)

        delays = []
        for instr in builder.instructions:
            assert not isinstance(instr, Reset)

            if isinstance(instr, Delay):
                delays.append(instr)

        # All pulse channels belong to a single qubit so only 1 reset -> delay.
        assert len(delays) == 1
        # All pulse channels in the qubit are active channels.
        assert len(delays[0].quantum_targets) == len(pulse_channels)
        assert delays[0].time == passive_reset_time

    @pytest.mark.parametrize("reset_chan", ["acquire", "measure"])
    def test_reset_with_no_drive_channel(self, passive_reset_time, reset_chan):
        model = EchoModelLoader().load()
        qubit_data = QubitDescription.random().model_copy(
            update={"passive_reset_time": passive_reset_time}
        )
        target_data = TargetData(
            max_shots=1000,
            default_shots=100,
            RESONATOR_DATA=ResonatorDescription.random(),
            QUBIT_DATA=qubit_data,
        )

        qubit = model.qubits[0]

        if reset_chan == "measure":
            reset_chan = qubit.get_measure_channel()
        else:
            reset_chan = qubit.get_acquire_channel()

        builder = model.create_builder()
        builder.pulse(qubit.get_measure_channel(), shape=PulseShapeType.SQUARE, width=80e-9)
        builder.acquire(qubit.get_acquire_channel(), time=80e-9, delay=0.0)
        builder.add(Reset(reset_chan))

        res = ActiveChannelResults()
        res.target_map[qubit.get_measure_channel()] = qubit
        res.target_map[qubit.get_acquire_channel()] = qubit

        res_mgr = ResultManager()
        res_mgr.add(res)

        builder = ResetsToDelays(target_data).run(builder, res_mgr)
        reset_instrs = [instr for instr in builder.instructions if isinstance(instr, Reset)]
        assert len(reset_instrs) == 0
        delay_instrs = [instr for instr in builder.instructions if isinstance(instr, Delay)]
        assert len(delay_instrs) == 1
        assert set(delay_instrs[0].quantum_targets) == set(
            [qubit.get_acquire_channel(), qubit.get_measure_channel()]
        )


class MockPulseShapeType(Enum):
    HEXAGON = "hexagon"
    SPAGHETTI = "spaghetti"


def mock_evaluate(data, t, phase_offset):
    if isinstance(data, Pulse) and data.shape in MockPulseShapeType:
        if data.shape == MockPulseShapeType.HEXAGON:
            return np.linspace(0, 1, len(t))
        elif data.shape == MockPulseShapeType.SPAGHETTI:
            return np.linspace(-1, 0, len(t))
    else:
        return evaluate_shape(data, t, phase_offset)


class TestEvaluatePulses:
    @pytest.mark.parametrize("scale", [True, False])
    def test_pulse_not_lowered(self, scale: bool):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        chan.scale = 0.05
        builder = model.create_builder()
        builder.pulse(
            chan,
            shape=PulseShapeType.SQUARE,
            width=400e-9,
            amp=1.0,
            ignore_channel_scale=scale,
        )

        EvaluatePulses().run(builder)
        assert len(builder.instructions) == 1
        pulse = builder.instructions[0]
        assert isinstance(pulse, Pulse)
        assert pulse.channel == chan
        assert pulse.ignore_channel_scale
        assert pulse.amp == 1.0 * (1.0 if scale else 0.05)

    @pytest.mark.parametrize("scale", [True, False])
    @pytest.mark.parametrize("attributes", pulse_attributes)
    def test_non_square_pulses_are_lowered(self, scale: bool, attributes: dict[str]):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        chan.scale = 0.05
        builder = model.create_builder()
        builder.pulse(chan, width=400e-9, amp=1.0, ignore_channel_scale=scale, **attributes)

        EvaluatePulses().run(builder)
        assert len(builder.instructions) == 1
        pulse = builder.instructions[0]
        assert isinstance(pulse, CustomPulse)
        assert pulse.ignore_channel_scale
        assert len(pulse.samples) == int(np.ceil(400e-9 / chan.sample_time - 1e-10))
        assert pulse.channel == chan
        # the pulse is never sampled at the peak, so amp is actually < 1.0
        assert np.max(np.abs(builder.instructions[0].samples)) >= 0.99 * (
            1.0 if scale else 0.05
        )

    @pytest.mark.parametrize("scale", [True, False])
    def test_custom_pulse(self, scale):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        chan.scale = 0.05
        builder = model.create_builder()
        builder.add(CustomPulse(chan, np.ones(80), ignore_channel_scale=scale))
        EvaluatePulses().run(builder)
        assert len(builder.instructions) == 1
        pulse = builder.instructions[0]
        assert isinstance(pulse, CustomPulse)
        assert pulse.ignore_channel_scale
        assert len(pulse.samples) == 80
        assert np.allclose(pulse.samples, 1.0 if scale else 0.05)

    @pytest.mark.parametrize("scale", [True, False])
    def test_acquire_is_lowered(self, scale):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_acquire_channel()
        chan.scale = 0.05
        builder = model.create_builder()
        pulse = Pulse(
            chan,
            shape=PulseShapeType.SQUARE,
            width=400e-9,
            amp=1.0,
            ignore_channel_scale=scale,
        )
        builder.acquire(chan, time=400e-9, filter=pulse, delay=0.0)

        EvaluatePulses().run(builder)
        assert len(builder.instructions) == 1
        acquire = builder.instructions[0]
        assert isinstance(acquire, Acquire)
        pulse = acquire.filter
        assert isinstance(pulse, CustomPulse)
        assert pulse.ignore_channel_scale
        assert len(pulse.samples) == int(np.ceil(400e-9 / chan.sample_time - 1e-10))
        assert pulse.channel == chan
        assert np.allclose(pulse.samples, (1.0 if scale else 0.05))

    @pytest.mark.parametrize("scale", [True, False])
    def test_acquire_is_not_lowered(self, scale):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_acquire_channel()
        chan.scale = 0.05
        builder = model.create_builder()
        pulse = Pulse(
            chan,
            shape=PulseShapeType.SQUARE,
            width=400e-9,
            amp=1.0,
            ignore_channel_scale=scale,
        )
        builder.acquire(chan, time=400e-9, filter=pulse, delay=0.0)

        EvaluatePulses(acquire_ignored_shapes=[PulseShapeType.SQUARE]).run(builder)
        assert len(builder.instructions) == 1
        acquire = builder.instructions[0]
        assert isinstance(acquire, Acquire)
        pulse = acquire.filter
        assert isinstance(pulse, Pulse)
        assert pulse.ignore_channel_scale
        assert pulse.shape == PulseShapeType.SQUARE

    def test_evaluate_injection_with_square(self):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        builder = model.create_builder()
        builder.pulse(chan, shape=PulseShapeType.SQUARE, width=400e-9, amp=1.0)
        builder.pulse(
            model.qubits[0].get_drive_channel(),
            shape=MockPulseShapeType.HEXAGON,
            width=400e-9,
            amp=1.0,
        )
        builder.pulse(
            model.qubits[0].get_drive_channel(),
            shape=MockPulseShapeType.SPAGHETTI,
            width=400e-9,
            amp=1.0,
        )

        EvaluatePulses(eval_function=mock_evaluate).run(builder)
        assert len(builder.instructions) == 3
        assert isinstance(builder.instructions[0], Pulse)
        assert builder.instructions[0].shape == PulseShapeType.SQUARE
        assert isinstance(builder.instructions[1], CustomPulse)
        assert isinstance(builder.instructions[2], CustomPulse)

    def test_evaluate_injection_with_custom_ignore(self):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        builder = model.create_builder()
        builder.pulse(chan, shape=PulseShapeType.SQUARE, width=400e-9, amp=1.0)
        builder.pulse(
            model.qubits[0].get_drive_channel(),
            shape=MockPulseShapeType.HEXAGON,
            width=400e-9,
            amp=1.0,
        )
        builder.pulse(
            model.qubits[0].get_drive_channel(),
            shape=MockPulseShapeType.SPAGHETTI,
            width=400e-9,
            amp=1.0,
        )

        EvaluatePulses(
            ignored_shapes=[MockPulseShapeType.HEXAGON], eval_function=mock_evaluate
        ).run(builder)
        assert len(builder.instructions) == 3
        assert isinstance(builder.instructions[0], CustomPulse)
        assert isinstance(builder.instructions[1], Pulse)
        assert builder.instructions[1].shape == MockPulseShapeType.HEXAGON
        assert isinstance(builder.instructions[2], CustomPulse)

    def test_evaluate_injection_with_two_custom_ignores(self):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        builder = model.create_builder()
        builder.pulse(chan, shape=PulseShapeType.SQUARE, width=400e-9, amp=1.0)
        builder.pulse(
            model.qubits[0].get_drive_channel(),
            shape=MockPulseShapeType.HEXAGON,
            width=400e-9,
            amp=1.0,
        )
        builder.pulse(
            model.qubits[0].get_drive_channel(),
            shape=MockPulseShapeType.SPAGHETTI,
            width=400e-9,
            amp=1.0,
        )

        EvaluatePulses(
            ignored_shapes=[PulseShapeType.SQUARE, MockPulseShapeType.HEXAGON],
            eval_function=mock_evaluate,
        ).run(builder)
        assert len(builder.instructions) == 3
        assert isinstance(builder.instructions[0], Pulse)
        assert builder.instructions[0].shape == PulseShapeType.SQUARE
        assert isinstance(builder.instructions[1], Pulse)
        assert builder.instructions[1].shape == MockPulseShapeType.HEXAGON
        assert isinstance(builder.instructions[2], CustomPulse)

    @pytest.mark.parametrize("shape1", list(PulseShapeType) + list(MockPulseShapeType))
    @pytest.mark.parametrize("shape2", list(PulseShapeType) + list(MockPulseShapeType))
    def test_hashing_for_different_shapes(self, shape1, shape2):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        pulse1 = Pulse(chan, shape1, width=400e-9)
        pulse2 = Pulse(chan, shape2, width=400e-9)
        pass_ = EvaluatePulses()
        if shape1 == shape2:
            assert pass_.hash_pulse(pulse1) == pass_.hash_pulse(pulse2)
        else:
            assert pass_.hash_pulse(pulse1) != pass_.hash_pulse(pulse2)

    def test_hashing_for_different_param(self):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        pulse1 = Pulse(chan, PulseShapeType.SQUARE, width=254e-9)
        pulse2 = Pulse(chan, PulseShapeType.SQUARE, width=454e-9)
        pass_ = EvaluatePulses()
        assert pass_.hash_pulse(pulse1) != pass_.hash_pulse(pulse2)

    def test_hashing_for_different_channels(self):
        model = EchoModelLoader().load()
        chan = model.qubits[0].get_drive_channel()
        chan2 = model.qubits[0].get_measure_channel()
        pulse1 = Pulse(chan, PulseShapeType.SQUARE, width=400e-9)
        pulse2 = Pulse(chan2, PulseShapeType.SQUARE, width=400e-9)
        pass_ = EvaluatePulses()
        assert pass_.hash_pulse(pulse1) != pass_.hash_pulse(pulse2)


@pytest.mark.parametrize("explicit_close", [False, True])
class TestRepeatTranslation:
    hw = EchoModelLoader(1).load()

    @staticmethod
    def _check_loop_start(ir: InstructionBuilder, indices: list[int]):
        for index in indices:
            # Create Variable
            assert isinstance(var := ir.instructions[index], Variable)
            assert var.var_type == LoopCount
            name_base = var.name.removesuffix("_count")

            # Assign with 0 to start
            assert isinstance(assign := ir.instructions[index + 1], Assign)
            assert name_base in assign.name
            assert assign.value == 0

            # Create Label
            assert isinstance(label := ir.instructions[index + 2], Label)
            assert name_base == label.name

    @staticmethod
    def _check_loop_close(ir: InstructionBuilder, indices: list[int], repeats: list[int]):
        for index, repeat in zip(indices, repeats):
            # Increment LoopCount Variable with 1
            assert isinstance(assign := ir.instructions[index], Assign)
            name_base = assign.name.removesuffix("_count")
            assert isinstance(plus := assign.value, Plus)
            assert isinstance(var := plus.left, Variable)
            assert name_base in var.name
            assert var.var_type == LoopCount
            assert plus.right == 1

            # Conditional Jump
            assert isinstance(jump := ir.instructions[index + 1], Jump)
            assert name_base == jump.target
            assert isinstance(condition := jump.condition, GreaterThan)
            assert condition.right == var
            assert condition.left == repeat

    def test_single_repeat(self, explicit_close):
        builder = self.hw.create_builder()
        builder.repeat(1000)

        if explicit_close:
            builder.add(EndRepeat())

        ir = RepeatTranslation(TargetData.default()).run(builder)

        assert len(ir.existing_names) == 1

        self._check_loop_start(ir, [0])
        self._check_loop_close(ir, [3], [1000])

    def test_multiple_repeat(self, explicit_close):
        builder = self.hw.create_builder()
        builder.repeat(1000).repeat(200)

        if explicit_close:
            builder.add([EndRepeat(), EndRepeat()])

        ir = RepeatTranslation(TargetData.default()).run(builder)

        assert len(ir.existing_names) == 2

        self._check_loop_start(ir, [0, 3])
        self._check_loop_close(ir, [8, 6], [1000, 200])

    @pytest.mark.parametrize("first", [0, 2])
    @pytest.mark.parametrize("second", [0, 3])
    @pytest.mark.parametrize("third", [0, 5])
    @pytest.mark.parametrize("fourth", [0, 1])
    @pytest.mark.parametrize("fifth", [0, 4])
    def test_with_other_instructions(
        self, explicit_close, first, second, third, fourth, fifth
    ):
        """Test all possible combinations of having additional instructions, before,
        between, and after the repeats, with and without explicitly closing scopes.

        [<first>, Repeat_a, <second>, Repeat_b, <third>, <close_b>, <fourth>, <close_a>, <fifth>]
        """
        builder = self.hw.create_builder()
        start_indices = [0, 3]
        close_indices = [8, 6]

        if first > 0:
            builder.add([Instruction()] * first)
            start_indices = [i + first for i in start_indices]
            close_indices = [i + first for i in close_indices]

        builder.repeat(1000)

        if second > 0:
            builder.add([Instruction()] * second)
            start_indices[1] += second
            close_indices = [i + second for i in close_indices]

        builder.repeat(300)

        if third > 0:
            builder.add([Instruction()] * third)
            close_indices = [i + third for i in close_indices]

        if explicit_close:
            builder.add(EndRepeat())

        if fourth > 0:
            builder.add([Instruction()] * fourth)
            if not explicit_close:
                close_indices[1] += fourth
            close_indices[0] += fourth

        if explicit_close:
            builder.add(EndRepeat())

        if fifth > 0:
            builder.add([Instruction()] * fifth)
            if not explicit_close:
                close_indices = [i + fifth for i in close_indices]

        ir = RepeatTranslation(TargetData.default()).run(builder)

        self._check_loop_start(ir, start_indices)
        self._check_loop_close(ir, close_indices, [1000, 300])


class TestRepeatSanitisation:
    @pytest.mark.parametrize("shots", [10, 100, 1000])
    @pytest.mark.parametrize("passive_reset_time", [1e-03, 2])
    @pytest.mark.parametrize("default", [True, False])
    def test_default_repeat(self, shots, passive_reset_time, default):
        model = EchoModelLoader().load()
        # TODO: Change to `passive_reset_time`. 428, 455
        qubit_data = QubitDescription.random().model_copy(
            update={"passive_reset_time": passive_reset_time}
        )
        target_data = TargetData(
            max_shots=1000,
            default_shots=shots,
            RESONATOR_DATA=ResonatorDescription.random(),
            QUBIT_DATA=qubit_data,
        )

        builder = QuantumInstructionBuilder(model).X(model.get_qubit(0))
        if not default:
            # To make sure the parameters are different from the default params.
            shots = shots - 1
            passive_reset_time = passive_reset_time - 1e-04
            builder.repeat(shots, passive_reset_time=passive_reset_time)
        else:
            default = target_data

        RepeatSanitisationValidation().run(builder)
        ir = RepeatSanitisation(model, target_data).run(builder)
        assert isinstance(repeat := ir.instructions[-1], Repeat)
        assert repeat.repeat_count == shots
        passive_reset_time
        assert repeat.passive_reset_time == passive_reset_time

    def test_repeat_instructions_are_sanitised(self):
        model = EchoModelLoader().load()
        target_data = TargetData.default()

        builder = model.create_builder()
        builder.X(model.qubits[0])
        builder.repeat(None, passive_reset_time=None)

        builder = RepeatSanitisation(model, target_data).run(builder)
        assert isinstance(repeat := builder.instructions[-1], Repeat)
        assert repeat.repeat_count == target_data.default_shots
        assert repeat.passive_reset_time == target_data.QUBIT_DATA.passive_reset_time


class TestFreqShiftSanitisation:
    @staticmethod
    def add_freq_shift_channels(hw, qubits: list[int] | int):
        """Adds frequency shift channels to given qubits and returns the set of new
        channels."""

        qubits = [qubits] if not isinstance(qubits, list) else qubits
        channels = set()
        for qubit_idx in qubits:
            qubit = hw.qubits[qubit_idx]
            freq_shift_pulse_ch = FreqShiftPulseChannel(
                id_=f"pulse_ch_Q{qubit_idx}", physical_channel=qubit.physical_channel
            )
            qubit.add_pulse_channel(
                freq_shift_pulse_ch, channel_type=ChannelType.freq_shift
            )
            channels.add(freq_shift_pulse_ch)
        return channels

    @staticmethod
    def get_classical_builder(hw):
        """Creates a builder with only classical instructions."""

        var = Variable("repeat_count", LoopCount)
        builder = hw.create_builder()
        builder.add(var)
        builder.assign("repeat_count", 0)
        builder.assign("repeat_count", Plus(var, 1))
        builder.returns("test")
        return builder

    @staticmethod
    def add_mock_x_and_measure(builder: QuantumInstructionBuilder, qubit):
        """Mocks an X gate and a measure for a builder."""

        drive_chan = qubit.get_drive_channel()
        measure_chan = qubit.get_measure_channel()
        acquire_chan = qubit.get_acquire_channel()
        builder.pulse(drive_chan, shape=PulseShapeType.SQUARE, width=40e-9)
        builder.pulse(drive_chan, shape=PulseShapeType.SQUARE, width=40e-9)
        builder.delay(measure_chan, 80e-9)
        builder.delay(acquire_chan, 128e-9)
        builder.pulse(measure_chan, shape=PulseShapeType.SQUARE, width=400e-9)
        builder.acquire(acquire_chan, delay=48e-9, time=352e-9)
        builder.delay(drive_chan, 400e-9)
        return builder

    def get_basic_builder(self, hw: QuantumHardwareModel):
        """Creates a basic builder free of control flow."""

        qubit = hw.qubits[0]
        builder = hw.create_builder()
        builder = self.add_mock_x_and_measure(builder, qubit)
        builder.returns("test")
        return builder

    def get_repeat_builder(self, hw):
        """Creates a builder with a repeat scope."""

        qubit = hw.qubits[0]
        builder = hw.create_builder()
        builder.add(Repeat(1000))
        builder.phase_shift(qubit, np.pi / 4)
        builder = self.add_mock_x_and_measure(builder, qubit)
        builder.add(EndRepeat())
        builder.returns("test")
        return builder

    def get_jump_builder(self, hw):
        """Creates a builder with a label and jump to replicate the behaviour of repeats."""

        qubit = hw.qubits[0]
        var = Variable("repeat_count", LoopCount)
        builder = hw.create_builder()
        builder.add(var)
        builder.assign("repeat_count", 0)
        builder.add(builder.create_label("repeats"))
        builder.phase_shift(qubit, np.pi / 4)
        builder = self.add_mock_x_and_measure(builder, qubit)
        builder.assign("repeat_count", Plus(var, 1))
        builder.jump("repeats", LessThan(var, 1000))
        builder.returns("test")
        return builder

    @pytest.mark.parametrize("num_qubits", [0, 1, 2, 3])
    def test_get_freq_shift_channels(self, num_qubits):
        hw = EchoModelLoader().load()
        channels = self.add_freq_shift_channels(hw, list(range(num_qubits)))
        found_channels = FreqShiftSanitisation(hw).get_freq_shift_channels()
        assert channels == set(found_channels.keys())
        assert set(found_channels.values()) == set(hw.qubits[0:num_qubits])

    def test_add_freq_shift_to_block_ignored_for_no_quantum_instructions(self):
        hw = EchoModelLoader().load()
        channels = self.add_freq_shift_channels(hw, [0, 1])
        builder = self.get_classical_builder(hw)
        builder = FreqShiftSanitisation(hw).add_freq_shift_to_ir(builder, channels)
        assert all([not isinstance(inst, Pulse) for inst in builder.instructions])

    def test_add_freq_shift_to_block_ignored_for_no_duration(self):
        hw = EchoModelLoader().load()
        channels = self.add_freq_shift_channels(hw, [0, 1])
        builder = hw.create_builder()
        qubit = hw.qubits[0]
        builder.phase_shift(qubit.get_drive_channel(), np.pi / 4)
        assert len(builder.instructions) == 1
        builder = FreqShiftSanitisation(hw).add_freq_shift_to_ir(builder, channels)
        assert len(builder.instructions) == 1
        assert isinstance(builder.instructions[0], PhaseShift)

    def test_active_results_updated(self):
        hw = EchoModelLoader().load()
        channels = self.add_freq_shift_channels(hw, [0, 1])
        qubits = hw.qubits[0:2]
        builder = hw.create_builder()
        builder.delay(qubits[0].get_drive_channel(), 80e-9)
        res = ActiveChannelResults(target_map={qubits[0].get_drive_channel(): qubits[0]})
        res_mgr = ResultManager()
        res_mgr.add(res)
        builder = FreqShiftSanitisation(hw).run(builder, res_mgr)
        for channel in channels:
            assert channel in res.target_map
            assert res.target_map[channel] in qubits

    @pytest.mark.parametrize("num_qubits", [1, 2, 3])
    @pytest.mark.parametrize(
        "factory", [get_basic_builder, get_repeat_builder, get_jump_builder]
    )
    def test_add_freq_shift(self, num_qubits, factory):
        """Considers builders that only have a single block with quantum instructions, and
        ensures the correct frequency shift pulses are added."""

        hw = EchoModelLoader().load()
        channels = self.add_freq_shift_channels(hw, list(range(num_qubits)))
        builder = factory(self, hw)
        pass_ = FreqShiftSanitisation(hw)
        builder = pass_.run(builder, ResultManager())

        first_quant_inst = [
            i
            for i, inst in enumerate(builder.instructions)
            if isinstance(inst, QuantumInstruction)
        ][0]

        for channel in channels:
            pulses = []
            for j, inst in enumerate(builder.instructions):
                if isinstance(inst, Pulse) and inst.channel == channel:
                    assert j >= first_quant_inst
                    pulses.append(inst)

            assert len(pulses) == 1
            assert pulses[0].duration == 480e-9

    def test_no_freq_shift_pulse_channel(self):
        hw = EchoModelLoader(8).load()
        builder = QuantumInstructionBuilder(hw)
        for qubit in hw.qubits:
            builder.X(qubit)

        ref_instructions = builder.instructions
        builder = FreqShiftSanitisation(hw).run(builder, ResultManager())

        # No instructions added since we do not have freq shift pulse channels.
        assert builder.instructions == ref_instructions

    def test_freq_shift_empty_target(self):
        hw = EchoModelLoader(8).load()
        res_mgr = ResultManager()
        ir = hw.create_builder()
        ir = FreqShiftSanitisation(hw).run(ir, res_mgr=res_mgr)
        assert len(ir.instructions) == 0


class TestLowerSyncsToDelays:
    def test_sync_with_two_channels(self):
        model = EchoModelLoader().load()
        chan1 = model.qubits[0].get_drive_channel()
        chan2 = model.qubits[1].get_drive_channel()

        builder = model.create_builder()
        builder.pulse(chan1, shape=PulseShapeType.SQUARE, width=120e-9)
        builder.delay(chan1, 48e-9)
        builder.delay(chan2, 72e-9)
        builder.pulse(chan2, shape=PulseShapeType.SQUARE, width=168e-9)
        builder.synchronize([chan1, chan2])

        LowerSyncsToDelays().run(builder)
        assert len(builder.instructions) == 5
        assert [type(inst) for inst in builder.instructions] == [
            Pulse,
            Delay,
            Delay,
            Pulse,
            Delay,
        ]
        times = [inst.duration for inst in builder.instructions]
        assert times[0] + times[1] + times[4] == times[2] + times[3]


class TestSquashDelaysOptimisation:
    @pytest.mark.parametrize("num_delays", [1, 2, 3, 4])
    @pytest.mark.parametrize("with_phase", [True, False])
    def test_multiple_delays_on_one_channel(self, num_delays, with_phase):
        delay_times = np.random.rand(num_delays)
        hw = EchoModelLoader().load()
        chan = hw.qubits[0].get_drive_channel()
        builder = hw.create_builder()
        for delay in delay_times:
            builder.delay(chan, delay)
            if with_phase:
                builder.phase_shift(chan, np.random.rand())

        builder = SquashDelaysOptimisation().run(builder, ResultManager(), MetricsManager())
        assert len(builder.instructions) == 1 + with_phase * num_delays
        delay_instructions = [
            inst for inst in builder.instructions if isinstance(inst, Delay)
        ]
        assert len(delay_instructions) == 1
        assert np.isclose(delay_instructions[0].time, sum(delay_times))

    @pytest.mark.parametrize("num_delays", [1, 2, 3, 4])
    @pytest.mark.parametrize("num_channels", [1, 2, 3, 4])
    @pytest.mark.parametrize("with_phase", [True, False])
    def test_multiple_delays_on_multiple_channels(
        self, num_delays, num_channels, with_phase
    ):
        hw = EchoModelLoader(num_channels).load()
        chans = [qubit.get_drive_channel() for qubit in hw.qubits]
        builder = hw.create_builder()
        accumulated_delays = defaultdict(float)
        for _ in range(num_delays):
            random.shuffle(chans)
            for chan in chans:
                delay = np.random.rand()
                accumulated_delays[chan] += delay
                builder.delay(chan, delay)
                if with_phase:
                    builder.phase_shift(chan, np.random.rand())

        builder = SquashDelaysOptimisation().run(builder, ResultManager(), MetricsManager())
        assert len(builder.instructions) == (1 + with_phase * num_delays) * num_channels
        delay_instructions = [
            inst for inst in builder.instructions if isinstance(inst, Delay)
        ]
        assert len(delay_instructions) == num_channels
        for delay in delay_instructions:
            assert delay.time == accumulated_delays[delay.quantum_targets[0]]

    def test_optimize_with_pulse(self):
        delay_times = np.random.rand(5)
        hw = EchoModelLoader().load()
        chan = hw.qubits[0].get_drive_channel()
        builder = hw.create_builder()
        builder.delay(chan, delay_times[0])
        builder.delay(chan, delay_times[1])
        builder.pulse(chan, width=80e-9, shape=PulseShapeType.SQUARE)
        builder.delay(chan, delay_times[2])
        builder.delay(chan, delay_times[3])
        builder.delay(chan, delay_times[4])
        builder = SquashDelaysOptimisation().run(builder, ResultManager(), MetricsManager())
        assert len(builder.instructions) == 3
        assert isinstance(builder.instructions[0], Delay)
        assert np.isclose(builder.instructions[0].time, np.sum(delay_times[0:2]))
        assert isinstance(builder.instructions[1], Pulse)
        assert isinstance(builder.instructions[2], Delay)
        assert np.isclose(builder.instructions[2].time, np.sum(delay_times[2:5]))

    def test_delay_with_multiple_channels(self):
        delay_times = np.random.rand(2)
        hw = EchoModelLoader().load()
        chan1 = hw.qubits[0].get_drive_channel()
        chan2 = hw.qubits[0].get_measure_channel()
        builder = hw.create_builder()
        builder.add(Delay([chan1, chan2], 5))
        builder.delay(chan1, delay_times[0])
        builder.delay(chan2, delay_times[1])
        builder = SquashDelaysOptimisation().run(builder, ResultManager(), MetricsManager())
        assert len(builder.instructions) == 2
        assert builder.instructions[0].time == 5 + delay_times[0]
        assert builder.instructions[1].time == 5 + delay_times[1]


class TestBatchedShots:
    model = EchoModelLoader().load()

    def test_with_no_repeat(self):
        builder = self.model.create_builder()
        builder.delay(self.model.qubits[0].get_drive_channel(), 80e-9)
        target_data = AbstractTargetData(max_shots=10000, default_shots=100)
        ir = BatchedShots(target_data).run(builder)
        assert len([inst for inst in ir.instructions if isinstance(inst, Repeat)]) == 0
        assert not hasattr(ir, "shots")
        assert not hasattr(ir, "compiled_shots")

    @pytest.mark.parametrize("num_shots", [1, 1000, 999, 10000])
    def test_not_batched_with_possible_amount(self, num_shots):
        builder = self.model.create_builder()
        builder.repeat(num_shots)
        target_data = AbstractTargetData(max_shots=10000)
        ir = BatchedShots(target_data).run(builder)
        repeats = [inst for inst in ir.instructions if isinstance(inst, Repeat)]
        assert len(repeats) == 1
        assert repeats[0].repeat_count == num_shots
        assert ir.shots == num_shots
        assert ir.compiled_shots == num_shots

    @pytest.mark.parametrize("num_shots", [1001, 1999, 2000, 4254])
    def test_shots_are_batched(self, num_shots):
        builder = self.model.create_builder()
        builder.repeat(num_shots)
        target_data = AbstractTargetData(max_shots=1000)
        ir = BatchedShots(target_data).run(builder)
        repeats = [inst for inst in ir.instructions if isinstance(inst, Repeat)]
        assert len(repeats) == 1
        assert repeats[0].repeat_count <= 1000
        assert ir.compiled_shots == repeats[0].repeat_count
        assert ir.shots == num_shots
        num_batches = np.ceil(ir.shots / ir.compiled_shots)
        assert num_batches * ir.compiled_shots >= num_shots
        assert (num_batches - 1) * ir.compiled_shots < num_shots


class TestScopeSanitisation:
    @pytest.mark.parametrize("num_repeats", [1, 2, 3])
    def test_repeats_are_shifted_to_the_beginning(self, num_repeats):
        model = EchoModelLoader().load()

        builder = model.create_builder()
        builder.X(model.qubits[0])
        for i in range(num_repeats):
            builder.repeat(i + 1, 100e-6)
        builder = ScopeSanitisation().run(builder)

        for i in range(num_repeats):
            assert isinstance(builder.instructions[i], Repeat)
            assert builder.instructions[i].repeat_count == i + 1

    @pytest.mark.parametrize("num_repeats", [1, 2, 3])
    def test_repeat_scopes_are_closed(self, num_repeats):
        model = EchoModelLoader().load()
        builder = model.create_builder()
        for i in range(num_repeats):
            builder.repeat(1000)
        builder.delay(model.qubits[0].get_drive_channel(), 80e-9)

        builder = ScopeSanitisation().run(builder)
        end_repeats = [inst for inst in builder.instructions if isinstance(inst, EndRepeat)]
        assert len(end_repeats) == num_repeats
        for j in range(num_repeats):
            assert isinstance(builder.instructions[-1 - j], EndRepeat)
