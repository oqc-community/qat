# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd
from copy import deepcopy

import numpy as np
import pytest
from compiler_config.config import CompilerConfig

from qat.core.metrics_base import MetricsManager
from qat.core.pass_base import AnalysisPass, PassManager, TransformPass, ValidationPass
from qat.core.result_base import ResultInfoMixin, ResultManager
from qat.middleend.middleends import CustomMiddleend, DefaultMiddleend, FallthroughMiddleend
from qat.middleend.passes.analysis import ActiveChannelResults
from qat.model.loaders.legacy import EchoModelLoader
from qat.model.target_data import TargetData
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.instructions import (
    CustomPulse,
    Delay,
    Jump,
    Label,
    PhaseSet,
    Pulse,
    PulseShapeType,
    QuantumInstruction,
    Repeat,
    Return,
    Synchronize,
)


class MockAnalysisResult(ResultInfoMixin):
    def __init__(self, value):
        self.value = value


class MockAnalysisPass(AnalysisPass):
    def run(self, ir, res_mgr, *args, **kwargs):
        res_mgr.add(MockAnalysisResult(ir))
        return ir


class MockTransformPass(TransformPass):
    def run(self, ir, *args, **kwargs):
        return 254


class MockValidationPass(ValidationPass):
    def run(self, ir, *args, **kwargs):
        if ir == 454:
            raise ValueError("testing...")
        return ir


class ValidateCompilerConfig(ValidationPass):
    def run(self, ir, *args, compiler_config, **kwargs):
        if not isinstance(compiler_config, CompilerConfig):
            raise ValueError("Invalid compiler configuration provided.")
        return ir


class TestCustomMiddleend:
    """Tests that the pipeline in the custom middleeend executes as expected with mock
    passes."""

    def mock_middleend(self):
        """Factory to create a mock CustomMiddleend."""
        pass_manager = PassManager()
        pass_manager.add(MockAnalysisPass())
        pass_manager.add(MockValidationPass())
        pass_manager.add(MockTransformPass())
        pass_manager.add(ValidateCompilerConfig())
        return CustomMiddleend(model=EchoModelLoader().load(), pipeline=pass_manager)

    def test_custom_middleend_with_anlysis_pass(self):
        """Test that the custom middleend runs an analysis pass and saves the result in
        the result manager."""
        middleend = self.mock_middleend()
        ir = 42
        res_mgr = ResultManager()
        middleend.emit(ir, res_mgr)
        analysis_result = res_mgr.lookup_by_type(MockAnalysisResult)
        assert analysis_result.value == ir

    def test_custom_middleend_with_transform_pass(self):
        """Test that the custom middleend runs the transform pass, altering the IR."""
        middleend = self.mock_middleend()
        ir = 42
        res_mgr = ResultManager()
        transformed_ir = middleend.emit(ir, res_mgr)
        assert transformed_ir == 254

    def test_custom_middleend_with_validation_pass(self):
        """Test that the custom middleend runs the validation pass and raises an error."""
        middleend = self.mock_middleend()
        ir = 454
        res_mgr = ResultManager()
        with pytest.raises(ValueError, match="testing..."):
            middleend.emit(ir, res_mgr)

    def test_custom_middleend_with_wrong_compiler_config(self):
        """Test that the custom middleend runs the validation pass with no compiler config."""
        middleend = self.mock_middleend()
        ir = 42
        res_mgr = ResultManager()
        with pytest.raises(ValueError, match="Invalid compiler configuration provided."):
            middleend.emit(ir, res_mgr, compiler_config=5)


class TestDefaultMiddleend:
    """Some basic sanity checks on the DefaultMiddleend to ensure passes integrate to give
    a valid pipeline, and passes run as expected. It doesn't do any deep testing of the
    passes themselves, as those are tested in their own unit tests.

    The DefaultMiddleend is likely to change, so this test is more of a smoke test that
    checks passes that synergise work as expected."""

    @pytest.fixture(scope="class")
    def model(self):
        return EchoModelLoader().load()

    @pytest.fixture(scope="class")
    def middleend(self, model):
        target_data = TargetData.default()
        target_data_blob = target_data.model_dump()
        target_data_blob["QUBIT_DATA"]["passive_reset_time"] = 1e-3
        target_data_blob["QUBIT_DATA"]["pulse_duration_max"] = 1e-4
        target_data = TargetData(**target_data_blob)
        return DefaultMiddleend(model=model, target_data=target_data)

    @pytest.fixture(scope="class")
    def preprocessed_ir(self, model):
        """Factory to create a mock instruction builder."""

        builder = model.create_builder()
        builder.X(model.qubits[0])
        builder.Z(model.qubits[0], np.pi / 4)
        builder.delay(model.qubits[1].get_drive_channel(), 120e-9)
        builder.cnot(model.qubits[0], model.qubits[1])
        builder.Z(model.qubits[0], -np.pi / 4)
        builder.Z(model.qubits[1], 3 * np.pi / 8)
        builder.U(model.qubits[0], np.pi / 2, 0, 0)
        builder.SX(model.qubits[1])
        builder.measure(model.qubits[0])
        builder.measure(model.qubits[1])
        return builder

    @pytest.fixture(scope="class")
    def res_mgr(self) -> ResultManager:
        return ResultManager()

    @pytest.fixture(scope="class")
    def met_mgr(self) -> MetricsManager:
        return MetricsManager()

    @pytest.fixture(scope="class")
    def processed_ir(self, middleend, preprocessed_ir, res_mgr, met_mgr):
        """Fixture to process the IR through the middleend to be reused in tests."""
        return middleend.emit(deepcopy(preprocessed_ir), res_mgr, met_mgr)

    def check_ir_has_type(self, ir, inst_type):
        """Helper function to check if the IR contains an instruction of a specific type."""
        return any(isinstance(inst, inst_type) for inst in ir.instructions)

    def test_default_middleend_runs_passes(self, processed_ir):
        """Test that the default middleend runs the passes in the pipeline."""
        assert isinstance(processed_ir, InstructionBuilder)

    def test_active_channel_result(self, processed_ir, res_mgr):
        """Test that the default middleend runs the ActivePulseChannelAnalysis pass and
        stores the result in the result manager, and non active channels are sanitised
        away."""
        active_channels = res_mgr.lookup_by_type(ActiveChannelResults)
        assert isinstance(active_channels, ActiveChannelResults)
        for inst in processed_ir.instructions:
            if isinstance(inst, QuantumInstruction):
                for target in inst.quantum_targets:
                    assert target in active_channels.targets

    def test_ir_has_return(self, preprocessed_ir, processed_ir):
        """Test that the default middleend adds a return instruction."""
        assert not self.check_ir_has_type(preprocessed_ir, Return)
        assert self.check_ir_has_type(processed_ir, Return)

    def test_ir_has_repeats(self, preprocessed_ir, processed_ir):
        """Test that the default middleend adds a repeat instruction and it is lowered."""
        for type in (Jump, Label, Repeat):
            assert not self.check_ir_has_type(preprocessed_ir, type)
        for type in (Jump, Label):
            assert self.check_ir_has_type(processed_ir, type)

    def test_ir_has_phase_set(self, preprocessed_ir, processed_ir):
        """Test that the default middleend adds a phase reset which is altered by phase
        optimisation."""
        assert not self.check_ir_has_type(preprocessed_ir, PhaseSet)
        assert self.check_ir_has_type(processed_ir, PhaseSet)

    def test_pulses_are_lowered(self, preprocessed_ir, processed_ir):
        """Test that the default middleend lowers pulses to the appropriate channels."""

        assert not self.check_ir_has_type(preprocessed_ir, CustomPulse)
        assert any(
            isinstance(inst, Pulse) and inst.shape != PulseShapeType.SQUARE
            for inst in preprocessed_ir.instructions
        )

        assert self.check_ir_has_type(processed_ir, CustomPulse)
        for inst in processed_ir.instructions:
            if isinstance(inst, Pulse):
                assert inst.shape == PulseShapeType.SQUARE

    def test_ir_has_no_syncs(self, preprocessed_ir, processed_ir):
        """Test that the default middleend does not add any sync instructions, as they
        are lowered to delays."""
        assert self.check_ir_has_type(preprocessed_ir, Synchronize)
        assert not self.check_ir_has_type(processed_ir, Synchronize)

    def test_passive_resets(self, middleend, processed_ir, res_mgr):
        """Tests that resets are added, lowered to delays and the instructions are split
        split up."""
        reset_time = middleend.target_data.QUBIT_DATA.passive_reset_time
        max_length = middleend.target_data.QUBIT_DATA.pulse_duration_max
        result = res_mgr.lookup_by_type(ActiveChannelResults)
        assert np.floor(reset_time / max_length) > 1
        expected_num_delays = len(result.targets) * np.floor(reset_time / max_length)
        actual_num_delays = sum(
            [
                isinstance(inst, Delay) and np.isclose(inst.duration, max_length)
                for inst in processed_ir.instructions
            ]
        )
        # less than equal because other delays from the circuit might contribute
        assert expected_num_delays <= actual_num_delays

    def test_metric_manager(self, processed_ir, met_mgr):
        """Test that the metrics manager is populated with the expected metrics."""
        # only instructions that "optimize" update this, but instructions that altar the IR
        # and the number of instructions are not accounted for. Could be handled with a
        # mixin of explicit pass type that specifies the number of instructions might be
        # altered
        assert met_mgr.optimized_instruction_count != None

    def test_delay_and_sync(self, model, middleend):
        """Regression test to test that a Delay on an inactive channel, followed by a sync
        with an active channel can be used to delay the active channel."""

        chan1 = model.qubits[0].get_drive_channel()
        chan2 = model.qubits[1].get_drive_channel()
        builder = model.create_builder()
        builder.delay(chan1, 80e-9)
        builder.synchronize([chan1, chan2])
        builder.pulse(chan2, width=160e-9, shape=PulseShapeType.SQUARE)
        ir = middleend.emit(builder)

        instructions = [
            inst
            for inst in ir.instructions
            if isinstance(inst, QuantumInstruction) and chan2 in inst.quantum_targets
        ]
        assert isinstance(instructions[1], Delay)
        assert instructions[1].duration == 80e-9
        assert isinstance(instructions[2], Pulse)
        assert instructions[2].duration == 160e-9


class TestFallthroughMiddleend:
    model = EchoModelLoader().load()

    @pytest.mark.parametrize("ir", [2.54, "test", None, model.create_builder()])
    def test_fallthrough_middleend_does_not_alter_ir(self, ir):
        """Test that the FallthroughMiddleend does not alter the input IR."""
        middleend = FallthroughMiddleend(model=self.model)
        res_mgr = ResultManager()
        met_mgr = MetricsManager()
        processed_ir = middleend.emit(ir, res_mgr)
        assert processed_ir is ir
        assert met_mgr.optimized_circuit is None
        assert met_mgr.optimized_instruction_count is None
        assert res_mgr.results == set()
