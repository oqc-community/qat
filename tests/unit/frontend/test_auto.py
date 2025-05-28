# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Oxford Quantum Circuits Ltd

import numpy as np
import pytest
from compiler_config.config import CompilerConfig, Languages, Tket

from qat.core.metrics_base import MetricsManager
from qat.core.result_base import ResultManager, ResultModel
from qat.frontend.auto import AutoFrontend
from qat.frontend.qasm import Qasm2Frontend, Qasm3Frontend, load_qasm_file
from qat.frontend.qir import QIRFrontend, load_qir_file
from qat.ir.instruction_builder import (
    QuantumInstructionBuilder as PydQuantumInstructionBuilder,
)
from qat.model.convert_legacy import convert_legacy_echo_hw_to_pydantic
from qat.model.loaders.legacy.echo import Connectivity, EchoModelLoader
from qat.purr.compiler.builders import QuantumInstructionBuilder
from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.instructions import Pulse, Repeat

from tests.unit.frontend.test_qasm import qasm3_tests
from tests.unit.utils.qasm_qir import (
    get_all_qasm2_paths,
    get_all_qir_paths,
    get_pulses_from_builder,
    get_qasm2,
    get_qasm3,
    get_qir,
)

# TODO: Update frontends to work with `Path`s, COMPILER-404
qasm2_tests = [str(path) for path in get_all_qasm2_paths()]
qasm3_tests = [str(path) for path in qasm3_tests]
qir_tests = [
    str(path)
    for path in get_all_qir_paths()
    if path.name not in ["base64_bitcode_ghz", "nonsense"]
]


class TestAutoFrontend:
    # The hardware model being required here isn't ideal. It's because on instantiation
    # of the parser, get_builder(model) is called. Something to resolve later.
    frontend = AutoFrontend(EchoModelLoader(32).load())

    @pytest.mark.parametrize("invalid_type", ["invalid", True, 3.14])
    def invalid_type(self, invalid_type):
        with pytest.raises(TypeError):
            AutoFrontend(invalid_type)

    @pytest.mark.parametrize("qasm2_path", qasm2_tests)
    def test_assign_frontend_qasm2(self, qasm2_path):
        # TODO: Update frontends to work with `Path`s, COMPILER-404
        assigned_frontend = self.frontend.assign_frontend(qasm2_path)
        assert isinstance(assigned_frontend, Qasm2Frontend)
        qasm2_str = load_qasm_file(qasm2_path)
        assigned_frontend = self.frontend.assign_frontend(qasm2_str)
        assert isinstance(assigned_frontend, Qasm2Frontend)

    @pytest.mark.parametrize("qasm3_path", qasm3_tests)
    def test_assign_frontend_qasm3(self, qasm3_path):
        # TODO: Update frontends to work with `Path`s, COMPILER-404
        assigned_frontend = self.frontend.assign_frontend(qasm3_path)
        assert isinstance(assigned_frontend, Qasm3Frontend)
        qasm3_str = load_qasm_file(qasm3_path)
        assigned_frontend = self.frontend.assign_frontend(qasm3_str)
        assert isinstance(assigned_frontend, Qasm3Frontend)

    @pytest.mark.parametrize("qir_path", qir_tests)
    def test_assign_frontend_qir(self, qir_path):
        # TODO: Update frontends to work with `Path`s, COMPILER-404
        assigned_frontend = self.frontend.assign_frontend(qir_path)
        assert isinstance(assigned_frontend, QIRFrontend)
        qir_str = load_qir_file(qir_path)
        assigned_frontend = self.frontend.assign_frontend(qir_str)
        assert isinstance(assigned_frontend, QIRFrontend)

    @pytest.mark.parametrize(
        "program",
        [get_qasm2("basic.qasm"), get_qasm3("basic.qasm"), get_qir("bell_psi_plus.ll")],
    )
    def test_check_and_return(self, program):
        """Tests that the frontend can check and return a source program."""
        assert self.frontend.check_and_return_source(program) == program

    @pytest.mark.parametrize(
        "program",
        [
            get_qasm2("nonsense/nonsense.qasm"),
            get_qasm3("nonsense/nonsense.qasm"),
            get_qir("nonsense/nonsense.ll"),
        ],
    )
    def test_check_and_return_with_bad_file(self, program):
        """Tests that the frontend can check and return a source program."""
        assert self.frontend.check_and_return_source(program) is False

    @pytest.mark.parametrize(
        "program",
        [get_qasm2("basic.qasm"), get_qasm3("basic.qasm"), get_qir("bell_psi_plus.ll")],
    )
    def test_emit(self, program):
        """Tests frontend-relevant details, such as successful parsing. Doesn't check the
        details of the IR as this is the responsibility of the respective parser tests."""
        builder = self.frontend.emit(program)
        assert isinstance(builder, QuantumInstructionBuilder)
        assert isinstance(builder.instructions[0], Repeat)

    @pytest.mark.parametrize(
        "reader, ext", [(get_qasm2, ".qasm"), (get_qasm3, ".qasm"), (get_qir, ".ll")]
    )
    @pytest.mark.parametrize(
        "program, n_gaussian, n_soft_square",
        [("bell_psi_plus", 6, 4), ("basic", 2, 0)],
    )
    def test_emit_contains_correct_default_gates(
        self, program, n_gaussian, n_soft_square, reader, ext
    ):
        """Tests frontend-relevant details, such as successful parsing. Doesn't check the
        details of the IR as this is the responsibility of the respective parser tests."""
        compiler_config = CompilerConfig(optimizations=Tket().disable())
        builder = self.frontend.emit(reader(program + ext), compiler_config=compiler_config)

        soft_square_pulses = get_pulses_from_builder(
            builder, shape_type=PulseShapeType.SOFT_SQUARE
        )
        gaussian_pulses = get_pulses_from_builder(
            builder, shape_type=PulseShapeType.GAUSSIAN
        )
        pulse_amp = 0.25 / (100e-9 * 1.0 / 3.0 * np.pi**0.5)
        assert len(soft_square_pulses) == n_soft_square
        assert len(gaussian_pulses) == n_gaussian
        if n_soft_square > 0:
            assert any(
                (p.amp == 1000000.0 and p.width == 1.25e-07) for p in soft_square_pulses
            )
        assert any((p.amp == pulse_amp and p.width == 100e-9) for p in gaussian_pulses)

    @pytest.mark.parametrize(
        "program",
        [
            get_qasm2("nonsense/nonsense.qasm"),
            get_qasm3("nonsense/nonsense.qasm"),
            get_qir("nonsense/nonsense.ll"),
        ],
    )
    def test_emit_raises_error_with_bad_file(self, program):
        with pytest.raises(ValueError):
            self.frontend.emit(program)

    @pytest.mark.parametrize("program", ["basic", "bell_psi_plus"])
    def test_basic_circuits_generate_the_same_instruction_list_for_all_ir_types(
        self, program
    ):
        compiler_config = CompilerConfig(optimizations=Tket().disable())
        builder_qasm2 = self.frontend.emit(
            get_qasm2(program + ".qasm"), compiler_config=compiler_config
        )
        builder_qasm3 = self.frontend.emit(get_qasm3(program + ".qasm"))
        builder_qir = self.frontend.emit(get_qir(program + ".ll"))

        assert len(builder_qasm2.instructions) == len(builder_qasm3.instructions)
        assert len(builder_qasm2.instructions) == len(builder_qir.instructions)

        for inst_qasm2, inst_qasm3, inst_qir in zip(
            builder_qasm2.instructions, builder_qasm3.instructions, builder_qir.instructions
        ):
            assert type(inst_qasm2) == type(inst_qasm3)
            assert type(inst_qasm2) == type(inst_qir)
            if isinstance(inst_qasm2, Pulse):
                assert inst_qasm2.shape == inst_qasm3.shape
                assert inst_qasm2.shape == inst_qir.shape
                assert inst_qasm2.amp == inst_qasm3.amp
                assert inst_qasm2.amp == inst_qir.amp
                assert inst_qasm2.width == inst_qasm3.width
                assert inst_qasm2.width == inst_qir.width

    @pytest.mark.parametrize(
        "program",
        [get_qasm2("basic.qasm"), get_qasm3("basic.qasm"), get_qir("bell_psi_plus.ll")],
    )
    def test_legacy_vs_pydantic_hw(self, program):
        leg_hw = EchoModelLoader(32, connectivity=Connectivity.Ring).load()
        pyd_hw = convert_legacy_echo_hw_to_pydantic(leg_hw)

        leg_frontend = AutoFrontend(leg_hw)
        leg_builder = leg_frontend.emit(program)
        assert isinstance(leg_builder, QuantumInstructionBuilder)
        assert len(leg_builder.instructions) > 0

        pyd_frontend = AutoFrontend(pyd_hw)
        pyd_builder = pyd_frontend.emit(program)
        assert isinstance(pyd_builder, PydQuantumInstructionBuilder)
        assert pyd_builder.number_of_instructions > 0

    @pytest.mark.parametrize("reader, ext", [(get_qasm2, ".qasm"), (get_qasm3, ".qasm")])
    @pytest.mark.parametrize("program", ["bell_psi_plus", "basic"])
    def test_metrics_manager_collects_metrics(self, reader, ext, program):
        compiler_config = CompilerConfig(optimizations=Tket().disable())
        metrics_manager = MetricsManager()
        ir_str = reader(program + ext)

        assert metrics_manager.optimized_circuit is None
        assert metrics_manager.optimized_instruction_count is None
        self.frontend.emit(ir_str, met_mgr=metrics_manager, compiler_config=compiler_config)
        assert metrics_manager.optimized_circuit == ir_str
        assert metrics_manager.optimized_instruction_count is None

    @pytest.mark.parametrize(
        "reader, ext, lang_type",
        [
            (get_qasm2, ".qasm", Languages.Qasm2),
            (get_qasm3, ".qasm", Languages.Qasm3),
        ],
    )
    @pytest.mark.parametrize("program", ["bell_psi_plus", "basic"])
    def test_results_manager_collects_results(self, reader, ext, lang_type, program):
        compiler_config = CompilerConfig(optimizations=Tket().disable())
        ir_str = reader(program + ext)
        result_manager = ResultManager()
        assert len(result_manager.results) == 0
        self.frontend.emit(ir_str, res_mgr=result_manager, compiler_config=compiler_config)
        assert len(result_manager.results) == 1
        result = list(result_manager.results)[0]
        assert isinstance(result, ResultModel)
        assert result.value.language == lang_type
        assert result.value.raw_input == ir_str
